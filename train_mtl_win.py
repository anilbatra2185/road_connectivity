from __future__ import print_function

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from model.models import MODELS
from road_dataset import DeepGlobeDataset, SpacenetDataset
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils.loss import CrossEntropyLoss2d, mIoULoss
from utils import util
from utils import viz_util
from multiprocessing import freeze_support

__dataset__ = {"spacenet": SpacenetDataset, "deepglobe": DeepGlobeDataset}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", required=True, type=str, help="config file path"
)
parser.add_argument(
    "--model_name",
    required=True,
    choices=sorted(MODELS.keys()),
    help="Name of Model = {}".format(MODELS.keys()),
)
parser.add_argument("--exp", required=True, type=str,
                    help="Experiment Name/Directory")
parser.add_argument(
    "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
)
parser.add_argument(
    "--dataset",
    required=True,
    choices=sorted(__dataset__.keys()),
    help="select dataset name from {}. (default: Spacenet)".format(
        __dataset__.keys()),
)
parser.add_argument(
    "--model_kwargs",
    default={},
    type=json.loads,
    help="parameters for the model",
)
parser.add_argument(
    "--multi_scale_pred",
    default=True,
    type=util.str2bool,
    help="perform multi-scale prediction (default: True)",
)


def initialize():
    args = parser.parse_args()
    config = None

    if args.resume is not None:
        if args.config is not None:
            print("Warning: --config overridden by --resume")
            config = torch.load(args.resume)["config"]
            config["train_batch_size"] = 32
    elif args.config is not None:
        config = json.load(open(args.config))

    assert config is not None

    util.setSeed(config)
    return args, config


def get_metric_files(args, config):
    experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)
    util.ensure_dir(experiment_dir)

    # Logging Files
    train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
    test_file = "{}/{}_test_loss.txt".format(experiment_dir, args.dataset)

    train_loss_file = open(train_file, "w")
    val_loss_file = open(test_file, "w")

    # Angle Metrics
    train_file_angle = "{}/{}_train_angle_loss.txt".format(
        experiment_dir, args.dataset)
    test_file_angle = "{}/{}_test_angle_loss.txt".format(
        experiment_dir, args.dataset)

    train_loss_angle_file = open(train_file_angle, "w")
    val_loss_angle_file = open(test_file_angle, "w")

    return [experiment_dir,
            train_loss_file,
            val_loss_file,
            train_loss_angle_file,
            val_loss_angle_file]


def get_model_optimizer(args, config):
    global best_miou
    start_epoch, best_miou = 1, 0.0
    num_gpus = torch.cuda.device_count()

    model = MODELS[args.model_name](
        config["task1_classes"], config["task2_classes"], **args.model_kwargs
    )
    if num_gpus > 1:
        print("Training with multiple GPUs ({})".format(num_gpus))
        model = nn.DataParallel(model)
    else:
        print("Single Cuda Node is avaiable")

    if args.resume is not None:
        print("Loading from existing FCN and copying weights to continue....")
        checkpoint = torch.load(
            args.resume, map_location=lambda storage, loc: storage)
        start_epoch = checkpoint["epoch"] + 1
        best_miou = checkpoint["miou"]
        model.load_state_dict(checkpoint["state_dict"])
        model = model.cuda()
        optimizer = optim.SGD(
            model.parameters(), lr=config["optimizer"]["lr"],
            momentum=0.9, weight_decay=0.0005
        )
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        util.weights_init(model, manual_seed=config["seed"])
        model = model.cuda()
        optimizer = optim.SGD(
            model.parameters(), lr=config["optimizer"]["lr"],
            momentum=0.9, weight_decay=0.0005
        )
    
    viz_util.summary(model, print_arch=False)

    return {"model": model,
            "start_epoch": start_epoch,
            "best_iou": best_miou,
            "optimizer": optimizer}


def get_loaders(args, config):
    # Load Dataset from root folder and intialize DataLoader
    train_loader = data.DataLoader(
        __dataset__[args.dataset](
            config["train_dataset"],
            seed=config["seed"],
            is_train=True,
            multi_scale_pred=args.multi_scale_pred,
        ),
        batch_size=config["train_batch_size"],
        num_workers=8,
        shuffle=True,
        pin_memory=False,
    )

    val_loader = data.DataLoader(
        __dataset__[args.dataset](
            config["val_dataset"],
            seed=config["seed"],
            is_train=False,
            multi_scale_pred=args.multi_scale_pred,
        ),
        batch_size=config["val_batch_size"],
        num_workers=8,
        shuffle=True,
        pin_memory=False,
    )
    print("Training with dataset => {}".format(
        train_loader.dataset.__class__.__name__))
    return train_loader, val_loader


def train(model, optimizer, train_loader, road_loss, angle_loss,
          train_loss_file, train_loss_angle_file, epoch):

    train_loss_iou = 0
    train_loss_vec = 0
    model.train()
    optimizer.zero_grad()
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["train_dataset"][args.dataset]["crop_size"]
    for i, data in enumerate(train_loader):
        inputsBGR, labels, vecmap_angles = data
        inputsBGR = Variable(inputsBGR.float().cuda())
        outputs, pred_vecmaps = model(inputsBGR)

        if args.multi_scale_pred:
            loss1 = road_loss(outputs[0], util.to_variable(labels[0]), False)
            num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
            for idx in range(num_stacks - 1):
                loss1 += road_loss(outputs[idx + 1],
                                   util.to_variable(labels[0]), False)
            for idx, output in enumerate(outputs[-2:]):
                loss1 += road_loss(output,
                                   util.to_variable(labels[idx + 1]), False)

            loss2 = angle_loss(
                pred_vecmaps[0], util.to_variable(vecmap_angles[0]))
            for idx in range(num_stacks - 1):
                loss2 += angle_loss(
                    pred_vecmaps[idx + 1], util.to_variable(vecmap_angles[0])
                )
            for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
                loss2 += angle_loss(pred_vecmap,
                                    util.to_variable(vecmap_angles[idx + 1]))

            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]
        else:
            loss1 = road_loss(outputs, util.to_variable(labels[-1]), False)
            loss2 = angle_loss(
                pred_vecmaps, util.to_variable(vecmap_angles[-1]))

        train_loss_iou += loss1.data.item()
        train_loss_vec += loss2.data.item()

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )

        _, predicted_angle = torch.max(pred_vecmaps.data, 1)
        correct_angles = vecmap_angles[-1].view(-1,
                                                crop_size, crop_size).long()
        hist_angles += util.fast_hist(
            predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
            correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
            config["task2_classes"],
        )

        _, miou, road_iou, _ = util.performMetrics(
            train_loss_file,
            None,
            epoch,
            hist,
            train_loss_iou / (i + 1),
            train_loss_vec / (i + 1),
        )
        _, miou_angle, _ = util.performAngleMetrics(
            train_loss_angle_file, None, epoch, hist_angles
        )

        viz_util.progress_bar(
            i,
            len(train_loader),
            "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%% "
            % (
                train_loss_iou / (i + 1),
                train_loss_vec / (i + 1),
                miou,
                road_iou,
                miou_angle,
            ),
        )

        torch.autograd.backward([loss1, loss2])

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        del (
            outputs,
            pred_vecmaps,
            predicted,
            correct_angles,
            correctLabel,
            inputsBGR,
            labels,
            vecmap_angles,
        )

    util.performMetrics(
        train_loss_file,
        None,
        epoch,
        hist,
        train_loss_iou / len(train_loader),
        train_loss_vec / len(train_loader),
        write=True,
    )
    util.performAngleMetrics(
        train_loss_angle_file, None, epoch, hist_angles, write=True
    )


def test(model, optimizer, val_loader, road_loss, angle_loss,
         val_loss_file, val_loss_angle_file, experiment_dir, epoch):

    global best_accuracy
    global best_miou
    model.eval()
    test_loss_iou = 0
    test_loss_vec = 0
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    hist_angles = np.zeros((config["task2_classes"], config["task2_classes"]))
    crop_size = config["val_dataset"][args.dataset]["crop_size"]
    for i, (inputsBGR, labels, vecmap_angles) in enumerate(val_loader, 0):
        inputsBGR = Variable(
            inputsBGR.float().cuda(), volatile=True, requires_grad=False
        )
        with torch.no_grad():
            outputs, pred_vecmaps = model(inputsBGR)
        if args.multi_scale_pred:
            loss1 = road_loss(outputs[0], util.to_variable(
                labels[0], True, False), True)
            num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
            for idx in range(num_stacks - 1):
                loss1 += road_loss(outputs[idx + 1],
                                   util.to_variable(labels[0], True, False), True)
            for idx, output in enumerate(outputs[-2:]):
                loss1 += road_loss(output, util.to_variable(labels[idx + 1], True, False),
                                   True)

            loss2 = angle_loss(pred_vecmaps[0], util.to_variable(
                vecmap_angles[0], True, False))
            for idx in range(num_stacks - 1):
                loss2 += angle_loss(
                    pred_vecmaps[idx + 1],
                    util.to_variable(vecmap_angles[0], True, False)
                )
            for idx, pred_vecmap in enumerate(pred_vecmaps[-2:]):
                loss2 += angle_loss(
                    pred_vecmap, util.to_variable(
                        vecmap_angles[idx + 1], True, False)
                )

            outputs = outputs[-1]
            pred_vecmaps = pred_vecmaps[-1]
        else:
            loss1 = road_loss(outputs, util.to_variable(
                labels[0], True, False), True)
            loss2 = angle_loss(
                pred_vecmaps, util.to_variable(labels[0], True, False))

        test_loss_iou += loss1.data.item()
        test_loss_vec += loss2.data.item()

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )

        _, predicted_angle = torch.max(pred_vecmaps.data, 1)
        correct_angles = vecmap_angles[-1].view(-1,
                                                crop_size, crop_size).long()
        hist_angles += util.fast_hist(
            predicted_angle.view(predicted_angle.size(0), -1).cpu().numpy(),
            correct_angles.view(correct_angles.size(0), -1).cpu().numpy(),
            config["task2_classes"],
        )

        _, miou, road_iou, _ = util.performMetrics(
            None,
            val_loss_file,
            epoch,
            hist,
            test_loss_iou / (i + 1),
            test_loss_vec / (i + 1),
            is_train=False,
        )
        _, miou_angle, _ = util.performAngleMetrics(
            None, val_loss_angle_file, epoch, hist_angles, is_train=False
        )

        viz_util.progress_bar(
            i,
            len(val_loader),
            "Loss: %.6f | VecLoss: %.6f | road miou: %.4f%%(%.4f%%) | angle miou: %.4f%%"
            % (
                test_loss_iou / (i + 1),
                test_loss_vec / (i + 1),
                miou,
                road_iou,
                miou_angle,
            ),
        )

        if i % 100 == 0 or i == len(val_loader) - 1:
            images_path = "{}/images/".format(experiment_dir)
            util.ensure_dir(images_path)
            util.savePredictedProb(
                inputsBGR.data.cpu(),
                labels[-1].cpu(),
                predicted.cpu(),
                F.softmax(outputs, dim=1).data.cpu()[:, 1, :, :],
                predicted_angle.cpu(),
                os.path.join(
                    images_path, "validate_pair_{}_{}.png".format(epoch, i)),
                norm_type=config["val_dataset"]["normalize_type"],
            )

        del inputsBGR, labels, predicted, outputs, pred_vecmaps, predicted_angle

    accuracy, miou, road_iou, _ = util.performMetrics(
        None,
        val_loss_file,
        epoch,
        hist,
        test_loss_iou / len(val_loader),
        test_loss_vec / len(val_loader),
        is_train=False,
        write=True,
    )
    util.performAngleMetrics(
        None,
        val_loss_angle_file,
        epoch,
        hist_angles,
        is_train=False,
        write=True,
    )

    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        util.save_checkpoint(epoch, test_loss_iou / len(val_loader), model,
                             optimizer, best_accuracy, best_miou, config, experiment_dir)

    return test_loss_iou / len(val_loader)


if __name__ == '__main__':
    freeze_support()
    best_accuracy = 0.0
    args, config = initialize()

    total_epochs = config["trainer"]["total_epochs"]
    model_optim_dict = get_model_optimizer(args, config)
    model = model_optim_dict["model"]
    optimizer = model_optim_dict["optimizer"]
    start_epoch = model_optim_dict["start_epoch"]
    train_loader, valid_loader = get_loaders(args, config)
    exp_dir, t_loss, v_loss, t_loss_angle, v_loss_angle = get_metric_files(
        args, config)
    scheduler = MultiStepLR(
        optimizer,
        milestones=eval(config["optimizer"]["lr_drop_epoch"]),
        gamma=config["optimizer"]["lr_step"],
    )

    weights = torch.ones(config["task1_classes"]).cuda()
    if config["task1_weight"] < 1:
        print("Roads are weighted.")
        weights[0] = 1 - config["task1_weight"]
        weights[1] = config["task1_weight"]

    weights_angles = torch.ones(config["task2_classes"]).cuda()
    if config["task2_weight"] < 1:
        print("Road angles are weighted.")
        weights_angles[-1] = config["task2_weight"]

    angle_loss = CrossEntropyLoss2d(
        weight=weights_angles, size_average=True, ignore_index=255, reduce=True
    ).cuda()
    road_loss = mIoULoss(
        weight=weights, size_average=True, n_classes=config["task1_classes"]
    ).cuda()

    for epoch in range(start_epoch, total_epochs + 1):
        start_time = datetime.now()
        print("\nTraining Epoch: %d" % epoch)
        train(model, optimizer, train_loader, road_loss,
              angle_loss, t_loss, t_loss_angle, epoch)
        if epoch % config["trainer"]["test_freq"] == 0:
            print("\nTesting Epoch: %d" % epoch)
            val_loss = test(model, optimizer, valid_loader, road_loss,
                            angle_loss, v_loss, v_loss_angle, exp_dir, epoch)
        scheduler.step()
        end_time = datetime.now()
        print("Time Elapsed for epoch {0} => {1}".format(
            epoch, end_time - start_time))
