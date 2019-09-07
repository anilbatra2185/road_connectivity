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
from model.models import MODELS_REFINE
from road_dataset import DeepGlobeDatasetCorrupt, SpacenetDatasetCorrupt
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
from utils.loss import CrossEntropyLoss2d, mIoULoss
from utils import util
from utils import viz_util


__dataset__ = {"spacenet": SpacenetDatasetCorrupt, "deepglobe": DeepGlobeDatasetCorrupt}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", required=True, type=str, help="config file path"
)
parser.add_argument(
    "--model_name",
    required=True,
    choices=sorted(MODELS_REFINE.keys()),
    help="Name of Model = {}".format(MODELS_REFINE.keys()),
)
parser.add_argument("--exp", required=True, type=str, help="Experiment Name/Directory")
parser.add_argument(
    "--resume", default=None, type=str, help="path to latest checkpoint (default: None)"
)
parser.add_argument(
    "--dataset",
    required=True,
    choices=sorted(__dataset__.keys()),
    help="select dataset name from {}. (default: Spacenet)".format(__dataset__.keys()),
)
parser.add_argument(
    "--model_kwargs",
    default={},
    type=json.loads,
    help="parameters for the model",
)
parser.add_argument(
    "--multi_scale_pred",
    default=False,
    type=util.str2bool,
    help="perform multi-scale prediction (default: False)",
)

args = parser.parse_args()
config = None

if args.resume is not None:
    if args.config is not None:
        print("Warning: --config overridden by --resume")
        config = torch.load(args.resume)["config"]
elif args.config is not None:
    config = json.load(open(args.config))

assert config is not None

util.setSeed(config)

experiment_dir = os.path.join(config["trainer"]["save_dir"], args.exp)
util.ensure_dir(experiment_dir)

###Logging Files
train_file = "{}/{}_train_loss.txt".format(experiment_dir, args.dataset)
test_file = "{}/{}_test_loss.txt".format(experiment_dir, args.dataset)

train_loss_file = open(train_file, "w", 0)
val_loss_file = open(test_file, "w", 0)

### Angle Metrics
train_file_angle = "{}/{}_train_angle_loss.txt".format(experiment_dir, args.dataset)
test_file_angle = "{}/{}_test_angle_loss.txt".format(experiment_dir, args.dataset)

train_loss_angle_file = open(train_file_angle, "w", 0)
val_loss_angle_file = open(test_file_angle, "w", 0)
################################################################################
num_gpus = torch.cuda.device_count()

model = MODELS_REFINE[args.model_name](
    in_channels=5, num_classes=config["task1_classes"]
)

if num_gpus > 1:
    print("Training with multiple GPUs ({})".format(num_gpus))
    model = nn.DataParallel(model).cuda()
else:
    print("Single Cuda Node is avaiable")
    model.cuda()
################################################################################

### Load Dataset from root folder and intialize DataLoader
train_loader = data.DataLoader(
    __dataset__[args.dataset](
        config["train_dataset"],
        seed=config["seed"],
        is_train=True
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
        is_train=False
    ),
    batch_size=config["val_batch_size"],
    num_workers=8,
    shuffle=True,
    pin_memory=False,
)

print("Training with dataset => {}".format(train_loader.dataset.__class__.__name__))
################################################################################

best_accuracy = 0
best_miou = 0
start_epoch = 1
total_epochs = config["trainer"]["total_epochs"]
optimizer = optim.SGD(
    model.parameters(), lr=config["optimizer"]["lr"], momentum=0.9, weight_decay=0.0005
)

if args.resume is not None:
    print("Loading from existing FCN and copying weights to continue....")
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint["epoch"] + 1
    best_miou = checkpoint["miou"]
    # stat_parallel_dict = util.getParllelNetworkStateDict(checkpoint['state_dict'])
    # model.load_state_dict(stat_parallel_dict)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    util.weights_init(model, manual_seed=config["seed"])

viz_util.summary(model, print_arch=False)

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


road_loss = mIoULoss(
    weight=weights, size_average=True, n_classes=config["task1_classes"]
).cuda()


def train(epoch):
    train_loss_iou = 0
    train_loss_vec = 0
    model.train()
    optimizer.zero_grad()
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    crop_size = config["train_dataset"][args.dataset]["crop_size"]
    for i, data in enumerate(train_loader, 0):
        inputs, labels, erased_label = data
        batch_size = inputs.size(0)

        inputs = Variable(inputs.float().cuda())
        erased_label = Variable(erased_label[-1].float().cuda()).unsqueeze(dim = 1)
        temp = erased_label

        for k in range(config['refinement']):
            in_ = torch.cat((inputs, erased_label, temp), dim=1)
            outputs = model(in_)
            if args.multi_scale_pred:
                loss1 = road_loss(outputs[0], labels[0].long().cuda(), False)
                num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
                for idx in range(num_stacks - 1):
                    loss1 += road_loss(outputs[idx + 1], labels[0].long().cuda(), False)
                for idx, output in enumerate(outputs[-2:]):
                    loss1 += road_loss(output, labels[idx + 1].long().cuda(), False)

                outputs = outputs[-1]
            else:
                loss1 = road_loss(outputs, labels[-1].long().cuda(), False)

            loss1.backward()
            temp = Variable(torch.max(outputs.data, 1)[1].float()).unsqueeze(dim = 1)

        train_loss_iou += loss1.data[0]

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )

        p_accu, miou, road_iou, fwacc = util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            train_loss_iou / (i + 1),
            0,
        )

        viz_util.progress_bar(
            i,
            len(train_loader),
            "Loss: %.6f | road miou: %.4f%%(%.4f%%)"
            % (
                train_loss_iou / (i + 1),
                miou,
                road_iou,
            ),
        )

        if i % config["trainer"]["iter_size"] == 0 or i == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        del (
            outputs,
            predicted,
            correctLabel,
            inputs,
            labels,
        )

    util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        train_loss_iou / len(train_loader),
        0,
        write=True,
    )


def test(epoch):
    global best_accuracy
    global best_miou
    model.eval()
    test_loss_iou = 0
    test_loss_vec = 0
    hist = np.zeros((config["task1_classes"], config["task1_classes"]))
    crop_size = config["val_dataset"][args.dataset]["crop_size"]
    for i, datas in enumerate(val_loader, 0):
        inputs, labels, erased_label = data
        batch_size = inputs.size(0)

        inputs = Variable(inputs.float().cuda(), volatile=True, requires_grad=False)
        erased_label = Variable(erased_label[-1].float().cuda(), volatile=True, requires_grad=False).unsqueeze(dim = 1)
        temp = erased_label

        for k in range(config['refinement']):
            in_ = torch.cat((inputs, erased_label, temp), dim=1)
            outputs = model(in_)
            if args.multi_scale_pred:
                loss1 = road_loss(outputs[0], labels[0].long().cuda(), False)
                num_stacks = model.module.num_stacks if num_gpus > 1 else model.num_stacks
                for idx in range(num_stacks - 1):
                    loss1 += road_loss(outputs[idx + 1], labels[0].long().cuda(), False)
                for idx, output in enumerate(outputs[-2:]):
                    loss1 += road_loss(output, labels[idx + 1].long().cuda(), False)

                outputs = outputs[-1]
            else:
                loss1 = road_loss(outputs, labels[-1].long().cuda(), False)

            temp = Variable(torch.max(outputs.data, 1)[1].float(), volatile=True, requires_grad=False).unsqueeze(dim = 1)

        test_loss_iou += loss1.data[0]

        _, predicted = torch.max(outputs.data, 1)

        correctLabel = labels[-1].view(-1, crop_size, crop_size).long()
        hist += util.fast_hist(
            predicted.view(predicted.size(0), -1).cpu().numpy(),
            correctLabel.view(correctLabel.size(0), -1).cpu().numpy(),
            config["task1_classes"],
        )

        p_accu, miou, road_iou, fwacc = util.performMetrics(
            train_loss_file,
            val_loss_file,
            epoch,
            hist,
            test_loss_iou / (i + 1),
            0,
            is_train=False,
        )

        viz_util.progress_bar(
            i,
            len(val_loader),
            "Loss: %.6f | road miou: %.4f%%(%.4f%%)"
            % (
                test_loss_iou / (i + 1),
                miou,
                road_iou,
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
                None,
                os.path.join(images_path, "validate_pair_{}_{}.png".format(epoch, i)),
                norm_type=config["val_dataset"]["normalize_type"],
            )

        del inputsBGR, labels, predicted, outputs

    accuracy, miou, road_iou, fwacc = util.performMetrics(
        train_loss_file,
        val_loss_file,
        epoch,
        hist,
        test_loss_iou / len(val_loader),
        0,
        is_train=False,
        write=True,
    )

    if miou > best_miou:
        best_accuracy = accuracy
        best_miou = miou
        util.save_checkpoint(epoch, test_loss_iou / len(val_loader), model, optimizer, best_accuracy, best_miou, config, experiment_dir)

    return test_loss_iou / len(val_loader)


for epoch in range(start_epoch, total_epochs + 1):
    start_time = datetime.now()
    scheduler.step(epoch)
    print("\nTraining Epoch: %d" % epoch)
    train(epoch)
    if epoch % config["trainer"]["test_freq"] == 0:
        print("\nTesting Epoch: %d" % epoch)
        val_loss = test(epoch)

    end_time = datetime.now()
    print("Time Elapsed for epoch => {1}".format(epoch, end_time - start_time))