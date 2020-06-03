## Improved Road Connectivity by Joint Learning of Orientation and Segmentation ##
#### In CVPR 2019 [[pdf]](https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf) [[supplementary]](https://anilbatra2185.github.io/papers/RoadConnectivity_CVPR_Supplementary.pdf)

## Overview
<img src='https://github.com/anilbatra2185/road_connectivity/blob/master/assests/images/overview.png' width="800">

## Migration to PyTorch 1.5.0 is in progress

## Requirements
* [PyTorch](https://pytorch.org/) (version = 0.3.0)
* [sknw](https://github.com/yxdragon/sknw)
* [networkx](https://networkx.github.io/) (version = 2.4)
* json
* skimage
* numpy
* tqdm

## Data Preparation

#### PreProcess Spacenet Data
- Convert Spacenet 11-bit images to 8-bit Images, country wise.
- Create Gaussian Road Masks, country wise.
- Move all data to single folder.

*Default Spacenet3 tree structure assumed.*
```
spacenet3
│
└───AOI_2_Vegas_Train
│   └───RGB-PanSharpen
│   └───geojson
│   └───summaryData
│
└───AOI_3_Paris_Train
│   └───RGB-PanSharpen
│   └───geojson
│   └───summaryData
|
└───AOI_4_Shanghai_Train
|   .
|
└───AOI_5_Khartoum_Train
|   .
|
```

```
cd preprocessing
bash prepare_spacenet.sh /spacenet3
```
#### Split Datasets
*Spacenet tree structure created by preprocessing.*
```
spacenet3
|
└───full
│   └───gt
│   └───images
```

*Download DeepGlobe Road dataset in the following tree structure.*
```
deepglobe
│
└───train
│   └───gt
│   └───images
```
*Script to split and save in **'/data/spacenet'** and **'/data/deepglobe'**.*
```
bash split_data.sh /spacenet3/full /data/spacenet/ .png .png
bash split_data.sh /deepglobe/train /data/deepglobe _sat.jpg _mask.png
```
#### Create Crops

```
data/spacenet
|   train.txt
|   val.txt
|   train_crops.txt   # created by script
|   val_crops.txt     # created by script
|
└───train
│   └───gt
│   └───images
└───val
│   └───gt
│   └───images
└───train_crops       # created by script
│   └───gt
│   └───images
└───val_crops         # created by script
│   └───gt
│   └───images
```
```
python create_crops.py --base_dir /data/spacenet/ --crop_size 650 --crop_overlap 215 --im_suffix .png --gt_suffix .png
python create_crops.py --base_dir /data/deepglobe/ --crop_size 512 --crop_overlap 256 --im_suffix _sat.jpg --gt_suffix _mask.png
```
## Visualize Data
* Road Orientation - [Notebook](https://github.com/anilbatra2185/road_connectivity/blob/master/visualize_tasks.ipynb)
* Training Dataset - [Notebook](https://github.com/anilbatra2185/road_connectivity/blob/master/visualize_dataset.ipynb)
* Linear Corruption (Connectivity Refinement) - [Notebook](https://github.com/anilbatra2185/road_connectivity/blob/master/visualize_dataset_corrupt.ipynb)

## Training

Train Multi-Task learning framework to predict road segmentation and road orientation.

__Training MTL Help__
```
usage: train_mtl.py [-h] --config CONFIG
                    --model_name {LinkNet34MTL,StackHourglassNetMTL}
                    --dataset {deepglobe,spacenet}
                    --exp EXP
                    [--resume RESUME]
                    [--model_kwargs MODEL_KWARGS]
                    [--multi_scale_pred MULTI_SCALE_PRED]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       config file path
  --model_name 			{LinkNet34MTL,StackHourglassNetMTL}
                        Name of Model = ['StackHourglassNetMTL',
                        'LinkNet34MTL']
  --exp EXP             Experiment Name/Directory
  --resume RESUME       path to latest checkpoint (default: None)
  --dataset 			{deepglobe,spacenet}
                        select dataset name from ['deepglobe', 'spacenet'].
                        (default: Spacenet)
  --model_kwargs 		MODEL_KWARGS
                        parameters for the model
  --multi_scale_pred 	MULTI_SCALE_PRED
                        perform multi-scale prediction (default: True)
```

__Sample Usage__

* Training with StackModule
```
CUDA_VISIBLE_DEVICES=0,1 python train_mtl.py --config config.json --dataset deepglobe --model_name "StackHourglassNetMTL" --exp dg_stak_mtl
```
* Training with LinkNet34
```
CUDA_VISIBLE_DEVICES=0,1 python train_mtl.py --config config.json --dataset deepglobe --model_name "LinkNet34MTL" --exp dg_L34_mtl --multi_scale_pred false
```

## Evaluate APLS

* Please use Java implementation to compute APLS provided by Spacenet Challenge. - [Visualizer tool](https://drive.google.com/file/d/1rwbj_o-ELBfruPZuVkCnEQxAX2-Pz5DX/view)
* For more info refer issue [#13](https://github.com/anilbatra2185/road_connectivity/issues/13)


## Connectivity Refinement

* Training with Linear Artifacts/Corruption (using LinkNe34 Architecture)
```
CUDA_VISIBLE_DEVICES=0,1 python train_refine_pre.py --config config.json --dataset spacenet --model_name "LinkNet34" --exp spacenet_L34_pre_train_with_corruption --multi_scale_pred false
```

## Citation
If you find our work useful in your research, please cite:

    @InProceedings{Batra_2019_CVPR,
		author = {Batra, Anil and Singh, Suriya and Pang, Guan and Basu, Saikat and Jawahar, C.V. and Paluri, Manohar},
		title = {Improved Road Connectivity by Joint Learning of Orientation and Segmentation},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {June},
		year = {2019}
	}

## Remaining Tasks
- [x] Dataset for Connectivity Refinement
- [ ] Training file for Road connectivity refinement
- [ ] Dataset for Junction Learning
