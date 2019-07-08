## Improved Road Connectivity by Joint Learning of Orientation and Segmentation ##
#### In CVPR 2019 [[pdf]](https://anilbatra2185.github.io/papers/RoadConnectivityCVPR2019.pdf)

## Overview
<img src='https://github.com/anilbatra2185/road_connectivity/blob/master/assests/images/overview.png' width="800">

## *Work in Progress*

## Requirements
* [PyTorch](https://pytorch.org/) (version >= 0.3.0)
* [sknw](https://github.com/yxdragon/sknw)
* [networkx](https://networkx.github.io/) (version = 1.11)

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
│   │
│   └───RGB-PanSharpen
│   │
│   └───geojson
│   │
│   └───summaryData
│   
└───AOI_3_Paris_Train
│   │
│   └───RGB-PanSharpen
│   │
│   └───geojson
│   │
│   └───summaryData
|
└───AOI_4_Shanghai_Train
|   .
|   .
|
└───AOI_5_Khartoum_Train
|   .
|   .
```

```
bash prepare_spacenet.sh /spacenet3
```
#### Split Datasets

```
spacenet3
│   
└───full
│   │
│   └───gt
│   │
│   └───images

deepglobe
│   
└───train
│   │
│   └───gt
│   │
│   └───images

```

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
## Training

## Citation
If you find our work useful in your research, please cite:

    @InProceedings{Batra_2019_CVPR,
		author = {Batra, Anil and Singh, Suriya and Pang, Guan and Basu, Saikat and Jawahar, C.V. and Paluri, Manohar},
		title = {Improved Road Connectivity by Joint Learning of Orientation and Segmentation},
		booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
		month = {June},
		year = {2019}
	}
