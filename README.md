# Learning a Invariant and Equivariant Network for Weakly Supervised Object Detection
By Xiaoxu Feng, Xiwen Yao, Hui Shen, Gong Cheng, Junwei Han
## Citation
```bash

```
## Overview

## Requirements
* python == 3.8 <br>
* Cuda == 11.0 <br>
* Pytorch == 1.7.0 <br>
* torchvision == 0.8.0 <br>
* Pillow <br>
* sklearn <br>
* opencv <br>
* scipy <br>
* cython <br>
* GPU: GeForce RTX 3090
## Installation
1. Clone the IENet repository
```bash
git clone https://github.com/XiaoxFeng/IENet.git
``` 
2. Install libraries
```bash
sh install.sh
```
3. Compile
```bash
cd IENet/lib
sh make.sh
```
4. Download the Dataset and rename it as VOCdevkit
```bash
cd $IENet_ROOT/data/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar
```
5. Extract all of these tars into one directory named VOCdevkit
```bash
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_18-May-2011.tar
```
6. Download pretrained ImageNet weights from [here](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ), and put it in the $IENet_ROOT/data/pretrained_model/
7. Download selective search proposals from [here](https://drive.google.com/drive/folders/1R4leOIYxP9qHJ2dVQJ4fKv2CoEHeEu41) and put it in the $IENet_ROOT/data/selective_search_data/
8. Create symlinks for the PASCAL VOC dataset
```bash
cd $IENet_ROOT/data
ln -s $VOCdevkit VOC2007
ln -s $VOCdevkit VOC2012
```
## Usage
**Train** a IENet. For example, train a VGG16 network on VOC 2007 trainval
```bash
CUDA_VISIBLE_DEVICES=0 python tools/train_net_step.py --dataset voc2007 \
  --cfg configs/baselines/vgg16_voc2007.yaml --bs 1 --nw 4 --iter_size 4
```
**Test** a PCL network. For example, test the VGG 16 network on VOC 2007:

**CorLoc**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg configs/baselines/vgg16_voc2007.yaml \
  --load_ckpt Outputs/vgg16_voc2007/$MODEL_PATH \
  --dataset voc2007trainval
```
**mAP**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --cfg configs/baselines/vgg16_voc2007.yaml \
  --load_ckpt Outputs/vgg16_voc2007/$model_path \
  --dataset voc2007test
```
## Models trained on PASCAL VOC 2007, MS COCO2014, and MS COCO 2017 can be downloaded here:[Google Drive.](https://drive.google.com/drive/folders/1xulStA_PCnd3bppfoKLQUc6OT_Wzbp2p?usp=sharing)
## Acknowledgement
We borrowed code from [PCL](https://github.com/ppengtang/pcl.pytorch), and [Faster-RCNN](https://github.com/jwyang/faster-rcnn.pytorch).
