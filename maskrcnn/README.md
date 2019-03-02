# Mask RCNN
Mask RCNN in TensorFlow

This repo attempts to reproduce this amazing work by Kaiming He et al. : 
[Mask R-CNN](https://arxiv.org/abs/1703.06870)

## Requirements

- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [COCO dataset](http://mscoco.org/dataset/#download)
- [Resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

## How-to
1. Go to `./libs/datasets/pycocotools` and run `make`
2. Download [COCO](http://mscoco.org/dataset/#download) dataset, place it into `./data`, then run `python download_and_convert_data.py` to build tf-records. It takes a while.
3. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz`, unzip it, place it into `./data/pretrained_models/`
4. Go to `./libs` and run `make`
5. run `python train/train.py` for training
 
