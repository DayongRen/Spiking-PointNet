# Spiking PointNet

Official PyTorch implementation for the following paper:

**Spiking PointNet: Spiking Neural Networks for Point Clouds**.

**TL;DR:** In this paper, we have presented Spiking PointNet, the first spiking neural network (SNN) specifically designed for efficient deep learning on point clouds.


## Install
The latest codes are tested on Ubuntu 18.04, CUDA10.1, PyTorch 1.6 and Python 3.7:
```shell
conda install pytorch==1.6.0 cudatoolkit=10.1 -c pytorch
```

## Classification (ModelNet10/40)
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
You can run different modes with following codes. 
* If you want to use offline processing of data, you can use `--process_data` in the first run. You can download pre-processd data [here](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing) and save it in `data/modelnet40_normal_resampled/`.
* If you want to train on ModelNet10, you can use `--num_category 10`.
```shell
# ModelNet40
## Select different models in ./models 

## e.g., Pointnet without normal features
python train_classification.py --model pointnet_cls --log_dir pointnet_cls
python test_classification.py --log_dir pointnet_cls

## e.g., Spiking Pointnet without normal features
python train_classification.py --model pointnet_cls --log_dir pointnet_cls --spike --step 1
python test_classification.py --log_dir pointnet_cls --spike --step 1

# ModelNet10
## Similar setting like ModelNet40, just using --num_category 10

## e.g., Pointnet without normal features
python train_classification.py --model pointnet_cls --log_dir pointnet_cls --num_category 10
python test_classification.py --log_dir pointnet_cls --num_category 10

## e.g., Pointnet without normal features
python train_classification.py --model pointnet_cls --log_dir pointnet_cls --num_category 10 --spike --step 1
python test_classification.py --log_dir pointnet_cls --num_category 10 --spike --step 1
```

### Performance
#### Comparison between our method and the vanilla SNN on ModelNet10/40 datasets

| Datasets | Methods | Training time steps | Testing time steps (1) | Testing time steps (2) | Testing time steps (3) | Testing time steps (4) |
| -------- | ------- | ------------------- | ---------------------- | ---------------------- | ---------------------- | ---------------------- |
| ModelNet10 | ANN | - | 92.98% |
| ModelNet10 | Vanilla SNN | 4 | 89.62% | 90.83% | 91.05% | 91.05% |
| ModelNet10 | Ours without MPP | 1 | 91.99% | 92.43% | 92.53% | 92.32% |
| ModelNet10 | Ours with MPP | 1 | 91.66% | 92.98% | 92.98% | **93.31%** |
| ModelNet40 | ANN | - | 89.20% |
| ModelNet40 | Vanilla SNN | 4 | 85.59% | 86.58% | 86.34% | 86.70% |
| ModelNet40 | Ours without MPP | 1 | 86.98% | 87.26% | 87.21% | 87.13% |
| ModelNet40 | Ours with MPP | 1 | 87.72% | 88.46% | 88.25% | **88.61%** |

### Acknowledgment
This library is inspired by [Re-Loss](https://github.com/yfguo91/Re-Loss).


### Citation
If you find Spiking PointNet codebase useful, please cite:
```tex
@inproceedings{
anonymous2023spiking,
title={Spiking PointNet: Spiking Neural Networks for Point Clouds},
author={Dayong Ren, Zhe Ma, Yuanpei Chen, Weihang Peng, Xiaode Liu, Yuhan Zhang, Yufei Guo},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=Ev2XuqvJCy}
}
```
