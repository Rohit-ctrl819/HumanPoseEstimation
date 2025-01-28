
# Deep High-Resolution Representation Learning for Human Pose Estimation (accepted to CVPR 2019)

## News

- If you are interested in internship or research positions related to computer vision in ByteDance AI Lab, feel free to contact me (leoxiaobin-at-gmail.com).
- Our new work [High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/abs/1904.04514) is available at [HRNet](https://github.com/HRNet). Our HRNet has been applied to a wide range of vision tasks, such as [image classification](https://github.com/HRNet/HRNet-Image-Classification), [object detection](https://github.com/HRNet/HRNet-Object-Detection), [semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation), and [facial landmark detection](https://github.com/HRNet/HRNet-Facial-Landmark-Detection).

## Introduction

This is the official PyTorch implementation of the paper *[Deep High-Resolution Representation Learning for Human Pose Estimation](https://arxiv.org/abs/1902.09212)*.

In this work, we address the human pose estimation problem, focusing on learning reliable high-resolution representations. While most existing methods **recover high-resolution representations** from low-resolution representations via high-to-low resolution networks, our proposed method **maintains high-resolution representations** throughout the process.

We begin with a high-resolution subnetwork in the first stage and progressively add high-to-low resolution subnetworks in parallel across multiple stages. We conduct **repeated multi-scale fusions** such that each high-to-low resolution representation receives information from other parallel representations multiple times, resulting in rich high-resolution representations. This leads to more accurate and spatially precise keypoint heatmaps.

Our method achieves superior pose estimation results over two benchmark datasets: COCO keypoint detection and the MPII Human Pose dataset.

![Illustrating the architecture of the proposed HRNet](/figures/hrnet.png)

## Main Results

### Results on MPII val
| Arch               | Head  | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Mean | Mean@0.1 |
|--------------------|-------|----------|-------|-------|-----|------|-------|------|----------|
| pose_resnet_50     | 96.4  | 95.3     | 89.0  | 83.2  | 88.4 | 84.0 | 79.6  | 88.5 | 34.0     |
| pose_resnet_101    | 96.9  | 95.9     | 89.5  | 84.4  | 88.4 | 84.5 | 80.7  | 89.1 | 34.0     |
| pose_resnet_152    | 97.0  | 95.9     | 90.0  | 85.0  | 89.2 | 85.3 | 81.3  | 89.6 | 35.0     |
| **pose_hrnet_w32** | **97.1** | **95.9** | **90.3** | **86.4** | **89.1** | **87.1** | **83.3** | **90.3** | **37.7** |

### Results on COCO val2017 (Human AP: 56.4)
| Arch               | Input size | #Params | GFLOPs | AP   | AP .5 | AP .75 | AP (M) | AP (L) | AR   | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|------|-------|--------|--------|--------|------|-------|--------|--------|--------|
| pose_resnet_50     | 256x192    | 34.0M   | 8.9    | 0.704 | 0.886 | 0.783  | 0.671  | 0.772  | 0.763| 0.929 | 0.834  | 0.721  | 0.824  |
| pose_resnet_101    | 256x192    | 53.0M   | 12.4   | 0.714 | 0.893 | 0.793  | 0.681  | 0.781  | 0.771| 0.934 | 0.840  | 0.730  | 0.832  |
| **pose_hrnet_w32** | 256x192    | 28.5M   | 7.1    | 0.744 | 0.905 | 0.819  | 0.708  | 0.810  | 0.798| 0.942 | 0.865  | 0.757  | 0.858  |

### Results on COCO test-dev2017 (Human AP: 60.9)
| Arch               | Input size | #Params | GFLOPs | AP   | AP .5 | AP .75 | AP (M) | AP (L) | AR   | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|------|-------|--------|--------|--------|------|-------|--------|--------|--------|
| **pose_hrnet_w48** | 384x288    | 63.6M   | 32.9   | 0.755 | 0.925 | 0.833  | 0.719  | 0.815  | 0.805| 0.957 | 0.874  | 0.763  | 0.863  |

## Environment

The code is developed with Python 3.6 on Ubuntu 16.04 and requires NVIDIA GPUs for execution. It was tested using 4 NVIDIA P100 GPU cards.

## Quick Start

### Installation
1. Install PyTorch >= v1.0.0 following the [official instructions](https://pytorch.org/).
   - For PyTorch versions < v1.0.0, disable cudnn's implementations of the BatchNorm layer.
2. Clone this repository and set the directory as `${POSE_ROOT}`.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Build libraries:
   ```bash
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install the [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```bash
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   make install
   ```
6. Create output and log directories:
   ```bash
   mkdir output
   mkdir log
   ```

### Download Pretrained Models
Download pretrained models from the model zoo ([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://onedrive.live.com)) and place them under `${POSE_ROOT}/models`.

### Data Preparation

**For MPII Dataset**: Download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). Convert annotation files to JSON format (if not already) and extract them under `${POSE_ROOT}/data/mpi/annot`.

**For COCO Dataset**: Download from [COCO](http://cocodataset.org/#download), and place the files under `${POSE_ROOT}/data/coco`.

### Training and Testing

#### Testing on MPII dataset
```bash
python tools/test.py --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_mpii/pose_hrnet_w32_256x256.pth
```

#### Training on MPII dataset
```bash
python tools/train.py --cfg experiments/mpii/hrnet/w32_256x256_adam_lr1e-3.yaml
```

#### Testing on COCO val2017 dataset
```bash
python tools/test.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset
```bash
python tools/train.py --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
```

### Citation

If you use our code or models in your research, please cite the following papers:

```
@inproceedings{sun2019deep,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Sun, Ke and Xiao, Bin and Liu, Dong and Wang, Jingdong},
  booktitle={CVPR},
  year={2019}
}

@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}
```

