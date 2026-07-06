# CACR

Official PyTorch implementation of

**Conflict-Aware Contributor Refinement Learning for Self-Supervised Underwater Image Enhancement**

This repository contains the implementation of the proposed CACR framework for self-supervised underwater image enhancement.

The paper is currently under review.

The code, pretrained models, and instructions will be continuously updated.

# Installation

Environment Requirements:

Python: 3.13.0
PyTorch: 2.7.1+cu118 (CUDA 11.8)
CUDA: 11.8 (recommended)

1. Clone the repository
git clone <your-repo-url>
cd <your-repo-name>

2. Install dependencies
pip install -r requirements.txt

(Optional) Use Tsinghua mirror for faster download in China:
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


# Pretrained Models
You can download pretrained models from the link below:
Download Pretrained Model
After downloading, place the model in the checkpoints/ folder or load it manually in your inference script.

# Usage / Training

1. Quick Start

Bashpython train.py \
  --nEpochs 200 \
  --batchSize 1 \
  --lr 1e-4 \
  --patch_size 128 \
  --threads 4 \
  --data_train ./Dataset/UIE/UIEBD/train/image \
  --label_train ./Dataset/UIE/UIEBD/train/label \
  --indicator my_experiment_v1

2. Examples

Quick test run:
Bashpython train.py --nEpochs 1 --batchSize 1 --debug True --indicator test_run
Full training example:
Bashpython train.py \
  --nEpochs 300 \
  --indicator UIEBD_refined_model_v2 \
  --lr 1e-4 \
  --threads 6
  
3. Checkpoints

Location: checkpoints/{indicator}/model_epoch_{epoch}.pth
Each checkpoint contains model weights, training config, and runtime state.


## Citation

Coming soon.
