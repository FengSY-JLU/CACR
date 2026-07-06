# CACR

Official PyTorch implementation of

**Conflict-Aware Contributor Refinement Learning for Self-Supervised Underwater Image Enhancement**

This repository contains the implementation of the proposed CACR framework for self-supervised underwater image enhancement.

The paper is currently under review.

The code, pretrained models, and instructions will be continuously updated.

## Project Workflow

<p align="center">
  <img src="figures/flowchart.png" width="800" alt="Project Workflow">
</p>

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

### Pretrained Models

You can download the pretrained model here:

[**Download Pretrained Model**](https://drive.google.com/file/d/1DkUYBEzvcB5S-o7Zzc0kdOl9hb2eYih5/view?usp=drive_link)

After downloading, place the `.pth` file into the `checkpoints/` folder (create it if it doesn't exist).  
Please also specify the corresponding `--learned_priors` combination when training or testing.

### Usage / Training

#### Quick Start

  ```bash
  python train.py \
    --nEpochs 200 \
    --batchSize 1 \
    --lr 1e-4 \
    --patch_size 128 \
    --threads 4 \
    --data_train ./Dataset/UIE/UIEBD/train/image \
    --label_train ./Dataset/UIE/UIEBD/train/label \
    --indicator my_experiment_v1
  ```
  Test Commands for Different Datasets
  Here are recommended --learned_priors combinations for different datasets:
  1. UIEBD
  ```bash
  python train.py --learned_priors udcp,clahee,ruie --indicator UIEBD_experiment
  ```
  2. UFO120
  ```bash
  python train.py --learned_priors clahe,unsharpmask,ruie --indicator UFO120_experiment
  ```
  3. RUIE
  ```bash
  python train.py --learned_priors msrcr,hdp --indicator RUIE_experiment
  ```
  4. OceanDark
  ```bash
  python train.py --learned_priors clahe,udcp,unsharpmask --indicator OceanDark_experiment
  ```
  Examples
  Quick test run:
  ```bash
  python train.py --nEpochs 1 --batchSize 1 --debug True --indicator test_run
  ```
  Standard training:
  ```bash
  python train.py \
    --nEpochs 300 \
    --indicator UIEBD_refined_model_v2 \
    --lr 1e-4 \
    --threads 6
  ```
  Checkpoints
  Checkpoints are automatically saved at:
  textcheckpoints/{indicator}/{dataset}_final.pth
  For example: checkpoints/UIEBD_refined_model_v2/UIEBD_final.pth
  

#### Citation

Coming soon.
