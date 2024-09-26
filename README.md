# Body Language Estimation in Interviews Using Human Mesh Reconstruction

<img src="https://camo.githubusercontent.com/2722992d519a722218f896d5f5231d49f337aaff4514e78bd59ac935334e916a/68747470733a2f2f692e696d6775722e636f6d2f77617856496d762e706e67" alt="Oryx Video-ChatGPT" data-canonical-src="https://i.imgur.com/waxVImv.png" style="max-width: 100%;">

# Overview

This paper introduces a Pyramid-based Mesh Refinement Reconstruction (PMRR) method for accurately reconstructing 3D human meshes from single images, crucial for analyzing body language in interviews. PMRR employs a feature pyramid to enhance neural network predictions, improving mesh-image alignment. It incorporates high-resolution features and pixel-wise supervision to reduce estimation errors. Validated on COCO and 3DPW datasets, PMRR demonstrates superior performance in mesh reconstruction, making it ideal for non-verbal communication studies.

# 👁️💬 Architecture

The comprehensive pipeline of the RealMock framework.

<img style="max-width: 100%;" src="https://github.com/swerizwan/PMRR/blob/main/resources/architecture.jpg" alt="PMRR Overview">

# PMRR Environment Setup

We evaluated PMRR using PVE, MPJPE, PA-MPJPE, and AP metrics on the COCO and 3DPW datasets. The method was compared against existing approaches, demonstrating superior performance in 3D pose and shape estimation.

The instructions for setting up a Conda environment named `pmrr` with the required dependencies:

## Requirements

- Python 3.8
```
conda create --no-default-packages -n pmrr python=3.8
conda activate pmrr
```

### Packages

- [PyTorch](https://www.pytorch.org) tested on version 1.9.0
```
conda install pytorch==1.9.0 torchvision==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

- [pytorch3d](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
```
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

- other packages listed in `requirements.txt`
```
pip install -r requirements.txt
```

### Necessary Files

> mesh_downsampling.npz & DensePose UV data

- Run the following script to fetch mesh_downsampling.npz & DensePose UV data from other repositories.

```
bash fetch_data.sh
```
> SMPL model files

- Collect SMPL model files from [https://smpl.is.tue.mpg.de](https://smpl.is.tue.mpg.de) and [UP](https://github.com/classner/up/blob/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl). Rename model files and put them into the `./data/smpl` directory.

> Fetch preprocessed data from [SPIN](https://github.com/nkolot/SPIN#fetch-data).

> Fetch final_fits data from [SPIN](https://github.com/nkolot/SPIN#final-fits). [important note: using [EFT](https://github.com/facebookresearch/eft) fits for training is much better. Compatible npz files are available [here](https://cloud.tsinghua.edu.cn/d/635c717375664cd6b3f5)]

> Download the [pre-trained model](https://drive.google.com/drive/folders/1R4_Vi4TpCQ26-6_b2PhjTBg-nBxZKjz6?usp=sharing) and put it into the `./data/pretrained_model` directory.

After collecting the above necessary files, the directory structure of `./data` is expected as follows.  
```
./data
├── dataset_extras
│   └── .npz files
├── J_regressor_extra.npy
├── J_regressor_h36m.npy
├── mesh_downsampling.npz
├── pretrained_model
│   └── emo-body-lang_checkpoint.pt
├── smpl
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
├── smpl_mean_params.npz
├── final_fits
│   └── .npy files
└── UV_data
    ├── UV_Processed.mat
    └── UV_symmetry_transforms.mat
```

## Preview of Demo Results:

### For Image Input:

```
python3 run_demo.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt --img_file input/Picture5.png
```

<p align="center">
    <img style="max-width: 100%;" src="https://github.com/swerizwan/PMRR/blob/main/resources/image.png" alt="PMRR Overview">
</p>

### For Video Input:

```
python3 run_demo.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt --vid_file input/dancer.mp4
```

<p align="left">
    <a href="https://github.com/swerizwan/PMRR/blob/main/resources/interview_result.mp4" target="_blank"></a>
</p>


## Evaluation

### COCO

1. Download the preprocessed data [coco_2014_val.npz](https://drive.google.com/drive/folders/1R4_Vi4TpCQ26-6_b2PhjTBg-nBxZKjz6?usp=sharing). Put it into the `./data/dataset_extras` directory. 

2. Run the COCO evaluation code.
```
python3 coco.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt
```

### 3DPW

Run the evaluation code. Using `--dataset` to specify the evaluation dataset.
```
# Example usage:
# 3DPW
python3 main.py --checkpoint=data/pretrained_model/emo_body_lang_checkpoint.pt --dataset=3dpw --log_freq=20
```

## Training

```
CUDA_VISIBLE_DEVICES=0 python3 trainer.py --regressor emo_body_lang --single_dataset --misc TRAIN.BATCH_SIZE 64
```
We can monitor the training process by setting up a TensorBoard in the directory `./logs`.
