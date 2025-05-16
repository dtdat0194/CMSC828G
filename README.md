# OpenSource approach to DISTMM

This project explores contrastive learning as a framework for aligning representa tions across modalities in a distributed training setting. Inspired by the Distributed Multimodal Model (DISTMM) Training system, we implement modality-aware strategies that account for the structural and computational differences between audio and video data.

## Directory Structure

```
.
├── DISTMM_implementation/
│   └── Contrastive_Learning_ResNet/
│       ├── ContrastiveLearningDISTMM_Test.py  # Main training script
│       ├── Backbones.py                       # Vision and Audio backbone models
│       └── requirements.txt                   # Python dependencies
```

## Setup on Zaratan

1. First, connect to Zaratan:
```bash
ssh your_username@zaratan.umd.edu
```

2. Create and activate a conda environment:
```bash
conda create -n contrastive_learning python=3.8
conda activate contrastive_learning
```

3. Install required packages:
```bash
pip install torch torchvision torchaudio
```

## Running the Training

The training script uses PyTorch's Distributed Data Parallel (DDP) to train across multiple GPUs. To run the training:

```bash
# Navigate to the project directory
cd DISTMM_implementation/Contrastive_Learning_ResNet

# Run 1 GPUs Training
python train.py

# Run 4 GPUs Naive Data Parallel Training
train_dataparallelism.py

# Run 4 GPUs Modality Aware Data Parallel Training
train_DistMM.py
```

### Datdaset

Dataset is available at https://drive.google.com/drive/folders/1vn_c4l1pctuUu5LASAQziAjSaGSX1k8V?usp=sharing
