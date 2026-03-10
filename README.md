## README.md for MAE Project:

```markdown
# Masked Autoencoder (MAE) Implementation

A PyTorch implementation of Masked Autoencoders (MAE) for self-supervised visual representation learning, based on the paper "Masked Autoencoders Are Scalable Vision Learners" by He et al. (2021).

## Overview
MAE is a self-supervised learning approach that masks random patches of an input image and reconstructs the missing pixels. The model learns rich visual representations without labeled data.

## Architecture
- **Encoder**: ViT-Base (12 Transformer blocks, embed_dim=768, 12 heads)
- **Masking Ratio**: 75% (147 out of 196 patches masked)
- **Patch Size**: 16×16
- **Image Size**: 224×224
- **Visible Patches**: 49 out of 196

## Project Structure
```
MAE/
├── mae.ipynb          ← main notebook
├── README.md
└── requirements.txt
```

## Key Components
- `patchify()` — converts image (B,3,224,224) into patches (B,196,768)
- `unpatchify()` — reconstructs image from patches
- `random_masking()` — randomly masks 75% of patches
- `MAEEncoder` — ViT-Base encoder with 12 transformer blocks
- `TransformerBlock` — Pre-Norm architecture with MHSA and MLP

## Requirements
```
torch
torchvision
numpy
matplotlib
tqdm
Pillow
```

## Installation
```bash
git clone https://github.com/username/MAE.git
cd MAE
pip install -r requirements.txt
```

## Dataset
- Trained on MNIST (converted to 3-channel, resized to 224×224)
- Uses custom `CustomDataSet` class with ImageNet normalization

## Training
```python
epochs = 10
batch_size = 32
optimizer = Adam
loss = MSELoss (on masked patches only)
```

## References
- [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
- He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021)
```


