# Masked Autoencoder (MAE) Implementation

A PyTorch implementation of **Masked Autoencoders (MAE)** for self-supervised visual representation learning, based on the paper:

**"Masked Autoencoders Are Scalable Vision Learners"** (He et al., 2021)

---

## 🚀 Live Demo

**Try it here:**
https://huggingface.co/spaces/ImranAliNaeem/mae-reconstruction

---

## 📖 Overview

Masked Autoencoders (MAE) are a self-supervised learning approach that trains a model to reconstruct missing image patches from a partially visible image.

By masking a large portion of the input image and learning to recover the missing content, the model acquires meaningful visual representations without requiring labeled data.

---

## ✨ Features

* Self-supervised visual representation learning
* Random masking of 75% of image patches
* Vision Transformer (ViT-Base) encoder
* Patch-based image reconstruction
* PyTorch implementation
* Interactive Hugging Face demo

---

## 🏗️ Architecture

| Component           | Configuration |
| ------------------- | ------------- |
| Encoder             | ViT-Base      |
| Transformer Blocks  | 12            |
| Embedding Dimension | 768           |
| Attention Heads     | 12            |
| Patch Size          | 16 × 16       |
| Image Size          | 224 × 224     |
| Total Patches       | 196           |
| Visible Patches     | 49            |
| Masked Patches      | 147           |
| Masking Ratio       | 75%           |



---

## 🔧 Key Components

### patchify()

Converts images from:

```text
(B, 3, 224, 224)
```

to

```text
(B, 196, 768)
```

### unpatchify()

Reconstructs the original image from patch embeddings.

### random_masking()

Randomly masks 75% of image patches during training.

### MAEEncoder

Vision Transformer encoder consisting of 12 transformer blocks.

### TransformerBlock

Pre-Norm Transformer architecture containing:

* Multi-Head Self Attention (MHSA)
* Feed Forward Network (MLP)
* Residual Connections
* Layer Normalization

---

## 📦 Requirements

```text
torch
torchvision
numpy
matplotlib
tqdm
Pillow
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/mae-implementation.git
cd mae-implementation
pip install -r requirements.txt
```

---

## 📊 Dataset

The model is trained on:

* MNIST dataset
* Converted to 3-channel RGB images
* Resized to 224 × 224
* ImageNet normalization applied
* Custom Dataset loader implementation

---

## 🏃 Training Configuration

```python
epochs = 10
batch_size = 32
optimizer = Adam
loss_function = MSELoss
```

### Reconstruction Loss

The model computes Mean Squared Error (MSE) only on the masked patches, encouraging the encoder to learn meaningful visual representations from limited visible information.

---

## 📈 Results

The model successfully reconstructs masked regions of input images and demonstrates the core principles of self-supervised representation learning through masked image modeling.

Example outputs can be explored in the Hugging Face demo.

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Vision Transformer (ViT)
* NumPy
* Matplotlib
* Hugging Face Spaces

---

## 📚 References

### Paper

Masked Autoencoders Are Scalable Vision Learners

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2021)

https://arxiv.org/abs/2111.06377

---

## 👨‍💻 Author

**Imran Ali**

Computer Science Student | AI & Machine Learning Enthusiast

If you found this project useful, consider giving it a ⭐ on GitHub.
