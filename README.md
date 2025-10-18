# Lightweight Image Denoising

**Project for CISC 473 - Deep Learning**

**Team Members:** Kevin Yao, Steven Li, Zhangzhengyang Song

## 🚀 Project Overview

This project aims to explore and develop **lightweight** neural network architectures for image denoising, especially suitable for deployment on resource-constrained devices (e.g. mobile, edge). We will benchmark existing lightweight models, propose one (or a variant) of our own, and evaluate the trade-offs between **denoising quality** (PSNR / SSIM) versus **efficiency** (parameters, FLOPs, inference time).

Our guiding questions:

- What architectural choices make a denoiser “lightweight” without sacrificing too much quality?
- Can simple pruning, quantization, or architectural tweaks yield significant performance gains?
- How do different noise types (Gaussian, real-world, smartphone) affect results?

## 📚 Key References

- [_DnCNN: Beyond a Gaussian Denoiser_ (Zhang et al., 2017)](https://arxiv.org/abs/1608.03981)
- [_Restormer: Efficient Transformer for High-Resolution Image Restoration_ (Zamir et al., 2021)](https://arxiv.org/abs/2111.09881)
- [_Simple Baselines for Image Restoration_](https://arxiv.org/abs/2204.04676)

Unsure

- _LIDIA: Lightweight Learned Image Denoising with Instance Adaptation_ (Vaksman, Elad, Milanfar, 2019)
- _Thunder: Thumbnail-based Fast Lightweight Image Denoising Network_ (Zhou et al., 2022)

## 🛠 Setup & Usage

### Requirements

- Python 3.8+
- PyTorch
- torchvision, numpy, PIL
- (Optional) CUDA for GPU training

Install via:

```bash
pip install -r requirements.txt
```

## How to Run the Training

Make sure you are in the project root folder (the same level as `src/`).

Then run the training script using module mode:

```bash
python -m src.train.train_dncnn
```

After training, run the visualization script to display results and compute PSNR/SSIM:

```bash
python -m src.visualize.visualize_denoising
```
