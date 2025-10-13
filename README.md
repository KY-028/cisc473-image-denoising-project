# Lightweight Image Denoising

**Team Members:** Kevin Yao, Steven Li, Zhangzhengyang Song
**Project for CISC 473**

---

## 🚀 Project Overview

This project aims to explore and develop **lightweight** neural network architectures for image denoising, especially suitable for deployment on resource-constrained devices (e.g. mobile, edge). We will benchmark existing lightweight models, propose one (or a variant) of our own, and evaluate the trade-offs between **denoising quality** (PSNR / SSIM) versus **efficiency** (parameters, FLOPs, inference time).

Our guiding questions:

- What architectural choices make a denoiser “lightweight” without sacrificing too much quality?
- Can simple pruning, quantization, or architectural tweaks yield significant performance gains?
- How do different noise types (Gaussian, real-world, smartphone) affect results?

---

## 📚 Key References

- *DnCNN: Beyond a Gaussian Denoiser* (Zhang et al., 2017)  
- *Restormer: Efficient Transformer for High-Resolution Image Restoration* (Zamir et al., 2021) :contentReference[oaicite:0]{index=0}  
- *NAFNet: Nonlinear Activation Free Network for Image Restoration*  
- *LIDIA: Lightweight Learned Image Denoising with Instance Adaptation* (Vaksman, Elad, Milanfar, 2019) :contentReference[oaicite:1]{index=1}  
- *Thunder: Thumbnail-based Fast Lightweight Image Denoising Network* (Zhou et al., 2022) :contentReference[oaicite:2]{index=2}  

We also examine relevant open-source repos such as [LIDIA-denoiser](https://github.com/grishavak/LIDIA-denoiser) :contentReference[oaicite:3]{index=3}, [LWDN](https://github.com/rami0205/LWDN) :contentReference[oaicite:4]{index=4}, and more.

---

## 🛠 Setup & Usage

### Requirements

- Python 3.8+  
- PyTorch  
- torchvision, numpy, PIL  
- (Optional) CUDA for GPU training  

Install via:

```bash
pip install -r requirements.txt
