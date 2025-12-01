# Lightweight Image Denoising

**Project for CISC 473 - Deep Learning**

**Team Members:** Kevin Yao, Steven Li, Zhangzhengyang Song

## 🚀 Project Overview

This project aims to explore and develop **lightweight** neural network architectures for image denoising, especially suitable for deployment on resource-constrained devices (e.g. mobile, edge). We will benchmark existing lightweight models, propose one (or a variant) of our own, and evaluate the trade-offs between **denoising quality** (PSNR / SSIM) versus **efficiency** (parameters, inference time, memory usage).

Our guiding questions:

- What architectural choices make a denoiser “lightweight” without sacrificing too much quality?
- Can simple pruning, quantization, or architectural tweaks yield significant performance gains?
- How do different noise types (Gaussian, real-world, smartphone) affect results?

## 📚 Key References

- [_DnCNN: Beyond a Gaussian Denoiser_ (Zhang et al., 2017)](https://arxiv.org/abs/1608.03981)
- [_Restormer: Efficient Transformer for High-Resolution Image Restoration_ (Zamir et al., 2021)](https://arxiv.org/abs/2111.09881)
- [_Simple Baselines for Image Restoration_](https://arxiv.org/abs/2204.04676)

## 📂 Data & Layout

We ship small samples of each dataset for quick experiments. For full runs, download the public datasets and mirror this layout:

```
BSDS300/images/{train,test}        # BSDS300 split with JPEGs
BSD68/                             # Optional: extra validation set
SIDD_Small_sRGB_Only/Data/<scene>/GT_SRGB_010.PNG and NOISY_SRGB_010.PNG
src/checkpoints/                   # Trained weights go here
src/checkpoints/quantized/         # Quantized weights go here
```

- BSDS300 (full): https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/  
- BSD68 (alt split): https://www.kaggle.com/code/mpwolke/berkeley-segmentation-dataset-68  
- SIDD Small sRGB: https://www.kaggle.com/datasets/rajat95gupta/smartphone-image-denoising-dataset  

If you skip downloads, you can still run the quick samples already under `BSDS300/` and `SIDD_Small_sRGB_Only/`.

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

## 🔁 Reproducibility (quick path)

The commands below reproduce the main figures/metrics on the included sample data. Set seeds where available (`python -m src.test.test_model ...` sets seeds internally).

1) Train (outputs checkpoints + training curves under `src/checkpoints/`):
```bash
python -m src.train.train_dncnn
python -m src.train.train_nafnet --dataset bsds
python -m src.train.train_nafnet --dataset sidd
```

2) Quantize NAFNet (INT8/PTQ and baselines under `src/checkpoints/quantized/`):
```bash
python -m src.quantize.quantize_nafnet --dataset sidd --checkpoint src/checkpoints/nafnet_small_best_sidd.pth
```
Note: for the full quantization experiment suite (extra configs, logs, and scripts), switch to the `quantization-experiments` branch first:
```bash
git checkout quantization-experiments
```
Then follow the branch-specific instructions in its README section.

3) Evaluate PSNR/SSIM + latency (prints table; uses 10 samples with fixed seeds):
```bash
python -m src.test.test_model dncnn bsds
python -m src.test.test_model nafnet sidd
```

4) Visualize qualitative results (matplotlib preview):
```bash
python -m src.visualize.visualize_denoising dncnn
python -m src.visualize.visualize_denoising nafnet_sidd
```

Artifacts:  
- Training curves: `src/checkpoints/dncnn_training_curve.png`, `src/checkpoints/nafnet_small_training_curve_{bsds|sidd}.png`  
- Example metrics: `src/results.txt` (sample PSNR/SSIM/inference time)  
- Quantized weights: `src/checkpoints/quantized/*.pth`

## How to Run the Training (details)

Make sure you are in the project root folder (the same level as `src/`).

Then run the training script using module mode:

```bash
python -m src.train.train_dncnn
```
```bash
python -m src.train.train_nafnet --dataset bsds
```
```bash
python -m src.train.train_nafnet --dataset sidd
```

After training, run the visualization script to display results and compute PSNR/SSIM:

```bash
python -m src.visualize.visualize_denoising dncnn
```
```bash
python -m src.visualize.visualize_denoising nafnet
```
```bash
python -m src.visualize.visualize_denoising nafnet_sidd
```

You may also run the test script with a set seed to replicate results shown in the report.

```bash
python -m src.test.test_model dncnn
```
```bash
python -m src.test.test_model nafnet bsds
```
```bash
python -m src.test.test_model nafnet sidd
```

## 📜 License & Data Sources

- Code: MIT License (see `LICENSE`).
- Datasets: Please follow the licenses/terms from the respective hosts (BSDS/BSD68 from Berkeley/Kaggle, SIDD Small from Kaggle). Data is used here for research/educational purposes only.
