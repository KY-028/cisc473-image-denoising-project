"""
Baseline PSNR/SSIM without a model.

For BSDS: compare synthetic noisy inputs (σ=25) against clean images.
For SIDD: compare provided noisy/clean pairs on the held-out test split.

Usage:
    python -m src.test.baseline_no_model --dataset bsds
    python -m src.test.baseline_no_model --dataset sidd
"""

import argparse
import os
import random
import numpy as np
import torch
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image


def evaluate_bsds(noise_level=25 / 255.0, num_samples=10):
    bsds300_dir = "BSDS300/images/test"
    if not os.path.exists(bsds300_dir):
        raise FileNotFoundError(f"BSDS300 directory not found at {bsds300_dir}")

    all_images = [f for f in os.listdir(bsds300_dir) if f.endswith(".jpg")]
    if len(all_images) < num_samples:
        raise ValueError(f"Not enough images in BSDS300 directory. Found {len(all_images)}, need at least {num_samples}")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    selected = random.sample(all_images, num_samples)
    psnr_values, ssim_values = [], []

    for img_name in selected:
        img_path = os.path.join(bsds300_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        clean = transform(img).unsqueeze(0)

        noisy = clean + noise_level * torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0.0, 1.0)

        psnr_values.append(psnr_metric(noisy, clean).item())
        ssim_values.append(ssim_metric(noisy, clean).item())

    return selected, psnr_values, ssim_values


def evaluate_sidd(num_samples=10):
    from src.data.sidd_dataset import SIDDDataset, split_sidd_dataset

    sidd_root = "SIDD_Small_sRGB_Only/Data"
    if not os.path.exists(sidd_root):
        raise FileNotFoundError(f"SIDD directory not found at {sidd_root}")

    dataset = SIDDDataset(dirs=[sidd_root], size=128, random_crop=False)
    _, _, test_set = split_sidd_dataset(dataset, seed=42)

    if len(test_set) < num_samples:
        raise ValueError(f"Not enough images in SIDD test split. Found {len(test_set)}, need at least {num_samples}")

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    sample_indices = random.sample(range(len(test_set)), num_samples)
    psnr_values, ssim_values, names = [], [], []

    for idx in sample_indices:
        noisy, clean = test_set[idx]
        noisy = noisy.unsqueeze(0)
        clean = clean.unsqueeze(0)

        psnr_values.append(psnr_metric(noisy, clean).item())
        ssim_values.append(ssim_metric(noisy, clean).item())

        orig_idx = test_set.indices[idx]
        noisy_path, _ = dataset.files[orig_idx]
        names.append(os.path.basename(os.path.dirname(noisy_path)))

    return names, psnr_values, ssim_values


def main():
    parser = argparse.ArgumentParser(description="Baseline PSNR/SSIM without a model.")
    parser.add_argument("--dataset", choices=["bsds", "sidd"], default="bsds", help="Dataset to evaluate.")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    if args.dataset == "sidd":
        names, psnr_vals, ssim_vals = evaluate_sidd()
    else:
        names, psnr_vals, ssim_vals = evaluate_bsds()

    avg_psnr, avg_ssim = np.mean(psnr_vals), np.mean(ssim_vals)
    std_psnr, std_ssim = np.std(psnr_vals), np.std(ssim_vals)

    print("BASELINE (no model) METRICS")
    print("-" * 40)
    print(f"Dataset: {args.dataset}")
    print(f"PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.3f} ± {std_ssim:.3f}")
    print("\nINDIVIDUAL SAMPLES")
    print("-" * 60)
    for name, p, s in zip(names, psnr_vals, ssim_vals):
        print(f"{name:<25} PSNR: {p:.2f} dB   SSIM: {s:.3f}")


if __name__ == "__main__":
    main()
