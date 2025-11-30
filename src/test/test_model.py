"""
Model Performance Evaluation Script

Evaluates a trained model on 10 randomly selected samples.
Supports BSDS300 (synthetic noise) or SIDD (real noisy/clean pairs).

Usage:
    python -m src.test.test_model dncnn bsds
    python -m src.test.test_model nafnet sidd
"""

import os
import time
import random
import torch
import numpy as np
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
import sys

MODEL_TYPE = sys.argv[1] if len(sys.argv) > 1 else "dncnn"
DATASET_TYPE = sys.argv[2] if len(sys.argv) > 2 else "bsds"

def test_performance():
    """
    Test model performance on BSDS300 (synthetic noise) or SIDD (real noisy/clean pairs).
    Calculates PSNR, SSIM, inference time, and CPU runtime metrics.
    """
    device = torch.device('cpu')

    # Select model type
    if MODEL_TYPE == "dncnn":
        from src.models.dncnn import DnCnn
        model = DnCnn(image_channels=3).to(device)
        checkpoint_path = "src/checkpoints/dncnn_best.pth"
    elif MODEL_TYPE == "nafnet":
        from src.models.nafnet import NAFNet
        model = NAFNet(
            image_channels=3,
            width=32,
            middle_blk_num=1,
            enc_blk_nums=[1, 1, 1, 8],
            dec_blk_nums=[1, 1, 1, 1]
        ).to(device)
        checkpoint_path = "src/checkpoints/nafnet_small_best_sidd.pth"
    else:
        raise ValueError(f"Unknown MODEL_TYPE '{MODEL_TYPE}'")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)

    psnr_values = []
    ssim_values = []
    inference_times = []
    cpu_runtimes = []
    sample_names = []

    noise_level = 25 / 255.0  # used for BSDS synthetic noise

    if DATASET_TYPE.lower() == "sidd":
        from src.data.sidd_dataset import SIDDDataset, split_sidd_dataset
        sidd_root = "SIDD_Small_sRGB_Only/Data"
        if not os.path.exists(sidd_root):
            raise FileNotFoundError(f"SIDD directory not found at {sidd_root}")

        dataset = SIDDDataset(
            dirs=[sidd_root],
            size=128,
            random_crop=False
        )
        train_set, val_set, test_set = split_sidd_dataset(dataset, seed=42)
        if len(test_set) < 10:
            raise ValueError(f"Not enough images in SIDD test split. Found {len(test_set)}, need at least 10")

        sample_indices = random.sample(range(len(test_set)), 10)

        for idx in sample_indices:
            cpu_start_time = time.time()

            noisy, clean = test_set[idx]
            noisy = noisy.unsqueeze(0)
            clean = clean.unsqueeze(0)

            inference_start = time.time()
            with torch.no_grad():
                denoised = model(noisy)
            inference_time = time.time() - inference_start

            psnr_value = psnr_metric(denoised, clean)
            ssim_value = ssim_metric(denoised, clean)

            cpu_runtime = time.time() - cpu_start_time

            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            inference_times.append(inference_time)
            cpu_runtimes.append(cpu_runtime)

            # Use scene folder name for readability
            orig_idx = test_set.indices[idx]
            noisy_path, _ = dataset.files[orig_idx]
            sample_names.append(os.path.basename(os.path.dirname(noisy_path)))

    else:
        bsds300_dir = "BSDS300/images/test"
        if not os.path.exists(bsds300_dir):
            raise FileNotFoundError(f"BSDS300 directory not found at {bsds300_dir}")

        all_images = [f for f in os.listdir(bsds300_dir) if f.endswith('.jpg')]
        if len(all_images) < 10:
            raise ValueError(f"Not enough images in BSDS300 directory. Found {len(all_images)}, need at least 10")

        selected_images = random.sample(all_images, 10)

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        for img_name in selected_images:
            img_path = os.path.join(bsds300_dir, img_name)

            cpu_start_time = time.time()

            img = Image.open(img_path)
            if img.mode != "RGB":
                img = img.convert("RGB")

            clean = transform(img).unsqueeze(0)

            noisy = clean + noise_level * torch.randn_like(clean)
            noisy = torch.clamp(noisy, 0., 1.)

            inference_start = time.time()
            with torch.no_grad():
                denoised = model(noisy)
            inference_time = time.time() - inference_start

            psnr_value = psnr_metric(denoised, clean)
            ssim_value = ssim_metric(denoised, clean)

            cpu_runtime = time.time() - cpu_start_time

            psnr_values.append(psnr_value.item())
            ssim_values.append(ssim_value.item())
            inference_times.append(inference_time)
            cpu_runtimes.append(cpu_runtime)
            sample_names.append(img_name)

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_inference_time = np.mean(inference_times)
    avg_cpu_runtime = np.mean(cpu_runtimes)

    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    std_inference_time = np.std(inference_times)
    std_cpu_runtime = np.std(cpu_runtimes)

    print("AVERAGE METRICS:")
    print("-" * 30)
    print(f"PSNR:           {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"SSIM:           {avg_ssim:.3f} ± {std_ssim:.3f}")
    print(f"Inference Time: {avg_inference_time:.6f} ± {std_inference_time:.6f} s")
    print(f"CPU Runtime:    {avg_cpu_runtime:.6f} ± {std_cpu_runtime:.6f} s")
    print()

    print("INDIVIDUAL RESULTS:")
    print("-" * 70)
    print(f"{'Image':<20} {'PSNR (dB)':<10} {'SSIM':<8} {'Inference Time (s)':<18} {'CPU Runtime (s)':<15}")
    print("-" * 70)

    for i, name in enumerate(sample_names):
        print(f"{name:<20} {psnr_values[i]:<10.2f} {ssim_values[i]:<8.3f} "
              f"{inference_times[i]:<18.6f} {cpu_runtimes[i]:<15.6f}")

    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_inference_time': avg_inference_time,
        'avg_cpu_runtime': avg_cpu_runtime,
        'std_psnr': std_psnr,
        'std_ssim': std_ssim,
        'std_inference_time': std_inference_time,
        'std_cpu_runtime': std_cpu_runtime,
        'individual_results': {
            'images': sample_names,
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'inference_times': inference_times,
            'cpu_runtimes': cpu_runtimes
        }
    }


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    results = test_performance()
