"""
DnCNN Model Performance Evaluation Script

Evaluates a trained DnCNN model on 10 randomly selected BSD68 images.
Calculates PSNR, SSIM, inference time, and CPU runtime metrics.

Usage: python -m src.test.test_dncnn
"""

import os
import time
import random
import torch
import numpy as np
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.models.dncnn import DnCnn
from PIL import Image

def test_dncnn_performance():
    """
    Test DnCNN model performance on randomly selected BSD68 images.
    Calculates PSNR, SSIM, inference time, and CPU runtime metrics.
    """
    # Initialize model on CPU
    device = torch.device('cpu')
    model = DnCnn().to(device)
    
    # Load trained model parameters from checkpoint
    checkpoint_path = "src/checkpoints/dncnn_best.pth"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get list of all BSD68 images
    bsd68_dir = "BSD68"
    if not os.path.exists(bsd68_dir):
        raise FileNotFoundError(f"BSD68 directory not found at {bsd68_dir}")
    
    all_images = [f for f in os.listdir(bsd68_dir) if f.endswith('.png')]
    if len(all_images) < 10:
        raise ValueError(f"Not enough images in BSD68 directory. Found {len(all_images)}, need at least 10")
    
    # Randomly select 10 images
    selected_images = random.sample(all_images, 10)
    
    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    
    # Metrics initialization
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    
    # Storage for metrics
    psnr_values = []
    ssim_values = []
    inference_times = []
    cpu_runtimes = []
    
    # Noise level (σ=25)
    noise_level = 25 / 255.0
    
    for img_name in selected_images:
        img_path = os.path.join(bsd68_dir, img_name)
        
        # Start measuring total CPU runtime
        cpu_start_time = time.time()
        
        # Load and preprocess image
        img = Image.open(img_path)
        if img.mode != 'L':
            img = img.convert('L')
        
        clean = transform(img).unsqueeze(0)
        
        # Add Gaussian noise
        noisy = clean + noise_level * torch.randn_like(clean)
        noisy = torch.clamp(noisy, 0., 1.)
        
        # Measure pure inference time (model forward pass only)
        inference_start = time.time()
        with torch.no_grad():
            denoised = model(noisy)
        inference_end = time.time()
        inference_time = inference_end - inference_start
        
        # Calculate PSNR and SSIM
        psnr_value = psnr_metric(denoised, clean)
        ssim_value = ssim_metric(denoised, clean)
        
        # End measuring total CPU runtime
        cpu_end_time = time.time()
        cpu_runtime = cpu_end_time - cpu_start_time
        
        # Store metrics
        psnr_values.append(psnr_value.item())
        ssim_values.append(ssim_value.item())
        inference_times.append(inference_time)
        cpu_runtimes.append(cpu_runtime)
    
    # Calculate averages
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_inference_time = np.mean(inference_times)
    avg_cpu_runtime = np.mean(cpu_runtimes)
    
    # Calculate standard deviations
    std_psnr = np.std(psnr_values)
    std_ssim = np.std(ssim_values)
    std_inference_time = np.std(inference_times)
    std_cpu_runtime = np.std(cpu_runtimes)
    
    # Print results
    print("AVERAGE METRICS:")
    print("-" * 30)
    print(f"PSNR:           {avg_psnr:.2f} ± {std_psnr:.2f} dB")
    print(f"SSIM:           {avg_ssim:.3f} ± {std_ssim:.3f}")
    print(f"Inference Time: {avg_inference_time:.6f} ± {std_inference_time:.6f} s")
    print(f"CPU Runtime:    {avg_cpu_runtime:.6f} ± {std_cpu_runtime:.6f} s")
    print()
    
    # Individual results
    print("INDIVIDUAL RESULTS:")
    print("-" * 70)
    print(f"{'Image':<15} {'PSNR (dB)':<10} {'SSIM':<8} {'Inference Time (s)':<18} {'CPU Runtime (s)':<15}")
    print("-" * 70)
    
    for i, img_name in enumerate(selected_images):
        print(f"{img_name:<15} {psnr_values[i]:<10.2f} {ssim_values[i]:<8.3f} "
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
            'images': selected_images,
            'psnr_values': psnr_values,
            'ssim_values': ssim_values,
            'inference_times': inference_times,
            'cpu_runtimes': cpu_runtimes
        }
    }

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the test
    results = test_dncnn_performance()
