"""
Model Visualization Script

Evaluates a trained model on a sample image from BSDS300 dataset.
Calculates PSNR, SSIM and display it in a matplotlib plot.

Usage: 
    python -m src.visualize.visualize_denoising dncnn
    python -m src.visualize.visualize_denoising nafnet
    python -m src.visualize.visualize_denoising nafnet_sidd
"""

import time
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from PIL import Image
import sys

if len(sys.argv) > 1:
    MODEL_TYPE = sys.argv[1]
else:
    MODEL_TYPE = "dncnn"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if MODEL_TYPE == "dncnn":
    from src.models.dncnn import DnCnn
    model = DnCnn(image_channels=3).to(device)
    # load the trained model parameters from checkpoint
    checkpoint = torch.load("src/checkpoints/dncnn_best.pth", map_location=device)

elif MODEL_TYPE == "nafnet":
    from src.models.nafnet import NAFNet
    model = NAFNet(image_channels=3).to(device)
    # load the trained model parameters from checkpoint
    checkpoint = torch.load("src/checkpoints/nafnet_small_best.pth", map_location=device)

elif MODEL_TYPE == "nafnet_sidd":
    from src.models.nafnet import NAFNet
    model = NAFNet(
        image_channels=3,
        width=32,
        middle_blk_num=1,
        enc_blk_nums=(1,1,1,8),
        dec_blk_nums=(1,1,1,1),
    ).to(device)
    checkpoint = torch.load("src/checkpoints/nafnet_small_best_sidd.pth", map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# load test image
if MODEL_TYPE == "nafnet_sidd":
    from src.data.sidd_dataset import SIDDDataset, split_sidd_dataset
    sidd_root = "SIDD_Small_sRGB_Only/Data"

    dataset = SIDDDataset(dirs=[sidd_root], size=128, random_crop=False)
    _, _, test_set = split_sidd_dataset(dataset, seed=42)

    noisy, clean = test_set[1]
    noisy = noisy.unsqueeze(0).to(device)
    clean = clean.unsqueeze(0).to(device)
else:
    img_path = "BSDS300/images/test/3096.jpg"
    img = Image.open(img_path)
    img = img.convert("RGB")

    # resize image and convert to tensor
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    clean = transform(img).unsqueeze(0).to(device)

if MODEL_TYPE == "nafnet_sidd":
    pass
else:
    # add Gaussian noise (σ=25)
    noise_level = 25 / 255.0
    noisy = clean + noise_level * torch.randn_like(clean)
    noisy = torch.clamp(noisy, 0., 1.)

# denoise image
with torch.no_grad():
    denoised = model(noisy)

# calculate PSNR and SSIM
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
psnr_value = psnr_metric(denoised, clean)
ssim_value = ssim_metric(denoised, clean)

def tensor_to_np(t):
    # convert tensor to numpy for visualization
    arr = t.squeeze(0).permute(1,2,0).detach().cpu().numpy()
    return np.clip(arr, 0, 1)

clean_np = tensor_to_np(clean)
noisy_np = tensor_to_np(noisy)
denoised_np = tensor_to_np(denoised)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(clean_np)  
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noisy_np)
if MODEL_TYPE == "nafnet_sidd":
    plt.title("Noisy (real SIDD)")
else:
    plt.title("Noisy (σ=25)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(denoised_np) 
plt.title(f"Denoised\nPSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.3f}")
plt.axis("off")

plt.tight_layout()
plt.show()
