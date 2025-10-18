import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.models.dncnn import DnCnn
from PIL import Image
import cv2

# python -m src.visualize.visualize_denoising

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DnCnn().to(device)
# load the trained model parameters from checkpoint
checkpoint = torch.load("src/checkpoints/dncnn_best.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# load test image
img_path = "BSD68/test001.png"
img = Image.open(img_path)

# resize image and convert to tensor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])
clean = transform(img).unsqueeze(0).to(device)

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
    arr = t.squeeze().detach().cpu().numpy()
    return np.clip(arr, 0, 1)

clean_np = tensor_to_np(clean)
noisy_np = tensor_to_np(noisy)
denoised_np = tensor_to_np(denoised)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.imshow(clean_np.squeeze(), cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(noisy_np.squeeze(), cmap='gray')
plt.title("Noisy (σ=25)")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(denoised_np.squeeze(), cmap='gray')
plt.title(f"Denoised\nPSNR: {psnr_value:.2f} dB\nSSIM: {ssim_value:.3f}")
plt.axis("off")

plt.tight_layout()
plt.show()
