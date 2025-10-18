import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader

class DenoiseDataset(Dataset):
    """
    loads grayscale images, adds Gaussian noise, and returns (noisy, clean) pairs for denoising.
    """
    def __init__(self, dir, size=64, sigma=25/255.0):
        """
        Initialize the dataset by collecting all image file paths.

        Arguments:
            dir: Path to the directory containing images.
            size: Resize all images to (size x size).
            sigma: Standard deviation of intended Gaussian noise generation.
        """
        self.files = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(".png")]
        self.size = size
        self.sigma = sigma

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        """
        Get a noisy-clean image pair based on index.
        
        Arguments:
            idx: Index of the image to load.
        Returns:
            A tuple (noisy_image, clean_image), both as torch tensors of shape (1, H, W).
        """
        # load grayscale image
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        
        # normalize pixels to [0,1]
        img = img.astype(np.float32) / 255.0

        # add Gaussian noise
        noise = np.random.randn(*img.shape) * self.sigma
        noisy = np.clip(img + noise, 0, 1)

        # convert to torch tensors, shape (1, H, W)
        clean = torch.tensor(img.copy(), dtype=torch.float32).unsqueeze(0)
        noisy = torch.tensor(noisy.copy(), dtype=torch.float32).unsqueeze(0)
        return noisy, clean