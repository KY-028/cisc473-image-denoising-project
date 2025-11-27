import torch
import torch.nn as nn
import os
import numpy as np
import cv2
from torch.utils.data import Dataset

class DenoiseDataset(Dataset):
    """
    loads grayscale images, adds Gaussian noise, and returns (noisy, clean) pairs for denoising.
    """
    def __init__(self, dirs, size=64, sigma=25/255.0, channels=3):
        """
        Initialize the dataset by collecting all image file paths.

        Arguments:
            dirs: Path to the directory containing images.
            size: Resize all images to (size x size).
            sigma: Standard deviation of intended Gaussian noise generation.
        """

        if isinstance(dirs, str):
            dirs = [dirs]
        self.files = []
        for d in dirs:
            for f in os.listdir(d):
                if f.lower().endswith((".png", ".jpg")):
                    self.files.append(os.path.join(d, f))
        self.size = size
        self.sigma = sigma
        self.channels = channels

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
        # load image
        img = cv2.imread(self.files[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))

        # normalize pixels to [0,1]
        img = img.astype(np.float32) / 255.0

        # add Gaussian noise
        noise = np.random.randn(*img.shape) * self.sigma
        noisy = np.clip(img + noise, 0, 1)

        # convert to torch tensors, shape (1, H, W)
        clean = torch.tensor(img.copy(), dtype=torch.float32).permute(2,0,1)
        noisy = torch.tensor(noisy.copy(), dtype=torch.float32).permute(2,0,1)
        return noisy, clean