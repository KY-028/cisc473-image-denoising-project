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
    def __init__(self, folder, size=64, sigma=25/255.0, augment=True):
        """
        Initialize the dataset by collecting all image file paths.
        folder : Path to the folder containing images.
        size : Resize all images to (size x size).
        sigma : Standard deviation of Gaussian noise.
        augment : Whether to apply random flips for data augmentation.
        """
        self.files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]
        self.size = size
        self.sigma = sigma
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # load grayscale image
        img = cv2.imread(self.files[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.size, self.size))
        # normalize to [0,1]
        img = img.astype(np.float32) / 255.0

        # flip for simple augmentation
        if self.augment:
            if np.random.rand() < 0.5:
                # flip left-right
                img = np.fliplr(img)
            if np.random.rand() < 0.5:
                # flip up-down
                img = np.flipud(img)

        # add Gaussian noise
        noise = np.random.randn(*img.shape) * self.sigma
        noisy = np.clip(img + noise, 0, 1)

        # convert to torch tensors, shape (1, H, W)
        clean = torch.tensor(img).unsqueeze(0)
        noisy = torch.tensor(noisy).unsqueeze(0)
        return noisy, clean