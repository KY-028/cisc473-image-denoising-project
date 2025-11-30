import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, random_split


class SIDDDataset(Dataset):
    """
    Loads paired (noisy, clean) images from the SIDD Small sRGB dataset.
    API is kept close to DenoiseDataset: returns (noisy, clean) tensors
    resized/cropped to (size x size) and normalized to [0, 1].
    """
    def __init__(self, dirs, size=128, random_crop=True, channels=3):
        """
        Arguments:
            dirs: Path or list of paths pointing to SIDD scene folders (or the Data root).
            size: Crop size (and fallback resize) to produce patches of shape size x size.
            random_crop: If True, draw random crops; otherwise take center crop.
            channels: Kept for interface consistency with DenoiseDataset.
        """
        if isinstance(dirs, str):
            dirs = [dirs]

        self.files = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for scene in sorted(os.listdir(d)):
                scene_dir = os.path.join(d, scene)
                if not os.path.isdir(scene_dir):
                    continue
                gt_path = os.path.join(scene_dir, "GT_SRGB_010.PNG")
                noisy_path = os.path.join(scene_dir, "NOISY_SRGB_010.PNG")
                if os.path.isfile(gt_path) and os.path.isfile(noisy_path):
                    self.files.append((noisy_path, gt_path))

        if not self.files:
            raise FileNotFoundError(f"No GT/NOISY pairs found in {dirs}")

        self.size = size
        self.random_crop = random_crop
        self.channels = channels

    def __len__(self):
        return len(self.files)

    def _crop(self, noisy, clean):
        h, w, _ = noisy.shape
        if h < self.size or w < self.size:
            noisy = cv2.resize(noisy, (self.size, self.size))
            clean = cv2.resize(clean, (self.size, self.size))
            return noisy, clean

        if self.random_crop:
            x = random.randint(0, w - self.size)
            y = random.randint(0, h - self.size)
        else:
            x = (w - self.size) // 2
            y = (h - self.size) // 2

        noisy = noisy[y:y + self.size, x:x + self.size]
        clean = clean[y:y + self.size, x:x + self.size]
        return noisy, clean

    def __getitem__(self, idx):
        noisy_path, clean_path = self.files[idx]

        noisy = cv2.imread(noisy_path, cv2.IMREAD_COLOR)
        clean = cv2.imread(clean_path, cv2.IMREAD_COLOR)
        noisy = cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB)
        clean = cv2.cvtColor(clean, cv2.COLOR_BGR2RGB)

        noisy, clean = self._crop(noisy, clean)

        noisy = noisy.astype(np.float32) / 255.0
        clean = clean.astype(np.float32) / 255.0

        noisy = torch.tensor(noisy.copy(), dtype=torch.float32).permute(2, 0, 1)
        clean = torch.tensor(clean.copy(), dtype=torch.float32).permute(2, 0, 1)

        return noisy, clean


def split_sidd_dataset(dataset, seed=42):
    """
    Deterministically split SIDD into train/val/test with ratios 75/15/15.
    """
    total = len(dataset)
    train_len = int(total * 0.75)
    val_len = int(total * 0.15)
    test_len = total - train_len - val_len
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)

