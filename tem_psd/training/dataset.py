from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import brightness_jitter, elastic_deform, random_flip_rot


class SegmentationDataset(Dataset):
    def __init__(self, image_dir: str | Path, mask_dir: str | Path, augment: bool = False):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        self.files = sorted([p for p in self.image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])
        self.rng = np.random.default_rng(42)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / (img_path.stem + ".png")
        img = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255.0
        mask = (np.array(Image.open(mask_path).convert("L"), dtype=np.float32) > 127).astype(np.float32)
        if self.augment:
            img, mask = random_flip_rot(img, mask, self.rng)
            if self.rng.random() < 0.5:
                img, mask = elastic_deform(img, mask, self.rng)
            img = brightness_jitter(img, self.rng)
        return torch.from_numpy(img[None, ...]), torch.from_numpy(mask[None, ...])
