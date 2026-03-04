from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
from skimage.draw import disk


def generate_synthetic_dataset(out_dir: str | Path, n: int = 100, size: int = 512):
    out_dir = Path(out_dir)
    image_dir = out_dir / "images"
    mask_dir = out_dir / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for i in range(n):
        img = rng.normal(0.55, 0.08, (size, size)).astype(np.float32)
        mask = np.zeros((size, size), dtype=np.uint8)
        count = int(rng.integers(25, 70))
        for _ in range(count):
            r = int(rng.integers(5, 20))
            cy = int(rng.integers(r, size - r))
            cx = int(rng.integers(r, size - r))
            rr, cc = disk((cy, cx), r, shape=img.shape)
            img[rr, cc] -= rng.uniform(0.2, 0.45)
            mask[rr, cc] = 255
        img = gaussian_filter(img, sigma=1.0)
        img = np.clip(img, 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(image_dir / f"sample_{i:04d}.png")
        Image.fromarray(mask).save(mask_dir / f"sample_{i:04d}.png")

    return image_dir, mask_dir
