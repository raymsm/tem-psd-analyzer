from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    from ncempy.io import dm
except Exception:  # pragma: no cover
    dm = None


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".dm3":
        if dm is None:
            raise ImportError("ncempy is required for .dm3 loading")
        data = dm.fileDM(str(path)).getDataset(0)["data"]
        img = np.asarray(data, dtype=np.float32)
    else:
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    return img


def preprocess_image(image: np.ndarray) -> np.ndarray:
    image_u8 = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_u8)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    norm = denoised.astype(np.float32) / 255.0
    return norm


def detect_scale_bar_nm_per_px(image: np.ndarray) -> Optional[float]:
    """Simple OCR-free heuristic based on bright horizontal bar near image bottom.

    Returns None if not confidently detected.
    """
    h, w = image.shape
    strip = image[int(h * 0.85) :, :]
    thresh = np.percentile(strip, 99)
    mask = strip >= thresh
    row_counts = mask.sum(axis=1)
    if row_counts.size == 0:
        return None
    r = int(np.argmax(row_counts))
    if row_counts[r] < w * 0.1:
        return None
    row = mask[r]
    starts = np.where(np.diff(np.concatenate([[0], row.view(np.int8), [0]])) == 1)[0]
    ends = np.where(np.diff(np.concatenate([[0], row.view(np.int8), [0]])) == -1)[0]
    if len(starts) == 0:
        return None
    lengths = ends - starts
    bar_px = int(lengths.max())
    if bar_px < 20:
        return None
    # Unknown absolute nm without text; assume common 100 nm bar as fallback heuristic.
    return 100.0 / bar_px


def pad_to_square(image: np.ndarray, size: int = 512) -> Tuple[np.ndarray, Tuple[int, int]]:
    h, w = image.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    return padded, (h, w)
