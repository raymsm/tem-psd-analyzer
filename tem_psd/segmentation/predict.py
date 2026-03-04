from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .classical import classical_segment
from .unet import UNet

MODEL_PATH = Path.home() / ".tem_psd" / "models" / "unet.pth"


def _gaussian_weight(tile_size: int = 512) -> np.ndarray:
    y, x = np.mgrid[0:tile_size, 0:tile_size]
    c = (tile_size - 1) / 2
    sigma = tile_size / 6
    w = np.exp(-((x - c) ** 2 + (y - c) ** 2) / (2 * sigma**2))
    return w.astype(np.float32)


def tiled_predict(model: UNet, image: np.ndarray, device: torch.device, tile: int = 512, overlap: int = 64) -> np.ndarray:
    h, w = image.shape
    stride = tile - overlap
    weight = _gaussian_weight(tile)
    out = np.zeros((h, w), dtype=np.float32)
    den = np.zeros((h, w), dtype=np.float32)
    for y in range(0, max(1, h - overlap), stride):
        for x in range(0, max(1, w - overlap), stride):
            y2, x2 = min(y + tile, h), min(x + tile, w)
            patch = image[y:y2, x:x2]
            pad = np.pad(patch, ((0, tile - patch.shape[0]), (0, tile - patch.shape[1])), mode="reflect")
            inp = torch.from_numpy(pad[None, None, ...]).float().to(device)
            with torch.no_grad():
                pred = torch.sigmoid(model(inp)).cpu().numpy()[0, 0]
            pred = pred[: patch.shape[0], : patch.shape[1]]
            ww = weight[: patch.shape[0], : patch.shape[1]]
            out[y:y2, x:x2] += pred * ww
            den[y:y2, x:x2] += ww
    return (out / np.maximum(den, 1e-8)) > 0.5


def segment_particles(image: np.ndarray, model_path: Path = MODEL_PATH) -> np.ndarray:
    model_path = Path(model_path)
    if not model_path.exists():
        return classical_segment(image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return tiled_predict(model, image, device)
