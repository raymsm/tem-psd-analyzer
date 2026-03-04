from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates


def random_flip_rot(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator):
    if rng.random() < 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        image = np.flipud(image)
        mask = np.flipud(mask)
    k = int(rng.integers(0, 4))
    image = np.rot90(image, k)
    mask = np.rot90(mask, k)
    return image, mask


def elastic_deform(image: np.ndarray, mask: np.ndarray, rng: np.random.Generator, alpha: float = 30, sigma: float = 4):
    shape = image.shape
    dx = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((rng.random(shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    idx = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    img_d = map_coordinates(image, idx, order=1, mode="reflect").reshape(shape)
    msk_d = map_coordinates(mask, idx, order=0, mode="reflect").reshape(shape)
    return img_d, msk_d


def brightness_jitter(image: np.ndarray, rng: np.random.Generator):
    factor = rng.uniform(0.8, 1.2)
    return np.clip(image * factor, 0, 1)
