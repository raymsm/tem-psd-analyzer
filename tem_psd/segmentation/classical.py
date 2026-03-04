from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import morphology
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed


def classical_segment(image: np.ndarray) -> np.ndarray:
    thresh = threshold_otsu(image)
    binary = image < thresh if image.mean() > thresh else image > thresh
    clean = morphology.remove_small_objects(binary, min_size=25)
    clean = morphology.remove_small_holes(clean, area_threshold=25)
    distance = ndi.distance_transform_edt(clean)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=clean)
    markers = np.zeros_like(clean, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    markers, _ = ndi.label(markers > 0)
    labels = watershed(-distance, markers, mask=clean)
    return labels > 0
