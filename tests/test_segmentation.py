import numpy as np

from tem_psd.segmentation.classical import classical_segment


def test_classical_segment_returns_mask():
    img = np.random.rand(128, 128).astype(np.float32)
    mask = classical_segment(img)
    assert mask.shape == img.shape
    assert mask.dtype == bool
