import numpy as np

from tem_psd.agglomeration import detect_agglomerates


def test_agglomeration_runs():
    m = np.zeros((64, 64), dtype=bool)
    m[10:20, 10:20] = True
    m[22:32, 22:32] = True
    classes, idx = detect_agglomerates(m, scale_nm_per_px=1.0)
    assert isinstance(classes, dict)
    assert idx >= 0
