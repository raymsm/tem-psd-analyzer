import numpy as np

from tem_psd.measurement import measure_particles


def test_measure_particles_columns():
    m = np.zeros((32, 32), dtype=bool)
    m[5:10, 5:10] = True
    df = measure_particles(m, scale_nm_per_px=1.0)
    assert "ecd_nm" in df.columns
