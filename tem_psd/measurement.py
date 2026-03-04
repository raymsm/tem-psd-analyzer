from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops


@dataclass
class ParticleMeasurement:
    label: int
    ecd_px: float
    ecd_nm: float
    major_axis_px: float
    minor_axis_px: float
    major_axis_nm: float
    minor_axis_nm: float
    aspect_ratio: float
    circularity: float
    area_px2: float
    area_nm2: float
    centroid_y: float
    centroid_x: float


def measure_particles(mask: np.ndarray, scale_nm_per_px: float, min_ecd_nm: float = 0.0) -> pd.DataFrame:
    lbl = label(mask)
    rows = []
    for p in regionprops(lbl):
        area = float(p.area)
        per = float(p.perimeter) if p.perimeter > 0 else 1.0
        ecd_px = math.sqrt(4 * area / math.pi)
        ecd_nm = ecd_px * scale_nm_per_px
        if ecd_nm < min_ecd_nm:
            continue
        major = float(p.major_axis_length)
        minor = max(float(p.minor_axis_length), 1e-8)
        rows.append(
            ParticleMeasurement(
                label=int(p.label),
                ecd_px=ecd_px,
                ecd_nm=ecd_nm,
                major_axis_px=major,
                minor_axis_px=minor,
                major_axis_nm=major * scale_nm_per_px,
                minor_axis_nm=minor * scale_nm_per_px,
                aspect_ratio=major / minor,
                circularity=float((4 * math.pi * area) / (per**2)),
                area_px2=area,
                area_nm2=area * (scale_nm_per_px**2),
                centroid_y=float(p.centroid[0]),
                centroid_x=float(p.centroid[1]),
            )
        )
    return pd.DataFrame(asdict(r) for r in rows)
