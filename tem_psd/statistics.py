from __future__ import annotations

import numpy as np
import pandas as pd


def compute_statistics(df: pd.DataFrame, image_area_nm2: float) -> dict:
    if df.empty:
        return {
            "count": 0,
            "mean_nm": 0.0,
            "std_nm": 0.0,
            "median_nm": 0.0,
            "d10_nm": 0.0,
            "d50_nm": 0.0,
            "d90_nm": 0.0,
            "cv_percent": 0.0,
            "number_density_per_nm2": 0.0,
        }
    d = df["ecd_nm"].to_numpy()
    mean = float(np.mean(d))
    std = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    return {
        "count": int(len(d)),
        "mean_nm": mean,
        "std_nm": std,
        "median_nm": float(np.median(d)),
        "d10_nm": float(np.percentile(d, 10)),
        "d50_nm": float(np.percentile(d, 50)),
        "d90_nm": float(np.percentile(d, 90)),
        "cv_percent": float((std / mean) * 100) if mean else 0.0,
        "number_density_per_nm2": float(len(d) / max(image_area_nm2, 1e-8)),
    }


def format_stats(stats: dict) -> str:
    lines = [
        f"Total particle count: {stats['count']}",
        f"Mean diameter: {stats['mean_nm']:.3f} ± {stats['std_nm']:.3f} nm",
        f"Median diameter: {stats['median_nm']:.3f} nm",
        f"D10/D50/D90: {stats['d10_nm']:.3f} / {stats['d50_nm']:.3f} / {stats['d90_nm']:.3f} nm",
        f"CV%: {stats['cv_percent']:.2f}",
        f"Number density: {stats['number_density_per_nm2']:.6e} particles/nm²",
    ]
    return "\n".join(lines)
