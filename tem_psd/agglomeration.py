from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image


def detect_agglomerates(mask: np.ndarray, scale_nm_per_px: float, min_gap_nm: float = 2.0):
    lbl = label(mask)
    props = regionprops(lbl)
    classes = {p.label: "single" for p in props}

    for p in props:
        region = lbl == p.label
        hull = convex_hull_image(region)
        hull_area = max(hull.sum(), 1)
        ratio = p.area / hull_area
        if ratio < 0.75:
            classes[p.label] = "agglomerate"

    if len(props) > 1:
        centroids = np.array([p.centroid for p in props], dtype=float)
        labels = np.array([p.label for p in props], dtype=int)
        tree = KDTree(centroids)
        px_thresh = min_gap_nm / scale_nm_per_px
        for idx, c in enumerate(centroids):
            d, j = tree.query(c, k=2)
            if d[1] < px_thresh:
                for l in (labels[idx], labels[j[1]]):
                    if classes[l] == "single":
                        classes[l] = "uncertain"

    total = len(props)
    agg_count = sum(v == "agglomerate" for v in classes.values())
    index = (agg_count / total) * 100 if total else 0.0
    return classes, index


def save_agglomeration_report(classes: dict, index: float, out_path: Path):
    counts = {
        "single": sum(v == "single" for v in classes.values()),
        "agglomerate": sum(v == "agglomerate" for v in classes.values()),
        "uncertain": sum(v == "uncertain" for v in classes.values()),
    }
    text = (
        "Agglomeration report\n"
        f"single: {counts['single']}\n"
        f"agglomerate: {counts['agglomerate']}\n"
        f"uncertain: {counts['uncertain']}\n"
        f"Agglomeration index (%): {index:.2f}\n"
    )
    out_path.write_text(text)


def save_agglomeration_labels(classes: dict[int, str], out_path: Path):
    rows = [{"label": int(k), "class": v} for k, v in sorted(classes.items())]
    pd.DataFrame(rows).to_csv(out_path, index=False)
