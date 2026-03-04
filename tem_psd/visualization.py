from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from skimage import measure
from skimage.measure import label, regionprops


def save_histogram(df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if not df.empty:
        sns.histplot(df["ecd_nm"], kde=True, bins=20)
    plt.xlabel("Equivalent Circular Diameter (nm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_dir / "histogram.png", dpi=200)
    plt.savefig(out_dir / "histogram.svg")
    plt.close()


def save_scatter(df, out_dir: Path):
    plt.figure(figsize=(6, 5))
    if not df.empty:
        sns.scatterplot(data=df, x="ecd_nm", y="circularity", s=25)
    plt.xlabel("ECD (nm)")
    plt.ylabel("Circularity")
    plt.tight_layout()
    plt.savefig(out_dir / "scatter.png", dpi=200)
    plt.savefig(out_dir / "scatter.svg")
    plt.close()


def save_overlay(image: np.ndarray, mask: np.ndarray, out_path: Path):
    rgb = np.dstack([image, image, image])
    rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-8)
    boundaries = measure.find_contours(mask.astype(float), 0.5)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, cmap="gray")
    for contour in boundaries:
        ax.plot(contour[:, 1], contour[:, 0], color="lime", linewidth=1)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def save_agglomeration_overlay(image: np.ndarray, mask: np.ndarray, classes: dict[int, str], out_path: Path):
    rgb = np.dstack([image, image, image])
    rgb = (rgb - rgb.min()) / max(rgb.max() - rgb.min(), 1e-8)
    lbl = label(mask)
    colors = {"single": "lime", "agglomerate": "red", "uncertain": "gold"}

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(rgb, cmap="gray")
    for prop in regionprops(lbl):
        color = colors.get(classes.get(prop.label, "single"), "white")
        contours = measure.find_contours((lbl == prop.label).astype(float), 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1.2)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)
