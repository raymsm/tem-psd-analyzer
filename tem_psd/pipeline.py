from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .agglomeration import detect_agglomerates, save_agglomeration_labels, save_agglomeration_report
from .measurement import measure_particles
from .preprocessing import detect_scale_bar_nm_per_px, load_image, preprocess_image
from .segmentation.predict import segment_particles
from .statistics import compute_statistics, format_stats
from .visualization import save_agglomeration_overlay, save_histogram, save_overlay, save_scatter


def timestamped_output(base: str | Path) -> Path:
    t = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(base) / t
    out.mkdir(parents=True, exist_ok=True)
    return out


def analyze_image(input_path: str | Path, scale: float | None, unit: str, output: str | Path, min_size_nm: float = 0.0):
    out_dir = timestamped_output(output)
    img = load_image(input_path)
    proc = preprocess_image(img)
    nm_per_px = scale if scale is not None else detect_scale_bar_nm_per_px(img)
    if nm_per_px is None:
        raise ValueError("Scale not provided and automatic scale-bar detection failed.")
    if unit.lower() != "nm":
        raise ValueError("Only nm is currently supported for --unit")

    mask = segment_particles(proc)
    df = measure_particles(mask, nm_per_px, min_ecd_nm=min_size_nm)
    image_area_nm2 = img.shape[0] * img.shape[1] * (nm_per_px**2)
    stats = compute_statistics(df, image_area_nm2)

    df.to_csv(out_dir / "particles.csv", index=False)
    (out_dir / "results_summary.txt").write_text(format_stats(stats))
    save_histogram(df, out_dir)
    save_scatter(df, out_dir)
    save_overlay(proc, mask, out_dir / "overlay.png")
    return out_dir, stats


def analyze_batch(input_dir: str | Path, scale: float | None, unit: str, output: str | Path, min_size_nm: float = 0.0):
    input_dir = Path(input_dir)
    files = [p for p in input_dir.iterdir() if p.suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".dm3"}]
    batch_out = timestamped_output(output)
    all_rows = []
    for fp in tqdm(files, desc="Batch analyzing"):
        out, _ = analyze_image(fp, scale, unit, batch_out, min_size_nm=min_size_nm)
        df = pd.read_csv(out / "particles.csv")
        if not df.empty:
            df["source_file"] = fp.name
            all_rows.append(df)
    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(batch_out / "batch_particles.csv", index=False)
    return batch_out


def run_agglomeration(input_path: str | Path, scale: float, output: str | Path, min_gap_nm: float = 2.0):
    out_dir = timestamped_output(output)
    img = load_image(input_path)
    proc = preprocess_image(img)
    mask = segment_particles(proc)
    classes, index = detect_agglomerates(mask, scale, min_gap_nm=min_gap_nm)
    save_agglomeration_overlay(proc, mask, classes, out_dir / "overlay.png")
    save_agglomeration_report(classes, index, out_dir / "agglomeration_report.txt")
    save_agglomeration_labels(classes, out_dir / "agglomeration_labels.csv")
    return out_dir, index
