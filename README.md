# tem-psd-analyzer

Fully offline CLI tool for TEM/HRTEM nanoparticle segmentation and particle size distribution (PSD) analysis.

## Features
- Single-image and batch analysis
- U-Net segmentation (PyTorch) with automatic fallback to classical segmentation
- Particle measurements (ECD, axes, circularity, area)
- Statistics (mean/std, median, D10/D50/D90, CV, number density)
- Agglomeration detection (concavity + nearest-neighbor)
- Synthetic data generator for immediate training/testing
- Fully offline (no external API calls)

## Installation
```bash
pip install -e .
```

## CLI Usage
```bash
# Analyze one image
tem-psd analyze --input image.tif --scale 0.42 --unit nm --output ./results

# Batch analyze folder
tem-psd batch --input ./images --scale 0.42 --unit nm --output ./results

# Train U-Net
tem-psd train --data ./dataset --epochs 50 --output ./models

# Train with synthetic bootstrapping
tem-psd train --data ./dataset --epochs 20 --synthetic --output ./models

# Agglomeration detection
tem-psd agglomerate --input image.tif --scale 0.42 --output ./results
```

## Demo
```bash
python demo.py
```
This generates one synthetic TEM-like image and runs the full analysis pipeline into `./demo_output/`.

## Expected Dataset Layout for Training
```text
dataset/
  images/
    sample_0001.png
  masks/
    sample_0001.png
```

## Architecture (ASCII)
```text
                    +------------------+
Input TEM image --->| preprocessing    |---> normalized image
                    | CLAHE + denoise  |
                    +---------+--------+
                              |
                              v
                    +------------------+
                    | segmentation     |
                    | U-Net or fallback|
                    +---------+--------+
                              |
                              v
                    +------------------+
                    | measurement      |
                    | regionprops      |
                    +---------+--------+
                              |
                 +------------+-------------+
                 v                          v
        +------------------+        +------------------+
        | statistics       |        | visualization    |
        | PSD metrics      |        | histogram/overlay|
        +------------------+        +------------------+
```

## Output Artifacts
Each run writes to a timestamped folder:
`results/YYYYMMDD_HHMMSS/`

- `results_summary.txt`
- `particles.csv`
- `histogram.png`, `histogram.svg`
- `scatter.png`, `scatter.svg`
- `overlay.png`
- `agglomeration_report.txt` (agglomerate command)
- `agglomeration_labels.csv` (per-particle class: single/agglomerate/uncertain)

## Notes
- If model weights are not available at `~/.tem_psd/models/unet.pth`, segmentation falls back to Otsu + morphology + watershed.
- If `--scale` is omitted, a simple pixel-based scale bar estimator attempts detection from the bottom image strip.
