from pathlib import Path

from tem_psd.pipeline import analyze_image
from tem_psd.training.synthetic import generate_synthetic_dataset


def main():
    demo_root = Path("./demo_output")
    ds = demo_root / "synthetic_dataset"
    images, _ = generate_synthetic_dataset(ds, n=1, size=512)
    image_path = next(images.glob("*.png"))
    out_dir, stats = analyze_image(image_path, scale=0.42, unit="nm", output=demo_root)
    print("Demo complete:", out_dir)
    print(stats)


if __name__ == "__main__":
    main()
