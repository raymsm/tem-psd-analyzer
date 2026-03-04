from __future__ import annotations

from pathlib import Path

import click

from .pipeline import analyze_batch, analyze_image, run_agglomeration
from .training.trainer import train_model


@click.group()
def cli():
    """TEM/HRTEM particle size distribution analyzer."""


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--scale", type=float, required=False)
@click.option("--unit", default="nm")
@click.option("--output", default="./results")
@click.option("--min-size", "min_size_nm", default=0.0, type=float)
def analyze(input_path, scale, unit, output, min_size_nm):
    out_dir, _ = analyze_image(input_path, scale, unit, output, min_size_nm)
    click.echo(f"Analysis complete: {out_dir}")


@cli.command()
@click.option("--input", "input_dir", required=True, type=click.Path(exists=True))
@click.option("--scale", type=float, required=False)
@click.option("--unit", default="nm")
@click.option("--output", default="./results")
@click.option("--min-size", "min_size_nm", default=0.0, type=float)
def batch(input_dir, scale, unit, output, min_size_nm):
    out = analyze_batch(input_dir, scale, unit, output, min_size_nm=min_size_nm)
    click.echo(f"Batch complete: {out}")


@cli.command()
@click.option("--data", "data_dir", required=True, type=click.Path())
@click.option("--epochs", default=50, type=int)
@click.option("--output", "output_dir", default="./models")
@click.option("--synthetic", is_flag=True)
def train(data_dir, epochs, output_dir, synthetic):
    model = train_model(data_dir, epochs, output_dir, synthetic=synthetic)
    target = Path.home() / ".tem_psd" / "models"
    target.mkdir(parents=True, exist_ok=True)
    final = target / "unet.pth"
    final.write_bytes(Path(model).read_bytes())
    click.echo(f"Training complete. Model saved to {final}")


@cli.command(name="agglomerate")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True))
@click.option("--scale", required=True, type=float)
@click.option("--output", default="./results")
@click.option("--min-gap", "min_gap_nm", default=2.0, type=float)
def agglomerate_cmd(input_path, scale, output, min_gap_nm):
    out_dir, index = run_agglomeration(input_path, scale, output, min_gap_nm=min_gap_nm)
    click.echo(f"Agglomeration complete: {out_dir} (index={index:.2f}%)")


if __name__ == "__main__":
    cli()
