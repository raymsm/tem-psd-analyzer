from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from tem_psd.segmentation.unet import UNet

from .dataset import SegmentationDataset
from .synthetic import generate_synthetic_dataset


def dice_coeff(pred, target, eps=1e-8):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return ((2 * inter + eps) / (union + eps)).mean()


def bce_dice_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice_loss = 1 - ((2 * inter + 1e-8) / (union + 1e-8)).mean()
    return bce + dice_loss


def train_model(data_dir: str | Path, epochs: int, output_dir: str | Path, synthetic: bool = False):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if synthetic:
        generate_synthetic_dataset(data_dir)

    ds = SegmentationDataset(data_dir / "images", data_dir / "masks", augment=True)
    if len(ds) < 2:
        raise ValueError("Need at least 2 samples")
    n_val = max(1, int(0.2 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_dl = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    opt = Adam(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    best_dice = -1.0
    hist = {"train_loss": [], "val_loss": [], "val_dice": []}

    for epoch in range(epochs):
        model.train()
        tr_loss = 0.0
        for x, y in tqdm(train_dl, desc=f"Train {epoch+1}/{epochs}"):
            x, y = x.float().to(device), y.float().to(device)
            opt.zero_grad()
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            loss.backward()
            opt.step()
            tr_loss += loss.item()

        model.eval()
        vl_loss, vl_dice = 0.0, 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.float().to(device), y.float().to(device)
                logits = model(x)
                vl_loss += bce_dice_loss(logits, y).item()
                vl_dice += dice_coeff(torch.sigmoid(logits), y).item()

        tr_loss /= max(len(train_dl), 1)
        vl_loss /= max(len(val_dl), 1)
        vl_dice /= max(len(val_dl), 1)
        sched.step()

        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(vl_loss)
        hist["val_dice"].append(vl_dice)

        if vl_dice > best_dice:
            best_dice = vl_dice
            torch.save(model.state_dict(), output_dir / "unet_best.pth")

    _plot_training_curves(hist, output_dir)
    return output_dir / "unet_best.pth"


def _plot_training_curves(hist: dict, output_dir: Path):
    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(epochs, hist["train_loss"], label="train")
    ax[0].plot(epochs, hist["val_loss"], label="val")
    ax[0].set_title("Loss")
    ax[0].legend()
    ax[1].plot(epochs, hist["val_dice"], label="val dice")
    ax[1].set_title("Dice")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "training_curves.png", dpi=200)
    fig.savefig(output_dir / "training_curves.svg")
    plt.close(fig)
