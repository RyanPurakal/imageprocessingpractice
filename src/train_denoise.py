"""Train a small U-Net to denoise synthetic scalar fields."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.metrics import batch_psnr
from src.synthetic_fields import add_gaussian_noise, sample_scalar_field
from src.unet import UNetDenoise1ch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net denoiser on synthetic fields.")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--noise", type=float, default=0.12, help="Gaussian noise sigma in [0,1] scale.")
    p.add_argument("--size", type=int, default=128, help="Spatial size (multiple of 8).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    root = Path(__file__).resolve().parents[1]
    ckpt_dir = root / "checkpoints"
    out_dir = root / "outputs"
    ckpt_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)

    val_rng = np.random.default_rng(12345)
    val_clean = sample_scalar_field(32, args.size, args.size, rng=val_rng).to(device)

    model = UNetDenoise1ch(base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_psnr"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 50
        bar = tqdm(range(n_batches), desc=f"epoch {epoch}/{args.epochs}")
        for _ in bar:
            clean = sample_scalar_field(args.batch, args.size, args.size).to(device)
            noisy = add_gaussian_noise(clean, args.noise)
            pred = model(noisy)
            loss = loss_fn(pred, clean)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += float(loss.item())
            bar.set_postfix(loss=f"{loss.item():.5f}")

        train_loss /= n_batches
        model.eval()
        with torch.no_grad():
            noisy_v = add_gaussian_noise(val_clean, args.noise)
            pred_v = model(noisy_v)
            val_psnr = batch_psnr(pred_v, val_clean)

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, val_psnr])

        tqdm.write(f"epoch {epoch}: train_loss={train_loss:.6f} val_psnr={val_psnr:.2f} dB")

    ckpt_path = ckpt_dir / "denoiser.pt"
    torch.save({"model": model.state_dict(), "noise": args.noise, "size": args.size}, ckpt_path)
    print(f"Saved {ckpt_path}")


if __name__ == "__main__":
    main()
