"""Run the trained denoiser and save a visual comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.field_io import expand_glob, load_image_paths_as_batch
from src.metrics import batch_psnr
from src.synthetic_fields import add_gaussian_noise, sample_scalar_field
from src.unet import UNetDenoise1ch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run denoising U-Net on synthetic fields or on PNG/PGM exports from simulation."
    )
    p.add_argument("--checkpoint", type=str, default="checkpoints/denoiser.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--png",
        nargs="+",
        metavar="PATH",
        help="One or more grayscale/RGB images (PNG, PGM, …). Resized to checkpoint size. Up to 6 shown.",
    )
    p.add_argument(
        "--png-glob",
        type=str,
        default="",
        help="Glob of images (e.g. sample_data/field_*.png). Sorted; first 6 used.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="",
        help="Output figure path (default: outputs/denoise_comparison.png or denoise_from_files.png).",
    )
    p.add_argument(
        "--noise-sigma",
        type=float,
        default=None,
        help="Override additive noise σ (default: value stored in checkpoint).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    ckpt_path = root / args.checkpoint
    if not ckpt_path.is_file():
        raise SystemExit(f"Missing checkpoint: {ckpt_path} — run python train.py first.")

    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    noise = float(args.noise_sigma if args.noise_sigma is not None else payload.get("noise", 0.12))
    size = int(payload.get("size", 128))

    device = torch.device(args.device)
    model = UNetDenoise1ch(base=32).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    from_png = bool(args.png or args.png_glob)
    if from_png:
        paths: list[Path] = []
        if args.png:
            paths.extend(Path(p) for p in args.png)
        if args.png_glob:
            paths.extend(expand_glob(args.png_glob))
        # unique, preserve order
        seen: set[str] = set()
        uniq: list[Path] = []
        for p in paths:
            k = str(p.resolve())
            if k not in seen:
                seen.add(k)
                uniq.append(p)
        paths = uniq[:6]
        if not paths:
            raise SystemExit("No image files matched --png / --png-glob.")
        clean = load_image_paths_as_batch(paths, size).to(device)
        n = clean.shape[0]
        title_clean = "clean (from file)"
    else:
        rng = np.random.default_rng(args.seed)
        clean = sample_scalar_field(6, size, size, rng=rng).to(device)
        n = 6
        title_clean = "clean (synthetic field)"

    noisy = add_gaussian_noise(clean, noise)
    with torch.no_grad():
        restored = model(noisy)
    psnr_noisy = batch_psnr(noisy, clean)
    psnr_rest = batch_psnr(restored, clean)

    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)

    def to_vis(t: torch.Tensor, i: int) -> np.ndarray:
        return t[i, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(3, n, figsize=(2 * n + 2, 6), constrained_layout=True)
    axes = np.asarray(axes)
    if axes.ndim == 1:
        axes = axes.reshape(3, 1)
    for i in range(n):
        axes[0, i].imshow(to_vis(clean, i), cmap="viridis", vmin=0, vmax=1)
        axes[1, i].imshow(to_vis(noisy, i), cmap="viridis", vmin=0, vmax=1)
        axes[2, i].imshow(to_vis(restored, i), cmap="viridis", vmin=0, vmax=1)
        for r in range(3):
            axes[r, i].axis("off")
    axes[0, 0].set_title(title_clean, loc="left")
    axes[1, 0].set_title(f"noisy σ={noise:.2f}", loc="left")
    axes[2, 0].set_title("U-Net restored", loc="left")
    fig.suptitle(f"PSNR noisy: {psnr_noisy:.2f} dB → restored: {psnr_rest:.2f} dB")
    if args.output:
        out_png = Path(args.output)
        if not out_png.is_absolute():
            out_png = root / out_png
    else:
        name = "denoise_from_files.png" if from_png else "denoise_comparison.png"
        out_png = out_dir / name
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
