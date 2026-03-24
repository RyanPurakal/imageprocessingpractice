#!/usr/bin/env python3
"""
Optional: 2× super-resolution with a pretrained open Swin2SR checkpoint from Hugging Face.
Accepts grayscale or RGB; converts to RGB for the model. First run downloads weights (~tens of MB).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    from transformers import AutoImageProcessor, Swin2SRForImageSuperResolution
except ImportError as e:
    raise SystemExit("Install transformers: pip install -r requirements.txt") from e


DEFAULT_MODEL = "caidas/swin2SR-lightweight-x2-64"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Swin2SR 2× super-resolution via Hugging Face Transformers.")
    p.add_argument("image", type=str, help="Input image path (PNG/JPEG).")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output path (default: outputs/swin2sr_<name>).",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_image_rgb(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img


def tensor_to_uint8_image(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).round().astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")


def main() -> None:
    args = parse_args()
    in_path = Path(args.image)
    if not in_path.is_file():
        raise SystemExit(f"Not found: {in_path}")

    device = torch.device(args.device)
    image = load_image_rgb(in_path)

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = Swin2SRForImageSuperResolution.from_pretrained(args.model)
    model.to(device)
    model.eval()

    inputs = processor(image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)

    out = outputs.reconstruction
    if out.dim() == 4:
        out = out[0]
    out = out.float().cpu().clamp(0, 1).numpy()
    if out.shape[0] in (1, 3):
        out = np.moveaxis(out, 0, -1)
    if out.shape[-1] == 1:
        out = out[:, :, 0]
    pil = tensor_to_uint8_image(out)

    root = Path(__file__).resolve().parent
    out_dir = root / "outputs"
    out_dir.mkdir(exist_ok=True)
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = out_dir / f"swin2sr_{in_path.stem}.png"
    pil.save(out_path)
    print(f"Saved super-resolved image to {out_path.resolve()}")


if __name__ == "__main__":
    main()
