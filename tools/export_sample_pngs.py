#!/usr/bin/env python3
"""
Write a few grayscale PNGs (synthetic scalar fields) into sample_data/ for testing
the PNG inference path without a C++ postprocessor.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image

from src.synthetic_fields import sample_scalar_field


def main() -> None:
    out_dir = ROOT / "sample_data"
    out_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(2025)
    batch = sample_scalar_field(6, 256, 256, rng=rng)
    for i in range(6):
        arr = (batch[i, 0].numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(arr, mode="L").save(out_dir / f"field_{i:02d}.png")
    print(f"Wrote 6 PNGs to {out_dir}/field_00.png … field_05.png")


if __name__ == "__main__":
    main()
