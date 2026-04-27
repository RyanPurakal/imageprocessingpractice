#!/usr/bin/env python3
"""
Writes six 256×256 grayscale PNGs to `sample_data/` using `synthetic_fields.sample_scalar_field` so the `infer.py --png-glob` path can be tested without a real simulation postprocessor.
Files are gitignored; re-run this script (or `scripts/demo.sh`) after `make clean-outputs` to regenerate them.
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
