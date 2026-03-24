"""Load scalar fields from disk (PNG, PGM, etc.) for inference — matches simulation/C++ export workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def load_image_paths_as_batch(paths: list[Path], size: int) -> torch.Tensor:
    """
    Load images as a single-channel batch in [0, 1], shape (B, 1, size, size).

    Grayscale PNG/PGM uses the intensity channel. RGB is converted to luminance (PIL ``L``).
    """
    if not paths:
        raise ValueError("paths must be non-empty")
    tensors: list[np.ndarray] = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
        img = Image.open(p)
        img = img.convert("L")
        img = img.resize((size, size), Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensors.append(arr)
    stacked = np.stack(tensors, axis=0)
    return torch.from_numpy(stacked).unsqueeze(1)


def expand_glob(pattern: str) -> list[Path]:
    from glob import glob

    return sorted(Path(p) for p in glob(pattern) if Path(p).is_file())
