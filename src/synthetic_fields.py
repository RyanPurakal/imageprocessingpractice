"""
Synthetic 2D scalar fields reminiscent of simulation slices (e.g. pressure or vorticity); produces clean (B,1,H,W) tensors via superimposed Gaussian blobs and adds AWGN via `add_gaussian_noise`.
No solver is involved — values are normalised to [0,1] before being returned.
"""

from __future__ import annotations

import numpy as np
import torch


def _gaussian_blob(h: int, w: int, cx: float, cy: float, sigma: float, amp: float) -> np.ndarray:
    ys, xs = np.ogrid[:h, :w]
    return amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))


def sample_scalar_field(
    batch: int,
    height: int,
    width: int,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Return a batch of single-channel fields in [0, 1]. Shape (B, 1, H, W)."""
    rng = rng or np.random.default_rng()
    out = np.zeros((batch, height, width), dtype=np.float32)
    for b in range(batch):
        n_blobs = int(rng.integers(3, 9))
        acc = np.zeros((height, width), dtype=np.float32)
        for _ in range(n_blobs):
            acc += _gaussian_blob(
                height,
                width,
                float(rng.uniform(0, width)),
                float(rng.uniform(0, height)),
                float(rng.uniform(width, 3 * width) / 20.0),
                float(rng.uniform(0.2, 1.0)),
            )
        # Gentle plane + normalize
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        acc += 0.05 * (yy / height - 0.5) + 0.05 * (xx / width - 0.5)
        acc -= acc.min()
        m = acc.max()
        if m > 1e-6:
            acc /= m
        out[b] = acc
    t = torch.from_numpy(out).unsqueeze(1)
    return t


def add_gaussian_noise(clean: torch.Tensor, sigma: float, rng: torch.Generator | None = None) -> torch.Tensor:
    noise = torch.randn(clean.shape, generator=rng, device=clean.device, dtype=clean.dtype)
    noisy = (clean + sigma * noise).clamp(0.0, 1.0)
    return noisy
