"""Image quality metrics for tensors in [0,1]; currently exposes only `batch_psnr`, which is used for validation logging and not during the training loss computation."""

from __future__ import annotations

import torch


def batch_psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> float:
    """Peak SNR assuming values in [0, 1]. Returns mean PSNR over batch."""
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    mse = torch.clamp(mse, min=eps)
    psnr = 10.0 * torch.log10(1.0 / mse)
    return float(psnr.mean().item())
