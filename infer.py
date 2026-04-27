#!/usr/bin/env python3
"""CLI entry point for inference — delegates entirely to `src.infer_denoise.main`; requires a trained checkpoint at `checkpoints/denoiser.pt` (run `train.py` first)."""

from src.infer_denoise import main

if __name__ == "__main__":
    main()
