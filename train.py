#!/usr/bin/env python3
"""CLI entry point for training — delegates entirely to `src.train_denoise.main`; run from the project root so relative checkpoint/output paths resolve correctly."""

from src.train_denoise import main

if __name__ == "__main__":
    main()
