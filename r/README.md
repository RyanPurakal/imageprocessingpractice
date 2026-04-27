# r/ — R plotting scripts

Single responsibility: read the `outputs/metrics.csv` file written by `train_denoise.py` and produce `outputs/training_curves.png` showing training loss and validation PSNR side by side.

## Usage

```bash
cd r && Rscript plot_metrics.R
# or from the project root:
Rscript r/plot_metrics.R
```

Requires only base R (no additional packages). Run `train.py` first so `outputs/metrics.csv` exists.

## Key design decisions

- **Base R only** — deliberately avoids `ggplot2` or any CRAN dependency so the script works in any R installation without a package setup step.
- **Path resolution via `--file=` arg** — the script derives the project root from its own path so it works correctly whether invoked from `r/`, the project root, or via `make`.
