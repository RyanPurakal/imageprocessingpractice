# outputs/ — Generated artifacts

Single responsibility: receive all files written by the training and inference scripts so that the project root stays clean and `make clean-outputs` has a single target to clear.

## Contents

| File | Written by | Description |
|------|-----------|-------------|
| `denoise_comparison.png` | `infer.py` (synthetic mode) | Side-by-side: clean / noisy / U-Net restored for 6 synthetic fields |
| `denoise_from_files.png` | `infer.py --png-glob …` | Same layout but sourced from `sample_data/field_*.png` |
| `denoise_from_pgm.png` | `infer.py --png …` | Same layout for a single PGM file (e.g. from the C++ stub) |
| `swin2sr_denoise_comparison.png` | `hf_swin2sr.py` | 2× super-resolved output from the Swin2SR transformer |
| `training_curves.png` | `r/plot_metrics.R` | Train MSE and validation PSNR curves over epochs |
| `metrics.csv` | `src/train_denoise.py` | Per-epoch `epoch,train_loss,val_psnr` — input to the R plot |

## Commit policy

`.gitignore` selectively un-ignores `*.png` and `metrics.csv` so rendered figures appear on GitHub without committing large binary blobs that change every run. Re-run `scripts/demo.sh` and commit if you want to refresh the figures in the repository.
