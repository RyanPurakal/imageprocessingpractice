# src/ — Core ML package

Single responsibility: everything the training and inference entry points need to do their jobs, with no I/O to disk or CLI argument parsing at this layer.

## Module responsibilities

| Module | Receives | Produces |
|--------|----------|----------|
| `unet.py` | A noisy `(B,1,H,W)` float tensor | Restored `(B,1,H,W)` tensor; the network is trained end-to-end with MSE |
| `synthetic_fields.py` | Random-seed parameters | Clean `(B,1,H,W)` tensors from Gaussian-blob superpositions; `add_gaussian_noise` corrupts them |
| `field_io.py` | File paths to PNG/PGM images | A `(B,1,H,W)` tensor normalised to `[0,1]`; converts RGB to grayscale via PIL `"L"` mode |
| `metrics.py` | A predicted and a target tensor | Mean PSNR (dB) over the batch — used for validation logging only, not during training |
| `train_denoise.py` | CLI args (epochs, lr, noise σ, …) | `checkpoints/denoiser.pt` and `outputs/metrics.csv`; runs the full training loop |
| `infer_denoise.py` | CLI args + a checkpoint | `outputs/denoise_*.png` comparison figure; can source images from `synthetic_fields` or `field_io` |

## Key design decisions

- **Spatial size must be a multiple of 8** — the U-Net applies three rounds of 2× downsampling, so non-multiples produce shape mismatches on the skip connections.
- **Single-channel only** — the model and synthetic data are intentionally 1-channel throughout; adapt `DoubleConv` input channels in `unet.py` to extend to RGB.
- **Checkpoint schema** — `denoiser.pt` stores `{"model": state_dict, "noise": float, "size": int}`; `infer_denoise.py` reads `noise` and `size` from this dict to reproduce the exact training conditions.
