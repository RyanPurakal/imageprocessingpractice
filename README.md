# Image restoration lab

Small demo that ties together **PyTorch**, **Hugging Face Transformers**, synthetic **mechanics-flavored** 2D fields (smooth scalar “slices” you might get from CFD/postprocessing), **optional super-resolution**, a minimal **R** plotting script, and a tiny **open-weights language model** example (DistilGPT-2).

It does **not** run FEM, SPH, or a CFD solver; it is meant as a bridge project between your simulation/visualization world and modern ML tooling. You can swap the synthetic tensors for PNG exports from a C++ or Python postprocessor.

## Prep for a finished run (one command)

**Prerequisites:** Python **3.10+** (3.11 or 3.12 recommended), `pip`, and ~2–4 GB disk for PyTorch + Transformers. Optional: **R** (`Rscript`) for metric plots; **network** on first run (HF / pip).

From the `image-restoration-lab` directory:

```bash
chmod +x scripts/demo.sh   # once
./scripts/demo.sh
```

Or: `make demo` (requires `make` + bash).

This creates `.venv` if needed, installs dependencies, trains a small denoiser (`DEMO_EPOCHS` defaults to `20`; override e.g. `DEMO_EPOCHS=40 ./scripts/demo.sh`), writes sample PNGs, and saves figures under `outputs/`.

**What “done” looks like** — you should have at least:

| Artifact | Role |
|----------|------|
| `checkpoints/denoiser.pt` | Trained U-Net weights |
| `outputs/metrics.csv` | Training loss + validation PSNR |
| `outputs/denoise_comparison.png` | Synthetic clean / noisy / restored |
| `outputs/denoise_from_files.png` | Same pipeline on `sample_data/field_*.png` |

**Handing in / zipping the project:** exclude the virtualenv and large caches so the archive stays small. Recipients recreate the env with `pip install -r requirements.txt`. Example:

```bash
cd ..
zip -r image-restoration-lab.zip image-restoration-lab \
  -x 'image-restoration-lab/.venv/*' \
  -x 'image-restoration-lab/**/__pycache__/*' \
  -x 'image-restoration-lab/.mplconfig/*'
```

Include **either** regenerated `outputs/` + `checkpoints/` from your demo run **or** a short note to run `./scripts/demo.sh` after unzipping.

**Reproducible installs (optional):** after a good run, `pip freeze > requirements.lock.txt` and document “install with `pip install -r requirements.lock.txt`” for bit-for-bit parity on the same OS/Python.

## Setup (manual)

```bash
cd image-restoration-lab
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 1. Train a denoiser (PyTorch U-Net)

Trains on random smooth fields with additive Gaussian noise; writes `checkpoints/denoiser.pt` and `outputs/metrics.csv`.

```bash
python train.py
# faster tryout: python train.py --epochs 8 --batch 8
```

## 2. Visualize denoising

Synthetic fields (default):

```bash
python infer.py
```

Saves `outputs/denoise_comparison.png`.

### Simulation / C++ hook (PNG or PGM on disk)

Export scalar fields from your postprocessor as **grayscale** (or RGB) images, then run inference on those files instead of synthetic data.

1. **Optional — generate sample PNGs** (no C++ needed):

   ```bash
   python tools/export_sample_pngs.py
   ```

2. **Optional — minimal C++ export** (writes a PGM without extra libraries):

   ```bash
   cd examples
   c++ -std=c++17 -O2 -o export_pgm export_pgm_stub.cpp
   ./export_pgm ../sample_data/from_cpp_stub.pgm
   cd ..
   ```

3. **Denoise** (paths are resized to the checkpoint’s training size, e.g. 128×128):

   ```bash
   python infer.py --png sample_data/field_00.png sample_data/field_01.png
   # or
   python infer.py --png-glob 'sample_data/field_*.png'
   ```

   Writes `outputs/denoise_from_files.png`.

4. **Super-resolution** (Transformers / Swin2SR) on the same exports:

   ```bash
   python hf_swin2sr.py sample_data/field_00.png -o outputs/field_00_swin2sr.png
   ```

Your real pipeline can mirror the stub: write one file per time slice or region, then point `--png` or `--png-glob` at those paths.

## 3. Plot metrics in R

From the `r` folder:

```bash
cd r
Rscript plot_metrics.R
```

Writes `outputs/training_curves.png`.

## 4. Hugging Face Swin2SR (2×, pretrained)

Uses an open checkpoint from the Hub (downloads on first run):

```bash
python hf_swin2sr.py outputs/denoise_comparison.png
```

## 5. Optional LLM blurb (DistilGPT-2)

```bash
python optional_llm_blurb.py
```

If Matplotlib warns about a non-writable cache directory (some sandboxed or locked-down environments), set a local config path before running `infer.py`:

```bash
export MPLCONFIGDIR="$(pwd)/.mplconfig"
mkdir -p "$MPLCONFIGDIR"
```

## Connecting to your research stack

- **C++ / simulation**: export grayscale PNG/PGM (see **Simulation / C++ hook** above), then `python infer.py --png …` and/or `python hf_swin2sr.py …`.
- **Stronger restoration models**: browse the Hub for SwinIR / Swin2SR / diffusion pipelines and swap the model id in `hf_swin2sr.py`.
