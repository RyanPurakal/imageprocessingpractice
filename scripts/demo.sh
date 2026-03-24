#!/usr/bin/env bash
# End-to-end demo: venv, deps, train, sample PNGs, inference figures.
# Run from anywhere:  bash scripts/demo.sh
# Or:                  cd image-restoration-lab && ./scripts/demo.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ ! -d .venv ]]; then
  echo "Creating .venv …"
  python3 -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

echo "Installing dependencies …"
pip install -q -r requirements.txt

EPOCHS="${DEMO_EPOCHS:-20}"
echo "Training denoiser (${EPOCHS} epochs; set DEMO_EPOCHS to change) …"
python train.py --epochs "${EPOCHS}" --batch 8

echo "Writing sample PNGs …"
python tools/export_sample_pngs.py

echo "Inference (synthetic) …"
python infer.py

echo "Inference (from sample PNGs) …"
python infer.py --png-glob 'sample_data/field_*.png'

echo ""
echo "Done. Typical artifacts:"
echo "  checkpoints/denoiser.pt"
echo "  outputs/metrics.csv"
echo "  outputs/denoise_comparison.png"
echo "  outputs/denoise_from_files.png"
echo ""
echo "Optional next steps:"
echo "  python hf_swin2sr.py outputs/denoise_comparison.png"
echo "  cd r && Rscript plot_metrics.R   # if R is installed"
