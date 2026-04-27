# scripts/ — Shell automation

Single responsibility: provide a single `demo.sh` script that runs the full end-to-end pipeline (venv creation, dependency install, training, sample-PNG generation, and both inference modes) from a clean checkout with one command.

## Usage

```bash
bash scripts/demo.sh          # from anywhere; script resolves the project root
# or
make demo                     # via Makefile convenience target
```

## Key design decisions

- **Self-contained bootstrap** — the script creates `.venv` if absent and installs `requirements.txt`, so a new contributor needs only Python 3 and `bash`.
- **`DEMO_EPOCHS` env var** — override the default 20-epoch training run without editing the script: `DEMO_EPOCHS=5 bash scripts/demo.sh`.
- **`MPLCONFIGDIR` redirect** — Matplotlib's config directory is set to `.mplconfig/` inside the project root to avoid polluting `~/.config/matplotlib` in shared or CI environments.
