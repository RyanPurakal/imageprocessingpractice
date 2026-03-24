# Image restoration lab — convenience targets (requires make + bash)

.PHONY: help venv install demo clean-outputs

help:
	@echo "Targets:"
	@echo "  make venv          Create .venv (if missing)"
	@echo "  make install       Create venv + pip install -r requirements.txt"
	@echo "  make demo          Full pipeline (same as scripts/demo.sh)"
	@echo "  make clean-outputs Remove outputs/* and checkpoints/* (keeps .venv)"

venv:
	@test -d .venv || python3 -m venv .venv

install: venv
	.venv/bin/pip install -r requirements.txt

demo:
	bash scripts/demo.sh

clean-outputs:
	rm -rf outputs/* checkpoints/*
	@echo "Removed generated outputs (venv unchanged)."
