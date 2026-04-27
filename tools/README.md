# tools/ — Development utilities

Single responsibility: generate synthetic test assets that are needed by `infer.py --png-glob` but are not committed to the repository.

## Scripts

### `export_sample_pngs.py`

Writes six 256×256 grayscale PNGs (`sample_data/field_00.png` … `field_05.png`) using the same `synthetic_fields.sample_scalar_field` function that the training pipeline uses, so the inference PNG path is exercised on data with the same statistical character as the training data.

Run directly or via `demo.sh`:

```bash
python tools/export_sample_pngs.py
```

## Key design decisions

- **Fixed seed (2025)** — reproducible output so `sample_data/` contents are deterministic across machines; change the seed if you want different field shapes.
- **256×256 vs 128×128 training size** — the PNGs are larger than the default training resolution; `field_io.load_image_paths_as_batch` resizes them to the checkpoint's `size` at inference time using bicubic interpolation.
