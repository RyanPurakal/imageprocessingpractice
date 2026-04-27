# sample_data/ — Input images for the PNG inference path

Single responsibility: hold grayscale PNG (or PGM) files that `infer.py --png-glob 'sample_data/field_*.png'` can consume, providing a concrete test of the file-loading path without a real simulation postprocessor.

## How files get here

Run `tools/export_sample_pngs.py` (or `scripts/demo.sh`) to generate `field_00.png` … `field_05.png` from the synthetic scalar-field generator. The files are gitignored (see `.gitignore`) to avoid committing binary blobs that regenerate trivially.

## How to add new entries

- **Synthetic:** adjust the seed or field count in `tools/export_sample_pngs.py` and re-run.
- **From a C++ postprocessor:** see `examples/export_pgm_stub.cpp` for the PGM byte layout; drop the resulting `.pgm` file here and pass it to `--png`.
- **Any grayscale or RGB image:** `field_io.load_image_paths_as_batch` converts RGB to luminance via PIL `"L"` mode and resizes to the checkpoint's spatial resolution, so any standard image format supported by Pillow works.
