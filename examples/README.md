# examples/ — Reference integrations

Single responsibility: demonstrate how external tools (simulation postprocessors, C++ renderers) can produce image files that the Python inference pipeline can consume.

## Files

### `export_pgm_stub.cpp`

A minimal C++17 program that writes a 256×256 PGM (binary grayscale) image with a synthetic sinusoidal pattern — no external libraries required.

```bash
# Build
c++ -std=c++17 -O2 -o export_pgm examples/export_pgm_stub.cpp

# Run — writes sample_data/from_cpp_stub.pgm
./export_pgm sample_data/from_cpp_stub.pgm

# Feed into inference
python infer.py --png sample_data/from_cpp_stub.pgm
```

## Key design decisions

- **PGM (P5 binary) format** — portable, no compression, trivially parseable by PIL; real postprocessors typically use libpng or VTK, but PGM shows the byte layout explicitly.
- **Stub only** — this is documentation-as-code: it shows the contract (one grayscale scalar per pixel, row-major, 8-bit) without coupling to any real solver.
