// Minimal C++ stub: writes a 256×256 grayscale PGM (no external libs).
// Build:  c++ -std=c++17 -O2 -o export_pgm export_pgm_stub.cpp
// Run:    ./export_pgm ../sample_data/from_cpp_stub.pgm
// Then:   python infer.py --png ../sample_data/from_cpp_stub.pgm
//
// Real postprocessors typically use libpng or VTK; this only shows a portable scalar export.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  const int w = 256;
  const int h = 256;
  const std::string path =
      (argc >= 2) ? argv[1] : "sample_data/from_cpp_stub.pgm";

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    std::cerr << "Cannot open " << path << "\n";
    return 1;
  }
  out << "P5\n" << w << " " << h << "\n255\n";
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      double cx = (x + 0.5) / w - 0.5;
      double cy = (y + 0.5) / h - 0.5;
      double r = std::sqrt(cx * cx + cy * cy);
      double v = 0.5 + 0.5 * std::sin(12.0 * r);
      auto u = static_cast<std::uint8_t>(std::clamp(v * 255.0, 0.0, 255.0));
      out.put(static_cast<char>(u));
    }
  }
  std::cout << "Wrote " << path << " (" << w << "x" << h << " PGM)\n";
  return 0;
}
