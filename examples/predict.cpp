// predict.cpp — minimal CLI for running inference with an OMLE model.
//
// Usage:
//   omle-predict <model.omle> [features.csv]
//
// If no CSV is supplied, prints model metadata and exits.
// CSV format: one sample per line, comma-separated float values, no header.

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "omle/runtime.h"

static std::vector<std::vector<float>> read_csv(const std::string& path) {
  std::vector<std::vector<float>> rows;
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("Cannot open CSV: " + path);

  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;
    while (std::getline(ss, cell, ',')) row.push_back(std::stof(cell));
    rows.push_back(std::move(row));
  }
  return rows;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(stderr, "Usage: %s <model.omle> [features.csv]\n", argv[0]);
    return 1;
  }

  auto load_result = omle::rt::Model::load(argv[1]);
  if (!load_result.ok()) {
    std::fprintf(stderr, "Error loading model: %s\n",
                 load_result.message().c_str());
    return 1;
  }
  auto model = std::move(*load_result);

  std::printf("Model loaded: %d features, %d outputs\n", model->num_inputs(),
              model->num_outputs());

  std::printf("\nInputs (%zu):\n", model->inputs().size());
  for (const auto& inp : model->inputs()) {
    std::printf("  %-20s  dtype=%-8d  shape=[", inp.name.c_str(),
                static_cast<int>(inp.dtype));
    for (std::size_t i = 0; i < inp.shape.size(); ++i) {
      if (i) std::putchar(',');
      std::printf("%lld", static_cast<long long>(inp.shape[i]));
    }
    std::puts("]");
  }

  std::printf("\nOutputs (%zu):\n", model->outputs().size());
  for (const auto& out : model->outputs()) {
    std::printf("  %-20s  dtype=%-8d  role=%-4d  shape=[", out.name.c_str(),
                static_cast<int>(out.dtype), static_cast<int>(out.role));
    for (std::size_t i = 0; i < out.shape.size(); ++i) {
      if (i) std::putchar(',');
      std::printf("%lld", static_cast<long long>(out.shape[i]));
    }
    std::puts("]");
  }

  if (argc < 3) return 0;

  std::vector<std::vector<float>> rows;
  try {
    rows = read_csv(argv[2]);
  } catch (const std::exception& ex) {
    std::fprintf(stderr, "Error reading CSV: %s\n", ex.what());
    return 1;
  }
  if (rows.empty()) {
    std::puts("CSV is empty.");
    return 0;
  }

  const int n_feat = model->num_inputs();
  const int n_out = model->num_outputs();

  std::vector<float> features(rows.size() * n_feat, 0.0f);
  for (int i = 0; i < static_cast<int>(rows.size()); ++i) {
    const int n = std::min(static_cast<int>(rows[i].size()), n_feat);
    std::copy(rows[i].begin(), rows[i].begin() + n,
              features.data() + i * n_feat);
  }

  const int n_rows = static_cast<int>(rows.size());
  auto session = model->create_session();
  session->bind_input("input",
                      omle::rt::Tensor::from_floats(n_rows, n_feat, features));
  auto run_st = session->run();
  if (!run_st.ok()) {
    std::fprintf(stderr, "Error running inference: %s\n",
                 run_st.message().c_str());
    return 1;
  }

  for (int i = 0; i < n_rows; ++i) {
    int col = 0;
    for (const auto& [name, t] : session->results()) {
      for (int j = 0; j < t.n_cols; ++j, ++col) {
        if (col) std::putchar(',');
        std::printf("%.6f", t.row(i)[j]);
      }
    }
    std::putchar('\n');
  }

  return 0;
}
