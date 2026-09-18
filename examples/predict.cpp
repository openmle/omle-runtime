// predict.cpp — minimal CLI for running inference with an OMLE model.
//
// Usage:
//   omle-predict <model.omle> [features.csv] [predictions.csv]
//
// If no input CSV is supplied, prints model metadata and exits.
// Input CSV: one sample per line, comma-separated float values, no header.
//
// Predictions go to the output CSV, never to stdout, so the scores stay
// machine-readable and stdout stays a report: model metadata, then how many
// records were scored and where they went. Omitting the output path scores the
// input and reports on it without writing anything.

#include <cstdio>
#include <chrono>
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
    // Basename, not argv[0]: the tool is reached through `omle predict` as
    // well as directly, and in that case argv[0] is the resolved absolute path
    // of the executable, which makes for a usage line nobody can read.
    const char* prog = argv[0];
    for (const char* c = argv[0]; *c; ++c) {
      if (*c == '/' || *c == '\\') prog = c + 1;
    }
    std::fprintf(stderr,
                 "Usage: %s <model.omle> [features.csv] [predictions.csv]\n",
                 prog);
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

  const auto t0 = std::chrono::steady_clock::now();
  auto run_st = session->run();
  const auto elapsed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t0)
          .count();
  if (!run_st.ok()) {
    std::fprintf(stderr, "Error running inference: %s\n",
                 run_st.message().c_str());
    return 1;
  }

  // Header names for the output file. An output producing several columns gets
  // an index suffix, so every column can be identified on its own.
  std::vector<std::string> col_names;
  for (const auto& [name, t] : session->results()) {
    if (t.n_cols == 1) {
      col_names.push_back(name);
    } else {
      for (int j = 0; j < t.n_cols; ++j) {
        col_names.push_back(name + "_" + std::to_string(j));
      }
    }
  }

  // Scoring is done at this point, so report it before the write: if the write
  // then fails, the summary is still an accurate account of what happened.
  std::printf("\nScored %d record%s (%d feature%s in, %zu column%s out) in %.2f ms\n",
              n_rows, n_rows == 1 ? "" : "s",
              n_feat, n_feat == 1 ? "" : "s",
              col_names.size(), col_names.size() == 1 ? "" : "s",
              elapsed_ms);

  if (argc >= 4) {
    const char* out_path = argv[3];
    std::ofstream out(out_path);
    if (!out) {
      std::fprintf(stderr, "Error: cannot open %s for writing\n", out_path);
      return 1;
    }

    for (std::size_t c = 0; c < col_names.size(); ++c) {
      if (c) out.put(',');
      out << col_names[c];
    }
    out.put('\n');

    char buf[32];
    for (int i = 0; i < n_rows; ++i) {
      int col = 0;
      for (const auto& [name, t] : session->results()) {
        for (int j = 0; j < t.n_cols; ++j, ++col) {
          if (col) out.put(',');
          std::snprintf(buf, sizeof(buf), "%.6f", t.row(i)[j]);
          out << buf;
        }
      }
      out.put('\n');
    }

    // A stream error here means a short write — a truncated predictions file
    // reported as success is the one outcome worth failing loudly for.
    out.flush();
    if (!out) {
      std::fprintf(stderr, "Error: failed writing %s\n", out_path);
      return 1;
    }
    std::printf("Wrote %d row%s to %s\n", n_rows, n_rows == 1 ? "" : "s",
                out_path);
  } else {
    std::puts("No output path given, so predictions were discarded.");
  }

  return 0;
}
