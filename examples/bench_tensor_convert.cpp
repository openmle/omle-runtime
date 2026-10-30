// bench_tensor_convert.cpp — measure Tensor::to_float32() / to_float64()
// across dtypes and sizes.
//
// Verifies the dtype-hoisted dispatch in tensor.h is materially faster than a
// scalar reference loop that switches on dtype per element.

#include <omle/tensor.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using omle::rt::DataType;
using omle::rt::Tensor;
using clk = std::chrono::high_resolution_clock;

static double ms_since(clk::time_point t0) {
  return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
}

// Reference scalar loop: per-element switch on dtype, mimicking the OLD
// implementation. Used purely as a baseline to demonstrate the dispatch-hoist
// win — not part of the library.
static std::vector<float> ref_to_float32(const Tensor& t) {
  const std::size_t n = t.numel();
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    const int r = static_cast<int>(i / t.n_cols);
    const int c = static_cast<int>(i % t.n_cols);
    out[i] = static_cast<float>(t.get(r, c));
  }
  return out;
}

template <typename T>
static Tensor make_dense(DataType dt, int rows, int cols, std::mt19937& rng) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  std::vector<uint8_t> bytes(n * sizeof(T));
  auto* p = reinterpret_cast<T*>(bytes.data());
  if constexpr (std::is_floating_point_v<T>) {
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<T>(d(rng));
  } else {
    std::uniform_int_distribution<int64_t> d(-100, 100);
    for (std::size_t i = 0; i < n; ++i) p[i] = static_cast<T>(d(rng));
  }
  return Tensor::from_raw(dt, rows, cols, std::move(bytes));
}

struct Result {
  double ref_ms;
  double opt_ms;
  float opt_checksum;
  float ref_checksum;
};

static Result bench(const Tensor& t, int reps) {
  Result r{0, 0, 0, 0};

  // Warm
  auto warm = t.to_float32();
  (void)warm;

  auto t0 = clk::now();
  float opt_sum = 0;
  for (int i = 0; i < reps; ++i) {
    Tensor out = t.to_float32();
    opt_sum += out.f32_ptr()[0] + out.f32_ptr()[out.numel() - 1];
  }
  r.opt_ms = ms_since(t0) / reps;
  r.opt_checksum = opt_sum;

  auto t1 = clk::now();
  float ref_sum = 0;
  for (int i = 0; i < reps; ++i) {
    std::vector<float> out = ref_to_float32(t);
    ref_sum += out[0] + out[out.size() - 1];
  }
  r.ref_ms = ms_since(t1) / reps;
  r.ref_checksum = ref_sum;

  return r;
}

int main(int argc, char** argv) {
  const int rows = (argc > 1) ? std::atoi(argv[1]) : 1'000'000;
  const int cols = (argc > 2) ? std::atoi(argv[2]) : 1;
  const int reps = (argc > 3) ? std::atoi(argv[3]) : 20;

  std::printf(
      "Tensor::to_float32() — dispatch-hoisted vs per-element switch\n");
  std::printf("Shape: %d x %d   reps/case: %d\n", rows, cols, reps);
  std::printf("%-12s %12s %12s %10s\n", "dtype", "ref ms/iter", "opt ms/iter",
              "speedup");
  std::printf("%-12s %12s %12s %10s\n", "------------", "------------",
              "------------", "----------");

  std::mt19937 rng(0xC0FFEE);

  struct Case {
    const char* name;
    Tensor t;
  };
  std::vector<Case> cases;
  cases.push_back(
      {"Float32", make_dense<float>(DataType::Float32, rows, cols, rng)});
  cases.push_back(
      {"Float64", make_dense<double>(DataType::Float64, rows, cols, rng)});
  cases.push_back(
      {"Int32", make_dense<int32_t>(DataType::Int32, rows, cols, rng)});
  cases.push_back(
      {"Int64", make_dense<int64_t>(DataType::Int64, rows, cols, rng)});
  cases.push_back(
      {"Int8", make_dense<int8_t>(DataType::Int8, rows, cols, rng)});
  cases.push_back(
      {"UInt8", make_dense<uint8_t>(DataType::UInt8, rows, cols, rng)});
  cases.push_back(
      {"Bool", make_dense<uint8_t>(DataType::Bool, rows, cols, rng)});

  for (const auto& c : cases) {
    Result r = bench(c.t, reps);
    const double speedup = r.ref_ms / r.opt_ms;
    std::printf("%-12s %12.3f %12.3f %9.2fx\n", c.name, r.ref_ms, r.opt_ms,
                speedup);
    if (std::fabs(r.opt_checksum - r.ref_checksum) >
        1e-3f * std::max(1.f, std::fabs(r.ref_checksum))) {
      std::fprintf(stderr, "  WARNING: checksum mismatch (opt=%g ref=%g)\n",
                   r.opt_checksum, r.ref_checksum);
    }
  }
  return 0;
}
