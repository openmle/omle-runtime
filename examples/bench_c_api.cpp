// bench_c_api.cpp — micro-benchmark comparing the OLD vs NEW tensor
// construction paths from c_api.cpp.  Uses the Tensor C++ API directly to avoid
// the protobuf linker chain.
//
// OLD path: vector(n, 0) zero-init then memcpy / alloc_zeroed then overwrite
// NEW path: vector(src, src+n) range-ctor (no zero-init) / alloc then memcpy
//
// Also benchmarks omle_session_get_output data-structure change:
//   OLD: unordered_map linear scan
//   NEW: vector<pair<>> linear scan
//
// Usage: bench_c_api [reps=400]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "omle/tensor.h"

using omle::rt::DataType;
using omle::rt::Tensor;
using clk = std::chrono::high_resolution_clock;

static double min_us(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

template <typename Fn>
static double bench_min(Fn&& fn, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) fn();
  double best = 1e18;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clk::now();
    fn();
    double us = min_us(t0);
    if (us < best) best = us;
  }
  return best;
}

template <typename Fn>
static double bench_mean(Fn&& fn, int warmup, int reps) {
  for (int i = 0; i < warmup; ++i) fn();
  double sum = 0;
  for (int r = 0; r < reps; ++r) {
    auto t0 = clk::now();
    fn();
    sum += min_us(t0);
  }
  return sum / reps;
}

static void row(const char* label, const char* path, int n, double mn,
                double me, double speedup_vs = 0) {
  if (speedup_vs > 0)
    std::printf(
        "  [%s] %-35s  n=%-7d  min=%8.2f us  mean=%8.2f us  speedup=%.2fx\n",
        path, label, n, mn, me, speedup_vs);
  else
    std::printf("  [%s] %-35s  n=%-7d  min=%8.2f us  mean=%8.2f us\n", path,
                label, n, mn, me);
}

// -----------------------------------------------------------------------
// Simulate OLD c_api create_f32 path: f32(n,1) then memcpy
// -----------------------------------------------------------------------
static Tensor old_create_f32(int n, const float* data) {
  Tensor t = Tensor::f32(n, 1);  // alloc_zeroed
  if (data)
    std::memcpy(t.f32_ptr(), data, static_cast<std::size_t>(n) * sizeof(float));
  return t;
}

// NEW c_api create_f32 path: from_ptr (single alloc + memcpy, no zero-init)
static Tensor new_create_f32(int n, const float* data) {
  if (n == 1) return Tensor::f32_scalar(data ? *data : 0.f);
  if (data) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
    return Tensor::from_ptr(DataType::Float32, n, 1, data, bytes);
  }
  return Tensor::f32(n, 1);
}

// -----------------------------------------------------------------------
// Simulate OLD c_api create_f64 path: dense(Float64, n,1) then memcpy
// -----------------------------------------------------------------------
static Tensor old_create_f64(int n, const double* data) {
  Tensor t = Tensor::dense(DataType::Float64, n, 1);  // alloc_zeroed
  if (data)
    std::memcpy(t.f64_ptr(), data,
                static_cast<std::size_t>(n) * sizeof(double));
  return t;
}

static Tensor new_create_f64(int n, const double* data) {
  if (n == 1) return Tensor::scalar(DataType::Float64, data ? *data : 0.0);
  if (data) {
    const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
    return Tensor::from_ptr(DataType::Float64, n, 1, data, bytes);
  }
  return Tensor::dense(DataType::Float64, n, 1);
}

// -----------------------------------------------------------------------
// Simulate OLD c_api create_dense path: vector(bytes, 0) then memcpy
// -----------------------------------------------------------------------
static Tensor old_create_dense(DataType dt, int n, const void* data,
                               std::size_t bytes) {
  std::vector<uint8_t> raw(bytes, 0);  // zero-init
  if (data && bytes > 0) std::memcpy(raw.data(), data, bytes);
  return Tensor::from_raw(dt, n, 1, std::move(raw));
}

static Tensor new_create_dense(DataType dt, int n, const void* data,
                               std::size_t bytes) {
  std::vector<uint8_t> raw;
  if (data && bytes > 0) {
    const auto* src = static_cast<const uint8_t*>(data);
    raw = std::vector<uint8_t>(src, src + bytes);  // range ctor, no zero-init
  } else {
    raw.assign(bytes, 0);
  }
  return Tensor::from_raw(dt, n, 1, std::move(raw));
}

// -----------------------------------------------------------------------
// Simulate OLD c_api create_sparse_csr: vector<uint8_t>(bytes,0) then memcpy
// -----------------------------------------------------------------------
static Tensor old_create_sparse(int n, const float* vals, int nnz,
                                const int32_t* idx, const int32_t* ptr) {
  const std::size_t vbytes = static_cast<std::size_t>(nnz) * sizeof(float);
  std::vector<uint8_t> vv(vbytes, 0);  // zero-init
  if (vals && vbytes > 0) std::memcpy(vv.data(), vals, vbytes);
  std::vector<int32_t> vi(idx, idx + nnz);
  std::vector<int32_t> vp(ptr, ptr + n + 1);
  return Tensor::sparse_csr(DataType::Float32, n, 500, std::move(vv),
                            std::move(vi), std::move(vp), 0.0);
}

static Tensor new_create_sparse(int n, const float* vals, int nnz,
                                const int32_t* idx, const int32_t* ptr) {
  const std::size_t vbytes = static_cast<std::size_t>(nnz) * sizeof(float);
  std::vector<uint8_t> vv;
  if (vals && vbytes > 0) {
    const auto* src = reinterpret_cast<const uint8_t*>(vals);
    vv = std::vector<uint8_t>(src, src + vbytes);  // range ctor, no zero-init
  }
  std::vector<int32_t> vi(idx, idx + nnz);
  std::vector<int32_t> vp(ptr, ptr + n + 1);
  return Tensor::sparse_csr(DataType::Float32, n, 500, std::move(vv),
                            std::move(vi), std::move(vp), 0.0);
}

// -----------------------------------------------------------------------
// session_get_output: OLD unordered_map scan vs NEW vector<pair<>> scan
// -----------------------------------------------------------------------

struct OldSession {
  std::unordered_map<std::string, Tensor> outputs;
};
struct NewSession {
  std::vector<std::pair<std::string, Tensor>> outputs;
};

static const Tensor* old_get_output(const OldSession& s, const char* name) {
  // OLD: manual linear scan over unordered_map (as in original c_api.cpp)
  for (const auto& [k, v] : s.outputs)
    if (k == name) return &v;
  return nullptr;
}
static const Tensor* new_get_output(const NewSession& s, const char* name) {
  for (const auto& p : s.outputs)
    if (p.first == name) return &p.second;
  return nullptr;
}

int main(int argc, char** argv) {
  const int reps = argc > 1 ? std::atoi(argv[1]) : 400;
  const int warmup = 20;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> fdist(-1.f, 1.f);
  std::uniform_real_distribution<double> ddist(-1.0, 1.0);

  std::printf(
      "bench_c_api: OLD (committed) vs NEW (patched) construction paths\n");
  std::printf("reps=%d  warmup=%d\n\n", reps, warmup);

  const int sizes[] = {1000, 10000, 100000, 500000};

  for (int n : sizes) {
    std::printf("n = %d\n", n);

    std::vector<float> fdata(n);
    std::vector<double> ddata(n);
    for (int i = 0; i < n; ++i) {
      fdata[i] = fdist(rng);
      ddata[i] = ddist(rng);
    }

    // ---- create_f32 with data ----
    double old_mn, old_me, new_mn, new_me;
    {
      old_mn = bench_min(
          [&] {
            auto t = old_create_f32(n, fdata.data());
            (void)t;
          },
          warmup, reps);
      old_me = bench_mean(
          [&] {
            auto t = old_create_f32(n, fdata.data());
            (void)t;
          },
          warmup, reps);
      new_mn = bench_min(
          [&] {
            auto t = new_create_f32(n, fdata.data());
            (void)t;
          },
          warmup, reps);
      new_me = bench_mean(
          [&] {
            auto t = new_create_f32(n, fdata.data());
            (void)t;
          },
          warmup, reps);
      row("create_f32(data)", "old", n, old_mn, old_me);
      row("create_f32(data)", "new", n, new_mn, new_me, old_mn / new_mn);
    }

    // ---- create_f64 with data ----
    {
      old_mn = bench_min(
          [&] {
            auto t = old_create_f64(n, ddata.data());
            (void)t;
          },
          warmup, reps);
      old_me = bench_mean(
          [&] {
            auto t = old_create_f64(n, ddata.data());
            (void)t;
          },
          warmup, reps);
      new_mn = bench_min(
          [&] {
            auto t = new_create_f64(n, ddata.data());
            (void)t;
          },
          warmup, reps);
      new_me = bench_mean(
          [&] {
            auto t = new_create_f64(n, ddata.data());
            (void)t;
          },
          warmup, reps);
      row("create_f64(data)", "old", n, old_mn, old_me);
      row("create_f64(data)", "new", n, new_mn, new_me, old_mn / new_mn);
    }

    // ---- create_dense Float32 with data ----
    {
      const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
      old_mn = bench_min(
          [&] {
            auto t =
                old_create_dense(DataType::Float32, n, fdata.data(), bytes);
            (void)t;
          },
          warmup, reps);
      old_me = bench_mean(
          [&] {
            auto t =
                old_create_dense(DataType::Float32, n, fdata.data(), bytes);
            (void)t;
          },
          warmup, reps);
      new_mn = bench_min(
          [&] {
            auto t =
                new_create_dense(DataType::Float32, n, fdata.data(), bytes);
            (void)t;
          },
          warmup, reps);
      new_me = bench_mean(
          [&] {
            auto t =
                new_create_dense(DataType::Float32, n, fdata.data(), bytes);
            (void)t;
          },
          warmup, reps);
      row("create_dense(data)", "old", n, old_mn, old_me);
      row("create_dense(data)", "new", n, new_mn, new_me, old_mn / new_mn);
    }

    // ---- create_sparse_csr 10% density ----
    {
      const int nnz = n / 10;
      std::vector<float> sp_vals(nnz);
      std::vector<int32_t> sp_idx(nnz);
      std::vector<int32_t> sp_ptr(n + 1);
      for (int i = 0; i < nnz; ++i) sp_vals[i] = fdist(rng);
      for (int i = 0; i <= n; ++i)
        sp_ptr[i] = static_cast<int32_t>(i * nnz / n);
      for (int i = 0; i < nnz; ++i) sp_idx[i] = i % 500;

      old_mn = bench_min(
          [&] {
            auto t = old_create_sparse(n, sp_vals.data(), nnz, sp_idx.data(),
                                       sp_ptr.data());
            (void)t;
          },
          warmup, reps);
      old_me = bench_mean(
          [&] {
            auto t = old_create_sparse(n, sp_vals.data(), nnz, sp_idx.data(),
                                       sp_ptr.data());
            (void)t;
          },
          warmup, reps);
      new_mn = bench_min(
          [&] {
            auto t = new_create_sparse(n, sp_vals.data(), nnz, sp_idx.data(),
                                       sp_ptr.data());
            (void)t;
          },
          warmup, reps);
      new_me = bench_mean(
          [&] {
            auto t = new_create_sparse(n, sp_vals.data(), nnz, sp_idx.data(),
                                       sp_ptr.data());
            (void)t;
          },
          warmup, reps);
      row("create_sparse_csr(10%)", "old", n, old_mn, old_me);
      row("create_sparse_csr(10%)", "new", n, new_mn, new_me, old_mn / new_mn);
    }

    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // session_get_output: OLD unordered_map scan vs NEW vector<pair<>> scan
  // -----------------------------------------------------------------------
  std::printf("session_get_output (last output in list):\n");
  const char* output_names[] = {"output_0", "output_1", "output_2", "output_3",
                                "output_4"};
  for (int n_outputs : {1, 3, 5}) {
    OldSession old_s;
    NewSession new_s;
    for (int i = 0; i < n_outputs; ++i) {
      old_s.outputs.emplace(output_names[i], Tensor::f32_scalar(0.f));
      new_s.outputs.emplace_back(output_names[i], Tensor::f32_scalar(0.f));
    }
    const char* target = output_names[n_outputs - 1];

    double old_mn = bench_min(
        [&] {
          auto* p = old_get_output(old_s, target);
          (void)p;
        },
        warmup, reps);
    double old_me = bench_mean(
        [&] {
          auto* p = old_get_output(old_s, target);
          (void)p;
        },
        warmup, reps);
    double new_mn = bench_min(
        [&] {
          auto* p = new_get_output(new_s, target);
          (void)p;
        },
        warmup, reps);
    double new_me = bench_mean(
        [&] {
          auto* p = new_get_output(new_s, target);
          (void)p;
        },
        warmup, reps);

    char label[64];
    std::snprintf(label, sizeof(label), "get_output(%d outputs)", n_outputs);
    row(label, "old", 0, old_mn, old_me);
    row(label, "new", 0, new_mn, new_me, old_mn / new_mn);
  }
  std::printf("\n");

  return 0;
}
