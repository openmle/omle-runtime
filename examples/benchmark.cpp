// benchmark.cpp — throughput benchmark for the OMLE runtime.
//
// Usage:
//   omle-benchmark <model.omle> [n_samples=100000] [n_threads=1] [n_warmup=3]
//   [n_reps=10]
//
// n_threads=0 uses hardware_concurrency().
// Warmup and measurement both use Model::predict() (thread-pool path).

#if defined(_WIN32)
// getrusage and <sys/resource.h> are POSIX-only; Windows reports process
// memory through psapi instead.
//
// NOMINMAX first: without it <windows.h> defines min and max as macros, and
// every std::min / std::max in this file and in tensor.h then fails to parse
// (C2589: illegal token on right side of '::'). WIN32_LEAN_AND_MEAN drops the
// socket and RPC headers we do not use.
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
//
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "omle/runtime.h"

#if defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>

#include <fstream>
#endif

// Current RSS of this process in bytes.
static std::size_t rss_bytes() {
#if defined(__APPLE__)
  mach_task_basic_info info{};
  mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &cnt) == KERN_SUCCESS)
    return static_cast<std::size_t>(info.resident_size);
  return 0;
#elif defined(__linux__)
  std::ifstream f("/proc/self/statm");
  std::size_t size_pages = 0, rss_pages = 0;
  f >> size_pages >> rss_pages;
  return rss_pages * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#else
  return 0;
#endif
}

// Peak RSS of this process in bytes (monotonically increasing).
static std::size_t peak_rss_bytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS pmc{};
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
    return static_cast<std::size_t>(pmc.PeakWorkingSetSize);
  return 0;
#else
  struct rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  // macOS reports ru_maxrss in bytes; Linux reports kilobytes.
  return static_cast<std::size_t>(ru.ru_maxrss);
#else
  return static_cast<std::size_t>(ru.ru_maxrss) * 1024;
#endif
#endif
}

static void print_mb(const char* label, std::size_t bytes,
                     const char* suffix = "") {
  std::printf("  %-18s: %7.1f MB%s\n", label, bytes / 1048576.0, suffix);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::fprintf(
        stderr,
        "Usage: %s <model.omle> [n_samples] [n_threads] [n_warmup] [n_reps]\n",
        argv[0]);
    return 1;
  }

  const int n_samples = argc >= 3 ? std::atoi(argv[2]) : 100'000;
  const int n_threads = argc >= 4 ? std::atoi(argv[3]) : 1;
  const int n_warmup = argc >= 5 ? std::atoi(argv[4]) : 3;
  const int n_reps = argc >= 6 ? std::atoi(argv[5]) : 10;

  const int effective_threads =
      n_threads == 0 ? static_cast<int>(std::thread::hardware_concurrency())
                     : n_threads;

  const std::size_t rss_baseline = rss_bytes();

  omle::rt::LoadOptions opts;
  opts.n_threads = n_threads;

  auto load_result = omle::rt::Model::load(argv[1], opts);
  if (!load_result.ok()) {
    std::fprintf(stderr, "Error: %s\n", load_result.message().c_str());
    return 1;
  }
  auto model = std::move(*load_result);

  const std::size_t rss_after_load = rss_bytes();

  const int n_feat = model->num_inputs();
  const int n_out = model->num_outputs();
  std::printf("Model    : %d features, %d outputs\n", n_feat, n_out);
  std::printf("Batch    : %d samples\n", n_samples);
  std::printf("Threads  : %d\n\n", effective_threads);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> feat_data(static_cast<std::size_t>(n_samples) * n_feat);
  for (float& v : feat_data) v = dist(rng);

  omle::rt::Tensor input =
      omle::rt::Tensor::from_floats(n_samples, n_feat, feat_data);

  auto run_predict =
      [&]() -> std::unordered_map<std::string, omle::rt::Tensor> {
    auto r = model->predict({{"input", input}});
    if (!r.ok()) throw std::runtime_error(r.message());
    return std::move(*r);
  };

  for (int i = 0; i < n_warmup; ++i) run_predict();

  std::unordered_map<std::string, omle::rt::Tensor> last_result;
  using clock = std::chrono::high_resolution_clock;
  std::vector<double> elapsed_ms(n_reps);
  for (int r = 0; r < n_reps; ++r) {
    auto t0 = clock::now();
    last_result = run_predict();
    auto t1 = clock::now();
    elapsed_ms[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }

  const std::size_t rss_after_bench = rss_bytes();
  const std::size_t peak_rss = peak_rss_bytes();

  const double mean_ms =
      std::accumulate(elapsed_ms.begin(), elapsed_ms.end(), 0.0) / n_reps;
  const double min_ms = *std::min_element(elapsed_ms.begin(), elapsed_ms.end());
  const double max_ms = *std::max_element(elapsed_ms.begin(), elapsed_ms.end());

  // ---- memory report ----
  char model_delta[32];
  std::snprintf(model_delta, sizeof(model_delta), "  (+%.1f MB model)",
                (static_cast<double>(rss_after_load) -
                 static_cast<double>(rss_baseline)) /
                    1048576.0);

  std::printf("Memory (RSS):\n");
  print_mb("baseline", rss_baseline);
  print_mb("after load", rss_after_load, model_delta);
  print_mb("after benchmark", rss_after_bench);
  print_mb("peak", peak_rss);
  std::printf("\n");

  // ---- timing report ----
  std::printf("Results over %d reps:\n", n_reps);
  std::printf("  Mean  : %.2f ms   (%.0f samples/sec)\n", mean_ms,
              static_cast<double>(n_samples) / (mean_ms / 1000.0));
  std::printf("  Min   : %.2f ms\n", min_ms);
  std::printf("  Max   : %.2f ms\n", max_ms);
  std::printf("  Best  : %.0f samples/sec\n",
              static_cast<double>(n_samples) / (min_ms / 1000.0));

  if (!last_result.empty()) {
    const auto& t = last_result.begin()->second;
    std::printf("\nFirst 5 outputs (sample 0):");
    for (int j = 0; j < std::min(t.n_cols, 5); ++j)
      std::printf(" %.6f", t.row(0)[j]);
    std::printf("\n");
  }

  return 0;
}
