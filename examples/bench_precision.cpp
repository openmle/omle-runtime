// bench_precision.cpp — compare float32 vs float64 throughput for a tree
// ensemble.
//
// Also measures the overhead of the Pimpl virtual dispatch vs a direct
// impl->compute call, to verify the refactoring did not regress the float32 hot
// path.
//
// Usage:
//   bench_precision [n_trees=200] [depth=6] [n_features=30] [n_reps=10]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#include "nodes/tree_ensemble_node.h"

using namespace omle::rt::impl;
using hrclock = std::chrono::high_resolution_clock;

// -----------------------------------------------------------------------
// Synthetic model builder
// -----------------------------------------------------------------------

template <typename T>
static FlatTree<T> build_tree(int depth, int n_features, std::mt19937& rng) {
  std::uniform_int_distribution<int> feat_dist(0, n_features - 1);
  std::uniform_real_distribution<T> thr_dist(T(0.1), T(0.9));
  std::normal_distribution<T> leaf_dist(T(0), T(0.1));

  const int n_nodes = (1 << (depth + 1)) - 1;
  FlatTree<T> t;
  t.n_nodes = n_nodes;
  t.leaf_width = 1;
  t.feature.resize(n_nodes, -1);
  t.threshold.resize(n_nodes, T(0));
  t.split_op.resize(n_nodes, SplitOp::LessThan);
  t.left_child.resize(n_nodes, -1);
  t.right_child.resize(n_nodes, -1);
  t.default_child.resize(n_nodes, -1);
  t.leaf_value.resize(n_nodes, T(0));
  t.cat_offset.resize(n_nodes, 0);
  t.cat_count.resize(n_nodes, 0);

  const int first_leaf = (1 << depth) - 1;
  for (int i = 0; i < n_nodes; ++i) {
    if (i >= first_leaf) {
      t.leaf_value[i] = leaf_dist(rng);
    } else {
      t.feature[i] = feat_dist(rng);
      t.threshold[i] = thr_dist(rng);
      t.left_child[i] = 2 * i + 1;
      t.right_child[i] = 2 * i + 2;
      t.default_child[i] = 2 * i + 2;
    }
  }
  t.all_less_than = true;
  t.default_right = true;
  return t;
}

template <typename T>
static TreeEnsembleModel<T> build_model(int n_trees, int depth, int n_features,
                                        std::mt19937& rng) {
  TreeEnsembleModel<T> m;
  m.n_features = n_features;
  m.n_outputs = 1;
  m.n_trees = n_trees;
  m.aggregation = Aggregation::Sum;
  m.post_transform = PostTransform::Identity;
  m.trees.reserve(n_trees);
  for (int i = 0; i < n_trees; ++i)
    m.trees.push_back(build_tree<T>(depth, n_features, rng));
  return m;
}

// -----------------------------------------------------------------------
// Benchmark helpers
// -----------------------------------------------------------------------

struct Result {
  double mean_ms;
  double min_ms;
  double throughput;  // samples/sec at best (min) latency
};

template <typename T, typename Fn>
static Result bench(Fn&& fn, int n_samples, int n_reps, int n_warmup) {
  for (int i = 0; i < n_warmup; ++i) fn();

  std::vector<double> elapsed(n_reps);
  for (int r = 0; r < n_reps; ++r) {
    auto t0 = hrclock::now();
    fn();
    auto t1 = hrclock::now();
    elapsed[r] = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  const double mean =
      std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / n_reps;
  const double minv = *std::min_element(elapsed.begin(), elapsed.end());
  return {mean, minv, static_cast<double>(n_samples) / (minv / 1000.0)};
}

static void print_row(const char* label, const Result& r) {
  std::printf("  %-22s  mean=%7.3f ms   min=%7.3f ms   best=%10.0f samp/s\n",
              label, r.mean_ms, r.min_ms, r.throughput);
}

int main(int argc, char* argv[]) {
  const int n_trees = argc >= 2 ? std::atoi(argv[1]) : 200;
  const int depth = argc >= 3 ? std::atoi(argv[2]) : 6;
  const int n_features = argc >= 4 ? std::atoi(argv[3]) : 30;
  const int n_reps = argc >= 5 ? std::atoi(argv[4]) : 10;
  const int n_warmup = 3;

  std::mt19937 rng(1234);
  auto model_f32 = build_model<float>(n_trees, depth, n_features, rng);
  rng.seed(1234);
  auto model_f64 = build_model<double>(n_trees, depth, n_features, rng);

  const int nodes_per_tree = (1 << (depth + 1)) - 1;
  std::printf("Model  : %d trees  depth=%d  %d nodes/tree  %d features\n",
              n_trees, depth, nodes_per_tree, n_features);
  std::printf("Reps   : %d (+ %d warmup)\n\n", n_reps, n_warmup);

  // Build two separate f32 impls: one for the executor, one for direct calls.
  auto impl_f32_exec = make_tree_ensemble_impl(model_f32);
  auto impl_f32_direct = make_tree_ensemble_impl(model_f32);
  auto impl_f64_exec = make_tree_ensemble_impl_f64(model_f64);

  TreeEnsembleExecutor exec_f32(std::move(impl_f32_exec), n_features, 1);
  TreeEnsembleExecutor exec_f64(std::move(impl_f64_exec), n_features, 1);

  // Keep a raw pointer to the direct impl (it's managed by impl_f32_direct).
  TreeEnsembleNode::Impl* raw_impl_f32 = impl_f32_direct.get();

  std::mt19937 data_rng(99);
  std::uniform_real_distribution<double> dist(0.0, 1.0);

  const int batch_sizes[] = {1, 100, 1000, 10000, 100000};
  for (int n_samples : batch_sizes) {
    const std::size_t n_vals = static_cast<std::size_t>(n_samples) * n_features;

    std::vector<float> feat_f32(n_vals);
    std::vector<double> feat_f64(n_vals);
    data_rng.seed(99);
    for (std::size_t i = 0; i < n_vals; ++i) {
      double v = dist(data_rng);
      feat_f32[i] = static_cast<float>(v);
      feat_f64[i] = v;
    }
    std::vector<float> out_f32(n_samples);
    std::vector<double> out_f64(n_samples);

    std::printf("Batch = %6d samples:\n", n_samples);

    // (A) float32 executor — 1 field read (dtype_) + 1 virtual call (compute)
    auto rA = bench<float>(
        [&] { exec_f32.predict(feat_f32.data(), n_samples, out_f32.data()); },
        n_samples, n_reps, n_warmup);
    print_row("f32 executor", rA);

    // (B) float32 impl->compute directly — 1 virtual call only, no dtype check
    auto rB = bench<float>(
        [&] {
          raw_impl_f32->compute(feat_f32.data(), n_features, n_samples,
                                out_f32.data(), nullptr);
        },
        n_samples, n_reps, n_warmup);
    print_row("f32 impl->compute", rB);

    std::printf("  executor vs direct: %.2fx  (overhead = %.3f us)\n",
                rB.min_ms / rA.min_ms, (rA.min_ms - rB.min_ms) * 1000.0);

    // (C) float64 executor
    auto rC = bench<double>(
        [&] { exec_f64.predict(feat_f64.data(), n_samples, out_f64.data()); },
        n_samples, n_reps, n_warmup);
    print_row("f64 executor", rC);

    std::printf("  f32/f64 speedup   : %.2fx\n\n", rC.min_ms / rA.min_ms);
  }

  return 0;
}
