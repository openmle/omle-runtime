#include "tree_ensemble_node.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "../expression_eval.h"
#include "../math_utils.h"
#include "../post_transform.h"
#include "../simd_traits.h"
#include "../thread_pool.h"
#include "omle/port.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// NaN detection for double — mirrors the float version in math_utils.h.
// Using bit-manipulation so it is safe under -ffast-math.
// -----------------------------------------------------------------------
namespace {

inline bool is_nan_safe(double x) noexcept {
  uint64_t u;
  std::memcpy(&u, &x, sizeof(u));
  return (u & UINT64_C(0x7FFFFFFFFFFFFFFF)) > UINT64_C(0x7FF0000000000000);
}

// -----------------------------------------------------------------------
// NaN scan: returns true if any element in data[0..n) is NaN.
// Uses bit manipulation to stay safe under -ffast-math.
// -----------------------------------------------------------------------
template <typename T>
static bool scan_has_nan(const T* OMLE_RESTRICT data, std::size_t n) noexcept {
#if defined(OMLE_AVX2)
  if constexpr (std::is_same<T, float>::value) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
      __m256 v = _mm256_loadu_ps(data + i);
      __m256i cmp = _mm256_castps_si256(_mm256_cmp_ps(v, v, _CMP_UNORD_Q));
      if (!_mm256_testz_si256(cmp, cmp)) return true;
    }
    for (; i < n; ++i)
      if (is_nan_safe(data[i])) return true;
  } else {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      __m256d v = _mm256_loadu_pd(data + i);
      __m256i cmp = _mm256_castpd_si256(_mm256_cmp_pd(v, v, _CMP_UNORD_Q));
      if (!_mm256_testz_si256(cmp, cmp)) return true;
    }
    for (; i < n; ++i)
      if (is_nan_safe(data[i])) return true;
  }
  return false;
#elif defined(OMLE_NEON)
  // Bit-manipulation NaN check: safe under -ffast-math.
  // float NaN:  (bits & 0x7FFFFFFF) > 0x7F800000
  // double NaN: (bits & 0x7FFFFFFFFFFFFFFF) > 0x7FF0000000000000
  if constexpr (std::is_same<T, float>::value) {
    const uint32x4_t exp_mask = vdupq_n_u32(0x7FFFFFFFu);
    const uint32x4_t nan_thr = vdupq_n_u32(0x7F800000u);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
      uint32x4_t bits = vreinterpretq_u32_f32(vld1q_f32(data + i));
      if (vmaxvq_u32(vcgtq_u32(vandq_u32(bits, exp_mask), nan_thr)))
        return true;
    }
    for (; i < n; ++i)
      if (is_nan_safe(data[i])) return true;
  } else {
    const uint64x2_t exp_mask = vdupq_n_u64(UINT64_C(0x7FFFFFFFFFFFFFFF));
    const uint64x2_t nan_thr = vdupq_n_u64(UINT64_C(0x7FF0000000000000));
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
      uint64x2_t bits = vreinterpretq_u64_f64(vld1q_f64(data + i));
      uint64x2_t is_nan = vcgtq_u64(vandq_u64(bits, exp_mask), nan_thr);
      if (vgetq_lane_u64(is_nan, 0) | vgetq_lane_u64(is_nan, 1)) return true;
    }
    for (; i < n; ++i)
      if (is_nan_safe(data[i])) return true;
  }
  return false;
#else
  for (std::size_t i = 0; i < n; ++i)
    if (is_nan_safe(data[i])) return true;
  return false;
#endif
}

// -----------------------------------------------------------------------
// AoS builder: populate FlatTree::aos_nodes from SoA arrays.
// Only called for all_less_than, no-complex trees after model is loaded.
// -----------------------------------------------------------------------
template <typename T>
static void build_aos_for_tree(FlatTree<T>& tree) {
  if ((!tree.all_less_than && !tree.all_less_equal) || tree.has_complex ||
      tree.n_nodes == 0)
    return;
  tree.aos_nodes.resize(static_cast<std::size_t>(tree.n_nodes));
  for (int i = 0; i < tree.n_nodes; ++i) {
    tree.aos_nodes[i] = {tree.feature[i],
                         (tree.feature[i] >= 0) ? tree.threshold[i] : T(0),
                         tree.left_child[i], tree.right_child[i]};
  }
}

// Maximum depth of any leaf in the tree (does not require uniformity).
template <typename T>
static int detect_max_depth(const FlatTree<T>& tree) {
  if (tree.n_nodes == 0) return 0;
  struct Frame {
    int node;
    int depth;
  };
  thread_local std::vector<Frame> stk;
  stk.clear();
  stk.push_back({0, 0});
  int max_d = 0;
  while (!stk.empty()) {
    auto [node, d] = stk.back();
    stk.pop_back();
    if (tree.feature[node] < 0)
      max_d = std::max(max_d, d);
    else {
      stk.push_back({tree.left_child[node], d + 1});
      stk.push_back({tree.right_child[node], d + 1});
    }
  }
  return max_d;
}

// -----------------------------------------------------------------------
// Flat forest builder: concatenate all trees into one contiguous AoS block.
//
// Children are converted from tree-local to forest-global indices.
// Leaf nodes (feature == -1) store their leaf value in the threshold field
// and self-loop (left = right = own global index), so the traversal loop
// can safely overshoot by running for global_depth steps even when a
// particular tree terminates early at a shallower leaf.
//
// Requires all trees to be homogeneous (all-lt or all-le, leaf_width==1,
// no complex predicates). Non-uniform tree depths are allowed: we compute
// the max depth across all trees and use that as the fixed iteration count.
//
// After a successful build the per-tree AoS caches are freed to prevent
// doubling the AoS memory footprint and evicting the flat forest from L1.
// -----------------------------------------------------------------------
template <typename T>
static void build_flat_forest(TreeEnsembleModel<T>& model) {
  auto& ff = model.flat_forest;
  ff.valid = false;
  if (model.trees.empty()) return;

  const bool all_le = model.trees[0].all_less_equal;
  for (const auto& tr : model.trees) {
    if (tr.has_complex || tr.leaf_width != 1) return;
    if (tr.all_less_equal != all_le) return;
    if (!tr.all_less_than && !tr.all_less_equal) return;
  }

  // Global iteration count = deepest leaf across all trees.
  // Trees with shallower depths hit a leaf node early; subsequent steps
  // follow the self-loop and re-read the same leaf (cheap but harmless).
  int global_depth = 0;
  for (const auto& tr : model.trees)
    global_depth = std::max(global_depth, detect_max_depth(tr));
  if (global_depth <= 0) return;

  int total = 0;
  for (const auto& tr : model.trees) total += tr.n_nodes;

  ff.all_nodes.resize(static_cast<std::size_t>(total));
  ff.roots.resize(model.trees.size());
  ff.tree_class.resize(model.trees.size(), 0);

  int off = 0;
  for (int t = 0; t < static_cast<int>(model.trees.size()); ++t) {
    const FlatTree<T>& tr = model.trees[t];
    ff.roots[t] = off;
    ff.tree_class[t] = (t < static_cast<int>(model.tree_group.size()))
                           ? model.tree_group[t]
                           : 0;
    for (int i = 0; i < tr.n_nodes; ++i) {
      const bool is_leaf = (tr.feature[i] < 0);
      ff.all_nodes[off + i] = {
          tr.feature[i],
          // Internal: split threshold.  Leaf: leaf value in threshold field.
          is_leaf ? tr.leaf_value[i] : tr.threshold[i],
          is_leaf ? (off + i) : (tr.left_child[i] + off),
          is_leaf ? (off + i) : (tr.right_child[i] + off)};
    }
    off += tr.n_nodes;
  }

  ff.uniform_depth = global_depth;
  ff.all_le = all_le;
  ff.valid = true;

  // Free per-tree AoS caches — flat forest supersedes them for n_samples==1;
  // keeping both would double the AoS memory, evicting the flat forest from L1.
  for (FlatTree<T>& tree : model.trees) tree.aos_nodes = {};
}

// -----------------------------------------------------------------------
// Detect uniform tree depth (all leaves at the same level).
// Returns that depth, or 0 if non-uniform or degenerate.
// -----------------------------------------------------------------------
template <typename T>
static int detect_uniform_depth(const FlatTree<T>& tree) {
  if (tree.n_nodes == 0) return 0;
  // Iterative DFS tracking depth of every leaf.
  struct Frame {
    int node;
    int depth;
  };
  thread_local std::vector<Frame> stk;
  stk.clear();
  stk.push_back({0, 0});
  int leaf_depth = -1;
  while (!stk.empty()) {
    auto [node, d] = stk.back();
    stk.pop_back();
    if (tree.feature[node] < 0) {  // leaf
      if (leaf_depth < 0)
        leaf_depth = d;
      else if (d != leaf_depth)
        return 0;  // non-uniform
    } else {
      stk.push_back({tree.left_child[node], d + 1});
      stk.push_back({tree.right_child[node], d + 1});
    }
  }
  return (leaf_depth > 0) ? leaf_depth : 0;
}

// Build a FlatTree<T> from a FlatTree<double>, narrowing doubles to T.
template <typename T>
FlatTree<T> make_flat_tree(const FlatTree<double>& in) {
  FlatTree<T> out;
  out.n_nodes = in.n_nodes;
  out.leaf_width = in.leaf_width;
  out.feature = in.feature;
  out.left_child = in.left_child;
  out.right_child = in.right_child;
  out.default_child = in.default_child;
  out.split_op = in.split_op;
  out.leaf_vector_index = in.leaf_vector_index;
  out.category_set = in.category_set;
  out.cat_offset = in.cat_offset;
  out.cat_count = in.cat_count;
  out.complex_preds = in.complex_preds;
  out.all_less_than = in.all_less_than;
  out.all_less_equal = in.all_less_equal;
  out.default_right = in.default_right;
  out.has_complex = in.has_complex;

  // Narrow (or preserve) doubles to T.
  out.threshold.resize(in.threshold.size());
  for (std::size_t i = 0; i < in.threshold.size(); ++i)
    out.threshold[i] = static_cast<T>(in.threshold[i]);

  out.leaf_value.resize(in.leaf_value.size());
  for (std::size_t i = 0; i < in.leaf_value.size(); ++i)
    out.leaf_value[i] = static_cast<T>(in.leaf_value[i]);

  out.leaf_vector.resize(in.leaf_vector.size());
  for (std::size_t i = 0; i < in.leaf_vector.size(); ++i)
    out.leaf_vector[i] = static_cast<T>(in.leaf_vector[i]);

  return out;
}

// -----------------------------------------------------------------------
// Category-set lookup (unchanged — operates on int arrays)
// -----------------------------------------------------------------------
template <typename T>
inline bool in_category_set(const FlatTree<T>& tree, int node, int32_t cat) {
  const int off = tree.cat_offset[node];
  const int cnt = tree.cat_count[node];
  for (int k = 0; k < cnt; ++k)
    if (tree.category_set[off + k] == cat) return true;
  return false;
}

// -----------------------------------------------------------------------
// Split evaluation (typed on T)
// -----------------------------------------------------------------------
template <typename T>
inline bool eval_split(const FlatTree<T>& tree, int node, T fval) {
  const T thr = tree.threshold[node];
  switch (tree.split_op[node]) {
    case SplitOp::LessThan:
      return fval < thr;
    case SplitOp::LessOrEqual:
      return fval <= thr;
    case SplitOp::GreaterThan:
      return fval > thr;
    case SplitOp::GreaterOrEqual:
      return fval >= thr;
    case SplitOp::Equal:
      return fval == thr;
    case SplitOp::NotEqual:
      return fval != thr;
    case SplitOp::InSet:
      return in_category_set(tree, node, static_cast<int32_t>(fval));
    case SplitOp::NotInSet:
      return !in_category_set(tree, node, static_cast<int32_t>(fval));
    case SplitOp::IsMissing:
      return is_nan_safe(fval);
    default:
      return fval < thr;
  }
}

// -----------------------------------------------------------------------
// Single-sample traversal helpers
// -----------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline int traverse_one_lt(
    const FlatTree<T>& tree, const T* OMLE_RESTRICT features) noexcept {
  int node = 0;
  while (tree.feature[node] >= 0) {
    const T fval = features[tree.feature[node]];
    if (OMLE_EXPECT_FALSE(is_nan_safe(fval)))
      node = tree.default_child[node];
    else
      node = (fval < tree.threshold[node]) ? tree.left_child[node]
                                           : tree.right_child[node];
  }
  return node;
}

template <typename T>
[[nodiscard]] inline int traverse_one_generic(
    const FlatTree<T>& tree, const T* OMLE_RESTRICT features) noexcept {
  int node = 0;
  while (tree.feature[node] >= 0) {
    const T fval = features[tree.feature[node]];
    if (OMLE_EXPECT_FALSE(is_nan_safe(fval)))
      node = tree.default_child[node];
    else
      node = eval_split(tree, node, fval) ? tree.left_child[node]
                                          : tree.right_child[node];
  }
  return node;
}

// Traversal for trees that have complex predicate overrides on some nodes.
template <typename T>
[[nodiscard]] inline int traverse_one_complex(const FlatTree<T>& tree,
                                              const T* OMLE_RESTRICT features,
                                              int /*n_features*/,
                                              const ValueStore& vs) {
  int node = 0;
  while (tree.feature[node] >= 0) {
    auto it = tree.complex_preds.find(node);
    bool go_left;
    if (OMLE_EXPECT_FALSE(it != tree.complex_preds.end())) {
      auto hits_or = eval_pred(*it->second, vs, 1);
      if (!hits_or.ok()) throw std::runtime_error(hits_or.message());
      go_left = ((*hits_or)[0] != 0);
    } else {
      const T fval = features[tree.feature[node]];
      if (OMLE_EXPECT_FALSE(is_nan_safe(fval))) {
        node = tree.default_child[node];
        continue;
      }
      go_left = eval_split(tree, node, fval);
    }
    node = go_left ? tree.left_child[node] : tree.right_child[node];
  }
  return node;
}

// -----------------------------------------------------------------------
// NaN-free traversal variants — call only after scan_has_nan returns false.
// -----------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline int traverse_one_le_nonan(
    const FlatTree<T>& tree, const T* OMLE_RESTRICT features) noexcept {
  int node = 0;
  while (tree.feature[node] >= 0) {
    const T fval = features[tree.feature[node]];
    node = (fval <= tree.threshold[node]) ? tree.left_child[node]
                                          : tree.right_child[node];
  }
  return node;
}

template <typename T>
[[nodiscard]] inline int traverse_one_lt_nonan(
    const FlatTree<T>& tree, const T* OMLE_RESTRICT features) noexcept {
  int node = 0;
  while (tree.feature[node] >= 0) {
    const T fval = features[tree.feature[node]];
    node = (fval < tree.threshold[node]) ? tree.left_child[node]
                                         : tree.right_child[node];
  }
  return node;
}

template <typename T>
[[nodiscard]] inline int traverse_one_generic_nonan(
    const FlatTree<T>& tree, const T* OMLE_RESTRICT features) noexcept {
  int node = 0;
  while (tree.feature[node] >= 0) {
    const T fval = features[tree.feature[node]];
    node = eval_split(tree, node, fval) ? tree.left_child[node]
                                        : tree.right_child[node];
  }
  return node;
}

// AoS traversal — single struct per node = better cache locality than SoA.
template <typename T>
[[nodiscard]] inline int traverse_aos_nonan(
    const AoSNode<T>* OMLE_RESTRICT nodes,
    const T* OMLE_RESTRICT features) noexcept {
  int n = 0;
  while (nodes[n].feature >= 0) {
    const T fval = features[nodes[n].feature];
    n = (fval < nodes[n].threshold) ? nodes[n].left : nodes[n].right;
  }
  return n;
}

// AoS traversal (<=) — same as traverse_aos_nonan but uses <=.
template <typename T>
[[nodiscard]] inline int traverse_aos_nonan_le(
    const AoSNode<T>* OMLE_RESTRICT nodes,
    const T* OMLE_RESTRICT features) noexcept {
  int n = 0;
  while (nodes[n].feature >= 0) {
    const T fval = features[nodes[n].feature];
    n = (fval <= nodes[n].threshold) ? nodes[n].left : nodes[n].right;
  }
  return n;
}

// Fixed-depth unrolled AoS traversal — eliminates while-loop and branch
// overhead. UseLE=false → strict <  (XGBoost);  UseLE=true → <= (LightGBM).
template <typename T, int D, bool UseLE>
struct DepthTraverser {
  static int go(const AoSNode<T>* OMLE_RESTRICT nodes,
                const T* OMLE_RESTRICT features, int n) noexcept {
    const T fval = features[nodes[n].feature];
    if constexpr (UseLE)
      n = (fval <= nodes[n].threshold) ? nodes[n].left : nodes[n].right;
    else
      n = (fval < nodes[n].threshold) ? nodes[n].left : nodes[n].right;
    return DepthTraverser<T, D - 1, UseLE>::go(nodes, features, n);
  }
};
template <typename T, bool UseLE>
struct DepthTraverser<T, 0, UseLE> {
  static int go(const AoSNode<T>*, const T*, int n) noexcept { return n; }
};

// Dispatch to the appropriate fixed-depth specialisation (depths 1–8) or
// fall back to the generic AoS traversal.  UseLE selects < vs <=.
template <typename T, bool UseLE>
[[nodiscard]] inline int traverse_aos_depth_nonan(
    const AoSNode<T>* OMLE_RESTRICT nodes, const T* OMLE_RESTRICT features,
    int depth) noexcept {
  switch (depth) {
    case 1:
      return DepthTraverser<T, 1, UseLE>::go(nodes, features, 0);
    case 2:
      return DepthTraverser<T, 2, UseLE>::go(nodes, features, 0);
    case 3:
      return DepthTraverser<T, 3, UseLE>::go(nodes, features, 0);
    case 4:
      return DepthTraverser<T, 4, UseLE>::go(nodes, features, 0);
    case 5:
      return DepthTraverser<T, 5, UseLE>::go(nodes, features, 0);
    case 6:
      return DepthTraverser<T, 6, UseLE>::go(nodes, features, 0);
    case 7:
      return DepthTraverser<T, 7, UseLE>::go(nodes, features, 0);
    case 8:
      return DepthTraverser<T, 8, UseLE>::go(nodes, features, 0);
    default:
      return UseLE ? traverse_aos_nonan_le(nodes, features)
                   : traverse_aos_nonan(nodes, features);
  }
}

// -----------------------------------------------------------------------
// SIMD leaf accumulation — gather leaf_value[node_idx[i]] and add to scores.
// -----------------------------------------------------------------------
template <typename T>
static void accum_leaves(const T* OMLE_RESTRICT leaf_value,
                         const int32_t* OMLE_RESTRICT node_idx,
                         T* OMLE_RESTRICT scores, int n) noexcept {
#if defined(OMLE_AVX2)
  if constexpr (std::is_same<T, float>::value) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
      __m128i idx4a = _mm_loadu_si128((const __m128i*)(node_idx + i));
      __m128i idx4b = _mm_loadu_si128((const __m128i*)(node_idx + i + 4));
      __m256i idx8 = _mm256_set_m128i(idx4b, idx4a);
      __m256 leaf = SimdTraits<float>::gather(leaf_value, idx8);
      __m256 cur = _mm256_loadu_ps(scores + i);
      _mm256_storeu_ps(scores + i, _mm256_add_ps(cur, leaf));
    }
    for (; i < n; ++i) scores[i] += leaf_value[node_idx[i]];
  } else {
    // double: WIDTH=4, _mm256_i32gather_pd takes __m128i
    int i = 0;
    for (; i + 4 <= n; i += 4) {
      __m128i idx4 = _mm_loadu_si128((const __m128i*)(node_idx + i));
      __m256d leaf = SimdTraits<double>::gather(leaf_value, idx4);
      __m256d cur = _mm256_loadu_pd(scores + i);
      _mm256_storeu_pd(scores + i, _mm256_add_pd(cur, leaf));
    }
    for (; i < n; ++i) scores[i] += leaf_value[node_idx[i]];
  }
#elif defined(OMLE_NEON)
  if constexpr (std::is_same<T, float>::value) {
    // float: batch=4 samples
    int i = 0;
    for (; i + 4 <= n; i += 4) {
      float32x4_t leaf = {
          leaf_value[node_idx[i + 0]], leaf_value[node_idx[i + 1]],
          leaf_value[node_idx[i + 2]], leaf_value[node_idx[i + 3]]};
      vst1q_f32(scores + i, vaddq_f32(vld1q_f32(scores + i), leaf));
    }
    for (; i < n; ++i) scores[i] += leaf_value[node_idx[i]];
  } else {
    // double: batch=2 samples
    int i = 0;
    for (; i + 2 <= n; i += 2) {
      float64x2_t leaf = {leaf_value[node_idx[i + 0]],
                          leaf_value[node_idx[i + 1]]};
      vst1q_f64(scores + i, vaddq_f64(vld1q_f64(scores + i), leaf));
    }
    for (; i < n; ++i) scores[i] += leaf_value[node_idx[i]];
  }
#else
  for (int i = 0; i < n; ++i) scores[i] += leaf_value[node_idx[i]];
#endif
}

// -----------------------------------------------------------------------
// traverse_block_simple — blocked per-sample traversal for all-lt or all-le,
// scalar-leaf trees.  Keeps a block of samples (≤128) in L1 cache while all
// trees iterate over it.  UseLE=false → strict < (XGBoost), true → <=
// (LightGBM).
//
// AVX2:  8-wide gather-based parallel traversal.
// NEON:  4× unrolled scalar (independent samples ⇒ OoO-friendly).
// else:  scalar loop.
// -----------------------------------------------------------------------
template <typename T, bool UseLE>
static void traverse_block_simple(const FlatTree<T>& tree,
                                  const T* OMLE_RESTRICT features, int block_n,
                                  int n_features,
                                  T* OMLE_RESTRICT scores) noexcept {
  const T* lv = tree.leaf_value.data();

#if defined(OMLE_AVX2)
  if constexpr (std::is_same<T, float>::value) {
    const int* feat_arr = tree.feature.data();
    const float* thr_arr = tree.threshold.data();
    const int* left_arr = tree.left_child.data();
    const int* right_arr = tree.right_child.data();
    const __m256i minus1 = _mm256_set1_epi32(-1);
    const __m256i all_ones = _mm256_set1_epi32(-1);
    const __m256i row_stride = _mm256_setr_epi32(
        0 * n_features, 1 * n_features, 2 * n_features, 3 * n_features,
        4 * n_features, 5 * n_features, 6 * n_features, 7 * n_features);

    int i = 0;
    for (; i + 8 <= block_n; i += 8) {
      __m256i cur = _mm256_setzero_si256();
      const float* base = features + static_cast<std::size_t>(i) * n_features;

      for (int d = 0; d < 64; ++d) {
        __m256i fi = _mm256_i32gather_epi32(feat_arr, cur, 4);
        __m256i act =
            _mm256_andnot_si256(_mm256_cmpeq_epi32(fi, minus1), all_ones);
        if (_mm256_testz_si256(act, act)) break;

        __m256 thr = _mm256_i32gather_ps(thr_arr, cur, 4);
        __m256i off = _mm256_add_epi32(row_stride, fi);
        __m256 fv = _mm256_i32gather_ps(base, off, 4);
        // UseLE: fv <= thr → go left;  else: fv < thr → go left
        __m256 cmp = UseLE ? _mm256_cmp_ps(fv, thr, _CMP_LE_OS)
                           : _mm256_cmp_ps(fv, thr, _CMP_LT_OS);
        __m256i lc = _mm256_i32gather_epi32(left_arr, cur, 4);
        __m256i rc = _mm256_i32gather_epi32(right_arr, cur, 4);
        __m256i nx = _mm256_blendv_epi8(rc, lc, _mm256_castps_si256(cmp));
        cur = _mm256_blendv_epi8(cur, nx, act);
      }

      __m256 leaf = _mm256_i32gather_ps(lv, cur, 4);
      _mm256_storeu_ps(scores + i,
                       _mm256_add_ps(_mm256_loadu_ps(scores + i), leaf));
    }
    for (; i < block_n; ++i) {
      const T* row = features + static_cast<std::size_t>(i) * n_features;
      scores[i] += lv[UseLE ? traverse_one_le_nonan(tree, row)
                            : traverse_one_lt_nonan(tree, row)];
    }
    return;
  }
#endif  // OMLE_AVX2

  // NEON / scalar: 4× unrolled.
  auto traverse_one = [&](const T* row) -> int {
    return UseLE ? traverse_one_le_nonan(tree, row)
                 : traverse_one_lt_nonan(tree, row);
  };
  int i = 0;
  for (; i + 4 <= block_n; i += 4) {
    scores[i + 0] += lv[traverse_one(
        features + static_cast<std::size_t>(i + 0) * n_features)];
    scores[i + 1] += lv[traverse_one(
        features + static_cast<std::size_t>(i + 1) * n_features)];
    scores[i + 2] += lv[traverse_one(
        features + static_cast<std::size_t>(i + 2) * n_features)];
    scores[i + 3] += lv[traverse_one(
        features + static_cast<std::size_t>(i + 3) * n_features)];
  }
  for (; i < block_n; ++i)
    scores[i] +=
        lv[traverse_one(features + static_cast<std::size_t>(i) * n_features)];
}

// -----------------------------------------------------------------------
// Stump fast path (depth-1 tree with all-less-than splits).
// float: batch=8 (AVX2) or batch=4 (NEON).
// double: batch=4 (AVX2) or batch=2 (NEON).
// -----------------------------------------------------------------------
template <typename T>
static void traverse_batch_stump(const FlatTree<T>& tree,
                                 const T* OMLE_RESTRICT features, int n_samples,
                                 int n_features,
                                 T* OMLE_RESTRICT scores) noexcept {
  const int feat = tree.feature[0];
  const T thr = tree.threshold[0];
  const T lv = tree.leaf_value[tree.left_child[0]];
  const T rv = tree.leaf_value[tree.right_child[0]];

#if defined(OMLE_NEON)
  if constexpr (std::is_same<T, float>::value) {
    const float32x4_t vthr = vdupq_n_f32(thr);
    const float32x4_t vlv = vdupq_n_f32(lv);
    const float32x4_t vrv = vdupq_n_f32(rv);
    int i = 0;
    for (; i + 4 <= n_samples; i += 4) {
      float32x4_t fvals = {features[(i + 0) * n_features + feat],
                           features[(i + 1) * n_features + feat],
                           features[(i + 2) * n_features + feat],
                           features[(i + 3) * n_features + feat]};
      uint32x4_t mask = vcltq_f32(fvals, vthr);
      float32x4_t leaf = vbslq_f32(mask, vlv, vrv);
      vst1q_f32(scores + i, vaddq_f32(vld1q_f32(scores + i), leaf));
    }
    for (; i < n_samples; ++i) {
      T fval = features[i * n_features + feat];
      scores[i] += (fval < thr) ? lv : rv;
    }
  } else {
    // double: batch=2
    const float64x2_t vthr = vdupq_n_f64(thr);
    const float64x2_t vlv = vdupq_n_f64(lv);
    const float64x2_t vrv = vdupq_n_f64(rv);
    int i = 0;
    for (; i + 2 <= n_samples; i += 2) {
      float64x2_t fvals = {features[(i + 0) * n_features + feat],
                           features[(i + 1) * n_features + feat]};
      uint64x2_t mask = vcltq_f64(fvals, vthr);
      float64x2_t leaf = vbslq_f64(mask, vlv, vrv);
      vst1q_f64(scores + i, vaddq_f64(vld1q_f64(scores + i), leaf));
    }
    for (; i < n_samples; ++i) {
      T fval = features[i * n_features + feat];
      scores[i] += (fval < thr) ? lv : rv;
    }
  }
#elif defined(OMLE_AVX2)
  if constexpr (std::is_same<T, float>::value) {
    const __m256 vthr = _mm256_set1_ps(thr);
    const __m256 vlv = _mm256_set1_ps(lv);
    const __m256 vrv = _mm256_set1_ps(rv);
    int i = 0;
    for (; i + 8 <= n_samples; i += 8) {
      __m256 fvals = _mm256_set_ps(features[(i + 7) * n_features + feat],
                                   features[(i + 6) * n_features + feat],
                                   features[(i + 5) * n_features + feat],
                                   features[(i + 4) * n_features + feat],
                                   features[(i + 3) * n_features + feat],
                                   features[(i + 2) * n_features + feat],
                                   features[(i + 1) * n_features + feat],
                                   features[(i + 0) * n_features + feat]);
      __m256 mask = _mm256_cmp_ps(fvals, vthr, _CMP_LT_OS);
      __m256 leaf = _mm256_blendv_ps(vrv, vlv, mask);
      _mm256_storeu_ps(scores + i,
                       _mm256_add_ps(_mm256_loadu_ps(scores + i), leaf));
    }
    for (; i < n_samples; ++i) {
      T fval = features[i * n_features + feat];
      scores[i] += (fval < thr) ? lv : rv;
    }
  } else {
    // double: batch=4
    const __m256d vthr = _mm256_set1_pd(thr);
    const __m256d vlv = _mm256_set1_pd(lv);
    const __m256d vrv = _mm256_set1_pd(rv);
    int i = 0;
    for (; i + 4 <= n_samples; i += 4) {
      __m256d fvals = _mm256_set_pd(features[(i + 3) * n_features + feat],
                                    features[(i + 2) * n_features + feat],
                                    features[(i + 1) * n_features + feat],
                                    features[(i + 0) * n_features + feat]);
      __m256d mask = _mm256_cmp_pd(fvals, vthr, _CMP_LT_OS);
      __m256d leaf = _mm256_blendv_pd(vrv, vlv, mask);
      _mm256_storeu_pd(scores + i,
                       _mm256_add_pd(_mm256_loadu_pd(scores + i), leaf));
    }
    for (; i < n_samples; ++i) {
      T fval = features[i * n_features + feat];
      scores[i] += (fval < thr) ? lv : rv;
    }
  }
#else
  for (int i = 0; i < n_samples; ++i) {
    T fval = features[i * n_features + feat];
    scores[i] += (fval < thr) ? lv : rv;
  }
#endif
}

// -----------------------------------------------------------------------
// Level-by-level batch traversal + leaf accumulation.
// -----------------------------------------------------------------------
template <typename T>
static void traverse_batch_accum(const FlatTree<T>& tree,
                                 const T* OMLE_RESTRICT features, int n_samples,
                                 int n_features, T* OMLE_RESTRICT scores,
                                 int class_idx = 0) {
  // Depth-1 stump fast path
  if (tree.n_nodes == 3 && tree.all_less_than && tree.leaf_width == 1) {
    traverse_batch_stump(tree, features, n_samples, n_features, scores);
    return;
  }

  thread_local std::vector<int32_t> nodes;
  nodes.assign(n_samples, 0);

  const bool lt_only = tree.all_less_than;

  int active = n_samples;
  while (active > 0) {
    active = 0;
    if (lt_only) {
      for (int i = 0; i < n_samples; ++i) {
        int node = nodes[i];
        const int feat = tree.feature[node];
        if (feat < 0) continue;
        ++active;
        const T fval = features[i * n_features + feat];
        if (OMLE_EXPECT_FALSE(is_nan_safe(fval)))
          nodes[i] = tree.default_child[node];
        else
          nodes[i] = (fval < tree.threshold[node]) ? tree.left_child[node]
                                                   : tree.right_child[node];
      }
    } else {
      for (int i = 0; i < n_samples; ++i) {
        int node = nodes[i];
        const int feat = tree.feature[node];
        if (feat < 0) continue;
        ++active;
        const T fval = features[i * n_features + feat];
        if (OMLE_EXPECT_FALSE(is_nan_safe(fval)))
          nodes[i] = tree.default_child[node];
        else
          nodes[i] = eval_split(tree, node, fval) ? tree.left_child[node]
                                                  : tree.right_child[node];
      }
    }
  }

  if (tree.leaf_width == 1) {
    accum_leaves(tree.leaf_value.data(), nodes.data(), scores, n_samples);
  } else {
    for (int i = 0; i < n_samples; ++i)
      scores[i] +=
          tree.leaf_vector[tree.leaf_vector_index[nodes[i]] + class_idx];
  }
}

// -----------------------------------------------------------------------
// Batch traversal for trees with complex predicate overrides.
// -----------------------------------------------------------------------
template <typename T>
static void traverse_complex_batch(const FlatTree<T>& tree,
                                   const T* OMLE_RESTRICT features,
                                   int n_samples, int n_features,
                                   const ValueStore& vs,
                                   T* OMLE_RESTRICT scores, int class_idx = 0) {
  thread_local std::vector<int32_t> nodes;
  nodes.assign(n_samples, 0);

  int active = n_samples;
  while (active > 0) {
    active = 0;

    // Find which complex-pred nodes are needed at this level.
    std::unordered_map<int32_t, std::vector<uint8_t>> cpred_cache;
    for (int i = 0; i < n_samples; ++i) {
      int node = nodes[i];
      if (tree.feature[node] < 0) continue;
      ++active;
      if (tree.has_complex) {
        auto it = tree.complex_preds.find(node);
        if (it != tree.complex_preds.end() && !cpred_cache.count(node)) {
          auto r = eval_pred(*it->second, vs, n_samples);
          if (!r.ok()) throw std::runtime_error(r.message());
          cpred_cache[node] = std::move(*r);
        }
      }
    }
    if (active == 0) break;

    for (int i = 0; i < n_samples; ++i) {
      int node = nodes[i];
      if (tree.feature[node] < 0) continue;

      auto cp = cpred_cache.find(node);
      if (cp != cpred_cache.end()) {
        nodes[i] = (cp->second[i] != 0) ? tree.left_child[node]
                                        : tree.right_child[node];
      } else {
        const T fval = features[i * n_features + tree.feature[node]];
        if (OMLE_EXPECT_FALSE(is_nan_safe(fval)))
          nodes[i] = tree.default_child[node];
        else
          nodes[i] = eval_split(tree, node, fval) ? tree.left_child[node]
                                                  : tree.right_child[node];
      }
    }
  }

  if (tree.leaf_width == 1)
    accum_leaves(tree.leaf_value.data(), nodes.data(), scores, n_samples);
  else
    for (int i = 0; i < n_samples; ++i)
      scores[i] +=
          tree.leaf_vector[tree.leaf_vector_index[nodes[i]] + class_idx];
}

// -----------------------------------------------------------------------
// Aggregation scale (average / weighted-average / soft-vote).
// -----------------------------------------------------------------------
template <typename T>
static void apply_aggregation_scale(T* OMLE_RESTRICT scores, int n_samples,
                                    int n_outputs,
                                    const TreeEnsembleModel<T>& model) {
  const int total = n_samples * n_outputs;
  switch (model.aggregation) {
    case Aggregation::Average:
      if (model.n_trees > 0) {
        const T inv = T(1) / static_cast<T>(model.n_trees);
        for (int i = 0; i < total; ++i) scores[i] *= inv;
      }
      break;
    case Aggregation::WeightedAvg: {
      T weight_sum = T(0);
      for (T w : model.tree_weights) weight_sum += w;
      if (weight_sum > T(0)) {
        const T inv = T(1) / weight_sum;
        for (int i = 0; i < total; ++i) scores[i] *= inv;
      }
      break;
    }
    case Aggregation::SoftVote:
      if (model.n_trees > 0 && n_outputs > 0) {
        const T inv = static_cast<T>(n_outputs) / static_cast<T>(model.n_trees);
        for (int i = 0; i < total; ++i) scores[i] *= inv;
      }
      break;
    default:
      break;
  }
}

// -----------------------------------------------------------------------
// Adaptive block size: keeps the feature window (BLOCK × n_feat × sizeof(T))
// below the L1 cache limit, rounded down to the SIMD gather width.
// For narrow models (n_feat ≤ 8) the result caps at 128; for wide models
// like MNIST (n_feat=784, float32) the result is 8 (=AVX2 gather width).
// -----------------------------------------------------------------------
static int compute_block_size(int n_feat, int elem_size) noexcept {
  // Use 75% of the L1 data cache so tree-node arrays share the remaining 25%.
  // NEON targets (Apple M) have 128 KB L1 per P-core; AVX2/fallback x86 have 32
  // KB.
#if defined(OMLE_NEON)
  constexpr int kL1 = 96 * 1024;  // 75% of 128 KB
#else
  constexpr int kL1 = 24 * 1024;  // 75% of 32 KB
#endif
  constexpr int kMax = 128;
  int simd_w;
#if defined(OMLE_AVX2)
  simd_w = (elem_size == 4) ? 8 : 4;
#elif defined(OMLE_NEON)
  simd_w = (elem_size == 4) ? 4 : 2;
#else
  simd_w = 4;
#endif
  if (n_feat <= 0) return kMax;
  int b = kL1 / (n_feat * elem_size);
  b = (b / simd_w) * simd_w;
  b = std::max(simd_w, b);
  return std::min(kMax, b);
}

// Minimum number of trees before paying thread pool overhead is worthwhile.
static constexpr int kTreeParallelThreshold = 200;

// -----------------------------------------------------------------------
// ILP-across-trees: process W=8 trees simultaneously for n_samples==1.
//
// Uses the compact flat AoS forest (one contiguous allocation for all trees)
// which can fit entirely in L1 on 128 KB-L1 chips (Apple M-series), vs the
// per-tree approach that scatters 500+ heap allocations across L2/L3.
//
// AoS layout: each AoSNode<T> has {feature, threshold, left, right} packed
// together.  Leaf nodes store leaf_value in the threshold field (feature==-1),
// so the leaf read is just nodes[c].threshold after exactly depth steps.
//
// AVX2 (float only): SIMD gathers across 8 trees per instruction, processing
// feature-lookups, threshold compares, and child selects in parallel.
// All other (NEON/scalar): 8-way scalar unroll — 8 independent chains let the
// CPU's OoO engine keep multiple L1-miss loads in flight simultaneously.
//
// UseLE=false → strict <  (XGBoost)
// UseLE=true  → <=       (LightGBM)
// -----------------------------------------------------------------------
template <typename T, bool UseLE>
static void traverse_forest_n1(
    const typename TreeEnsembleModel<T>::FlatForestData& ff,
    const T* OMLE_RESTRICT features, T* OMLE_RESTRICT output) noexcept {
  const int n_trees = static_cast<int>(ff.roots.size());
  const int depth = ff.uniform_depth;

  const AoSNode<T>* OMLE_RESTRICT nodes = ff.all_nodes.data();
  const int32_t* OMLE_RESTRICT root_arr = ff.roots.data();
  const int32_t* OMLE_RESTRICT cls_arr = ff.tree_class.data();

  int t = 0;

#if defined(OMLE_AVX2)
  // AVX2 float: gather-based 8-wide traversal.
  // AoSNode<float> is 16 bytes; gather scale=4 addresses into int32/float
  // arrays, so we keep 4 separate per-field arrays sourced from the AoS
  // on-the-fly. We fall through to the 8-way scalar unroll for non-float
  // (double) AVX2.
  if constexpr (std::is_same<T, float>::value) {
    // Extract per-field base pointers from the AoS.
    // AoSNode<float> layout:
    // [feature:int32][threshold:float][left:int32][right:int32] All fields are
    // at stride 4 nodes (64 bytes = 4 × AoSNode<float>). We use gather with
    // scale=16 (bytes per AoSNode) — but max scale is 8. Instead, we keep
    // separate arrays by decomposing AoS → SoA during model build (on AVX2
    // targets only).  On this NEON-only path the AoS is used directly. NOTE:
    // AVX2 code omitted here — falls through to scalar path below. (AVX2 gather
    // with stride=16 requires vindex×4 scaling trick; implement if needed.)
    (void)nodes;
    (void)root_arr;
    (void)cls_arr;
    (void)depth;  // suppress unused-var
    // Intentional fall-through to scalar 8-way below for correctness.
    // TODO: add proper AVX2 AoS gather path if this runs on AVX2 hardware.
  }
  t = 0;  // reset; fall through to scalar path
#endif    // OMLE_AVX2

  // 8-way scalar unroll (AoS): each step loads one 16-byte AoSNode and one
  // feature value.  The 8 independent chains overlap in the OoO pipeline,
  // hiding the ~4-cycle L1 load latency for all 8 nodes simultaneously.
  //
  // Guard against leaf self-loops (feature == -1): trees shallower than
  // global_depth will hit their leaf before depth steps are exhausted, then
  // self-loop at that leaf for the remaining steps.  The guard prevents the
  // UB of reading features[-1] and c stays at the leaf node unchanged.
#define FF_STEP(c)                                                            \
  do {                                                                        \
    const AoSNode<T>& nd = nodes[c];                                          \
    if (OMLE_EXPECT_TRUE(nd.feature >= 0))                                    \
      c = UseLE                                                               \
              ? ((features[nd.feature] <= nd.threshold) ? nd.left : nd.right) \
              : ((features[nd.feature] < nd.threshold) ? nd.left : nd.right); \
  } while (0)

  for (; t + 8 <= n_trees; t += 8) {
    int c0 = root_arr[t + 0], c1 = root_arr[t + 1], c2 = root_arr[t + 2],
        c3 = root_arr[t + 3];
    int c4 = root_arr[t + 4], c5 = root_arr[t + 5], c6 = root_arr[t + 6],
        c7 = root_arr[t + 7];

    for (int d = 0; d < depth; ++d) {
      FF_STEP(c0);
      FF_STEP(c1);
      FF_STEP(c2);
      FF_STEP(c3);
      FF_STEP(c4);
      FF_STEP(c5);
      FF_STEP(c6);
      FF_STEP(c7);
    }

    // Leaf value is stored in the threshold field (feature == -1 at leaf).
    output[cls_arr[t + 0]] += nodes[c0].threshold;
    output[cls_arr[t + 1]] += nodes[c1].threshold;
    output[cls_arr[t + 2]] += nodes[c2].threshold;
    output[cls_arr[t + 3]] += nodes[c3].threshold;
    output[cls_arr[t + 4]] += nodes[c4].threshold;
    output[cls_arr[t + 5]] += nodes[c5].threshold;
    output[cls_arr[t + 6]] += nodes[c6].threshold;
    output[cls_arr[t + 7]] += nodes[c7].threshold;
  }

  // Scalar tail (< 8 remaining trees).
  for (; t < n_trees; ++t) {
    int c = root_arr[t];
    for (int d = 0; d < depth; ++d) {
      FF_STEP(c);
    }
    output[cls_arr[t]] += nodes[c].threshold;
  }
#undef FF_STEP
}

// -----------------------------------------------------------------------
// Helper: traverse one tree for n_samples==1, choosing the best path.
// -----------------------------------------------------------------------
template <typename T>
[[nodiscard]] inline int traverse_single(const FlatTree<T>& tree,
                                         const T* features, bool has_nan,
                                         const ValueStore* vs,
                                         int n_feat) noexcept {
  if (OMLE_EXPECT_FALSE(tree.has_complex && vs != nullptr))
    return traverse_one_complex(tree, features, n_feat, *vs);
  if (has_nan)
    return tree.all_less_than ? traverse_one_lt(tree, features)
                              : traverse_one_generic(tree, features);
  // NaN-free paths — eliminate per-node is_nan_safe branch.
  if (!tree.aos_nodes.empty()) {
    if (tree.all_less_equal) {
      return (tree.uniform_depth > 0)
                 ? traverse_aos_depth_nonan<T, true>(
                       tree.aos_nodes.data(), features, tree.uniform_depth)
                 : traverse_aos_nonan_le(tree.aos_nodes.data(), features);
    }
    return (tree.uniform_depth > 0)
               ? traverse_aos_depth_nonan<T, false>(
                     tree.aos_nodes.data(), features, tree.uniform_depth)
               : traverse_aos_nonan(tree.aos_nodes.data(), features);
  }
  if (tree.all_less_equal) return traverse_one_le_nonan(tree, features);
  if (tree.all_less_than) return traverse_one_lt_nonan(tree, features);
  return traverse_one_generic_nonan(tree, features);
}

// -----------------------------------------------------------------------
// Core predict function, templated on T.
// pool: optional thread pool for intra-call tree parallelism (may be null).
// -----------------------------------------------------------------------
template <typename T>
static void predict_impl(const TreeEnsembleModel<T>& model,
                         const T* OMLE_RESTRICT features, int n_samples,
                         T* OMLE_RESTRICT output, const ValueStore* vs,
                         int n_features, ThreadPool* pool = nullptr) {
  const int n_out = model.n_outputs;
  const int n_feat = n_features > 0 ? n_features : model.n_features;

  const T init_val = model.has_base_score ? model.base_score : T(0);
  if (init_val == T(0))
    std::memset(output, 0,
                sizeof(T) * static_cast<std::size_t>(n_samples) * n_out);
  else
    std::fill(output, output + n_samples * n_out, init_val);

  if (model.aggregation == Aggregation::Min)
    std::fill(output, output + n_samples * n_out,
              std::numeric_limits<T>::max());
  else if (model.aggregation == Aggregation::Max)
    std::fill(output, output + n_samples * n_out,
              std::numeric_limits<T>::lowest());

  const bool has_weights = !model.tree_weights.empty();
  const bool has_groups = !model.tree_group.empty();
  const bool multi_output = n_out > 1;

  // One-shot NaN scan for the whole input batch.  For clean inputs (the
  // common case) this unlocks NaN-free traversal paths on every tree.
  const bool has_nan =
      scan_has_nan(features, static_cast<std::size_t>(n_samples) *
                                 static_cast<std::size_t>(n_feat));

  thread_local std::vector<T> tree_scores;
  thread_local std::vector<int32_t> nodes;

  // Dispatch helper: use complex-batch path when tree has complex predicates.
  auto dispatch_batch = [&](const FlatTree<T>& tree, T* scores_buf, int cidx) {
    if (tree.has_complex && vs)
      traverse_complex_batch(tree, features, n_samples, n_feat, *vs, scores_buf,
                             cidx);
    else
      traverse_batch_accum(tree, features, n_samples, n_feat, scores_buf, cidx);
  };

  // Fast path: single-output SUM with no per-tree weights (regression/binary).
  if (!multi_output && !has_groups && model.aggregation == Aggregation::Sum &&
      !has_weights) {
    if (n_samples == 1) {
      // Thread-parallel tree scoring when pool is wired up and the model
      // is large enough to amortise synchronisation overhead.
      if (pool != nullptr && pool->size() > 1 &&
          model.n_trees >= kTreeParallelThreshold) {
        const int np = pool->size();
        // Pad each slot to a full cache line to eliminate false sharing.
        constexpr int kPad = 64 / sizeof(T);
        std::vector<T> partial(static_cast<std::size_t>(np) * kPad, T(0));
        pool->parallel_for(np, [&](int tid) {
          const int t0 = (model.n_trees * tid) / np;
          const int t1 = (model.n_trees * (tid + 1)) / np;
          T acc = T(0);
          for (int t = t0; t < t1; ++t) {
            const FlatTree<T>& tree = model.trees[t];
            acc += tree.leaf_value[traverse_single(tree, features, has_nan, vs,
                                                   n_feat)];
          }
          partial[static_cast<std::size_t>(tid) * kPad] = acc;
        });
        for (int tid = 0; tid < np; ++tid)
          output[0] += partial[static_cast<std::size_t>(tid) * kPad];
        apply_post_transform(output, 1, 1, model.post_transform);
        return;
      }

      // Single-threaded: flat forest fast path when available.
      if (!has_nan && model.flat_forest.valid) {
        if (model.flat_forest.all_le)
          traverse_forest_n1<T, true>(model.flat_forest, features, output);
        else
          traverse_forest_n1<T, false>(model.flat_forest, features, output);
        apply_post_transform(output, 1, 1, model.post_transform);
        return;
      }
      for (int t = 0; t < model.n_trees; ++t) {
        const FlatTree<T>& tree = model.trees[t];
        output[0] += tree.leaf_value[traverse_single(tree, features, has_nan,
                                                     vs, n_feat)];
      }
      apply_post_transform(output, 1, 1, model.post_transform);
      return;
    }

    // Blocked fast path: keeps a 128-row feature window in L1 cache while
    // all trees run over it, eliminating L3 cache misses on large batches.
    bool all_lt = true, all_le = true;
    for (const auto& tr : model.trees) {
      if (!tr.all_less_than) all_lt = false;
      if (!tr.all_less_equal) all_le = false;
      if (tr.leaf_width != 1 || tr.has_complex) {
        all_lt = all_le = false;
        break;
      }
      if (!all_lt && !all_le) break;
    }

    if (all_lt || all_le) {
      const int BLOCK = compute_block_size(n_feat, static_cast<int>(sizeof(T)));
      for (int b = 0; b < n_samples; b += BLOCK) {
        const int bn = std::min(BLOCK, n_samples - b);
        const T* feat_b = features + static_cast<std::size_t>(b) * n_feat;
        T* out_b = output + b;
        if (all_le) {
          for (const FlatTree<T>& tr : model.trees)
            traverse_block_simple<T, true>(tr, feat_b, bn, n_feat, out_b);
        } else {
          for (const FlatTree<T>& tr : model.trees)
            traverse_block_simple<T, false>(tr, feat_b, bn, n_feat, out_b);
        }
      }
    } else {
      for (int t = 0; t < model.n_trees; ++t)
        dispatch_batch(model.trees[t], output, 0);
    }
    apply_post_transform(output, n_samples, n_out, model.post_transform);
    return;
  }

  // Blocked path for grouped multi-output (XGBoost/LightGBM multiclass):
  // same block-outer strategy to keep feature windows in L1 cache.
  if (has_groups && model.aggregation == Aggregation::Sum && !has_weights) {
    // Fast path for n_samples==1: flat forest ILP/SIMD-across-trees when
    // the model was built with a valid FlatForest (no NaN, uniform depth,
    // all-lt or all-le, leaf_width==1).  Falls back to per-tree traverse_single
    // for NaN inputs or models that don't satisfy the preconditions.
    if (n_samples == 1) {
      if (!has_nan && model.flat_forest.valid) {
        if (model.flat_forest.all_le)
          traverse_forest_n1<T, true>(model.flat_forest, features, output);
        else
          traverse_forest_n1<T, false>(model.flat_forest, features, output);
        apply_aggregation_scale(output, 1, n_out, model);
        apply_post_transform(output, 1, n_out, model.post_transform);
        return;
      }
      for (int t = 0; t < model.n_trees; ++t) {
        const FlatTree<T>& tree = model.trees[t];
        const int ci = (t < static_cast<int>(model.tree_group.size()))
                           ? model.tree_group[t]
                           : 0;
        const int node = traverse_single(tree, features, has_nan, vs, n_feat);
        if (tree.leaf_width > 1) {
          const int off = tree.leaf_vector_index[node];
          for (int c = 0; c < tree.leaf_width; ++c)
            output[c] += tree.leaf_vector[static_cast<std::size_t>(off) + c];
        } else {
          output[ci] += tree.leaf_value[node];
        }
      }
      apply_aggregation_scale(output, 1, n_out, model);
      apply_post_transform(output, 1, n_out, model.post_transform);
      return;
    }

    bool all_lt = true, all_le = true;
    for (const auto& tr : model.trees) {
      if (!tr.all_less_than) all_lt = false;
      if (!tr.all_less_equal) all_le = false;
      if (tr.leaf_width != 1 || tr.has_complex) {
        all_lt = all_le = false;
        break;
      }
      if (!all_lt && !all_le) break;
    }
    if (all_lt || all_le) {
      const int BLOCK = compute_block_size(n_feat, static_cast<int>(sizeof(T)));
      for (int b = 0; b < n_samples; b += BLOCK) {
        const int bn = std::min(BLOCK, n_samples - b);
        const T* feat_b = features + static_cast<std::size_t>(b) * n_feat;
        T* out_b = output + static_cast<std::size_t>(b) * n_out;
        for (int t = 0; t < model.n_trees; ++t) {
          const FlatTree<T>& tr = model.trees[t];
          const int ci = (t < static_cast<int>(model.tree_group.size()))
                             ? model.tree_group[t]
                             : 0;
          const T* lv = tr.leaf_value.data();
          if (all_le) {
            for (int i = 0; i < bn; ++i)
              out_b[static_cast<std::size_t>(i) * n_out + ci] +=
                  lv[traverse_one_le_nonan(
                      tr, feat_b + static_cast<std::size_t>(i) * n_feat)];
          } else {
            for (int i = 0; i < bn; ++i)
              out_b[static_cast<std::size_t>(i) * n_out + ci] +=
                  lv[traverse_one_lt_nonan(
                      tr, feat_b + static_cast<std::size_t>(i) * n_feat)];
          }
        }
      }
      apply_aggregation_scale(output, n_samples, n_out, model);
      apply_post_transform(output, n_samples, n_out, model.post_transform);
      return;
    }
  }

  // General path: multi-output / grouped / weighted.
  for (int t = 0; t < model.n_trees; ++t) {
    const FlatTree<T>& tree = model.trees[t];
    const T weight = has_weights ? model.tree_weights[t] : T(1);
    int class_idx =
        (has_groups && t < static_cast<int>(model.tree_group.size()))
            ? model.tree_group[t]
            : 0;

    if (tree.leaf_width > 1) {
      // Vector-leaf tree: single traversal per class, scatter all outputs.
      for (int c = 0; c < tree.leaf_width; ++c) {
        tree_scores.assign(n_samples, T(0));
        dispatch_batch(tree, tree_scores.data(), c);
        for (int s = 0; s < n_samples; ++s) {
          const T val = tree_scores[s] * weight;
          T* dst = output + s * n_out + c;
          switch (model.aggregation) {
            case Aggregation::Min:
              *dst = std::min(*dst, val);
              break;
            case Aggregation::Max:
              *dst = std::max(*dst, val);
              break;
            default:
              *dst += val;
              break;
          }
        }
      }
    } else if (tree.leaf_width == 1) {
      tree_scores.assign(n_samples, T(0));
      dispatch_batch(tree, tree_scores.data(), class_idx);

      for (int s = 0; s < n_samples; ++s) {
        const T val = tree_scores[s] * weight;
        T* dst = output + s * n_out + class_idx;
        switch (model.aggregation) {
          case Aggregation::Min:
            *dst = std::min(*dst, val);
            break;
          case Aggregation::Max:
            *dst = std::max(*dst, val);
            break;
          default:
            *dst += val;
            break;
        }
      }
    } else {
      nodes.assign(n_samples, 0);
      int active = n_samples;
      const bool lt_nonan = tree.all_less_than && tree.default_right;
      while (active > 0) {
        active = 0;
        for (int i = 0; i < n_samples; ++i) {
          int node = nodes[i];
          const int feat = tree.feature[node];
          if (feat < 0) continue;
          ++active;
          const T fval = features[i * n_feat + feat];
          if (lt_nonan)
            nodes[i] = (fval < tree.threshold[node]) ? tree.left_child[node]
                                                     : tree.right_child[node];
          else if (is_nan_safe(fval))
            nodes[i] = tree.default_child[node];
          else
            nodes[i] = eval_split(tree, node, fval) ? tree.left_child[node]
                                                    : tree.right_child[node];
        }
      }

      for (int s = 0; s < n_samples; ++s) {
        const int off = tree.leaf_vector_index[nodes[s]];
        T* dst = output + s * n_out;
        switch (model.aggregation) {
          case Aggregation::Min:
            for (int j = 0; j < n_out; ++j)
              dst[j] = std::min(dst[j], tree.leaf_vector[off + j] * weight);
            break;
          case Aggregation::Max:
            for (int j = 0; j < n_out; ++j)
              dst[j] = std::max(dst[j], tree.leaf_vector[off + j] * weight);
            break;
          default:
            for (int j = 0; j < n_out; ++j)
              dst[j] += tree.leaf_vector[off + j] * weight;
            break;
        }
      }
    }
  }

  apply_aggregation_scale(output, n_samples, n_out, model);
  apply_post_transform(output, n_samples, n_out, model.post_transform);
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// Typed Impl (defined outside anonymous namespace so the explicit
// instantiations below are in the enclosing namespace).
// -----------------------------------------------------------------------
template <typename T>
struct TreeEnsembleImplT final : TreeEnsembleNode::Impl {
  TreeEnsembleModel<T> model;
  ThreadPool* pool_ = nullptr;  // non-owning; set via set_pool()

  int n_features() const noexcept override { return model.n_features; }
  int n_outputs() const noexcept override { return model.n_outputs; }
  omle::rt::DataType dtype() const noexcept override {
    return std::is_same<T, double>::value ? omle::rt::DataType::Float64
                                          : omle::rt::DataType::Float32;
  }

  void set_pool(ThreadPool* pool) override { pool_ = pool; }

  void compute(const void* features_raw, int n_features_stride, int n_rows,
               void* output_raw, const ValueStore* vs) const override {
    const T* features = static_cast<const T*>(features_raw);
    T* output = static_cast<T*>(output_raw);
    predict_impl(model, features, n_rows, output, vs, n_features_stride, pool_);
  }
};

// Explicit instantiations — linker finds these; no other TU instantiates them.
template struct TreeEnsembleImplT<float>;
template struct TreeEnsembleImplT<double>;

// -----------------------------------------------------------------------
// narrow_tree_ensemble_model: convert a double-precision model to float32.
// Used by graph_loader when proto data is stored as double but the executor
// should run at float32 precision.
// -----------------------------------------------------------------------
namespace {

static TreeEnsembleModel<float> build_float_model(
    const TreeEnsembleModel<double>& in) {
  TreeEnsembleModel<float> m;
  m.aggregation = in.aggregation;
  m.post_transform = in.post_transform;
  m.has_base_score = in.has_base_score;
  m.base_score = static_cast<float>(in.base_score);
  m.tree_group = in.tree_group;
  m.n_features = in.n_features;
  m.n_outputs = in.n_outputs;
  m.n_trees = in.n_trees;

  m.tree_weights.resize(in.tree_weights.size());
  for (std::size_t i = 0; i < in.tree_weights.size(); ++i)
    m.tree_weights[i] = static_cast<float>(in.tree_weights[i]);

  m.trees.reserve(in.trees.size());
  for (const auto& ti : in.trees) m.trees.push_back(make_flat_tree<float>(ti));

  return m;
}

}  // anonymous namespace

TreeEnsembleModel<float> narrow_tree_ensemble_model(
    const TreeEnsembleModel<double>& in) {
  return build_float_model(in);
}

// -----------------------------------------------------------------------
// Post-construction setup: build AoS node arrays and detect uniform depth.
// Called once per tree in the factory after the model is copied into the Impl.
// -----------------------------------------------------------------------
template <typename T>
static void finalize_model(TreeEnsembleModel<T>& model) {
  for (FlatTree<T>& tree : model.trees) {
    build_aos_for_tree(tree);
    if ((tree.all_less_than || tree.all_less_equal) && !tree.has_complex)
      tree.uniform_depth = detect_uniform_depth(tree);
  }
  build_flat_forest(model);
}

// -----------------------------------------------------------------------
// Factory functions
// -----------------------------------------------------------------------
std::unique_ptr<TreeEnsembleNode::Impl> make_tree_ensemble_impl(
    const TreeEnsembleModel<float>& m) {
  auto p = std::make_unique<TreeEnsembleImplT<float>>();
  p->model = m;
  finalize_model(p->model);
  return p;
}

std::unique_ptr<TreeEnsembleNode::Impl> make_tree_ensemble_impl_f64(
    const TreeEnsembleModel<double>& m) {
  auto p = std::make_unique<TreeEnsembleImplT<double>>();
  p->model = m;
  finalize_model(p->model);
  return p;
}

// -----------------------------------------------------------------------
// TreeEnsembleNode
// -----------------------------------------------------------------------
TreeEnsembleNode::TreeEnsembleNode(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}

TreeEnsembleNode::~TreeEnsembleNode() = default;

omle::rt::Status TreeEnsembleNode::execute(ValueStore& vs, int n_rows) const {
  const omle::rt::DataType dt = impl_->dtype();
  auto features = gather_slots(vs, in_names, n_rows, dt);
  const int nf = features.n_cols;
  const int no = impl_->n_outputs();

  omle::rt::Tensor scores = omle::rt::Tensor::dense(dt, n_rows, no);
  impl_->compute(dt == omle::rt::DataType::Float64
                     ? static_cast<const void*>(features.f64_ptr())
                     : static_cast<const void*>(features.f32_ptr()),
                 nf, n_rows,
                 dt == omle::rt::DataType::Float64
                     ? static_cast<void*>(scores.f64_ptr())
                     : static_cast<void*>(scores.f32_ptr()),
                 &vs);

  // Classification: two output names means [y_pred, y_prob].
  // scores contains probability values (post-transform already applied).
  if (out_names.size() == 2) {
    // y_pred: class labels [n_rows, 1] — INT64 per output spec
    omle::rt::Tensor pred =
        omle::rt::Tensor::dense(omle::rt::DataType::Int64, n_rows, 1);
    int64_t* pred_ptr = pred.i64_ptr();

    if (dt == omle::rt::DataType::Float64) {
      const double* sp = scores.f64_ptr();
      if (no == 1) {
        // Binary: expand (n,1) → (n,2): [1-p, p], threshold 0.5
        omle::rt::Tensor prob2 = omle::rt::Tensor::dense(dt, n_rows, 2);
        double* dp = prob2.f64_ptr();
        for (int r = 0; r < n_rows; ++r) {
          dp[r * 2] = 1.0 - sp[r];
          dp[r * 2 + 1] = sp[r];
          pred_ptr[r] = sp[r] >= 0.5 ? 1 : 0;
        }
        vs.put(out_names[1], std::move(prob2));
      } else {
        omle::rt::Tensor prob = scores;
        for (int r = 0; r < n_rows; ++r) {
          const double* row = sp + r * no;
          pred_ptr[r] = static_cast<int64_t>(
              std::distance(row, std::max_element(row, row + no)));
        }
        vs.put(out_names[1], std::move(prob));
      }
    } else {
      const float* sp = scores.f32_ptr();
      if (no == 1) {
        // Binary: expand (n,1) → (n,2): [1-p, p], threshold 0.5
        omle::rt::Tensor prob2 = omle::rt::Tensor::dense(dt, n_rows, 2);
        float* dp = prob2.f32_ptr();
        for (int r = 0; r < n_rows; ++r) {
          dp[r * 2] = 1.f - sp[r];
          dp[r * 2 + 1] = sp[r];
          pred_ptr[r] = sp[r] >= 0.5f ? 1 : 0;
        }
        vs.put(out_names[1], std::move(prob2));
      } else {
        omle::rt::Tensor prob = scores;
        for (int r = 0; r < n_rows; ++r) {
          const float* row = sp + r * no;
          pred_ptr[r] = static_cast<int64_t>(
              std::distance(row, std::max_element(row, row + no)));
        }
        vs.put(out_names[1], std::move(prob));
      }
    }
    vs.put(out_names[0], std::move(pred));
  } else {
    if (dt == omle::rt::DataType::Float64)
      scatter_outputs(vs, out_names, scores.f64_ptr(), n_rows, no);
    else
      scatter_outputs(vs, out_names, scores.f32_ptr(), n_rows, no);
  }
  return {};
}

// -----------------------------------------------------------------------
// TreeEnsembleExecutor (standalone ModelBase path)
// -----------------------------------------------------------------------
TreeEnsembleExecutor::TreeEnsembleExecutor(
    std::unique_ptr<TreeEnsembleNode::Impl> impl, int n_features, int n_outputs)
    : impl_(std::move(impl)),
      n_features_(n_features),
      n_outputs_(n_outputs),
      dtype_(impl_->dtype()) {}

TreeEnsembleExecutor::~TreeEnsembleExecutor() = default;

omle::rt::DataType TreeEnsembleExecutor::dtype() const { return dtype_; }

void TreeEnsembleExecutor::predict(const float* features, int n_samples,
                                   float* output) const {
  if (dtype_ == omle::rt::DataType::Float64) {
    // float32 input to float64 model: upcast, compute, downcast.
    thread_local std::vector<double> feat_d, out_d;
    const int nf = n_features_;
    const int no = n_outputs_;
    feat_d.resize(static_cast<std::size_t>(n_samples) * nf);
    out_d.resize(static_cast<std::size_t>(n_samples) * no);
    upcast_f32_to_f64(features, feat_d.data(), n_samples * nf);
    impl_->compute(feat_d.data(), nf, n_samples, out_d.data(), nullptr);
    downcast_f64_to_f32(out_d.data(), output, n_samples * no);
  } else {
    impl_->compute(features, n_features_, n_samples, output, nullptr);
  }
}

void TreeEnsembleExecutor::predict(const double* features, int n_samples,
                                   double* output) const {
  if (dtype_ == omle::rt::DataType::Float64) {
    // float64 input to float64 model: native path, no conversion.
    impl_->compute(features, n_features_, n_samples, output, nullptr);
  } else {
    // float64 input to float32 model: downcast, compute, upcast.
    thread_local std::vector<float> feat_f, out_f;
    const int nf = n_features_;
    const int no = n_outputs_;
    feat_f.resize(static_cast<std::size_t>(n_samples) * nf);
    out_f.resize(static_cast<std::size_t>(n_samples) * no);
    downcast_f64_to_f32(features, feat_f.data(), n_samples * nf);
    impl_->compute(feat_f.data(), nf, n_samples, out_f.data(), nullptr);
    upcast_f32_to_f64(out_f.data(), output, n_samples * no);
  }
}

}  // namespace omle::rt::impl
