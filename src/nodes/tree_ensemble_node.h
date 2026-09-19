#ifndef OMLE_NODES_TREE_ENSEMBLE_NODE_H_
#define OMLE_NODES_TREE_ENSEMBLE_NODE_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

#include "../ast_types.h"
#include "../graph_node.h"
#include "../model_base.h"
#include "../runtime_tensor.h"
#include "../thread_pool.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// AoS node — packs {feature, threshold, left, right} into one record
// for cache-friendly traversal on the NaN-free fast path.
// Only built for all_less_than trees without complex predicates.
// -----------------------------------------------------------------------
template <typename T>
struct AoSNode {
  int32_t feature;  // -1 → leaf
  T threshold;
  int32_t left;
  int32_t right;
  // float: 16 bytes (1/4 cache line)  double: 24 bytes
};

// -----------------------------------------------------------------------
// Typed flat tree (SoA layout).
// T is float (float32 precision) or double (float64 precision).
// All split/leaf values and thresholds are stored as T.
// Integer arrays and flags are type-independent.
// -----------------------------------------------------------------------
template <typename T>
struct FlatTree {
  int32_t n_nodes = 0;
  int32_t leaf_width = 1;  // >1 for multi-output trees (LightGBM DART etc.)
  int32_t uniform_depth =
      0;  // 0 = unknown/non-uniform; >0 = all leaves at this depth

  std::vector<int32_t> feature;
  std::vector<T> threshold;
  std::vector<int32_t> left_child;
  std::vector<int32_t> right_child;
  std::vector<int32_t> default_child;
  std::vector<SplitOp> split_op;
  std::vector<T> leaf_value;

  std::vector<T> leaf_vector;
  std::vector<int32_t> leaf_vector_index;

  std::vector<int32_t> category_set;
  std::vector<int32_t> cat_offset;
  std::vector<int32_t> cat_count;

  std::unordered_map<int32_t, PredPtr> complex_preds;

  // AoS cache: built by make_tree_ensemble_impl for all_less_than trees.
  std::vector<AoSNode<T>> aos_nodes;

  bool all_less_than = false;   // all splits use strict <  (XGBoost default)
  bool all_less_equal = false;  // all splits use <=        (LightGBM default)
  bool default_right = false;
  bool has_complex = false;
};

// -----------------------------------------------------------------------
// Typed tree ensemble model.
// T is float or double; all floating-point parameters use T.
// -----------------------------------------------------------------------
template <typename T>
struct TreeEnsembleModel {
  std::vector<FlatTree<T>> trees;

  Aggregation aggregation = Aggregation::Sum;
  PostTransform post_transform = PostTransform::Identity;

  std::vector<T> tree_weights;
  // Base scores added before post_transform. Empty when the model has none;
  // otherwise either one value shared by every output, or one per output. A
  // vector rather than a scalar because an ensemble may fit an intercept per
  // class, and emptiness already encodes absence without a separate flag.
  std::vector<T> base_scores;

  std::vector<int32_t> tree_group;

  int n_features = 0;
  int n_outputs = 1;
  int n_trees = 0;

  // -----------------------------------------------------------------------
  // Compact flat forest: all trees' AoS nodes in one contiguous block.
  // Built by finalize_model for n_samples==1 ILP-across-trees fast path.
  // Valid when: all trees share the same uniform_depth, are all-less-than or
  // all-less-equal, have leaf_width==1, and no complex predicates.
  //
  // Leaf nodes store their leaf value in the threshold field (feature == -1).
  // This keeps all per-node data in a single 16-byte struct (float) or 24-byte
  // (double), so the entire forest fits in L1 on 128KB-L1 M-series chips
  // (e.g. MNIST small = 500 × 15 × 16 = 120 KB).
  // -----------------------------------------------------------------------
  struct FlatForestData {
    std::vector<AoSNode<T>>
        all_nodes;               // all trees' nodes, global child indices
    std::vector<int32_t> roots;  // global root index per tree
    std::vector<int32_t>
        tree_class;  // output class index per tree (0 if no groups)
    int uniform_depth = 0;
    bool all_le = false;  // true = all <=, false = all <
    bool valid = false;
  } flat_forest;
};

// -----------------------------------------------------------------------
// DAG node wrapping a typed tree ensemble implementation.
// -----------------------------------------------------------------------
class TreeEnsembleNode final : public GraphNode {
 public:
  // Opaque typed implementation — defined in tree_ensemble_node.cpp only.
  struct Impl {
    virtual int n_features() const noexcept = 0;
    virtual int n_outputs() const noexcept = 0;
    virtual omle::rt::DataType dtype() const noexcept = 0;
    virtual void compute(const void* features, int n_features_stride,
                         int n_rows, void* output,
                         const ValueStore* vs = nullptr) const = 0;
    // Wire up an optional thread pool for intra-call tree parallelism.
    virtual void set_pool(ThreadPool* /*pool*/) {}
    virtual ~Impl() = default;
  };

  explicit TreeEnsembleNode(std::unique_ptr<Impl> impl);
  ~TreeEnsembleNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;
  void set_thread_pool(ThreadPool* pool) override { impl_->set_pool(pool); }

 private:
  std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// Factory functions (defined in tree_ensemble_node.cpp; no templates leak).
//
//   make_tree_ensemble_impl     → float32 Impl from a float32 model
//   make_tree_ensemble_impl_f64 → float64 Impl from a float64 model
//
// When the source data is double (e.g. from the graph loader), call
// narrow_tree_ensemble_model first to produce a TreeEnsembleModel<float>.
// -----------------------------------------------------------------------
std::unique_ptr<TreeEnsembleNode::Impl> make_tree_ensemble_impl(
    const TreeEnsembleModel<float>&);
std::unique_ptr<TreeEnsembleNode::Impl> make_tree_ensemble_impl_f64(
    const TreeEnsembleModel<double>&);

// Narrow a double-precision model to float32 (used by the graph loader when
// proto data is stored as doubles but the executor should run at float32).
TreeEnsembleModel<float> narrow_tree_ensemble_model(
    const TreeEnsembleModel<double>&);

// -----------------------------------------------------------------------
// ModelBase adapter used by the standalone (non-DAG) Runtime path.
// Exposes both float32 and float64 predict paths; the native path is
// determined by the precision of the wrapped Impl.
// -----------------------------------------------------------------------
class TreeEnsembleExecutor final : public ModelBase {
 public:
  explicit TreeEnsembleExecutor(std::unique_ptr<TreeEnsembleNode::Impl> impl,
                                int n_features, int n_outputs);

  ~TreeEnsembleExecutor();

  omle::rt::DataType dtype() const override;
  int num_inputs() const override { return n_features_; }
  int num_outputs() const override { return n_outputs_; }

  // Float32 path: native for float32 Impl; upcast/compute/downcast for float64
  // Impl.
  void predict(const float* features, int n_samples,
               float* output) const override;

  // Float64 path: native for float64 Impl; downcast/compute/upcast for float32
  // Impl.
  void predict(const double* features, int n_samples,
               double* output) const override;

 private:
  std::unique_ptr<TreeEnsembleNode::Impl> impl_;
  int n_features_;
  int n_outputs_;
  omle::rt::DataType dtype_;  // cached at construction — dtype never changes
};

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_TREE_ENSEMBLE_NODE_H_
