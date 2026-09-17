#ifndef OMLE_NODES_LINEAR_NODE_H_
#define OMLE_NODES_LINEAR_NODE_H_

#include <memory>
#include <vector>

#include "../graph_node.h"
#include "../model_base.h"

namespace omle::rt::impl {

// Opaque typed implementation — defined in linear_node.cpp only.
// T is either float or double; never exposed in this header.
class LinearNode final : public GraphNode {
 public:
  struct Impl;
  explicit LinearNode(std::unique_ptr<Impl> impl);
  ~LinearNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;

 private:
  std::unique_ptr<Impl> impl_;
};

// ModelBase adapter used by the standalone (non-DAG) Runtime path.
class LinearExecutor final : public ModelBase {
 public:
  explicit LinearExecutor(std::unique_ptr<LinearNode::Impl> impl,
                          int n_features, int n_outputs);
  ~LinearExecutor();

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
  std::unique_ptr<LinearNode::Impl> impl_;
  int n_features_;
  int n_outputs_;
  omle::rt::DataType dtype_;  // cached at construction — dtype never changes
};

// -----------------------------------------------------------------------
// Factory functions (defined in linear_node.cpp; no templates leak here).
// -----------------------------------------------------------------------

// Build a float32 LinearNode::Impl from float weight data.
std::unique_ptr<LinearNode::Impl> make_linear_impl(
    std::vector<float> coefficients, std::vector<float> intercept,
    PostTransform post_transform, int n_features, int n_outputs);

// Build a float64 LinearNode::Impl from double weight data.
std::unique_ptr<LinearNode::Impl> make_linear_impl_f64(
    std::vector<double> coefficients, std::vector<double> intercept,
    PostTransform post_transform, int n_features, int n_outputs);

// Node-level factories — return the fully constructed node.
// Use these from TUs where LinearNode::Impl is incomplete.
std::unique_ptr<LinearNode> make_linear_node(std::vector<float> coefficients,
                                             std::vector<float> intercept,
                                             PostTransform post_transform,
                                             int n_features, int n_outputs);

std::unique_ptr<LinearNode> make_linear_node_f64(
    std::vector<double> coefficients, std::vector<double> intercept,
    PostTransform post_transform, int n_features, int n_outputs);

// Executor-level factories — return the fully constructed executor.
// Use these from TUs where LinearNode::Impl is incomplete.
std::unique_ptr<LinearExecutor> make_linear_executor(
    std::vector<float> coefficients, std::vector<float> intercept,
    PostTransform post_transform, int n_features, int n_outputs);

std::unique_ptr<LinearExecutor> make_linear_executor_f64(
    std::vector<double> coefficients, std::vector<double> intercept,
    PostTransform post_transform, int n_features, int n_outputs);

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_LINEAR_NODE_H_
