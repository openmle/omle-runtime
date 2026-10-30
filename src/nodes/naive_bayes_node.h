#ifndef OMLE_NODES_NAIVE_BAYES_NODE_H_
#define OMLE_NODES_NAIVE_BAYES_NODE_H_

#include <memory>
#include <optional>
#include <vector>

#include "../graph_node.h"
#include "../model_base.h"

namespace omle::rt::impl {

// Variant enum lives here so factory callers and readers share it.
enum class NaiveBayesVariant : uint8_t {
  Gaussian,
  Multinomial,
  Bernoulli,
  Categorical,
};

// Opaque typed implementation — defined in naive_bayes_node.cpp only.
// The node outputs two tensors: prediction (Float32 class index) and
// class probabilities (Float32 after softmax of log-posteriors).
// Input features are Float32 or Float64 depending on which factory is used.
class NaiveBayesNode final : public GraphNode {
 public:
  struct Impl;
  explicit NaiveBayesNode(std::unique_ptr<Impl> impl);
  ~NaiveBayesNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;

 private:
  std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// Factory functions (defined in naive_bayes_node.cpp; no templates leak).
// -----------------------------------------------------------------------

// Build a float32 NaiveBayesNode::Impl.
// variance_epsilon and binarize_threshold remain double regardless of T
// (they are scalar options, not feature arrays).
std::unique_ptr<NaiveBayesNode::Impl> make_naive_bayes_impl(
    NaiveBayesVariant variant, std::vector<float> class_log_priors,
    std::vector<float> means, std::vector<float> variances,
    std::optional<double> variance_epsilon, std::vector<float> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<float> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features);

// Build a float64 NaiveBayesNode::Impl.
std::unique_ptr<NaiveBayesNode::Impl> make_naive_bayes_impl_f64(
    NaiveBayesVariant variant, std::vector<double> class_log_priors,
    std::vector<double> means, std::vector<double> variances,
    std::optional<double> variance_epsilon,
    std::vector<double> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<double> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features);

// Node-level factories — return the fully constructed node.
// Use these from TUs where NaiveBayesNode::Impl is incomplete.
std::unique_ptr<NaiveBayesNode> make_naive_bayes_node(
    NaiveBayesVariant variant, std::vector<float> class_log_priors,
    std::vector<float> means, std::vector<float> variances,
    std::optional<double> variance_epsilon, std::vector<float> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<float> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features);

std::unique_ptr<NaiveBayesNode> make_naive_bayes_node_f64(
    NaiveBayesVariant variant, std::vector<double> class_log_priors,
    std::vector<double> means, std::vector<double> variances,
    std::optional<double> variance_epsilon,
    std::vector<double> feature_log_prob,
    std::optional<double> binarize_threshold,
    std::vector<double> category_log_prob, std::vector<int32_t> category_offset,
    std::vector<int32_t> category_count, int n_classes, int n_features);

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_NAIVE_BAYES_NODE_H_
