#ifndef OMLE_NODES_SVM_NODE_H_
#define OMLE_NODES_SVM_NODE_H_

#include <memory>
#include <optional>
#include <vector>

#include "../graph_node.h"
#include "../model_base.h"

namespace omle::rt::impl {

// Enums live here so factory callers and readers share them without templates.
enum class SVMKind : uint8_t { Linear, Kernel };

enum class KernelType : uint8_t {
  Linear = 0,
  Poly = 1,
  RBF = 2,
  Sigmoid = 3,
};

// Opaque typed implementation — defined in svm_node.cpp only.
class SVMNode final : public GraphNode {
 public:
  struct Impl;
  explicit SVMNode(std::unique_ptr<Impl> impl);
  ~SVMNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;

 private:
  std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// Factory functions (defined in svm_node.cpp; no templates leak here).
// -----------------------------------------------------------------------

// Build a float32 SVMNode::Impl.
std::unique_ptr<SVMNode::Impl> make_svm_impl(
    SVMKind kind, KernelType kernel_type, std::vector<float> coefficients,
    std::vector<float> intercept, std::vector<float> support_vectors,
    std::vector<float> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<float> prob_a = {},
    std::vector<float> prob_b = {}, std::vector<int> n_support = {});

// Build a float64 SVMNode::Impl.
std::unique_ptr<SVMNode::Impl> make_svm_impl_f64(
    SVMKind kind, KernelType kernel_type, std::vector<double> coefficients,
    std::vector<double> intercept, std::vector<double> support_vectors,
    std::vector<double> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<double> prob_a = {},
    std::vector<double> prob_b = {}, std::vector<int> n_support = {});

// Node-level factories — return the fully constructed node.
// Use these from TUs where SVMNode::Impl is incomplete.
std::unique_ptr<SVMNode> make_svm_node(
    SVMKind kind, KernelType kernel_type, std::vector<float> coefficients,
    std::vector<float> intercept, std::vector<float> support_vectors,
    std::vector<float> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<float> prob_a = {},
    std::vector<float> prob_b = {}, std::vector<int> n_support = {});

std::unique_ptr<SVMNode> make_svm_node_f64(
    SVMKind kind, KernelType kernel_type, std::vector<double> coefficients,
    std::vector<double> intercept, std::vector<double> support_vectors,
    std::vector<double> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<double> prob_a = {},
    std::vector<double> prob_b = {}, std::vector<int> n_support = {});

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_SVM_NODE_H_
