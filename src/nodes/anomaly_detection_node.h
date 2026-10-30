#ifndef OMLE_NODES_ANOMALY_DETECTION_NODE_H_
#define OMLE_NODES_ANOMALY_DETECTION_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "../graph_node.h"
#include "../model_base.h"

namespace omle::rt::impl {

// Which structured variant of the anomaly_detection body is populated.
enum class AnomalyKind : uint8_t {
  IsolationForest,
  OneClassSVM,
  LinearOneClassSVM,
  LocalOutlierFactor,
  EllipticEnvelope,
};

enum class AnomalyKernel : uint8_t { Linear, Poly, RBF, Sigmoid };

// One decision tree of an isolation forest, flattened.
//
// leaf_value already holds `depth + c(n_node_samples)` for the leaf — the
// converter folds the average-path-length correction in at conversion time, so
// scoring only has to walk to a leaf and read the value.
struct IsolationTree {
  std::vector<uint8_t> is_leaf;
  std::vector<int> split_feature;
  std::vector<double> split_threshold;
  std::vector<int> children_index;   // flat child id pool
  std::vector<int> children_offset;  // per node: index into children_index
  std::vector<int> children_count;
  std::vector<double> leaf_value;
};

// Scores anomaly-detection models. Output is always float64.
//
// The three declared outputs follow the omle.ml/AnomalyDetection contract and
// are emitted positionally for however many outputs the node declares:
//   score           — sklearn's score_samples: higher means more normal
//   decision_value  — score - offset: negative means outlier
//   prediction      — +1 inlier / -1 outlier
// A node that declares a single output gets the one matching its declared role,
// defaulting to score.
class AnomalyDetectionNode final : public GraphNode {
 public:
  struct Impl;
  explicit AnomalyDetectionNode(std::unique_ptr<Impl> impl);
  ~AnomalyDetectionNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;

 private:
  std::unique_ptr<Impl> impl_;
};

// Roles the caller wants emitted, in output order.
enum class AnomalyOutput : uint8_t { Score, DecisionValue, Prediction };

std::unique_ptr<AnomalyDetectionNode> make_isolation_forest_node(
    std::vector<IsolationTree> trees, int64_t max_samples, double offset,
    std::vector<AnomalyOutput> outputs);

std::unique_ptr<AnomalyDetectionNode> make_elliptic_envelope_node(
    std::vector<double> location, std::vector<double> precision, int n_features,
    double offset, std::vector<AnomalyOutput> outputs);

std::unique_ptr<AnomalyDetectionNode> make_linear_one_class_svm_node(
    std::vector<double> coefficients, double intercept, double offset,
    int n_features, std::vector<AnomalyOutput> outputs);

std::unique_ptr<AnomalyDetectionNode> make_one_class_svm_node(
    AnomalyKernel kernel, std::vector<double> support_vectors,
    std::vector<double> dual_coefficients, double intercept, double gamma,
    int degree, double coef0, int n_sv, int n_features, double offset,
    std::vector<AnomalyOutput> outputs);

std::unique_ptr<AnomalyDetectionNode> make_local_outlier_factor_node(
    std::vector<double> reference_samples, int n_reference, int n_features,
    int n_neighbors, const std::string& metric, double p, double offset,
    std::vector<AnomalyOutput> outputs);

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_ANOMALY_DETECTION_NODE_H_
