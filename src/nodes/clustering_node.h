#ifndef OMLE_NODES_CLUSTERING_NODE_H_
#define OMLE_NODES_CLUSTERING_NODE_H_

#include <memory>
#include <string>
#include <vector>

#include "../graph_node.h"
#include "../model_base.h"

namespace omle::rt::impl {

// Enums live here so factory callers and readers share them without templates.
enum class ClusterKind : uint8_t { Prototype, GaussianMixture };

enum class ClusterDistance : uint8_t {
  Euclidean,
  SquaredEuclidean,
  Manhattan,
  Cosine,
};

enum class ClusterCovType : uint8_t { Full, Diagonal, Spherical };

// Opaque typed implementation — defined in clustering_node.cpp only.
// NOTE: Clustering always outputs Float32 (cluster indices and distances).
// The input features may be Float32 or Float64; the impl handles conversion
// internally by templating the distance/GMM kernels on T.
class ClusteringNode final : public GraphNode {
 public:
  struct Impl;
  explicit ClusteringNode(std::unique_ptr<Impl> impl);
  ~ClusteringNode();
  omle::rt::Status execute(ValueStore& vs, int n_rows) const override;

 private:
  std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------
// Factory functions (defined in clustering_node.cpp; no templates leak here).
// -----------------------------------------------------------------------

// Build a float32 ClusteringNode::Impl (prototype / k-means style).
std::unique_ptr<ClusteringNode::Impl> make_clustering_impl(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<float> centers, std::vector<float> weights,
    std::vector<float> means, std::vector<float> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features);

// Build a float64 ClusteringNode::Impl.
std::unique_ptr<ClusteringNode::Impl> make_clustering_impl_f64(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<double> centers, std::vector<double> weights,
    std::vector<double> means, std::vector<double> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features);

// Node-level factories — return the fully constructed node.
// Use these from TUs where ClusteringNode::Impl is incomplete.
std::unique_ptr<ClusteringNode> make_clustering_node(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<float> centers, std::vector<float> weights,
    std::vector<float> means, std::vector<float> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features);

std::unique_ptr<ClusteringNode> make_clustering_node_f64(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<double> centers, std::vector<double> weights,
    std::vector<double> means, std::vector<double> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features);

}  // namespace omle::rt::impl

#endif  // OMLE_NODES_CLUSTERING_NODE_H_
