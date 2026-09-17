#include <gtest/gtest.h>

#include "clustering_node.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

// -----------------------------------------------------------------------
// Prototype clustering (k-means style)
// -----------------------------------------------------------------------

TEST(Clustering, Euclidean_NearestCentroid) {
  // centroid 0 at (0,0), centroid 1 at (10,10)
  auto node = make_clustering_node(
      ClusterKind::Prototype, ClusterDistance::Euclidean, ClusterCovType::Full,
      {0.0f, 0.0f, 10.0f, 10.0f},  // centers
      {},                          // weights
      {},                          // means
      {},                          // covariances
      {},                          // labels
      2,                           // n_clusters
      2);                          // n_features
  node->in_names = {"x"};
  node->out_names = {"label", "distances"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(3, 2);
  x.at(0, 0) = 1.0f;
  x.at(0, 1) = 1.0f;  // near 0
  x.at(1, 0) = 9.0f;
  x.at(1, 1) = 9.0f;  // near 1
  x.at(2, 0) = 5.0f;
  x.at(2, 1) = 5.0f;  // equidistant → cluster 0 (first)
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 3).ok());

  const Tensor& label = vs.get("label");
  EXPECT_EQ(label.i32_at(0, 0), 0);
  EXPECT_EQ(label.i32_at(1, 0), 1);
  // equidistant: both clusters are equally close, cluster 0 wins (first
  // comparison)
  EXPECT_EQ(label.i32_at(2, 0), 0);
}

TEST(Clustering, SquaredEuclidean) {
  auto node = make_clustering_node(
      ClusterKind::Prototype, ClusterDistance::SquaredEuclidean,
      ClusterCovType::Full, {0.0f, 100.0f},  // centers (1D)
      {}, {}, {}, {}, 2, 1);
  node->in_names = {"x"};
  node->out_names = {"label"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 1);
  x.at(0, 0) = 1.0f;   // closer to 0
  x.at(1, 0) = 90.0f;  // closer to 100
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  EXPECT_EQ(vs.get("label").i32_at(0, 0), 0);
  EXPECT_EQ(vs.get("label").i32_at(1, 0), 1);
}

TEST(Clustering, Manhattan) {
  auto node = make_clustering_node(
      ClusterKind::Prototype, ClusterDistance::Manhattan, ClusterCovType::Full,
      {0.0f, 0.0f, 3.0f, 4.0f},  // centers
      {}, {}, {}, {}, 2, 2);
  node->in_names = {"x"};
  node->out_names = {"label"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(1, 2);
  x.at(0, 0) = 2.0f;
  x.at(0, 1) = 3.0f;
  // Manhattan to (0,0) = 5; to (3,4) = 2 → cluster 1
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 1).ok());

  EXPECT_EQ(vs.get("label").i32_at(0, 0), 1);
}

// -----------------------------------------------------------------------
// Gaussian mixture
// -----------------------------------------------------------------------

TEST(Clustering, GaussianMixture_Diagonal) {
  auto node = make_clustering_node(
      ClusterKind::GaussianMixture, ClusterDistance::Euclidean,
      ClusterCovType::Diagonal, {},  // centers (unused for GMM)
      {0.5f, 0.5f},                  // weights
      {0.0f, 0.0f, 20.0f, 20.0f},    // means
      {1.0f, 1.0f, 1.0f, 1.0f},      // covariances (diagonal)
      {}, 2, 2);
  node->in_names = {"x"};
  node->out_names = {"label", "prob"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 2);
  x.at(0, 0) = 0.0f;
  x.at(0, 1) = 0.0f;  // near component 0
  x.at(1, 0) = 20.0f;
  x.at(1, 1) = 20.0f;  // near component 1
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  EXPECT_EQ(vs.get("label").i32_at(0, 0), 0);
  EXPECT_EQ(vs.get("label").i32_at(1, 0), 1);

  // Probabilities should sum to 1
  const Tensor& prob = vs.get("prob");
  EXPECT_NEAR(prob.at(0, 0) + prob.at(0, 1), 1.0f, 1e-5f);
}
