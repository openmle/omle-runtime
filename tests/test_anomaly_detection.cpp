#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "nodes/anomaly_detection_node.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

namespace {

class AnomalyTest : public ::testing::Test {
 protected:
  ConstantStore cs;
  ValueStore make_vs() { return ValueStore(cs); }

  Tensor F64(int rows, int cols, std::vector<double> v) {
    Tensor t = Tensor::dense(omle::rt::DataType::Float64, rows, cols);
    double* p = t.f64_ptr();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
  }

  // Run a node over one input tensor and return the named outputs.
  std::vector<double> run(AnomalyDetectionNode& node, ValueStore& vs,
                          int n_rows, const std::string& out) {
    EXPECT_TRUE(node.execute(vs, n_rows).ok());
    const Tensor& t = vs.get(out);
    return std::vector<double>(t.f64_ptr(), t.f64_ptr() + n_rows);
  }
};

}  // namespace

// The isolation-forest and elliptic-envelope paths are additionally covered
// end-to-end by the embedded verification cases that omle-convert writes into
// converted sklearn models, which compare against sklearn's score_samples.

TEST_F(AnomalyTest, LinearOneClassSVMScore) {
  auto node = make_linear_one_class_svm_node({1.0, 2.0}, /*intercept=*/0.5,
                                             /*offset=*/0.0, /*n_features=*/2,
                                             {AnomalyOutput::Score});
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(2, 2, {1.0, 1.0, 2.0, -1.0}));

  auto got = run(*node, vs, 2, "score");
  EXPECT_NEAR(got[0], 1.0 * 1.0 + 2.0 * 1.0 + 0.5, 1e-12);
  EXPECT_NEAR(got[1], 1.0 * 2.0 + 2.0 * -1.0 + 0.5, 1e-12);
}

TEST_F(AnomalyTest, OneClassSVMRbfScore) {
  // Two support vectors at (0,0) and (1,1); both are distance^2 = 1 from (1,0).
  auto node = make_one_class_svm_node(
      AnomalyKernel::RBF, /*support_vectors=*/{0.0, 0.0, 1.0, 1.0},
      /*dual=*/{0.5, -0.25}, /*intercept=*/0.1, /*gamma=*/0.5, /*degree=*/3,
      /*coef0=*/0.0, /*n_sv=*/2, /*n_features=*/2, /*offset=*/0.0,
      {AnomalyOutput::Score});
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(1, 2, {1.0, 0.0}));

  const double k = std::exp(-0.5 * 1.0);
  auto got = run(*node, vs, 1, "score");
  EXPECT_NEAR(got[0], 0.1 + 0.5 * k + (-0.25) * k, 1e-12);
}

TEST_F(AnomalyTest, OneClassSVMLinearKernelScore) {
  auto node = make_one_class_svm_node(AnomalyKernel::Linear,
                                      {1.0, 0.0, 0.0, 1.0}, {2.0, 3.0},
                                      /*intercept=*/-1.0, /*gamma=*/1.0, 3, 0.0,
                                      2, 2, 0.0, {AnomalyOutput::Score});
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(1, 2, {4.0, 5.0}));

  // 2*(1*4 + 0*5) + 3*(0*4 + 1*5) - 1 = 8 + 15 - 1
  auto got = run(*node, vs, 1, "score");
  EXPECT_NEAR(got[0], 22.0, 1e-12);
}

TEST_F(AnomalyTest, EllipticEnvelopeIsNegativeMahalanobis) {
  // Identity precision → squared Euclidean distance from the location.
  auto node = make_elliptic_envelope_node(/*location=*/{1.0, 2.0},
                                          /*precision=*/{1.0, 0.0, 0.0, 1.0},
                                          /*n_features=*/2, /*offset=*/0.0,
                                          {AnomalyOutput::Score});
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(2, 2, {1.0, 2.0, 4.0, 6.0}));

  auto got = run(*node, vs, 2, "score");
  EXPECT_NEAR(got[0], 0.0, 1e-12);            // exactly at the centre
  EXPECT_NEAR(got[1], -(9.0 + 16.0), 1e-12);  // -(3^2 + 4^2)
}

TEST_F(AnomalyTest, DecisionValueAndPredictionUseOffset) {
  // score = x (1 feature, unit coefficient, no intercept); offset splits at 2.
  auto node = make_linear_one_class_svm_node(
      {1.0}, /*intercept=*/0.0, /*offset=*/2.0, /*n_features=*/1,
      {AnomalyOutput::Score, AnomalyOutput::DecisionValue,
       AnomalyOutput::Prediction});
  node->in_names = {"X"};
  node->out_names = {"score", "decision_value", "prediction"};

  auto vs = make_vs();
  vs.put("X", F64(3, 1, {5.0, 2.0, 0.5}));

  auto score = run(*node, vs, 3, "score");
  const Tensor& dec = vs.get("decision_value");
  const Tensor& pred = vs.get("prediction");

  EXPECT_NEAR(score[0], 5.0, 1e-12);
  EXPECT_NEAR(dec.f64_ptr()[0], 3.0, 1e-12);   // 5 - 2
  EXPECT_NEAR(dec.f64_ptr()[2], -1.5, 1e-12);  // 0.5 - 2
  EXPECT_DOUBLE_EQ(pred.f64_ptr()[0], 1.0);    // inlier
  EXPECT_DOUBLE_EQ(pred.f64_ptr()[1], 1.0);    // exactly at the offset
  EXPECT_DOUBLE_EQ(pred.f64_ptr()[2], -1.0);   // outlier
}

TEST_F(AnomalyTest, LocalOutlierFactorFlagsAnOutlier) {
  // A tight cluster near the origin plus one far point. The far point must get
  // a markedly lower (more abnormal) score than a point inside the cluster.
  std::vector<double> ref = {0.0, 0.0, 0.1,  0.0, 0.0, 0.1,
                             0.1, 0.1, -0.1, 0.0, 0.0, -0.1};
  auto node = make_local_outlier_factor_node(ref, /*n_reference=*/6,
                                             /*n_features=*/2,
                                             /*n_neighbors=*/3, "euclidean",
                                             /*p=*/2.0, /*offset=*/0.0,
                                             {AnomalyOutput::Score});
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(2, 2, {0.02, 0.02, 10.0, 10.0}));

  auto got = run(*node, vs, 2, "score");
  EXPECT_LT(got[1], got[0]);  // far point is more abnormal
  EXPECT_LT(got[1], -1.5);    // and clearly outlying, not marginal
  EXPECT_GT(got[0], -1.5);    // cluster member scores near -1
}

TEST_F(AnomalyTest, FeatureCountMismatchIsAnError) {
  auto node = make_linear_one_class_svm_node({1.0, 2.0}, 0.0, 0.0, 2,
                                             {AnomalyOutput::Score});
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(1, 3, {1.0, 2.0, 3.0}));
  EXPECT_FALSE(node->execute(vs, 1).ok());
}
