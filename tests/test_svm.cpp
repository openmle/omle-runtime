// Tests for SVMNode — the omle.ml/SVM body.
//
// Every expected value here is derived analytically from the model parameters
// rather than captured from a run, so a change in the scoring maths fails the
// test instead of being blessed by it. End-to-end parity against sklearn is
// covered separately by omle-convert's runtime tests; this suite pins the
// kernel maths, the post-transforms, and the output contract.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "runtime_tensor.h"
#include "svm_node.h"

using namespace omle::rt::impl;

namespace {

class SvmTest : public ::testing::Test {
 protected:
  ConstantStore cs;
  ValueStore make_vs() { return ValueStore(cs); }

  static omle::rt::Tensor F32(int rows, int cols, std::vector<float> v) {
    omle::rt::Tensor t =
        omle::rt::Tensor::dense(omle::rt::DataType::Float32, rows, cols);
    float* p = t.f32_ptr();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
  }

  static omle::rt::Tensor F64(int rows, int cols, std::vector<double> v) {
    omle::rt::Tensor t =
        omle::rt::Tensor::dense(omle::rt::DataType::Float64, rows, cols);
    double* p = t.f64_ptr();
    for (std::size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
  }

  // A linear SVM over `n_features` with the given weights, one output.
  static std::unique_ptr<SVMNode> linear(std::vector<float> coeff,
                                         std::vector<float> intercept,
                                         int n_features, int n_outputs,
                                         PostTransform pt,
                                         std::vector<float> prob_a = {},
                                         std::vector<float> prob_b = {}) {
    return make_svm_node(SVMKind::Linear, KernelType::Linear, std::move(coeff),
                         std::move(intercept), {}, {},
                         /*gamma=*/1.0, /*degree=*/3, /*coef0=*/0.0,
                         /*n_sv=*/0, n_features, n_outputs, pt,
                         std::move(prob_a), std::move(prob_b));
  }

  // A kernel SVM with one output.
  static std::unique_ptr<SVMNode> kernel(KernelType kt,
                                         std::vector<float> support_vectors,
                                         std::vector<float> dual,
                                         std::vector<float> intercept,
                                         double gamma, int degree, double coef0,
                                         int n_sv, int n_features) {
    return make_svm_node(SVMKind::Kernel, kt, {}, std::move(intercept),
                         std::move(support_vectors), std::move(dual), gamma,
                         degree, coef0, n_sv, n_features, /*n_outputs=*/1,
                         PostTransform::Identity);
  }

  static float logistic(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};

}  // namespace

// =============================================================================
// Linear SVM — decision values
// =============================================================================

TEST_F(SvmTest, LinearDecisionValue) {
  auto node = linear({2.0f, -1.0f}, {0.5f}, 2, 1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(2, 2, {1.0f, 1.0f, 0.0f, 3.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("score");
  ASSERT_EQ(y.n_rows, 2);
  ASSERT_EQ(y.n_cols, 1);
  EXPECT_NEAR(y.at(0, 0), 2.0f * 1 - 1.0f * 1 + 0.5f, 1e-6f);  // 1.5
  EXPECT_NEAR(y.at(1, 0), 2.0f * 0 - 1.0f * 3 + 0.5f, 1e-6f);  // -2.5
}

TEST_F(SvmTest, LinearWithoutInterceptTreatsBiasAsZero) {
  auto node = linear({1.0f, 2.0f, 3.0f}, {}, 3, 1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 3, {1.0f, 1.0f, 1.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  EXPECT_NEAR(vs.get("score").at(0, 0), 6.0f, 1e-6f);
}

TEST_F(SvmTest, LinearMultiOutputOneColumnPerClass) {
  // 3 one-vs-rest planes over 2 features.
  auto node = linear({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                     2, 3, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {2.0f, 5.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const auto& y = vs.get("score");
  ASSERT_EQ(y.n_cols, 3);
  EXPECT_NEAR(y.at(0, 0), 2.0f, 1e-6f);
  EXPECT_NEAR(y.at(0, 1), 5.0f, 1e-6f);
  EXPECT_NEAR(y.at(0, 2), 7.0f, 1e-6f);
}

TEST_F(SvmTest, MultipleOutputNamesSplitIntoOneColumnEach) {
  // Three output names (not two) is the scatter path, not the classifier path.
  auto node = linear({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                     2, 3, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"a", "b", "c"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {2.0f, 5.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  EXPECT_EQ(vs.get("a").n_cols, 1);
  EXPECT_NEAR(vs.get("a").at(0, 0), 2.0f, 1e-6f);
  EXPECT_NEAR(vs.get("b").at(0, 0), 5.0f, 1e-6f);
  EXPECT_NEAR(vs.get("c").at(0, 0), 7.0f, 1e-6f);
}

// =============================================================================
// Post-transforms
// =============================================================================

TEST_F(SvmTest, SigmoidPostTransform) {
  auto node = linear({1.0f}, {0.0f}, 1, 1, PostTransform::Sigmoid);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {0.0f, 2.0f, -2.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& y = vs.get("score");
  EXPECT_NEAR(y.at(0, 0), 0.5f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), logistic(2.0f), 1e-6f);
  EXPECT_NEAR(y.at(2, 0), logistic(-2.0f), 1e-6f);
}

TEST_F(SvmTest, SoftmaxPostTransformNormalisesEachRow) {
  auto node = linear({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                     2, 3, PostTransform::Softmax);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(2, 2, {2.0f, 5.0f, 0.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("score");
  ASSERT_EQ(y.n_cols, 3);
  for (int r = 0; r < 2; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < 3; ++c) {
      EXPECT_GE(y.at(r, c), 0.0f);
      sum += y.at(r, c);
    }
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }
  // Row 0 raw scores were [2, 5, 7]; softmax preserves the ordering.
  EXPECT_LT(y.at(0, 0), y.at(0, 1));
  EXPECT_LT(y.at(0, 1), y.at(0, 2));
  // Row 1 was all zeros → uniform.
  EXPECT_NEAR(y.at(1, 0), 1.0f / 3.0f, 1e-5f);
}

// =============================================================================
// Kernel SVM — one test per kernel, values worked out by hand
// =============================================================================

TEST_F(SvmTest, LinearKernel) {
  // dual · <x, sv> + b = 2*(1*4 + 0*5) + 3*(0*4 + 1*5) - 1
  auto node = kernel(KernelType::Linear, {1.0f, 0.0f, 0.0f, 1.0f}, {2.0f, 3.0f},
                     {-1.0f}, /*gamma=*/1.0, 3, 0.0, 2, 2);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {4.0f, 5.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  EXPECT_NEAR(vs.get("score").at(0, 0), 22.0f, 1e-5f);
}

TEST_F(SvmTest, RbfKernel) {
  // Support vectors (0,0) and (1,1) are both squared-distance 1 from (1,0).
  auto node = kernel(KernelType::RBF, {0.0f, 0.0f, 1.0f, 1.0f}, {0.5f, -0.25f},
                     {0.1f}, /*gamma=*/0.5, 3, 0.0, 2, 2);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {1.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const float k = std::exp(-0.5f);
  EXPECT_NEAR(vs.get("score").at(0, 0), 0.1f + 0.5f * k - 0.25f * k, 1e-6f);
}

TEST_F(SvmTest, RbfKernelPeaksAtTheSupportVector) {
  // A single support vector: the kernel is maximal at the vector itself and
  // decays monotonically away from it.
  auto node = kernel(KernelType::RBF, {0.0f}, {1.0f}, {0.0f},
                     /*gamma=*/1.0, 3, 0.0, 1, 1);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {0.0f, 1.0f, 3.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& y = vs.get("score");
  EXPECT_NEAR(y.at(0, 0), 1.0f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), std::exp(-1.0f), 1e-6f);
  EXPECT_NEAR(y.at(2, 0), std::exp(-9.0f), 1e-6f);
}

TEST_F(SvmTest, PolyKernel) {
  // (gamma*<x,sv> + coef0)^degree = (0.5*2 + 1)^2 = 4
  auto node = kernel(KernelType::Poly, {1.0f, 0.0f}, {1.0f}, {0.0f},
                     /*gamma=*/0.5, /*degree=*/2, /*coef0=*/1.0, 1, 2);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {2.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  EXPECT_NEAR(vs.get("score").at(0, 0), 4.0f, 1e-5f);
}

TEST_F(SvmTest, SigmoidKernel) {
  // tanh(gamma*<x,sv> + coef0) = tanh(0.5)
  auto node = kernel(KernelType::Sigmoid, {1.0f, 0.0f}, {1.0f}, {0.0f},
                     /*gamma=*/1.0, 3, /*coef0=*/0.0, 1, 2);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {0.5f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  EXPECT_NEAR(vs.get("score").at(0, 0), std::tanh(0.5f), 1e-6f);
}

TEST_F(SvmTest, KernelMultiOutputUsesOneDualRowPerOutput) {
  // 2 outputs × 2 support vectors: dual row 0 = {1,0}, row 1 = {0,1}.
  auto node =
      make_svm_node(SVMKind::Kernel, KernelType::Linear, {}, {0.0f, 10.0f},
                    /*support_vectors=*/{1.0f, 0.0f, 0.0f, 1.0f},
                    /*dual=*/{1.0f, 0.0f, 0.0f, 1.0f}, 1.0, 3, 0.0, /*n_sv=*/2,
                    /*n_features=*/2, /*n_outputs=*/2, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {4.0f, 5.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const auto& y = vs.get("score");
  ASSERT_EQ(y.n_cols, 2);
  EXPECT_NEAR(y.at(0, 0), 4.0f, 1e-5f);          // 1*<x,sv0> + 0 + 0
  EXPECT_NEAR(y.at(0, 1), 5.0f + 10.0f, 1e-5f);  // 1*<x,sv1> + 10
}

// =============================================================================
// Classification output contract: two output names → [y_pred, y_prob]
// =============================================================================

TEST_F(SvmTest, BinaryWithoutPlattThresholdsRawDecisionAtZero) {
  auto node = linear({1.0f}, {0.0f}, 1, 1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {1.5f, -0.5f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 1);
  EXPECT_EQ(pred[1], 0);
  EXPECT_EQ(pred[2], 1);  // exactly zero counts as the positive class

  // Without Platt parameters y_prob carries the raw decision values through.
  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 1);
  EXPECT_NEAR(prob.at(0, 0), 1.5f, 1e-6f);
  EXPECT_NEAR(prob.at(1, 0), -0.5f, 1e-6f);
}

TEST_F(SvmTest, BinaryPlattProducesCalibratedTwoColumnProbabilities) {
  // With A=-1, B=0 the calibration collapses to p(class 1) = logistic(d),
  // which makes the expected values checkable by hand.
  auto node = linear({1.0f}, {0.0f}, 1, 1, PostTransform::Identity,
                     /*prob_a=*/{-1.0f}, /*prob_b=*/{0.0f});
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {2.0f, -2.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 2);
  for (int r = 0; r < 3; ++r)
    EXPECT_NEAR(prob.at(r, 0) + prob.at(r, 1), 1.0f, 1e-5f);

  EXPECT_NEAR(prob.at(0, 1), logistic(2.0f), 1e-5f);
  EXPECT_NEAR(prob.at(1, 1), logistic(-2.0f), 1e-5f);
  EXPECT_NEAR(prob.at(2, 1), 0.5f, 1e-5f);

  // y_pred mirrors sklearn's predict(): the sign of the decision value, not
  // a 0.5 threshold on the calibrated probability.
  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 1);
  EXPECT_EQ(pred[1], 0);
  EXPECT_EQ(pred[2], 1);
}

TEST_F(SvmTest, MulticlassPredictionIsArgmaxOfTheScoreMatrix) {
  auto node = linear({1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
                     2, 3, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  // Row 0 → [2, 5, 7] (argmax 2); row 1 → [9, 1, 10] (argmax 2);
  // row 2 → [4, 1, 5]... use values that move the argmax around.
  vs.put("X", F32(3, 2, {2.0f, 5.0f, 9.0f, 1.0f, -3.0f, 1.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 3);
  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 2);  // [2, 5, 7]
  EXPECT_EQ(pred[1], 2);  // [9, 1, 10]
  EXPECT_EQ(pred[2], 1);  // [-3, 1, -2]
}

TEST_F(SvmTest, MulticlassOvoPlattYieldsANormalisedDistribution) {
  // Three classes, one support vector each, all dual coefficients zero — so
  // each pairwise decision value is exactly its intercept and the pairwise
  // probabilities are logistic(intercept) under A=-1, B=0.
  //
  //   pair (0,1): d =  0.0 → P(0 | 0,1) = 0.5   (a tie)
  //   pair (0,2): d = -3.0 → P(0 | 0,2) ≈ 0.047 (class 2 dominates)
  //   pair (1,2): d = -3.0 → P(1 | 1,2) ≈ 0.047 (class 2 dominates)
  //
  // so class 2 must come out on top. The Wu-Lin-Weng coupling that turns the
  // pairwise probabilities into a distribution is iterative, so this asserts
  // the properties it guarantees rather than exact values.
  auto node = make_svm_node(
      SVMKind::Kernel, KernelType::Linear, {},
      /*intercept=*/{0.0f, -3.0f, -3.0f},
      /*support_vectors=*/{0.0f, 1.0f, 2.0f},
      /*dual=*/std::vector<float>(6, 0.0f), 1.0, 3, 0.0, /*n_sv=*/3,
      /*n_features=*/1, /*n_outputs=*/3, PostTransform::Identity,
      /*prob_a=*/{-1.0f, -1.0f, -1.0f}, /*prob_b=*/{0.0f, 0.0f, 0.0f},
      /*n_support=*/{1, 1, 1});
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F32(1, 1, {0.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 3);

  float sum = 0.0f;
  for (int c = 0; c < 3; ++c) {
    EXPECT_GE(prob.at(0, c), 0.0f);
    EXPECT_LE(prob.at(0, c), 1.0f);
    sum += prob.at(0, c);
  }
  EXPECT_NEAR(sum, 1.0f, 1e-3f);

  EXPECT_GT(prob.at(0, 2), prob.at(0, 0));
  EXPECT_GT(prob.at(0, 2), prob.at(0, 1));
  // Classes 0 and 1 tie against each other and lose identically to class 2.
  EXPECT_NEAR(prob.at(0, 0), prob.at(0, 1), 1e-3f);

  EXPECT_EQ(vs.get("y_pred").i64_ptr()[0], 2);
}

// =============================================================================
// Float64 model path
// =============================================================================

TEST_F(SvmTest, Float64LinearMatchesTheFloat32Maths) {
  auto node = make_svm_node_f64(
      SVMKind::Linear, KernelType::Linear, {2.0, -1.0}, {0.5}, {}, {}, 1.0, 3,
      0.0, 0, /*n_features=*/2, /*n_outputs=*/1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(2, 2, {1.0, 1.0, 0.0, 3.0}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("score");
  ASSERT_EQ(y.dtype, omle::rt::DataType::Float64);
  EXPECT_NEAR(y.f64_at(0, 0), 1.5, 1e-12);
  EXPECT_NEAR(y.f64_at(1, 0), -2.5, 1e-12);
}

TEST_F(SvmTest, Float64KernelKeepsDoublePrecision) {
  // A weight that is not representable in float32: the float64 path must not
  // round it away.
  const double w = 1.0 + 1e-10;
  auto node = make_svm_node_f64(SVMKind::Kernel, KernelType::Linear, {}, {0.0},
                                /*support_vectors=*/{1.0}, /*dual=*/{w}, 1.0, 3,
                                0.0, /*n_sv=*/1, /*n_features=*/1,
                                /*n_outputs=*/1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(1, 1, {1.0}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  EXPECT_NEAR(vs.get("score").f64_at(0, 0), w, 1e-15);
  EXPECT_NE(vs.get("score").f64_at(0, 0), 1.0);
}

TEST_F(SvmTest, Float64InputIsConvertedForAFloat32Model) {
  // gather_slots is told the model's dtype, so a float64 input feeding a
  // float32 model must be converted rather than reinterpreted.
  auto node = linear({2.0f, -1.0f}, {0.5f}, 2, 1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("X", F64(1, 2, {1.0, 1.0}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const auto& y = vs.get("score");
  ASSERT_EQ(y.dtype, omle::rt::DataType::Float32);
  EXPECT_NEAR(y.at(0, 0), 1.5f, 1e-6f);
}

// =============================================================================
// Multiple input slots
// =============================================================================

TEST_F(SvmTest, FeaturesAreGatheredAcrossInputSlots) {
  // Two 1-column inputs must concatenate into the model's 2-feature vector.
  auto node = linear({2.0f, -1.0f}, {0.5f}, 2, 1, PostTransform::Identity);
  node->in_names = {"f0", "f1"};
  node->out_names = {"score"};

  auto vs = make_vs();
  vs.put("f0", F32(2, 1, {1.0f, 0.0f}));
  vs.put("f1", F32(2, 1, {1.0f, 3.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("score");
  EXPECT_NEAR(y.at(0, 0), 1.5f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), -2.5f, 1e-6f);
}

// The classification output contract has a separate float32 and float64
// implementation inside execute(); both need covering or a change to one of
// them slips through.

TEST_F(SvmTest, Float64BinaryPlattProducesCalibratedProbabilities) {
  auto node = make_svm_node_f64(
      SVMKind::Linear, KernelType::Linear, {1.0}, {0.0}, {}, {}, 1.0, 3, 0.0, 0,
      /*n_features=*/1, /*n_outputs=*/1, PostTransform::Identity,
      /*prob_a=*/{-1.0}, /*prob_b=*/{0.0});
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F64(3, 1, {2.0, -2.0, 0.0}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.dtype, omle::rt::DataType::Float64);
  ASSERT_EQ(prob.n_cols, 2);

  const auto logistic64 = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  for (int r = 0; r < 3; ++r)
    EXPECT_NEAR(prob.f64_at(r, 0) + prob.f64_at(r, 1), 1.0, 1e-12);
  EXPECT_NEAR(prob.f64_at(0, 1), logistic64(2.0), 1e-12);
  EXPECT_NEAR(prob.f64_at(1, 1), logistic64(-2.0), 1e-12);
  EXPECT_NEAR(prob.f64_at(2, 1), 0.5, 1e-12);

  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 1);
  EXPECT_EQ(pred[1], 0);
  EXPECT_EQ(pred[2], 1);
}

TEST_F(SvmTest, Float64BinaryWithoutPlattThresholdsRawDecisionAtZero) {
  auto node =
      make_svm_node_f64(SVMKind::Linear, KernelType::Linear, {1.0}, {0.0}, {},
                        {}, 1.0, 3, 0.0, 0, 1, 1, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F64(3, 1, {1.5, -0.5, 0.0}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 1);
  EXPECT_EQ(pred[1], 0);
  EXPECT_EQ(pred[2], 1);

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 1);
  EXPECT_NEAR(prob.f64_at(0, 0), 1.5, 1e-12);
  EXPECT_NEAR(prob.f64_at(1, 0), -0.5, 1e-12);
}

TEST_F(SvmTest, Float64MulticlassPredictionIsArgmax) {
  auto node = make_svm_node_f64(SVMKind::Linear, KernelType::Linear,
                                {1.0, 0.0, 0.0, 1.0, 1.0, 1.0}, {0.0, 0.0, 0.0},
                                {}, {}, 1.0, 3, 0.0, 0, /*n_features=*/2,
                                /*n_outputs=*/3, PostTransform::Identity);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F64(3, 2, {2.0, 5.0, 9.0, 1.0, -3.0, 1.0}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  ASSERT_EQ(vs.get("y_prob").n_cols, 3);
  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 2);  // [2, 5, 7]
  EXPECT_EQ(pred[1], 2);  // [9, 1, 10]
  EXPECT_EQ(pred[2], 1);  // [-3, 1, -2]
}
