// Tests for NeuralNetworkNode — the omle.ml/NeuralNetwork body.
//
// Weights are chosen so that each forward pass can be evaluated by hand, which
// keeps the assertions independent of the implementation. Coverage here is
// deliberately biased toward the classification path: omle-convert's runtime
// parity tests only exercise MLP regression, so softmax outputs, argmax
// predictions and the binary probability expansion are pinned here.

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "neural_network_node.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

namespace {

class NnTest : public ::testing::Test {
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

  // weights are [out_features, in_features] row-major.
  static NNLayerDataF32 layer(int in_features, int out_features,
                              std::vector<float> weights,
                              std::vector<float> bias, NNActivation act) {
    NNLayerDataF32 l;
    l.weights = std::move(weights);
    l.bias = std::move(bias);
    l.activation = act;
    l.in_features = in_features;
    l.out_features = out_features;
    return l;
  }

  static NNLayerDataF64 layer64(int in_features, int out_features,
                                std::vector<double> weights,
                                std::vector<double> bias, NNActivation act) {
    NNLayerDataF64 l;
    l.weights = std::move(weights);
    l.bias = std::move(bias);
    l.activation = act;
    l.in_features = in_features;
    l.out_features = out_features;
    return l;
  }

  static float logistic(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};

}  // namespace

// =============================================================================
// Forward pass
// =============================================================================

TEST_F(NnTest, SingleLayerIdentity) {
  auto node = make_neural_network_node(
      {layer(2, 1, {2.0f, 3.0f}, {1.0f}, NNActivation::Identity)},
      /*n_features=*/2, /*n_outputs=*/1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(2, 2, {1.0f, 1.0f, 2.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_rows, 2);
  ASSERT_EQ(y.n_cols, 1);
  EXPECT_NEAR(y.at(0, 0), 2.0f + 3.0f + 1.0f, 1e-6f);  // 6
  EXPECT_NEAR(y.at(1, 0), 4.0f + 0.0f + 1.0f, 1e-6f);  // 5
}

TEST_F(NnTest, EmptyBiasIsTreatedAsZero) {
  auto node = make_neural_network_node(
      {layer(2, 1, {2.0f, 3.0f}, {}, NNActivation::Identity)}, 2, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {1.0f, 1.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  EXPECT_NEAR(vs.get("y").at(0, 0), 5.0f, 1e-6f);
}

TEST_F(NnTest, HiddenReluClampsNegativePreActivations) {
  // h = relu([1*3 + 0*4, 0*3 + (-1)*4]) = relu([3, -4]) = [3, 0]
  // y = 1*3 + 1*0 = 3
  auto node = make_neural_network_node(
      {layer(2, 2, {1.0f, 0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}, NNActivation::Relu),
       layer(2, 1, {1.0f, 1.0f}, {0.0f}, NNActivation::Identity)},
      2, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(2, 2, {3.0f, 4.0f, 3.0f, -4.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 3.0f, 1e-6f);
  // Second row: relu([3, 4]) = [3, 4] → 7. Without the clamp both rows would
  // have produced the same value.
  EXPECT_NEAR(y.at(1, 0), 7.0f, 1e-6f);
}

TEST_F(NnTest, LogisticActivation) {
  auto node = make_neural_network_node(
      {layer(1, 1, {1.0f}, {0.0f}, NNActivation::Logistic)}, 1, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {0.0f, 2.0f, -2.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 0.5f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), logistic(2.0f), 1e-6f);
  EXPECT_NEAR(y.at(2, 0), logistic(-2.0f), 1e-6f);
}

TEST_F(NnTest, TanhActivation) {
  auto node = make_neural_network_node(
      {layer(1, 1, {1.0f}, {0.0f}, NNActivation::Tanh)}, 1, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {0.0f, 1.0f, -1.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 0.0f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), std::tanh(1.0f), 1e-6f);
  EXPECT_NEAR(y.at(2, 0), -std::tanh(1.0f), 1e-6f);
}

TEST_F(NnTest, SoftmaxOutputLayer) {
  // pre-activations [1, 2, 3] → softmax = [e^-2, e^-1, 1] / (e^-2 + e^-1 + 1)
  auto node = make_neural_network_node(
      {layer(1, 3, {1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f},
             NNActivation::Softmax)},
      /*n_features=*/1, /*n_outputs=*/3);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(2, 1, {1.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 3);

  const float denom = std::exp(-2.0f) + std::exp(-1.0f) + 1.0f;
  EXPECT_NEAR(y.at(0, 0), std::exp(-2.0f) / denom, 1e-6f);
  EXPECT_NEAR(y.at(0, 1), std::exp(-1.0f) / denom, 1e-6f);
  EXPECT_NEAR(y.at(0, 2), 1.0f / denom, 1e-6f);

  // Row 1 has all-zero pre-activations → uniform.
  for (int c = 0; c < 3; ++c) EXPECT_NEAR(y.at(1, c), 1.0f / 3.0f, 1e-6f);

  for (int r = 0; r < 2; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < 3; ++c) sum += y.at(r, c);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }
}

TEST_F(NnTest, ActivationsAreIndependentPerLayer) {
  // Logistic hidden layer feeding an identity output layer: the output must
  // not be squashed a second time.
  auto node = make_neural_network_node(
      {layer(1, 1, {0.0f}, {0.0f}, NNActivation::Logistic),
       layer(1, 1, {10.0f}, {0.0f}, NNActivation::Identity)},
      1, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(1, 1, {123.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  // hidden = logistic(0) = 0.5; output = 10 * 0.5 = 5, well outside [0,1].
  EXPECT_NEAR(vs.get("y").at(0, 0), 5.0f, 1e-6f);
}

TEST_F(NnTest, DeepNetworkChainsLayerBuffers) {
  // 2 → 3 → 2 → 1, so the ping-pong buffers change width in both directions.
  auto node = make_neural_network_node(
      {layer(2, 3, {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
             NNActivation::Identity),
       layer(3, 2, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}, {0.0f, 0.0f},
             NNActivation::Identity),
       layer(2, 1, {1.0f, 1.0f}, {0.0f}, NNActivation::Identity)},
      /*n_features=*/2, /*n_outputs=*/1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(1, 2, {1.0f, 2.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  // [1,2] → [1, 2, 3] → [1, 3] → 4
  EXPECT_NEAR(vs.get("y").at(0, 0), 4.0f, 1e-6f);
}

TEST_F(NnTest, HiddenLayerWiderThanInputAndOutput) {
  // Buffer sizing is driven by the widest layer, not by n_features/n_outputs.
  auto node = make_neural_network_node(
      {layer(1, 5, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
             {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, NNActivation::Identity),
       layer(5, 1, {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, {0.0f},
             NNActivation::Identity)},
      /*n_features=*/1, /*n_outputs=*/1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(1, 1, {2.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());
  // hidden = [2,4,6,8,10] → sum = 30
  EXPECT_NEAR(vs.get("y").at(0, 0), 30.0f, 1e-6f);
}

TEST_F(NnTest, RowsAreScoredIndependently) {
  // A stateful bug in the ping-pong buffers would leak row 0 into row 1.
  auto node = make_neural_network_node(
      {layer(1, 2, {1.0f, -1.0f}, {0.0f, 0.0f}, NNActivation::Relu),
       layer(2, 1, {1.0f, 1.0f}, {0.0f}, NNActivation::Identity)},
      1, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F32(4, 1, {5.0f, -5.0f, 5.0f, -5.0f}));
  ASSERT_TRUE(node->execute(vs, 4).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 5.0f, 1e-6f);  // relu([5, -5]) = [5, 0]
  EXPECT_NEAR(y.at(1, 0), 5.0f, 1e-6f);  // relu([-5, 5]) = [0, 5]
  EXPECT_NEAR(y.at(2, 0), 5.0f, 1e-6f);
  EXPECT_NEAR(y.at(3, 0), 5.0f, 1e-6f);
}

// =============================================================================
// Output contract
// =============================================================================

TEST_F(NnTest, BinaryClassificationExpandsToTwoProbabilityColumns) {
  auto node = make_neural_network_node(
      {layer(1, 1, {1.0f}, {0.0f}, NNActivation::Logistic)},
      /*n_features=*/1, /*n_outputs=*/1);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {2.0f, -2.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 2);
  for (int r = 0; r < 3; ++r)
    EXPECT_NEAR(prob.at(r, 0) + prob.at(r, 1), 1.0f, 1e-5f);

  EXPECT_NEAR(prob.at(0, 1), logistic(2.0f), 1e-6f);
  EXPECT_NEAR(prob.at(1, 1), logistic(-2.0f), 1e-6f);
  EXPECT_NEAR(prob.at(2, 1), 0.5f, 1e-6f);

  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 1);
  EXPECT_EQ(pred[1], 0);
  EXPECT_EQ(pred[2], 1);  // exactly 0.5 counts as the positive class
}

TEST_F(NnTest, MulticlassClassificationPredictsArgmax) {
  // Three output units with distinct weights, so the argmax moves with x.
  auto node = make_neural_network_node(
      {layer(1, 3, {1.0f, 0.0f, -1.0f}, {0.0f, 0.5f, 0.0f},
             NNActivation::Softmax)},
      /*n_features=*/1, /*n_outputs=*/3);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F32(3, 1, {5.0f, 0.0f, -5.0f}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.n_cols, 3);
  for (int r = 0; r < 3; ++r) {
    float sum = 0.0f;
    for (int c = 0; c < 3; ++c) sum += prob.at(r, c);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
  }

  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 0);  // pre = [5, 0.5, -5]
  EXPECT_EQ(pred[1], 1);  // pre = [0, 0.5, 0]
  EXPECT_EQ(pred[2], 2);  // pre = [-5, 0.5, 5]
}

TEST_F(NnTest, MultipleOutputNamesSplitIntoOneColumnEach) {
  auto node = make_neural_network_node(
      {layer(1, 3, {1.0f, 2.0f, 3.0f}, {0.0f, 0.0f, 0.0f},
             NNActivation::Identity)},
      1, 3);
  node->in_names = {"X"};
  node->out_names = {"a", "b", "c"};

  auto vs = make_vs();
  vs.put("X", F32(1, 1, {2.0f}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  EXPECT_EQ(vs.get("a").n_cols, 1);
  EXPECT_NEAR(vs.get("a").at(0, 0), 2.0f, 1e-6f);
  EXPECT_NEAR(vs.get("b").at(0, 0), 4.0f, 1e-6f);
  EXPECT_NEAR(vs.get("c").at(0, 0), 6.0f, 1e-6f);
}

TEST_F(NnTest, FeaturesAreGatheredAcrossInputSlots) {
  auto node = make_neural_network_node(
      {layer(2, 1, {2.0f, 3.0f}, {1.0f}, NNActivation::Identity)}, 2, 1);
  node->in_names = {"f0", "f1"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("f0", F32(2, 1, {1.0f, 2.0f}));
  vs.put("f1", F32(2, 1, {1.0f, 0.0f}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 6.0f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), 5.0f, 1e-6f);
}

// =============================================================================
// Float64 model path
// =============================================================================

TEST_F(NnTest, Float64ForwardPass) {
  auto node = make_neural_network_node_f64(
      {layer64(2, 2, {1.0, 0.0, 0.0, -1.0}, {0.0, 0.0}, NNActivation::Relu),
       layer64(2, 1, {1.0, 1.0}, {0.0}, NNActivation::Identity)},
      2, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F64(1, 2, {3.0, 4.0}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const auto& y = vs.get("y");
  ASSERT_EQ(y.dtype, omle::rt::DataType::Float64);
  EXPECT_NEAR(y.f64_at(0, 0), 3.0, 1e-12);
}

TEST_F(NnTest, Float64KeepsPrecisionAFloat32ModelWouldLose) {
  const double w = 1.0 + 1e-10;
  auto node = make_neural_network_node_f64(
      {layer64(1, 1, {w}, {0.0}, NNActivation::Identity)}, 1, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F64(2, 1, {1.0, 2.0}));
  ASSERT_TRUE(node->execute(vs, 2).ok());

  EXPECT_NEAR(vs.get("y").f64_at(0, 0), w, 1e-15);
  EXPECT_NE(vs.get("y").f64_at(0, 0), 1.0);
}

TEST_F(NnTest, Float64InputIsConvertedForAFloat32Model) {
  auto node = make_neural_network_node(
      {layer(2, 1, {2.0f, 3.0f}, {1.0f}, NNActivation::Identity)}, 2, 1);
  node->in_names = {"X"};
  node->out_names = {"y"};

  auto vs = make_vs();
  vs.put("X", F64(1, 2, {1.0, 1.0}));
  ASSERT_TRUE(node->execute(vs, 1).ok());

  const auto& y = vs.get("y");
  ASSERT_EQ(y.dtype, omle::rt::DataType::Float32);
  EXPECT_NEAR(y.at(0, 0), 6.0f, 1e-6f);
}

// As with SVMNode, execute() implements the classification contract twice —
// once for float32 and once for float64.

TEST_F(NnTest, Float64BinaryClassificationExpandsToTwoColumns) {
  auto node = make_neural_network_node_f64(
      {layer64(1, 1, {1.0}, {0.0}, NNActivation::Logistic)}, 1, 1);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F64(3, 1, {2.0, -2.0, 0.0}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  const auto& prob = vs.get("y_prob");
  ASSERT_EQ(prob.dtype, omle::rt::DataType::Float64);
  ASSERT_EQ(prob.n_cols, 2);

  const auto logistic64 = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
  EXPECT_NEAR(prob.f64_at(0, 1), logistic64(2.0), 1e-12);
  EXPECT_NEAR(prob.f64_at(1, 1), logistic64(-2.0), 1e-12);
  EXPECT_NEAR(prob.f64_at(2, 1), 0.5, 1e-12);
  for (int r = 0; r < 3; ++r)
    EXPECT_NEAR(prob.f64_at(r, 0) + prob.f64_at(r, 1), 1.0, 1e-12);

  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 1);
  EXPECT_EQ(pred[1], 0);
  EXPECT_EQ(pred[2], 1);
}

TEST_F(NnTest, Float64MulticlassPredictionIsArgmax) {
  auto node = make_neural_network_node_f64(
      {layer64(1, 3, {1.0, 0.0, -1.0}, {0.0, 0.5, 0.0}, NNActivation::Softmax)},
      1, 3);
  node->in_names = {"X"};
  node->out_names = {"y_pred", "y_prob"};

  auto vs = make_vs();
  vs.put("X", F64(3, 1, {5.0, 0.0, -5.0}));
  ASSERT_TRUE(node->execute(vs, 3).ok());

  ASSERT_EQ(vs.get("y_prob").n_cols, 3);
  const int64_t* pred = vs.get("y_pred").i64_ptr();
  EXPECT_EQ(pred[0], 0);
  EXPECT_EQ(pred[1], 1);
  EXPECT_EQ(pred[2], 2);
}
