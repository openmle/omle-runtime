#include <gtest/gtest.h>

#include "runtime_tensor.h"
#include "tree_ensemble_node.h"

using namespace omle::rt::impl;

// -----------------------------------------------------------------------
// Typed fixture — instantiated for both float and double.
// -----------------------------------------------------------------------
template <typename T>
class TreeEnsembleTest : public ::testing::Test {
 protected:
  static FlatTree<T> make_stump(T threshold, T left_val, T right_val) {
    FlatTree<T> t;
    t.n_nodes = 3;
    t.leaf_width = 1;
    t.feature = {0, -1, -1};
    t.threshold = {threshold, T(0), T(0)};
    t.split_op = {SplitOp::LessThan, SplitOp::LessThan, SplitOp::LessThan};
    t.left_child = {1, -1, -1};
    t.right_child = {2, -1, -1};
    t.default_child = {2, -1, -1};
    t.leaf_value = {T(0), left_val, right_val};
    t.cat_offset = {0, 0, 0};
    t.cat_count = {0, 0, 0};
    t.all_less_than = true;
    t.default_right = true;
    return t;
  }

  static TreeEnsembleModel<T> make_model(
      std::vector<FlatTree<T>> trees, int n_features, int n_outputs,
      Aggregation agg = Aggregation::Sum,
      PostTransform pt = PostTransform::Identity, T base_score = T(0),
      bool has_base = false) {
    TreeEnsembleModel<T> m;
    m.trees = std::move(trees);
    m.n_trees = static_cast<int>(m.trees.size());
    m.n_features = n_features;
    m.n_outputs = n_outputs;
    m.aggregation = agg;
    m.post_transform = pt;
    m.base_score = base_score;
    m.has_base_score = has_base;
    return m;
  }

  static TreeEnsembleExecutor make_executor(const TreeEnsembleModel<T>& m) {
    if constexpr (std::is_same_v<T, float>)
      return TreeEnsembleExecutor(make_tree_ensemble_impl(m), m.n_features,
                                  m.n_outputs);
    else
      return TreeEnsembleExecutor(make_tree_ensemble_impl_f64(m), m.n_features,
                                  m.n_outputs);
  }

  static std::unique_ptr<TreeEnsembleNode> make_node(
      const TreeEnsembleModel<T>& m) {
    if constexpr (std::is_same_v<T, float>)
      return std::make_unique<TreeEnsembleNode>(make_tree_ensemble_impl(m));
    else
      return std::make_unique<TreeEnsembleNode>(make_tree_ensemble_impl_f64(m));
  }

  static T pred_at(const omle::rt::Tensor& t, int r, int c) {
    if constexpr (std::is_same_v<T, float>)
      return t.at(r, c);
    else
      return t.f64_at(r, c);
  }
};

using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TreeEnsembleTest, Precisions);

// -----------------------------------------------------------------------
// Predict tests
// -----------------------------------------------------------------------

TYPED_TEST(TreeEnsembleTest, SingleStumpRegression) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(5), TypeParam(1), TypeParam(-1))}, 1, 1);
  auto exec = this->make_executor(m);

  TypeParam features[] = {TypeParam(3), TypeParam(7)};
  TypeParam output[2] = {};
  exec.predict(features, 2, output);

  EXPECT_EQ(output[0], TypeParam(1));   // 3 < 5 → left
  EXPECT_EQ(output[1], TypeParam(-1));  // 7 >= 5 → right
}

TYPED_TEST(TreeEnsembleTest, TwoStumpsSum) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(5), TypeParam(1), TypeParam(-1)),
       this->make_stump(TypeParam(5), TypeParam(0.5), TypeParam(0.5))},
      1, 1);
  auto exec = this->make_executor(m);

  TypeParam x = TypeParam(3);
  TypeParam output = TypeParam(0);
  exec.predict(&x, 1, &output);

  EXPECT_EQ(output, TypeParam(1.5));  // 1.0 + 0.5
}

TYPED_TEST(TreeEnsembleTest, NaNRoutesToDefault) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(5), TypeParam(1), TypeParam(-1))}, 1, 1);
  auto exec = this->make_executor(m);

  TypeParam nan_val = std::numeric_limits<TypeParam>::quiet_NaN();
  TypeParam output = TypeParam(0);
  exec.predict(&nan_val, 1, &output);

  EXPECT_EQ(output, TypeParam(-1));  // NaN → default_child = right
}

TYPED_TEST(TreeEnsembleTest, AverageAggregation) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(5), TypeParam(4), TypeParam(0)),
       this->make_stump(TypeParam(5), TypeParam(8), TypeParam(0))},
      1, 1, Aggregation::Average);
  auto exec = this->make_executor(m);

  TypeParam x = TypeParam(1);
  TypeParam output = TypeParam(0);
  exec.predict(&x, 1, &output);

  EXPECT_EQ(output, TypeParam(6));  // (4+8)/2
}

TYPED_TEST(TreeEnsembleTest, BaseScore) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(5), TypeParam(0.1), TypeParam(0))}, 1, 1,
      Aggregation::Sum, PostTransform::Identity, TypeParam(0.5), true);
  auto exec = this->make_executor(m);

  TypeParam x = TypeParam(1);
  TypeParam output = TypeParam(0);
  exec.predict(&x, 1, &output);

  EXPECT_NEAR(output, TypeParam(0.6), TypeParam(1e-6));  // 0.5 + 0.1
}

TYPED_TEST(TreeEnsembleTest, SigmoidPostTransform) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(5), TypeParam(0), TypeParam(0))}, 1, 1,
      Aggregation::Sum, PostTransform::Sigmoid);
  auto exec = this->make_executor(m);

  TypeParam x = TypeParam(1);
  TypeParam output = TypeParam(0);
  exec.predict(&x, 1, &output);

  EXPECT_NEAR(output, TypeParam(0.5), TypeParam(1e-5));  // sigmoid(0) = 0.5
}

// -----------------------------------------------------------------------
// DAG node test
// -----------------------------------------------------------------------

TYPED_TEST(TreeEnsembleTest, ExecuteViaValueStore) {
  auto m = this->make_model(
      {this->make_stump(TypeParam(0), TypeParam(10), TypeParam(-10))}, 1, 1);
  auto node = this->make_node(m);
  node->in_names = {"x"};
  node->out_names = {"pred"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 1);
  x.at(0, 0) = -1.0f;  // < 0 → left
  x.at(1, 0) = 1.0f;   // >= 0 → right
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  const omle::rt::Tensor& pred = vs.get("pred");
  EXPECT_EQ(this->pred_at(pred, 0, 0), TypeParam(10));
  EXPECT_EQ(this->pred_at(pred, 1, 0), TypeParam(-10));
}
