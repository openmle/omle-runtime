#include <gtest/gtest.h>

#include "linear_node.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

// -----------------------------------------------------------------------
// Typed fixture — instantiated for both float and double.
// -----------------------------------------------------------------------
template <typename T>
class LinearTest : public ::testing::Test {
 protected:
  static std::unique_ptr<LinearExecutor> make_executor(std::vector<T> coeff,
                                                       std::vector<T> intercept,
                                                       PostTransform pt,
                                                       int n_features,
                                                       int n_outputs) {
    if constexpr (std::is_same_v<T, float>)
      return make_linear_executor(std::move(coeff), std::move(intercept), pt,
                                  n_features, n_outputs);
    else
      return make_linear_executor_f64(std::move(coeff), std::move(intercept),
                                      pt, n_features, n_outputs);
  }

  static std::unique_ptr<LinearNode> make_node(std::vector<T> coeff,
                                               std::vector<T> intercept,
                                               PostTransform pt, int n_features,
                                               int n_outputs) {
    if constexpr (std::is_same_v<T, float>)
      return make_linear_node(std::move(coeff), std::move(intercept), pt,
                              n_features, n_outputs);
    else
      return make_linear_node_f64(std::move(coeff), std::move(intercept), pt,
                                  n_features, n_outputs);
  }

  static T output_at(const omle::rt::Tensor& t, int r, int c) {
    if constexpr (std::is_same_v<T, float>)
      return t.at(r, c);
    else
      return t.f64_at(r, c);
  }
};

using Precisions = ::testing::Types<float, double>;
TYPED_TEST_SUITE(LinearTest, Precisions);

// -----------------------------------------------------------------------
// Executor tests
// -----------------------------------------------------------------------

TYPED_TEST(LinearTest, SingleOutputNoIntercept) {
  auto exec = this->make_executor({TypeParam(1), TypeParam(2), TypeParam(3)},
                                  {}, PostTransform::Identity, 3, 1);

  TypeParam input[] = {TypeParam(1), TypeParam(0), TypeParam(0),
                       TypeParam(0), TypeParam(1), TypeParam(0),
                       TypeParam(1), TypeParam(1), TypeParam(1)};
  TypeParam output[3] = {};
  exec->predict(input, 3, output);

  EXPECT_EQ(output[0], TypeParam(1));
  EXPECT_EQ(output[1], TypeParam(2));
  EXPECT_EQ(output[2], TypeParam(6));
}

TYPED_TEST(LinearTest, SingleOutputWithIntercept) {
  auto exec = this->make_executor({TypeParam(2), TypeParam(3)}, {TypeParam(10)},
                                  PostTransform::Identity, 2, 1);

  TypeParam input[] = {TypeParam(1), TypeParam(1)};
  TypeParam output[1] = {};
  exec->predict(input, 1, output);

  EXPECT_EQ(output[0], TypeParam(15));  // 2+3+10
}

TYPED_TEST(LinearTest, MultiOutput) {
  auto exec = this->make_executor({TypeParam(1), TypeParam(0), TypeParam(0),
                                   TypeParam(1), TypeParam(1), TypeParam(1)},
                                  {TypeParam(0), TypeParam(0.5), TypeParam(0)},
                                  PostTransform::Identity, 2, 3);

  TypeParam input[] = {TypeParam(2), TypeParam(3)};
  TypeParam output[3] = {};
  exec->predict(input, 1, output);

  EXPECT_EQ(output[0], TypeParam(2));    // 1*2 + 0*3
  EXPECT_EQ(output[1], TypeParam(3.5));  // 0*2 + 1*3 + 0.5
  EXPECT_EQ(output[2], TypeParam(5));    // 1*2 + 1*3
}

TYPED_TEST(LinearTest, SigmoidPostTransform) {
  auto exec =
      this->make_executor({TypeParam(0)}, {}, PostTransform::Sigmoid, 1, 1);

  TypeParam input[] = {TypeParam(99)};
  TypeParam output[1] = {};
  exec->predict(input, 1, output);

  EXPECT_NEAR(output[0], TypeParam(0.5), TypeParam(1e-5));
}

// -----------------------------------------------------------------------
// DAG node test
// -----------------------------------------------------------------------

TYPED_TEST(LinearTest, ExecuteViaValueStore) {
  auto node = this->make_node({TypeParam(3), TypeParam(4)}, {},
                              PostTransform::Identity, 2, 1);
  node->in_names = {"x"};
  node->out_names = {"y"};

  ConstantStore cs;
  ValueStore vs(cs);
  Tensor x(2, 2);
  x.at(0, 0) = 1.0f;
  x.at(0, 1) = 0.0f;
  x.at(1, 0) = 0.0f;
  x.at(1, 1) = 1.0f;
  vs.put("x", x);

  ASSERT_TRUE(node->execute(vs, 2).ok());

  const omle::rt::Tensor& y = vs.get("y");
  EXPECT_EQ(this->output_at(y, 0, 0), TypeParam(3));
  EXPECT_EQ(this->output_at(y, 1, 0), TypeParam(4));
}
