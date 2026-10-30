#include <gtest/gtest.h>

#include <cmath>

#include "operator_node.h"
#include "operator_registry.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

// Ensure operators are registered before any test runs.
class OperatorTest : public ::testing::Test {
 protected:
  static constexpr float inf = std::numeric_limits<float>::infinity();
  void SetUp() override { register_builtin_operators(); }

  ConstantStore cs;

  ValueStore make_vs() { return ValueStore(cs); }

  Tensor make_tensor(int rows, int cols, std::vector<float> data) {
    Tensor t(rows, cols);
    t.set_floats(std::move(data));
    return t;
  }
};

// -----------------------------------------------------------------------
// omle.core
// -----------------------------------------------------------------------

TEST_F(OperatorTest, Identity) {
  auto vs = make_vs();
  vs.put("x", make_tensor(2, 2, {1, 2, 3, 4}));

  OperatorNode node;
  node.domain = "omle.core";
  node.op = "Identity";
  node.in_names = {"x"};
  node.out_names = {"y"};
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 3.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 4.0f);
}

TEST_F(OperatorTest, Concat) {
  auto vs = make_vs();
  vs.put("a", make_tensor(2, 2, {1, 2, 3, 4}));
  vs.put("b", make_tensor(2, 1, {5, 6}));

  OperatorNode node;
  node.domain = "omle.core";
  node.op = "Concat";
  node.in_names = {"a", "b"};
  node.out_names = {"out"};
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& out = vs.get("out");
  ASSERT_EQ(out.n_cols, 3);
  EXPECT_FLOAT_EQ(out.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(out.at(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(out.at(0, 2), 5.0f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 3.0f);
  EXPECT_FLOAT_EQ(out.at(1, 1), 4.0f);
  EXPECT_FLOAT_EQ(out.at(1, 2), 6.0f);
}

TEST_F(OperatorTest, Gather) {
  auto vs = make_vs();
  vs.put("x", make_tensor(2, 4, {10, 20, 30, 40, 50, 60, 70, 80}));

  OperatorNode node;
  node.domain = "omle.core";
  node.op = "Gather";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal indices;
  indices.kind = AttrVal::Kind::Ints;
  indices.ints = {0, 2};  // select columns 0 and 2
  node.attrs["indices"] = indices;
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 2);
  EXPECT_FLOAT_EQ(y.at(0, 0), 10.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 30.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 50.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 70.0f);
}

TEST_F(OperatorTest, Clip) {
  auto vs = make_vs();
  vs.put("x", make_tensor(1, 4, {-5, 0, 5, 10}));

  OperatorNode node;
  node.domain = "omle.core";
  node.op = "Clip";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal lo, hi;
  lo.kind = AttrVal::Kind::Float;
  lo.f = 0.0;
  hi.kind = AttrVal::Kind::Float;
  hi.f = 6.0;
  node.attrs["min"] = lo;
  node.attrs["max"] = hi;
  ASSERT_TRUE(node.execute(vs, 1).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 5.0f);
  EXPECT_FLOAT_EQ(y.at(0, 3), 6.0f);
}

// -----------------------------------------------------------------------
// omle.feature
// -----------------------------------------------------------------------

TEST_F(OperatorTest, StandardScaler) {
  // mean=[2,4], scale=[2,2] → (x - mean) / scale
  auto mean_t = make_tensor(1, 2, {2.0f, 4.0f});
  auto scale_t = make_tensor(1, 2, {2.0f, 2.0f});
  cs["mean"] = std::make_shared<Tensor>(std::move(mean_t));
  cs["scale"] = std::make_shared<Tensor>(std::move(scale_t));

  auto vs = make_vs();
  vs.put("x", make_tensor(2, 2, {4.0f, 8.0f, 0.0f, 0.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "StandardScaler";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal mean_attr;
  mean_attr.kind = AttrVal::Kind::Tensor;
  mean_attr.tensor_name = "mean";
  AttrVal scale_attr;
  scale_attr.kind = AttrVal::Kind::Tensor;
  scale_attr.tensor_name = "scale";
  node.attrs["mean"] = mean_attr;
  node.attrs["scale"] = scale_attr;
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 1.0f, 1e-5f);   // (4-2)/2
  EXPECT_NEAR(y.at(0, 1), 2.0f, 1e-5f);   // (8-4)/2
  EXPECT_NEAR(y.at(1, 0), -1.0f, 1e-5f);  // (0-2)/2
  EXPECT_NEAR(y.at(1, 1), -2.0f, 1e-5f);  // (0-4)/2
}

TEST_F(OperatorTest, Binarizer) {
  auto vs = make_vs();
  vs.put("x", make_tensor(1, 4, {0.0f, 0.5f, 1.0f, 2.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "Binarizer";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal thr;
  thr.kind = AttrVal::Kind::Float;
  thr.f = 0.5;
  node.attrs["threshold"] = thr;
  ASSERT_TRUE(node.execute(vs, 1).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);  // 0.0 <= 0.5
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);  // 0.5 not > 0.5
  EXPECT_FLOAT_EQ(y.at(0, 2), 1.0f);  // 1.0 > 0.5
  EXPECT_FLOAT_EQ(y.at(0, 3), 1.0f);
}

TEST_F(OperatorTest, Normalizer_L2) {
  auto vs = make_vs();
  vs.put("x", make_tensor(1, 2, {3.0f, 4.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "Normalizer";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal norm;
  norm.kind = AttrVal::Kind::String;
  norm.s = "l2";
  node.attrs["norm"] = norm;
  ASSERT_TRUE(node.execute(vs, 1).ok());

  const Tensor& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 0.6f, 1e-5f);  // 3/5
  EXPECT_NEAR(y.at(0, 1), 0.8f, 1e-5f);  // 4/5
}

TEST_F(OperatorTest, SimpleImputer_FillValue) {
  auto vs = make_vs();
  float nan = std::numeric_limits<float>::quiet_NaN();
  vs.put("x", make_tensor(1, 3, {1.0f, nan, 3.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "SimpleImputer";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal fill;
  fill.kind = AttrVal::Kind::Float;
  fill.f = 99.0;
  node.attrs["fill_value"] = fill;
  ASSERT_TRUE(node.execute(vs, 1).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 99.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 3.0f);
}

TEST_F(OperatorTest, OneHotEncoder) {
  // categories tensor = [0,1,2] (3 numeric categories), offsets = [0,3]
  cs["ohe_cats"] =
      std::make_shared<Tensor>(make_tensor(1, 3, {0.0f, 1.0f, 2.0f}));
  cs["ohe_offsets"] = std::make_shared<Tensor>(make_tensor(1, 2, {0.0f, 3.0f}));
  auto vs = make_vs();
  vs.put("x", make_tensor(3, 1, {0.0f, 1.0f, 2.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "OneHotEncoder";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal cats;
  cats.tensor_name = "ohe_cats";
  AttrVal offs;
  offs.tensor_name = "ohe_offsets";
  node.attrs["categories"] = cats;
  node.attrs["category_offsets"] = offs;
  ASSERT_TRUE(node.execute(vs, 3).ok());

  const Tensor& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 3);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 2), 0.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(2, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(2, 2), 1.0f);
}

TEST_F(OperatorTest, Discretize) {
  // Bins: (-inf, 20], (20, 50], (50, +inf)
  cs["bin_left"] =
      std::make_shared<Tensor>(make_tensor(1, 3, {-inf, 20.0f, 50.0f}));
  cs["bin_right"] =
      std::make_shared<Tensor>(make_tensor(1, 3, {20.0f, 50.0f, inf}));
  cs["bin_vals"] =
      std::make_shared<Tensor>(make_tensor(1, 3, {1.0f, 2.0f, 3.0f}));

  auto vs = make_vs();
  vs.put("x", make_tensor(5, 1, {10.0f, 25.0f, 20.0f, 50.0f, 75.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "Discretize";
  node.in_names = {"x", "bin_left", "bin_right", "bin_vals"};
  node.out_names = {"y"};
  AttrVal lc, rc;
  lc.kind = AttrVal::Kind::Ints;
  lc.ints = {0, 0, 0};  // left_closed: all false
  rc.kind = AttrVal::Kind::Ints;
  rc.ints = {1, 1, 0};  // right_closed: true, true, false
  node.attrs["left_closed"] = lc;
  node.attrs["right_closed"] = rc;
  ASSERT_TRUE(node.execute(vs, 5).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);  // 10  → bin 0 (≤20)
  EXPECT_FLOAT_EQ(y.at(1, 0), 2.0f);  // 25  → bin 1 (20,50]
  EXPECT_FLOAT_EQ(y.at(2, 0), 1.0f);  // 20  → bin 0 (right_closed)
  EXPECT_FLOAT_EQ(y.at(3, 0), 2.0f);  // 50  → bin 1 (right_closed)
  EXPECT_FLOAT_EQ(y.at(4, 0), 3.0f);  // 75  → bin 2 (>50)
}

TEST_F(OperatorTest, Discretize_DefaultValue) {
  cs["bl"] = std::make_shared<Tensor>(make_tensor(1, 1, {0.0f}));
  cs["br"] = std::make_shared<Tensor>(make_tensor(1, 1, {10.0f}));
  cs["bv"] = std::make_shared<Tensor>(make_tensor(1, 1, {99.0f}));

  auto vs = make_vs();
  vs.put("x", make_tensor(2, 1, {5.0f, 20.0f}));  // 20 falls outside bin

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "Discretize";
  node.in_names = {"x", "bl", "br", "bv"};
  node.out_names = {"y"};
  AttrVal lc, rc, def;
  lc.kind = AttrVal::Kind::Ints;
  lc.ints = {1};
  rc.kind = AttrVal::Kind::Ints;
  rc.ints = {1};
  def.kind = AttrVal::Kind::Float;
  def.f = -1.0;
  node.attrs["left_closed"] = lc;
  node.attrs["right_closed"] = rc;
  node.attrs["default_value"] = def;
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 99.0f);  // 5 in [0,10]
  EXPECT_FLOAT_EQ(y.at(1, 0), -1.0f);  // 20 outside → default
}

TEST_F(OperatorTest, NormDiscrete) {
  auto vs = make_vs();
  vs.put("x", make_tensor(4, 1, {1.0f, 2.0f, 3.0f, 2.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "NormDiscrete";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal val;
  val.kind = AttrVal::Kind::Float;
  val.f = 2.0;
  node.attrs["value"] = val;
  ASSERT_TRUE(node.execute(vs, 4).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);  // 1 ≠ 2
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);  // 2 = 2
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);  // 3 ≠ 2
  EXPECT_FLOAT_EQ(y.at(3, 0), 1.0f);  // 2 = 2
}

TEST_F(OperatorTest, NormDiscrete_MapMissing) {
  auto vs = make_vs();
  const float nan = std::numeric_limits<float>::quiet_NaN();
  vs.put("x", make_tensor(3, 1, {2.0f, nan, 5.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "NormDiscrete";
  node.in_names = {"x"};
  node.out_names = {"y"};
  AttrVal val, miss;
  val.kind = AttrVal::Kind::Float;
  val.f = 2.0;
  miss.kind = AttrVal::Kind::Float;
  miss.f = -1.0;
  node.attrs["value"] = val;
  node.attrs["map_missing_to"] = miss;
  ASSERT_TRUE(node.execute(vs, 3).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);   // 2 = 2
  EXPECT_FLOAT_EQ(y.at(1, 0), -1.0f);  // NaN → map_missing_to
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);   // 5 ≠ 2
}

TEST_F(OperatorTest, MapValues) {
  cs["keys"] = std::make_shared<Tensor>(make_tensor(1, 3, {1.0f, 2.0f, 3.0f}));
  cs["vals"] =
      std::make_shared<Tensor>(make_tensor(1, 3, {10.0f, 20.0f, 30.0f}));

  auto vs = make_vs();
  vs.put("x", make_tensor(4, 1, {2.0f, 3.0f, 1.0f, 4.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "MapValues";
  node.in_names = {"x", "keys", "vals"};
  node.out_names = {"y"};
  AttrVal def;
  def.kind = AttrVal::Kind::Float;
  def.f = -1.0;
  node.attrs["default_value"] = def;
  ASSERT_TRUE(node.execute(vs, 4).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 20.0f);  // key 2 → 20
  EXPECT_FLOAT_EQ(y.at(1, 0), 30.0f);  // key 3 → 30
  EXPECT_FLOAT_EQ(y.at(2, 0), 10.0f);  // key 1 → 10
  EXPECT_FLOAT_EQ(y.at(3, 0), -1.0f);  // key 4 → default
}

TEST_F(OperatorTest, MapValues_MapMissing) {
  cs["k2"] = std::make_shared<Tensor>(make_tensor(1, 2, {1.0f, 2.0f}));
  cs["v2"] = std::make_shared<Tensor>(make_tensor(1, 2, {100.0f, 200.0f}));

  const float nan = std::numeric_limits<float>::quiet_NaN();
  auto vs = make_vs();
  vs.put("x", make_tensor(3, 1, {1.0f, nan, 3.0f}));

  OperatorNode node;
  node.domain = "omle.feature";
  node.op = "MapValues";
  node.in_names = {"x", "k2", "v2"};
  node.out_names = {"y"};
  AttrVal def, miss;
  def.kind = AttrVal::Kind::Float;
  def.f = 0.0;
  miss.kind = AttrVal::Kind::Float;
  miss.f = -9.0;
  node.attrs["default_value"] = def;
  node.attrs["map_missing_to"] = miss;
  ASSERT_TRUE(node.execute(vs, 3).ok());

  const Tensor& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 100.0f);  // key 1 → 100
  EXPECT_FLOAT_EQ(y.at(1, 0), -9.0f);   // NaN → map_missing_to
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);    // key 3 not found → default
}
