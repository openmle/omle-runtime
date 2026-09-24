// Float64 fidelity for the operator set.
//
// Every operator used to read its inputs through the float32 accessors
// (at(), row(), f32_ptr()), which reinterpret double storage as float rather
// than converting it. A narrowing shim in OperatorNode hid that by casting
// float64 inputs down before dispatch; these tests pin the behaviour that
// replaced it, so a kernel that goes back to reading float32 fails here
// instead of silently returning a different answer.
//
// Two properties are checked throughout:
//   * a value that float32 cannot represent survives the operator, and
//   * an operator that only moves or selects data keeps the input's dtype.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "op_test_fixture.h"

using namespace omle::rt::impl;
using omle_test::OpTest;

namespace {

// 2^24+1 is the first integer float32 cannot represent: it rounds to 2^24.
constexpr double kBeyondF32Int = 16777217.0;
// Not representable in float32 either; narrowing perturbs it by ~1.5e-9.
constexpr double kBeyondF32Frac = 0.1;

class OpF64Test : public OpTest {
 protected:
  Tensor Td(int rows, int cols, std::vector<double> data) {
    Tensor t = Tensor::dense(omle::rt::DataType::Float64, rows, cols);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c)
        t.f64_at(r, c) = data[static_cast<std::size_t>(r) * cols + c];
    return t;
  }

  void put_const_f64(const std::string& name, int rows, int cols,
                     std::vector<double> data) {
    cs[name] = std::make_shared<Tensor>(Td(rows, cols, std::move(data)));
  }

  static bool is_f64(const Tensor& t) {
    return t.dtype == omle::rt::DataType::Float64;
  }
};

// ── pass-through operators keep both the value and the dtype ────────────────

TEST_F(OpF64Test, GatherKeepsFloat64) {
  auto vs = make_vs();
  vs.put("x", Td(1, 3, {kBeyondF32Int, 2.0, kBeyondF32Frac}));
  exec_n(vs, 1, "omle.core", "Gather", {"x"}, {"y"},
         {{"indices", aint({0, 2})}});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_EQ(y.get(0, 0), kBeyondF32Int);
  EXPECT_EQ(y.get(0, 1), kBeyondF32Frac);
}

TEST_F(OpF64Test, SplitKeepsFloat64) {
  auto vs = make_vs();
  vs.put("x", Td(1, 4, {kBeyondF32Int, 1.0, kBeyondF32Frac, 3.0}));
  exec_n(vs, 1, "omle.core", "Split", {"x"}, {"a", "b"},
         {{"sections", aint({2, 2})}});
  EXPECT_TRUE(is_f64(vs.get("a")));
  EXPECT_EQ(vs.get("a").get(0, 0), kBeyondF32Int);
  EXPECT_EQ(vs.get("b").get(0, 0), kBeyondF32Frac);
}

TEST_F(OpF64Test, ReshapeKeepsFloat64) {
  auto vs = make_vs();
  vs.put("x", Td(2, 2, {kBeyondF32Int, 1.0, 2.0, kBeyondF32Frac}));
  exec_n(vs, 2, "omle.core", "Reshape", {"x"}, {"y"},
         {{"new_shape", aint({1, 4})}});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_EQ(y.n_rows, 1);
  EXPECT_EQ(y.n_cols, 4);
  EXPECT_EQ(y.get(0, 0), kBeyondF32Int);
  EXPECT_EQ(y.get(0, 3), kBeyondF32Frac);
}

TEST_F(OpF64Test, SelectKeepsFloat64) {
  auto vs = make_vs();
  vs.put("sel", Td(2, 1, {0.0, 1.0}));
  vs.put("a", Td(2, 1, {kBeyondF32Int, 1.0}));
  vs.put("b", Td(2, 1, {2.0, kBeyondF32Frac}));
  exec_n(vs, 2, "omle.core", "Select", {"sel", "a", "b"}, {"y"});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_EQ(y.get(0, 0), kBeyondF32Int);
  EXPECT_EQ(y.get(1, 0), kBeyondF32Frac);
}

// Cast is the operator whose whole job is width, and it used to funnel
// everything through to_float32().
TEST_F(OpF64Test, CastToFloat64Widens) {
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {1.5f}));
  exec_n(vs, 1, "omle.core", "Cast", {"x"}, {"y"},
         {{"output_dtype", astr("float64")}});
  EXPECT_TRUE(is_f64(vs.get("y")));
  EXPECT_EQ(vs.get("y").get(0, 0), 1.5);
}

TEST_F(OpF64Test, CastToIntKeepsFloat64Width) {
  auto vs = make_vs();
  // Rounding this through float32 lands on 16777216, the wrong integer.
  vs.put("x", Td(1, 1, {kBeyondF32Int + 0.25}));
  exec_n(vs, 1, "omle.core", "Cast", {"x"}, {"y"},
         {{"output_dtype", astr("int64")}});
  EXPECT_TRUE(is_f64(vs.get("y")));
  EXPECT_EQ(vs.get("y").get(0, 0), kBeyondF32Int);
}

// ── decision-making operators: a narrowed read changes the answer ───────────

TEST_F(OpF64Test, ArgMaxUsesFullPrecision) {
  auto vs = make_vs();
  // The two scores are distinct doubles that collapse onto the same float32,
  // so a narrowed comparison keeps index 0 while the real maximum is index 1.
  const double a = 1.0000000000000002;
  vs.put("x", Td(1, 2, {1.0, a}));
  ASSERT_EQ(static_cast<float>(1.0), static_cast<float>(a));
  exec_n(vs, 1, "omle.core", "ArgMax", {"x"}, {"y"});
  EXPECT_EQ(vs.get("y").get(0, 0), 1.0);
}

TEST_F(OpF64Test, SelectByPrimarySecondaryScoreUsesFullPrecision) {
  auto vs = make_vs();
  const double a = 1.0000000000000002;
  vs.put("p", Td(1, 2, {1.0, a}));
  vs.put("s", Td(1, 2, {0.0, 0.0}));
  exec_n(vs, 1, "omle.core", "SelectByPrimarySecondaryScore", {"p", "s"},
         {"y"});
  EXPECT_EQ(vs.get("y").get(0, 0), 1.0);
}

TEST_F(OpF64Test, BucketizerBoundaryIsNotNarrowed) {
  auto vs = make_vs();
  // x sits just below the boundary in double, but both round to the same
  // float32 -- narrowed, the <= test puts it in the upper bucket.
  const double edge = 16777217.0;
  vs.put("x", Td(1, 1, {edge - 1.0}));
  exec_n(vs, 1, "omle.feature", "Bucketizer", {"x"}, {"y"},
         {{"boundaries", afloats({edge})}});
  EXPECT_EQ(vs.get("y").get(0, 0), 0.0);
}

// ── exact-equality lookups: probe and table must agree bit for bit ──────────

TEST_F(OpF64Test, OrdinalEncoderMatchesLargeIntegerCategories) {
  auto vs = make_vs();
  put_const_f64("cats", 1, 3,
                {kBeyondF32Int, kBeyondF32Int + 1.0, kBeyondF32Int + 2.0});
  put_const("offs", 1, 2, {0.f, 3.f});
  vs.put("x",
         Td(3, 1, {kBeyondF32Int + 2.0, kBeyondF32Int, kBeyondF32Int + 1.0}));
  exec_n(
      vs, 3, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
      {{"categories", atensor("cats")}, {"category_offsets", atensor("offs")}});
  const Tensor& y = vs.get("y");
  EXPECT_EQ(y.get(0, 0), 2.0);
  EXPECT_EQ(y.get(1, 0), 0.0);
  EXPECT_EQ(y.get(2, 0), 1.0);
}

TEST_F(OpF64Test, LabelEncoderMatchesLargeIntegerLabels) {
  auto vs = make_vs();
  put_const_f64("labels", 1, 2, {kBeyondF32Int, kBeyondF32Int + 1.0});
  put_const("offs", 1, 2, {0.f, 2.f});
  vs.put("x", Td(2, 1, {kBeyondF32Int + 1.0, kBeyondF32Int}));
  exec_n(vs, 2, "omle.feature", "LabelEncoder", {"x"}, {"y"},
         {{"labels", atensor("labels")}, {"label_offsets", atensor("offs")}});
  EXPECT_EQ(vs.get("y").get(0, 0), 1.0);
  EXPECT_EQ(vs.get("y").get(1, 0), 0.0);
}

TEST_F(OpF64Test, MapValuesMatchesFloat64Keys) {
  auto vs = make_vs();
  put_const_f64("keys", 1, 2, {kBeyondF32Frac, kBeyondF32Int});
  put_const_f64("vals", 1, 2, {7.0, 9.0});
  vs.put("x", Td(2, 1, {kBeyondF32Int, kBeyondF32Frac}));
  exec_n(vs, 2, "omle.feature", "MapValues", {"x"}, {"y"},
         {{"keys", atensor("keys")}, {"values", atensor("vals")}});
  EXPECT_EQ(vs.get("y").get(0, 0), 9.0);
  EXPECT_EQ(vs.get("y").get(1, 0), 7.0);
}

TEST_F(OpF64Test, NormDiscreteComparesAtFullPrecision) {
  auto vs = make_vs();
  // Equal in float32, different in double: the indicator must be 0.
  vs.put("x", Td(1, 1, {kBeyondF32Int + 1.0}));
  exec_n(vs, 1, "omle.feature", "NormDiscrete", {"x"}, {"y"},
         {{"value", aflt(kBeyondF32Int)}});
  EXPECT_EQ(vs.get("y").get(0, 0), 0.0);
}

TEST_F(OpF64Test, MultiLabelBinarizerMatchesFloat64Classes) {
  auto vs = make_vs();
  put_const_f64("classes", 1, 2, {kBeyondF32Int, kBeyondF32Int + 1.0});
  vs.put("x", Td(1, 1, {kBeyondF32Int + 1.0}));
  exec_n(vs, 1, "omle.feature", "MultiLabelBinarizer", {"x"}, {"y"},
         {{"classes", atensor("classes")}});
  EXPECT_EQ(vs.get("y").get(0, 0), 0.0);
  EXPECT_EQ(vs.get("y").get(0, 1), 1.0);
}

// ── arithmetic operators compute in double and keep the width ───────────────

TEST_F(OpF64Test, PcaProjectsInDoubleAndKeepsFloat64) {
  auto vs = make_vs();
  // A coefficient with more significant digits than float32 can hold.
  const double c = 0.9628150866775175;
  put_const_f64("comp", 1, 1, {c});
  put_const_f64("mean", 1, 1, {0.0});
  vs.put("x", Td(1, 1, {3.0}));
  exec_n(vs, 1, "omle.feature", "PCA", {"x"}, {"y"},
         {{"components", atensor("comp")}, {"mean", atensor("mean")}});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_DOUBLE_EQ(y.get(0, 0), 3.0 * c);
}

TEST_F(OpF64Test, PcaFloat32InputStillUsesExactCoefficients) {
  auto vs = make_vs();
  const double c = 0.9628150866775175;
  put_const_f64("comp", 1, 1, {c});
  vs.put("x", Tf(1, 1, {3.0f}));
  exec_n(vs, 1, "omle.feature", "PCA", {"x"}, {"y"},
         {{"components", atensor("comp")}});
  // float32 in, float32 out -- but the product is formed in double from the
  // undegraded coefficient and rounded once, not computed from a narrowed one.
  EXPECT_FALSE(is_f64(vs.get("y")));
  EXPECT_FLOAT_EQ(vs.get("y").get(0, 0), static_cast<float>(3.0 * c));
}

TEST_F(OpF64Test, NormContinuousKeepsFloat64) {
  auto vs = make_vs();
  put_const_f64("orig", 1, 2, {0.0, 1.0});
  put_const_f64("norm", 1, 2, {0.0, 1.0});
  put_const("offs", 1, 2, {0.f, 2.f});
  vs.put("x", Td(1, 1, {kBeyondF32Frac}));
  exec_n(vs, 1, "omle.feature", "NormContinuous", {"x"}, {"y"},
         {{"orig_points", atensor("orig")},
          {"norm_points", atensor("norm")},
          {"point_offsets", atensor("offs")}});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_DOUBLE_EQ(y.get(0, 0), kBeyondF32Frac);
}

TEST_F(OpF64Test, TfIdfTransformerKeepsFloat64) {
  auto vs = make_vs();
  const double idf = 1.4054651081081644;
  put_const_f64("idf", 1, 1, {idf});
  vs.put("x", Td(1, 1, {3.0}));
  exec_n(vs, 1, "omle.text", "TfIdfTransformer", {"x"}, {"y"},
         {{"idf", atensor("idf")}});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_DOUBLE_EQ(y.get(0, 0), 3.0 * idf);
}

TEST_F(OpF64Test, DeriveEvaluatesInDoubleAndKeepsFloat64) {
  auto vs = make_vs();
  vs.put("x", Td(1, 1, {kBeyondF32Int}));
  // y = x + 1 -- in float32 both 2^24+1 and 2^24+2 collapse, so the result
  // would come back equal to the input.
  auto add = std::make_shared<Expr>();
  ExprApply ap;
  ap.function = "add";
  auto col = std::make_shared<Expr>();
  col->node = ExprColumn{"x"};
  auto one = std::make_shared<Expr>();
  ScalarVal sv;
  sv.kind = ScalarVal::Kind::Double;
  sv.d = 1.0;
  one->node = ExprLiteral{sv};
  ap.args = {col, one};
  add->node = ap;

  AttrVal expr_attr;
  expr_attr.expr = add;
  exec_n(vs, 1, "omle.core", "Derive", {"x"}, {"y"}, {{"expr", expr_attr}});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_EQ(y.get(0, 0), kBeyondF32Int + 1.0);
}

TEST_F(OpF64Test, KnnRanksNeighboursInDouble) {
  auto vs = make_vs();
  // Two training points whose distances to the query differ only in double.
  put_const_f64("train_x", 3, 1, {0.0, kBeyondF32Frac, 100.0});
  put_const_f64("train_y", 3, 1, {10.0, 20.0, 30.0});
  vs.put("q", Td(1, 1, {kBeyondF32Frac}));
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"},
         {{"train_features", atensor("train_x")},
          {"train_targets", atensor("train_y")},
          {"n_neighbors", aint({1})},
          {"task", astr("regression")}});
  // The exact match must win: distance 0 to the second training row.
  EXPECT_DOUBLE_EQ(vs.get("y").get(0, 0), 20.0);
}

TEST_F(OpF64Test, ConcatOfMixedWidthsKeepsFloat64) {
  auto vs = make_vs();
  vs.put("a", Td(1, 1, {kBeyondF32Int}));
  vs.put("b", Tf(1, 1, {2.0f}));
  exec_n(vs, 1, "omle.core", "Concat", {"a", "b"}, {"y"});
  const Tensor& y = vs.get("y");
  EXPECT_TRUE(is_f64(y));
  EXPECT_EQ(y.get(0, 0), kBeyondF32Int);
}

}  // namespace
