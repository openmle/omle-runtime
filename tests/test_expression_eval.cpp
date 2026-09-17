#include <gtest/gtest.h>

#include <cmath>

#include "expression_eval.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

// Helpers to build test ValueStore with a named column.
static ValueStore make_vs(const std::string& name, std::vector<float> vals) {
  static ConstantStore cs;
  ValueStore vs(cs);
  Tensor t(static_cast<int>(vals.size()), 1);
  for (int i = 0; i < (int)vals.size(); ++i) t.at(i, 0) = vals[i];
  vs.put(name, std::move(t));
  return vs;
}

static ExprPtr col(const std::string& name) { return make_column(name); }
static ExprPtr lit(double v) { return make_literal(ScalarVal::from_double(v)); }
static ExprPtr app(const std::string& fn, std::vector<ExprPtr> args) {
  return make_apply(fn, std::move(args));
}

// -----------------------------------------------------------------------
// Expression evaluation
// -----------------------------------------------------------------------

TEST(ExprEval, Literal) {
  ConstantStore cs;
  ValueStore vs(cs);
  auto e = lit(7.0);
  auto r = eval_expr(*e, vs, 3).value();
  ASSERT_EQ((int)r.size(), 3);
  for (float v : r) EXPECT_FLOAT_EQ(v, 7.0f);
}

TEST(ExprEval, Column) {
  auto vs = make_vs("x", {1.0f, 2.0f, 3.0f});
  auto e = col("x");
  auto r = eval_expr(*e, vs, 3).value();
  EXPECT_FLOAT_EQ(r[0], 1.0f);
  EXPECT_FLOAT_EQ(r[1], 2.0f);
  EXPECT_FLOAT_EQ(r[2], 3.0f);
}

TEST(ExprEval, AddSubMulDiv) {
  auto vs = make_vs("x", {4.0f, 6.0f});
  int n = 2;
  EXPECT_FLOAT_EQ(eval_expr(*app("add", {col("x"), lit(2)}), vs, n).value()[0],
                  6.0f);
  EXPECT_FLOAT_EQ(
      eval_expr(*app("subtract", {col("x"), lit(2)}), vs, n).value()[0], 2.0f);
  EXPECT_FLOAT_EQ(
      eval_expr(*app("multiply", {col("x"), lit(3)}), vs, n).value()[0], 12.0f);
  EXPECT_FLOAT_EQ(
      eval_expr(*app("divide", {col("x"), lit(2)}), vs, n).value()[0], 2.0f);
}

TEST(ExprEval, Negate_Abs) {
  auto vs = make_vs("x", {-3.0f, 3.0f});
  auto r_neg = eval_expr(*app("negate", {col("x")}), vs, 2).value();
  EXPECT_FLOAT_EQ(r_neg[0], 3.0f);
  EXPECT_FLOAT_EQ(r_neg[1], -3.0f);

  auto r_abs = eval_expr(*app("abs", {col("x")}), vs, 2).value();
  EXPECT_FLOAT_EQ(r_abs[0], 3.0f);
  EXPECT_FLOAT_EQ(r_abs[1], 3.0f);
}

TEST(ExprEval, ExpLog) {
  auto vs = make_vs("x", {0.0f, 1.0f});
  auto r_exp = eval_expr(*app("exp", {col("x")}), vs, 2).value();
  EXPECT_NEAR(r_exp[0], 1.0f, 1e-5f);
  EXPECT_NEAR(r_exp[1], std::exp(1.0f), 1e-5f);

  auto r_log = eval_expr(*app("log", {col("x")}), vs, 2).value();
  EXPECT_NEAR(r_log[1], 0.0f, 1e-5f);  // log(1) = 0
}

TEST(ExprEval, Sqrt) {
  auto vs = make_vs("x", {4.0f, 9.0f});
  auto r = eval_expr(*app("sqrt", {col("x")}), vs, 2).value();
  EXPECT_NEAR(r[0], 2.0f, 1e-5f);
  EXPECT_NEAR(r[1], 3.0f, 1e-5f);
}

TEST(ExprEval, MinMax) {
  auto vs = make_vs("x", {3.0f, 7.0f});
  auto r_min = eval_expr(*app("min", {col("x"), lit(5)}), vs, 2).value();
  EXPECT_FLOAT_EQ(r_min[0], 3.0f);
  EXPECT_FLOAT_EQ(r_min[1], 5.0f);

  auto r_max = eval_expr(*app("max", {col("x"), lit(5)}), vs, 2).value();
  EXPECT_FLOAT_EQ(r_max[0], 5.0f);
  EXPECT_FLOAT_EQ(r_max[1], 7.0f);
}

TEST(ExprEval, Ternary_If) {
  auto vs = make_vs("x", {-1.0f, 1.0f});
  // if(x > 0, 10, -10) implemented as if(gt(x,0), 10, -10)
  auto cond = app("greater", {col("x"), lit(0)});
  auto e = app("if", {std::move(cond), lit(10), lit(-10)});
  auto r = eval_expr(*e, vs, 2).value();
  EXPECT_FLOAT_EQ(r[0], -10.0f);
  EXPECT_FLOAT_EQ(r[1], 10.0f);
}

TEST(ExprEval, Clamp) {
  auto vs = make_vs("x", {-5.0f, 0.5f, 10.0f});
  auto e = app("clamp", {col("x"), lit(0), lit(1)});
  auto r = eval_expr(*e, vs, 3).value();
  EXPECT_FLOAT_EQ(r[0], 0.0f);
  EXPECT_FLOAT_EQ(r[1], 0.5f);
  EXPECT_FLOAT_EQ(r[2], 1.0f);
}

TEST(ExprEval, Sigmoid) {
  auto vs = make_vs("x", {0.0f});
  auto r = eval_expr(*app("sigmoid", {col("x")}), vs, 1).value();
  EXPECT_NEAR(r[0], 0.5f, 1e-5f);
}

TEST(ExprEval, Relu) {
  auto vs = make_vs("x", {-2.0f, 0.0f, 3.0f});
  auto r = eval_expr(*app("relu", {col("x")}), vs, 3).value();
  EXPECT_FLOAT_EQ(r[0], 0.0f);
  EXPECT_FLOAT_EQ(r[1], 0.0f);
  EXPECT_FLOAT_EQ(r[2], 3.0f);
}

// -----------------------------------------------------------------------
// Predicate evaluation
// -----------------------------------------------------------------------

static PredPtr simple_pred(const std::string& col, SimpleOp op, double val) {
  PredSimple ps;
  ps.column = col;
  ps.op = op;
  ps.value = ScalarVal::from_double(val);
  return std::make_shared<Pred>(Pred{std::move(ps)});
}

TEST(PredEval, TrueFalse) {
  ConstantStore cs;
  ValueStore vs(cs);
  auto rt = eval_pred(*make_true_pred(), vs, 3).value();
  auto rf = eval_pred(*make_false_pred(), vs, 3).value();
  for (int i = 0; i < 3; ++i) {
    EXPECT_TRUE(rt[i]);
    EXPECT_FALSE(rf[i]);
  }
}

TEST(PredEval, SimpleComparisons) {
  auto vs = make_vs("x", {1.0f, 5.0f, 10.0f});

  auto lt = eval_pred(*simple_pred("x", SimpleOp::LT, 5.0), vs, 3).value();
  EXPECT_TRUE(lt[0]);
  EXPECT_FALSE(lt[1]);
  EXPECT_FALSE(lt[2]);

  auto le = eval_pred(*simple_pred("x", SimpleOp::LE, 5.0), vs, 3).value();
  EXPECT_TRUE(le[0]);
  EXPECT_TRUE(le[1]);
  EXPECT_FALSE(le[2]);

  auto eq = eval_pred(*simple_pred("x", SimpleOp::EQ, 5.0), vs, 3).value();
  EXPECT_FALSE(eq[0]);
  EXPECT_TRUE(eq[1]);
  EXPECT_FALSE(eq[2]);

  auto ne = eval_pred(*simple_pred("x", SimpleOp::NE, 5.0), vs, 3).value();
  EXPECT_TRUE(ne[0]);
  EXPECT_FALSE(ne[1]);
  EXPECT_TRUE(ne[2]);
}

TEST(PredEval, IsMissingIsNotMissing) {
  auto vs = make_vs("x", {1.0f, std::numeric_limits<float>::quiet_NaN()});

  auto is_miss =
      eval_pred(*simple_pred("x", SimpleOp::IsMissing, 0.0), vs, 2).value();
  auto is_not_miss =
      eval_pred(*simple_pred("x", SimpleOp::IsNotMissing, 0.0), vs, 2).value();

  EXPECT_FALSE(is_miss[0]);
  EXPECT_TRUE(is_miss[1]);
  EXPECT_TRUE(is_not_miss[0]);
  EXPECT_FALSE(is_not_miss[1]);
}

TEST(PredEval, SetPredicate) {
  auto vs = make_vs("x", {1.0f, 2.0f, 3.0f, 4.0f});

  PredSet ps;
  ps.column = "x";
  ps.op = SetOp::In;
  ps.values = {ScalarVal::from_double(2.0), ScalarVal::from_double(4.0)};
  auto pred = std::make_shared<Pred>(Pred{std::move(ps)});
  auto r = eval_pred(*pred, vs, 4).value();

  EXPECT_FALSE(r[0]);
  EXPECT_TRUE(r[1]);
  EXPECT_FALSE(r[2]);
  EXPECT_TRUE(r[3]);
}

TEST(PredEval, CompoundAnd) {
  auto vs = make_vs("x", {1.0f, 5.0f, 10.0f});

  // x >= 2 AND x <= 8
  PredCompound pc;
  pc.op = BoolOp::And;
  pc.children.push_back(simple_pred("x", SimpleOp::GE, 2.0));
  pc.children.push_back(simple_pred("x", SimpleOp::LE, 8.0));
  auto pred = std::make_shared<Pred>(Pred{std::move(pc)});
  auto r = eval_pred(*pred, vs, 3).value();

  EXPECT_FALSE(r[0]);  // 1 < 2
  EXPECT_TRUE(r[1]);   // 2 <= 5 <= 8
  EXPECT_FALSE(r[2]);  // 10 > 8
}

TEST(PredEval, CompoundOr) {
  auto vs = make_vs("x", {1.0f, 5.0f, 10.0f});

  // x < 3 OR x > 8
  PredCompound pc;
  pc.op = BoolOp::Or;
  pc.children.push_back(simple_pred("x", SimpleOp::LT, 3.0));
  pc.children.push_back(simple_pred("x", SimpleOp::GT, 8.0));
  auto pred = std::make_shared<Pred>(Pred{std::move(pc)});
  auto r = eval_pred(*pred, vs, 3).value();

  EXPECT_TRUE(r[0]);
  EXPECT_FALSE(r[1]);
  EXPECT_TRUE(r[2]);
}
