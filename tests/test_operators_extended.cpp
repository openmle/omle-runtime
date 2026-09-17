#include "op_test_fixture.h"

using namespace omle_test;

// =============================================================================
// omle.core
// =============================================================================

TEST_F(OpTest, ArgMax) {
  auto vs = make_vs();
  vs.put("x", Tf(3, 4, {1, 3, 2, 0, 5, 1, 4, 2, 0, 0, 0, 7}));
  exec_n(vs, 3, "omle.core", "ArgMax", {"x"}, {"y"});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 1);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);  // max=3 at col 1
  EXPECT_FLOAT_EQ(y.at(1, 0), 0.0f);  // max=5 at col 0
  EXPECT_FLOAT_EQ(y.at(2, 0), 3.0f);  // max=7 at col 3
}

TEST_F(OpTest, Cast_FloatPassthrough) {
  auto vs = make_vs();
  vs.put("x", Tf(1, 3, {1.7f, -2.3f, 0.0f}));
  exec_n(vs, 1, "omle.core", "Cast", {"x"}, {"y"},
         {{"output_dtype", astr("float32")}});
  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 1.7f, 1e-5f);
  EXPECT_NEAR(y.at(0, 1), -2.3f, 1e-5f);
}

TEST_F(OpTest, Cast_IntRounding) {
  auto vs = make_vs();
  vs.put("x", Tf(1, 3, {1.7f, -2.3f, 0.5f}));
  exec_n(vs, 1, "omle.core", "Cast", {"x"}, {"y"},
         {{"output_dtype", astr("int32")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 2.0f);   // round(1.7)
  EXPECT_FLOAT_EQ(y.at(0, 1), -2.0f);  // round(-2.3)
  EXPECT_FLOAT_EQ(y.at(0, 2), 1.0f);   // round(0.5)
}

TEST_F(OpTest, Reshape) {
  auto vs = make_vs();
  vs.put("x", Tf(2, 3, {1, 2, 3, 4, 5, 6}));
  exec_n(vs, 2, "omle.core", "Reshape", {"x"}, {"y"},
         {{"new_shape", aint({-1, 2})}});
  const auto& y = vs.get("y");
  EXPECT_EQ(y.n_rows, 3);
  EXPECT_EQ(y.n_cols, 2);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 1), 6.0f);
}

TEST_F(OpTest, Split) {
  auto vs = make_vs();
  vs.put("x", Tf(2, 4, {1, 2, 3, 4, 5, 6, 7, 8}));
  exec_n(vs, 2, "omle.core", "Split", {"x"}, {"a", "b"},
         {{"sections", aint({2, 2})}});
  const auto& a = vs.get("a");
  const auto& b = vs.get("b");
  ASSERT_EQ(a.n_cols, 2);
  ASSERT_EQ(b.n_cols, 2);
  EXPECT_FLOAT_EQ(a.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(a.at(0, 1), 2.0f);
  EXPECT_FLOAT_EQ(b.at(0, 0), 3.0f);
  EXPECT_FLOAT_EQ(b.at(0, 1), 4.0f);
  EXPECT_FLOAT_EQ(a.at(1, 0), 5.0f);
  EXPECT_FLOAT_EQ(b.at(1, 1), 8.0f);
}

TEST_F(OpTest, TakeSlots) {
  auto vs = make_vs();
  vs.put("a", Tf(2, 2, {1, 2, 3, 4}));
  vs.put("b", Tf(2, 2, {5, 6, 7, 8}));
  exec_n(vs, 2, "omle.core", "TakeSlots", {"a", "b"}, {"y"},
         {{"indices", aint({0, 2})}});  // col 0 from a, col 0 from b
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 2);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 5.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 3.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 7.0f);
}

TEST_F(OpTest, Select) {
  auto vs = make_vs();
  // sel: row 0 picks candidate 0, row 1 picks candidate 1
  vs.put("sel", Tf(2, 1, {0.0f, 1.0f}));
  vs.put("c0", Tf(2, 1, {10.0f, 10.0f}));
  vs.put("c1", Tf(2, 1, {20.0f, 20.0f}));
  exec_n(vs, 2, "omle.core", "Select", {"sel", "c0", "c1"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 10.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 20.0f);
}

TEST_F(OpTest, SparseToDense) {
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {1.0f, 2.0f, 3.0f, 4.0f}));
  exec_n(vs, 2, "omle.core", "SparseToDense", {"x"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 4.0f);
}

TEST_F(OpTest, Sum) {
  auto vs = make_vs();
  vs.put("a", Tf(2, 2, {1, 2, 3, 4}));
  vs.put("b", Tf(2, 2, {10, 20, 30, 40}));
  vs.put("c", Tf(2, 2, {100, 200, 300, 400}));
  exec_n(vs, 2, "omle.core", "Sum", {"a", "b", "c"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 111.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 222.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 333.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 444.0f);
}

TEST_F(OpTest, Average) {
  auto vs = make_vs();
  vs.put("a", Tf(1, 2, {0.0f, 6.0f}));
  vs.put("b", Tf(1, 2, {6.0f, 0.0f}));
  exec_n(vs, 1, "omle.core", "Average", {"a", "b"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 3.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 3.0f);
}

TEST_F(OpTest, Min) {
  auto vs = make_vs();
  vs.put("a", Tf(1, 2, {5.0f, 1.0f}));
  vs.put("b", Tf(1, 2, {3.0f, 7.0f}));
  vs.put("c", Tf(1, 2, {9.0f, 2.0f}));
  exec_n(vs, 1, "omle.core", "Min", {"a", "b", "c"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 3.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 1.0f);
}

TEST_F(OpTest, Max) {
  auto vs = make_vs();
  vs.put("a", Tf(1, 2, {5.0f, 1.0f}));
  vs.put("b", Tf(1, 2, {3.0f, 7.0f}));
  vs.put("c", Tf(1, 2, {9.0f, 2.0f}));
  exec_n(vs, 1, "omle.core", "Max", {"a", "b", "c"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 9.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 7.0f);
}

TEST_F(OpTest, Median_Odd) {
  auto vs = make_vs();
  vs.put("a", Tf(1, 1, {1.0f}));
  vs.put("b", Tf(1, 1, {5.0f}));
  vs.put("c", Tf(1, 1, {3.0f}));
  exec_n(vs, 1, "omle.core", "Median", {"a", "b", "c"}, {"y"});
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 3.0f);
}

TEST_F(OpTest, WeightedSum) {
  auto vs = make_vs();
  vs.put("a", Tf(1, 1, {2.0f}));
  vs.put("b", Tf(1, 1, {3.0f}));
  exec_n(vs, 1, "omle.core", "WeightedSum", {"a", "b"}, {"y"},
         {{"weights", afloats({0.5, 2.0})}});
  // 0.5*2 + 2.0*3 = 1 + 6 = 7
  EXPECT_NEAR(vs.get("y").at(0, 0), 7.0f, 1e-5f);
}

TEST_F(OpTest, WeightedAverage) {
  auto vs = make_vs();
  vs.put("a", Tf(1, 1, {0.0f}));
  vs.put("b", Tf(1, 1, {10.0f}));
  exec_n(vs, 1, "omle.core", "WeightedAverage", {"a", "b"}, {"y"},
         {{"weights", afloats({1.0, 3.0})}});
  // (1*0 + 3*10) / 4 = 7.5
  EXPECT_NEAR(vs.get("y").at(0, 0), 7.5f, 1e-5f);
}

TEST_F(OpTest, WeightedMedian) {
  // Values: 1,2,3 with weights 1,3,1 → weighted median = 2 (weight 3 > half of
  // 5)
  auto vs = make_vs();
  vs.put("a", Tf(1, 1, {1.0f}));
  vs.put("b", Tf(1, 1, {2.0f}));
  vs.put("c", Tf(1, 1, {3.0f}));
  exec_n(vs, 1, "omle.core", "WeightedMedian", {"a", "b", "c"}, {"y"},
         {{"weights", afloats({1.0, 3.0, 1.0})}});
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 2.0f);
}

TEST_F(OpTest, MajorityVote) {
  auto vs = make_vs();
  vs.put("v0", Tf(2, 1, {1.0f, 2.0f}));
  vs.put("v1", Tf(2, 1, {1.0f, 3.0f}));
  vs.put("v2", Tf(2, 1, {2.0f, 3.0f}));
  exec_n(vs, 2, "omle.core", "MajorityVote", {"v0", "v1", "v2"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);  // 1,1,2 → 1 wins
  EXPECT_FLOAT_EQ(y.at(1, 0), 3.0f);  // 2,3,3 → 3 wins
}

TEST_F(OpTest, WeightedMajorityVote) {
  auto vs = make_vs();
  vs.put("v0", Tf(1, 1, {1.0f}));
  vs.put("v1", Tf(1, 1, {2.0f}));
  vs.put("v2", Tf(1, 1, {2.0f}));
  exec_n(vs, 1, "omle.core", "WeightedMajorityVote", {"v0", "v1", "v2"}, {"y"},
         {{"weights", afloats({10.0, 1.0, 1.0})}});
  // weight of class 1 = 10, class 2 = 2 → class 1 wins
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 1.0f);
}

TEST_F(OpTest, SoftVote) {
  auto vs = make_vs();
  vs.put("p0", Tf(1, 3, {0.6f, 0.3f, 0.1f}));
  vs.put("p1", Tf(1, 3, {0.2f, 0.5f, 0.3f}));
  exec_n(vs, 1, "omle.core", "SoftVote", {"p0", "p1"}, {"y"});
  // average: [0.4, 0.4, 0.2]
  EXPECT_NEAR(vs.get("y").at(0, 0), 0.4f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 1), 0.4f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 2), 0.2f, 1e-5f);
}

TEST_F(OpTest, SoftVote_Weighted) {
  auto vs = make_vs();
  vs.put("p0", Tf(1, 2, {1.0f, 0.0f}));
  vs.put("p1", Tf(1, 2, {0.0f, 1.0f}));
  exec_n(vs, 1, "omle.core", "SoftVote", {"p0", "p1"}, {"y"},
         {{"weights", afloats({3.0, 1.0})}});
  // (3*1+1*0)/4=0.75, (3*0+1*1)/4=0.25
  EXPECT_NEAR(vs.get("y").at(0, 0), 0.75f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 1), 0.25f, 1e-5f);
}

// =============================================================================
// omle.feature — scalers
// =============================================================================

TEST_F(OpTest, MinMaxScaler) {
  // data_min=[0,10], data_scale=[10,10] (data_max-data_min), feat_range=[0,1]
  put_const("dmin", 1, 2, {0.0f, 10.0f});
  put_const("dscale", 1, 2, {10.0f, 10.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {0.0f, 10.0f, 5.0f, 20.0f}));
  exec_n(vs, 2, "omle.feature", "MinMaxScaler", {"x"}, {"y"},
         {{"data_min", atensor("dmin")}, {"data_max", atensor("dscale")}});
  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 0.0f, 1e-5f);  // (0-0)/10
  EXPECT_NEAR(y.at(0, 1), 0.0f, 1e-5f);  // (10-10)/10
  EXPECT_NEAR(y.at(1, 0), 0.5f, 1e-5f);  // (5-0)/10
  EXPECT_NEAR(y.at(1, 1), 1.0f, 1e-5f);  // (20-10)/10
}

TEST_F(OpTest, MinMaxScaler_CustomRange) {
  put_const("dmin2", 1, 1, {0.0f});
  put_const("dscale2", 1, 1, {10.0f});
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {5.0f}));
  exec_n(vs, 1, "omle.feature", "MinMaxScaler", {"x"}, {"y"},
         {{"data_min", atensor("dmin2")},
          {"data_max", atensor("dscale2")},
          {"feature_range_min", aflt(-1.0)},
          {"feature_range_max", aflt(1.0)}});
  // (5-0)/10 * (1-(-1)) + (-1) = 0.5*2 - 1 = 0.0
  EXPECT_NEAR(vs.get("y").at(0, 0), 0.0f, 1e-5f);
}

TEST_F(OpTest, RobustScaler) {
  put_const("center", 1, 2, {5.0f, 0.0f});
  put_const("rscale", 1, 2, {2.0f, 4.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {7.0f, 8.0f, 3.0f, -4.0f}));
  exec_n(vs, 2, "omle.feature", "RobustScaler", {"x"}, {"y"},
         {{"center", atensor("center")}, {"scale", atensor("rscale")}});
  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 1.0f, 1e-5f);   // (7-5)/2
  EXPECT_NEAR(y.at(0, 1), 2.0f, 1e-5f);   // (8-0)/4
  EXPECT_NEAR(y.at(1, 0), -1.0f, 1e-5f);  // (3-5)/2
  EXPECT_NEAR(y.at(1, 1), -1.0f, 1e-5f);  // (-4-0)/4
}

TEST_F(OpTest, MaxAbsScaler) {
  put_const("mabs", 1, 2, {4.0f, 8.0f});
  auto vs = make_vs();
  vs.put("x", Tf(1, 2, {2.0f, -4.0f}));
  exec_n(vs, 1, "omle.feature", "MaxAbsScaler", {"x"}, {"y"},
         {{"scale", atensor("mabs")}});
  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 0.5f, 1e-5f);   // 2/4
  EXPECT_NEAR(y.at(0, 1), -0.5f, 1e-5f);  // -4/8
}

TEST_F(OpTest, Normalizer_L1) {
  auto vs = make_vs();
  vs.put("x", Tf(1, 3, {1.0f, 2.0f, 3.0f}));
  exec_n(vs, 1, "omle.feature", "Normalizer", {"x"}, {"y"},
         {{"norm", astr("l1")}});
  // sum = 6; [1/6, 2/6, 3/6]
  EXPECT_NEAR(vs.get("y").at(0, 0), 1.0f / 6.0f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 1), 2.0f / 6.0f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 2), 3.0f / 6.0f, 1e-5f);
}

TEST_F(OpTest, Normalizer_Max) {
  auto vs = make_vs();
  vs.put("x", Tf(1, 3, {2.0f, 5.0f, 3.0f}));
  exec_n(vs, 1, "omle.feature", "Normalizer", {"x"}, {"y"},
         {{"norm", astr("max")}});
  // max = 5; [2/5, 5/5, 3/5]
  EXPECT_NEAR(vs.get("y").at(0, 0), 0.4f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 1), 1.0f, 1e-5f);
  EXPECT_NEAR(vs.get("y").at(0, 2), 0.6f, 1e-5f);
}

// =============================================================================
// omle.feature — transformers
// =============================================================================

TEST_F(OpTest, PowerTransformer_YeoJohnson_Positive) {
  // For x >= 0, lambda=2: ((x+1)^2 - 1) / 2
  put_const("pt_lam", 1, 1, {2.0f});
  put_const("pt_mean", 1, 1, {0.0f});
  put_const("pt_scale", 1, 1, {1.0f});
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {3.0f}));
  exec_n(vs, 1, "omle.feature", "PowerTransformer", {"x"}, {"y"},
         {{"lambdas", atensor("pt_lam")},
          {"mean", atensor("pt_mean")},
          {"scale", atensor("pt_scale")},
          {"method", astr("yeo_johnson")},
          {"standardize", abool(false)}});
  // ((3+1)^2 - 1)/2 = (16-1)/2 = 7.5
  EXPECT_NEAR(vs.get("y").at(0, 0), 7.5f, 1e-4f);
}

TEST_F(OpTest, PowerTransformer_BoxCox) {
  // For x>0, lambda=0: log(x)
  put_const("bc_lam", 1, 1, {0.0f});
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {std::exp(1.0f)}));
  exec_n(vs, 1, "omle.feature", "PowerTransformer", {"x"}, {"y"},
         {{"lambdas", atensor("bc_lam")},
          {"method", astr("box_cox")},
          {"standardize", abool(false)}});
  EXPECT_NEAR(vs.get("y").at(0, 0), 1.0f, 1e-4f);  // log(e) = 1
}

TEST_F(OpTest, QuantileTransformer_Uniform) {
  // Quantile levels: [0, 0.5, 1.0], references for 1 feature: [0, 5, 10]
  put_const("qt_q", 1, 3, {0.0f, 0.5f, 1.0f});
  put_const("qt_ref", 3, 1, {0.0f, 5.0f, 10.0f});  // [nq, nf] layout
  auto vs = make_vs();
  vs.put("x", Tf(3, 1, {0.0f, 5.0f, 10.0f}));
  exec_n(vs, 3, "omle.feature", "QuantileTransformer", {"x"}, {"y"},
         {{"quantiles", atensor("qt_q")}, {"references", atensor("qt_ref")}});
  EXPECT_NEAR(vs.get("y").at(0, 0), 0.0f, 1e-4f);
  EXPECT_NEAR(vs.get("y").at(1, 0), 0.5f, 1e-4f);
  EXPECT_NEAR(vs.get("y").at(2, 0), 1.0f, 1e-4f);
}

TEST_F(OpTest, PolynomialFeatures_Degree2_NoBias) {
  // 2 features [a,b], degree 2, no bias → [a, b, a^2, ab, b^2]
  auto vs = make_vs();
  vs.put("x", Tf(1, 2, {2.0f, 3.0f}));
  exec_n(vs, 1, "omle.feature", "PolynomialFeatures", {"x"}, {"y"},
         {{"max_degree", aint({2})}, {"include_bias", abool(false)}});
  const auto& y = vs.get("y");
  // Expected: 2, 3, 4, 6, 9
  std::vector<float> got;
  for (int c = 0; c < y.n_cols; ++c) got.push_back(y.at(0, c));
  std::sort(got.begin(), got.end());
  std::vector<float> exp = {2.0f, 3.0f, 4.0f, 6.0f, 9.0f};
  std::sort(exp.begin(), exp.end());
  ASSERT_EQ(got.size(), exp.size());
  for (size_t i = 0; i < got.size(); ++i) EXPECT_NEAR(got[i], exp[i], 1e-4f);
}

TEST_F(OpTest, Bucketizer) {
  auto vs = make_vs();
  vs.put("x", Tf(5, 1, {-1.0f, 0.0f, 1.5f, 3.0f, 5.0f}));
  exec_n(vs, 5, "omle.feature", "Bucketizer", {"x"}, {"y"},
         {{"boundaries", afloats({0.0, 2.0, 4.0})}});
  // boundaries: (-inf,0], (0,2], (2,4], (4,+inf) → buckets 0,1,2,3
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);  // -1  < 0
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);  // 0 not > 0 → bucket 1
  EXPECT_FLOAT_EQ(y.at(2, 0), 1.0f);  // 1.5 <= 2
  EXPECT_FLOAT_EQ(y.at(3, 0), 2.0f);  // 3 in (2,4]
  EXPECT_FLOAT_EQ(y.at(4, 0), 3.0f);  // 5 > 4
}

// =============================================================================
// omle.feature — imputers
// =============================================================================

TEST_F(OpTest, SimpleImputer_FillTensor) {
  put_const("stats", 1, 2, {99.0f, -1.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {1.0f, kNaN, kNaN, 5.0f}));
  exec_n(vs, 2, "omle.feature", "SimpleImputer", {"x"}, {"y"},
         {{"fill_tensor", atensor("stats")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), -1.0f);  // NaN → stats[1]
  EXPECT_FLOAT_EQ(y.at(1, 0), 99.0f);  // NaN → stats[0]
  EXPECT_FLOAT_EQ(y.at(1, 1), 5.0f);
}

TEST_F(OpTest, MissingIndicator_AllCols) {
  auto vs = make_vs();
  vs.put("x", Tf(2, 3, {1.0f, kNaN, 3.0f, kNaN, 2.0f, kNaN}));
  exec_n(vs, 2, "omle.feature", "MissingIndicator", {"x"}, {"y"});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 3);
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 2), 1.0f);
}

TEST_F(OpTest, MissingIndicator_SelectedCols) {
  put_const("fi", 1, 2, {0.0f, 2.0f});  // indicate columns 0 and 2
  auto vs = make_vs();
  vs.put("x", Tf(1, 3, {kNaN, kNaN, 3.0f}));
  exec_n(vs, 1, "omle.feature", "MissingIndicator", {"x"}, {"y"},
         {{"feature_indices", atensor("fi")}});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 2);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);  // col 0 is NaN
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);  // col 2 is 3.0
}

// =============================================================================
// omle.feature — encoders
// =============================================================================

TEST_F(OpTest, OrdinalEncoder_NumericCategories) {
  // categories: [10, 20, 30] → input 20 → ordinal 1
  put_const("oe_cats", 1, 3, {10.0f, 20.0f, 30.0f});
  put_const("oe_offsets", 1, 2, {0.0f, 3.0f});
  auto vs = make_vs();
  vs.put("x", Tf(3, 1, {10.0f, 20.0f, 30.0f}));
  exec_n(vs, 3, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
         {{"categories", atensor("oe_cats")},
          {"category_offsets", atensor("oe_offsets")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 2.0f);
}

TEST_F(OpTest, OrdinalEncoder_StringInput) {
  // String tensor input: "dog" is at index 1 in ["cat","dog","fish"] → 1.0
  put_const_str("oe_cats", {"cat", "dog", "fish"});
  put_const("oe_offsets", 1, 2, {0.0f, 3.0f});
  auto vs = make_vs();
  vs.put("x", Tstr(3, 1, {"cat", "dog", "fish"}));
  exec_n(vs, 3, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
         {{"categories", atensor("oe_cats")},
          {"category_offsets", atensor("oe_offsets")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 2.0f);
}

TEST_F(OpTest, OrdinalEncoder_StringInput_MultiFeature) {
  // 2 string features: cats_f0=["a","b"], cats_f1=["x","y","z"],
  // offsets=[0,2,5]
  put_const_str("oe_cats", {"a", "b", "x", "y", "z"});
  put_const("oe_offsets", 1, 3, {0.0f, 2.0f, 5.0f});
  auto vs = make_vs();
  vs.put("x", Tstr(2, 2, {"b", "z", "a", "x"}));
  exec_n(vs, 2, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
         {{"categories", atensor("oe_cats")},
          {"category_offsets", atensor("oe_offsets")}});
  // f0: cats[0..2)=["a","b"], f1: cats[2..5)=["x","y","z"]
  // row0: f0="b"→1, f1="z"→2
  // row1: f0="a"→0, f1="x"→0
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 1.0f);  // "b" → 1
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 1), 2.0f);  // "z" → 2
  EXPECT_FLOAT_EQ(vs.get("y").at(1, 0), 0.0f);  // "a" → 0
  EXPECT_FLOAT_EQ(vs.get("y").at(1, 1), 0.0f);  // "x" → 0
}

TEST_F(OpTest, OrdinalEncoder_StringCategories) {
  // Numeric float input with string categories (non-numeric → NaN-encoded):
  // input 0/1/2 matches by ordinal index (NaN match path).
  put_const_str("oe_cats", {"cat", "dog", "fish"});
  put_const("oe_offsets", 1, 2, {0.0f, 3.0f});
  auto vs = make_vs();
  vs.put("x", Tf(3, 1, {0.0f, 1.0f, 2.0f}));
  exec_n(vs, 3, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
         {{"categories", atensor("oe_cats")},
          {"category_offsets", atensor("oe_offsets")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 2.0f);
}

TEST_F(OpTest, OrdinalEncoder_MultiFeature) {
  // 2 features: cats_f0=[1,2], cats_f1=[10,20], offsets=[0,2,4]
  put_const("oe_cats", 1, 4, {1.0f, 2.0f, 10.0f, 20.0f});
  put_const("oe_offsets", 1, 3, {0.0f, 2.0f, 4.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {1.0f, 20.0f, 2.0f, 10.0f}));
  exec_n(vs, 2, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
         {{"categories", atensor("oe_cats")},
          {"category_offsets", atensor("oe_offsets")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);  // 1 → cat[0] in f0
  EXPECT_FLOAT_EQ(y.at(0, 1), 1.0f);  // 20 → cat[1] in f1
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);  // 2 → cat[1] in f0
  EXPECT_FLOAT_EQ(y.at(1, 1), 0.0f);  // 10 → cat[0] in f1
}

TEST_F(OpTest, OrdinalEncoder_Unknown_Returns_NaN) {
  put_const("oe_cats", 1, 3, {1.0f, 2.0f, 3.0f});
  put_const("oe_offsets", 1, 2, {0.0f, 3.0f});
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {99.0f}));  // 99 not in categories
  exec_n(vs, 1, "omle.feature", "OrdinalEncoder", {"x"}, {"y"},
         {{"categories", atensor("oe_cats")},
          {"category_offsets", atensor("oe_offsets")}});
  EXPECT_TRUE(std::isnan(vs.get("y").at(0, 0)));
}

TEST_F(OpTest, LabelEncoder_String) {
  // String tensor input: direct string→index lookup against labels list.
  // "C" is at index 2 in ["A","B","C"], so it returns 2.0 — not identity.
  put_const_str("le_labels", {"A", "B", "C"});
  put_const("le_offsets", 1, 2, {0.0f, 3.0f});
  auto vs = make_vs();
  vs.put("x", Tstr(3, 1, {"A", "C", "B"}));
  exec_n(vs, 3, "omle.feature", "LabelEncoder", {"x"}, {"y"},
         {{"labels", atensor("le_labels")},
          {"label_offsets", atensor("le_offsets")}});
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 0.0f);  // "A" → 0
  EXPECT_FLOAT_EQ(vs.get("y").at(1, 0), 2.0f);  // "C" → 2
  EXPECT_FLOAT_EQ(vs.get("y").at(2, 0), 1.0f);  // "B" → 1
}

TEST_F(OpTest, LabelEncoder_Numeric) {
  // Numeric float input: labels parsed as floats, matched by value.
  // labels=["1","3","5"]: input 3.0 → index 1 (not 3)
  put_const_str("le_labels", {"1", "3", "5"});
  put_const("le_offsets", 1, 2, {0.0f, 3.0f});
  auto vs = make_vs();
  vs.put("x", Tf(3, 1, {1.0f, 3.0f, 5.0f}));
  exec_n(vs, 3, "omle.feature", "LabelEncoder", {"x"}, {"y"},
         {{"labels", atensor("le_labels")},
          {"label_offsets", atensor("le_offsets")}});
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 0.0f);  // 1.0 → index 0
  EXPECT_FLOAT_EQ(vs.get("y").at(1, 0), 1.0f);  // 3.0 → index 1
  EXPECT_FLOAT_EQ(vs.get("y").at(2, 0), 2.0f);  // 5.0 → index 2
}

TEST_F(OpTest, LabelBinarizer_Binary) {
  // No classes → threshold at 0.5
  auto vs = make_vs();
  vs.put("x", Tf(3, 1, {0.0f, 0.8f, 0.3f}));
  exec_n(vs, 3, "omle.feature", "LabelBinarizer", {"x"}, {"y"});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);
}

TEST_F(OpTest, LabelBinarizer_Multiclass) {
  put_const("classes", 1, 3, {0.0f, 1.0f, 2.0f});
  auto vs = make_vs();
  vs.put("x", Tf(3, 1, {0.0f, 2.0f, 1.0f}));
  exec_n(vs, 3, "omle.feature", "LabelBinarizer", {"x", "classes"}, {"y"});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 3);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 2), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(2, 1), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 2), 0.0f);
}

TEST_F(OpTest, MultiLabelBinarizer) {
  put_const("mlb_classes", 1, 3, {10.0f, 20.0f, 30.0f});
  auto vs = make_vs();
  // Row 0 has labels {10, 30}, row 1 has {20}
  vs.put("x", Tf(2, 2, {10.0f, 30.0f, 20.0f, -1.0f}));  // -1 = no match
  exec_n(vs, 2, "omle.feature", "MultiLabelBinarizer", {"x", "mlb_classes"},
         {"y"});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 3);
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 2), 0.0f);
}

TEST_F(OpTest, TargetEncoder) {
  // 1 feature, 3 categories
  // cats_flat: [1,2,3], enc_flat: [10,20,30], default: [15]
  put_const("te_cats", 1, 3, {1.0f, 2.0f, 3.0f});
  put_const("te_enc", 1, 3, {10.0f, 20.0f, 30.0f});
  put_const("te_def", 1, 1, {15.0f});
  auto vs = make_vs();
  vs.put("x", Tf(4, 1, {1.0f, 3.0f, 2.0f, 99.0f}));
  exec_n(vs, 4, "omle.feature", "TargetEncoder", {"x"}, {"y"},
         {{"categories", atensor("te_cats")},
          {"encoded_values", atensor("te_enc")},
          {"default_values", atensor("te_def")}});
  const auto& y = vs.get("y");
  EXPECT_FLOAT_EQ(y.at(0, 0), 10.0f);  // cat 1 → 10
  EXPECT_FLOAT_EQ(y.at(1, 0), 30.0f);  // cat 3 → 30
  EXPECT_FLOAT_EQ(y.at(2, 0), 20.0f);  // cat 2 → 20
  EXPECT_FLOAT_EQ(y.at(3, 0), 15.0f);  // unknown → default
}

// =============================================================================
// omle.feature — dimensionality reduction
// =============================================================================

TEST_F(OpTest, TruncatedSVD) {
  // components: [2 components, 3 features]
  // [[1,0,0],[0,1,0]] → project to 2D, dropping 3rd feature
  put_const("svd_comp", 2, 3, {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 3, {3.0f, 5.0f, 7.0f, 1.0f, 2.0f, 9.0f}));
  exec_n(vs, 2, "omle.feature", "TruncatedSVD", {"x"}, {"y"},
         {{"components", atensor("svd_comp")}});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 2);
  EXPECT_NEAR(y.at(0, 0), 3.0f, 1e-4f);
  EXPECT_NEAR(y.at(0, 1), 5.0f, 1e-4f);
  EXPECT_NEAR(y.at(1, 0), 1.0f, 1e-4f);
  EXPECT_NEAR(y.at(1, 1), 2.0f, 1e-4f);
}

TEST_F(OpTest, FactorAnalysis) {
  // Y = (X - mean) @ components.T
  put_const("fa_mean", 1, 2, {1.0f, 2.0f});
  put_const("fa_comp", 1, 2, {1.0f, 0.0f});  // 1 component, 2 features
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {3.0f, 5.0f, 1.0f, 2.0f}));
  exec_n(vs, 2, "omle.feature", "FactorAnalysis", {"x"}, {"y"},
         {{"mean", atensor("fa_mean")}, {"components", atensor("fa_comp")}});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 1);
  EXPECT_NEAR(y.at(0, 0), 2.0f, 1e-4f);  // (3-1)*1+(5-2)*0 = 2
  EXPECT_NEAR(y.at(1, 0), 0.0f, 1e-4f);  // (1-1)*1+(2-2)*0 = 0
}

TEST_F(OpTest, NMF) {
  // Y = max(0, X @ components.T); components: [1 comp, 2 feats]
  put_const("nmf_comp", 1, 2, {1.0f, 2.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {1.0f, 1.0f, -1.0f, -1.0f}));
  exec_n(vs, 2, "omle.feature", "NMF", {"x"}, {"y"},
         {{"components", atensor("nmf_comp")}});
  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 3.0f, 1e-4f);  // max(0, 1*1+1*2) = 3
  EXPECT_NEAR(y.at(1, 0), 0.0f, 1e-4f);  // max(0, -1-2) = 0
}

TEST_F(OpTest, LDA) {
  // LDA: Y = softmax(X @ components.T)
  // 1 component, 2 features, with a very large value to test softmax
  put_const("lda_comp", 2, 2, {1.0f, 0.0f, 0.0f, 1.0f});  // identity
  auto vs = make_vs();
  vs.put("x", Tf(1, 2, {100.0f, 0.0f}));
  exec_n(vs, 1, "omle.feature", "LatentDirichletAllocation", {"x"}, {"y"},
         {{"components", atensor("lda_comp")}});
  const auto& y = vs.get("y");
  // Softmax([100,0]) ≈ [1.0, 0.0]
  EXPECT_NEAR(y.at(0, 0), 1.0f, 1e-3f);
  EXPECT_NEAR(y.at(0, 1), 0.0f, 1e-3f);
}

TEST_F(OpTest, KernelPCA_RBF) {
  // 1 training sample [0,0], 1 component
  // Query [0,0]: K = exp(0) = 1; K_centered ~ 0; result ~ 0 anyway
  put_const("kpca_train", 1, 2, {0.0f, 0.0f});
  put_const("kpca_dual", 1, 1, {1.0f});  // [1 comp, 1 train sample]
  auto vs = make_vs();
  vs.put("x", Tf(1, 2, {0.0f, 0.0f}));
  exec_n(vs, 1, "omle.feature", "KernelPCA", {"x"}, {"y"},
         {{"fit_samples", atensor("kpca_train")},
          {"dual_components", atensor("kpca_dual")},
          {"kernel", astr("rbf")},
          {"gamma", aflt(1.0)}});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 1);
  EXPECT_NEAR(y.at(0, 0), 1.0f, 1e-4f);  // K(0,0)=1, dual=1 → 1*1=1
}

TEST_F(OpTest, SplineTransformer_Linear) {
  // degree=1, augmented knots=[0,0,1,2,2] (boundary knots repeated degree
  // times). n_t=5, n_basis=5-1-1=3, n_out=3 (include_bias=true)
  put_const("spline_knots", 5, 1, {0.0f, 0.0f, 1.0f, 2.0f, 2.0f});
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {1.5f}));
  exec_n(vs, 1, "omle.feature", "SplineTransformer", {"x"}, {"y"},
         {{"knots", atensor("spline_knots")},
          {"degree", aint({1})},
          {"include_bias", abool(true)}});
  const auto& y = vs.get("y");
  // 3 basis functions, sum should be 1 at a valid interior point
  float bsum = 0.0f;
  for (int c = 0; c < y.n_cols; ++c) bsum += y.at(0, c);
  EXPECT_NEAR(bsum, 1.0f, 1e-4f);
}

// =============================================================================
// OHE with multi-feature (2 features)
// =============================================================================

TEST_F(OpTest, OneHotEncoder_MultiFeature) {
  // f0 has 3 categories, f1 has 2 categories; offsets=[0,3,5]
  put_const("ohe_cats", 1, 5, {0.0f, 1.0f, 2.0f, 0.0f, 1.0f});
  put_const("ohe_offsets", 1, 3, {0.0f, 3.0f, 5.0f});
  auto vs = make_vs();
  vs.put("x", Tf(2, 2, {0.0f, 1.0f, 2.0f, 0.0f}));
  exec_n(vs, 2, "omle.feature", "OneHotEncoder", {"x"}, {"y"},
         {{"categories", atensor("ohe_cats")},
          {"category_offsets", atensor("ohe_offsets")}});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 5);
  // Row 0: f0=0 → [1,0,0], f1=1 → [0,1]
  EXPECT_FLOAT_EQ(y.at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 2), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 3), 0.0f);
  EXPECT_FLOAT_EQ(y.at(0, 4), 1.0f);
  // Row 1: f0=2 → [0,0,1], f1=0 → [1,0]
  EXPECT_FLOAT_EQ(y.at(1, 0), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 1), 0.0f);
  EXPECT_FLOAT_EQ(y.at(1, 2), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 3), 1.0f);
  EXPECT_FLOAT_EQ(y.at(1, 4), 0.0f);
}

// -----------------------------------------------------------------------
// omle.text — elementwise string transforms
// -----------------------------------------------------------------------

namespace {

// Build a [n,1] string tensor, the rank-1 shape these operators declare.
Tensor str_col(const std::vector<std::string>& v) {
  Tensor t = Tensor::strings(static_cast<int>(v.size()), 1);
  for (int i = 0; i < static_cast<int>(v.size()); ++i) t.str_at(i, 0) = v[i];
  return t;
}

AttrVal s_attr(const std::string& v) {
  AttrVal a;
  a.kind = AttrVal::Kind::String;
  a.s = v;
  return a;
}

AttrVal b_attr(bool v) {
  AttrVal a;
  a.kind = AttrVal::Kind::Bool;
  a.b = v;
  return a;
}

}  // namespace

TEST_F(OpTest, TextLowercase) {
  auto vs = make_vs();
  vs.put("x", str_col({"Hello World", "ALL CAPS", "already"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "Lowercase";
  node.in_names = {"x"};
  node.out_names = {"y"};
  ASSERT_TRUE(node.execute(vs, 3).ok());

  const Tensor& y = vs.get("y");
  ASSERT_EQ(y.n_rows, 3);
  ASSERT_EQ(y.n_cols, 1);  // same_shape(x)
  EXPECT_EQ(y.str_at(0, 0), "hello world");
  EXPECT_EQ(y.str_at(1, 0), "all caps");
  EXPECT_EQ(y.str_at(2, 0), "already");
}

TEST_F(OpTest, TextTrim) {
  auto vs = make_vs();
  vs.put("x", str_col({"  padded  ", "\t tabs \n", "none", "   "}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "Trim";
  node.in_names = {"x"};
  node.out_names = {"y"};
  ASSERT_TRUE(node.execute(vs, 4).ok());

  const Tensor& y = vs.get("y");
  EXPECT_EQ(y.str_at(0, 0), "padded");
  EXPECT_EQ(y.str_at(1, 0), "tabs");
  EXPECT_EQ(y.str_at(2, 0), "none");
  EXPECT_EQ(y.str_at(3, 0), "");  // all-whitespace collapses to empty
}

TEST_F(OpTest, TextNormalizeWhitespace) {
  auto vs = make_vs();
  vs.put("x", str_col({"a   b\t\tc", "  lead and trail  ", "single space"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "NormalizeWhitespace";
  node.in_names = {"x"};
  node.out_names = {"y"};
  ASSERT_TRUE(node.execute(vs, 3).ok());

  const Tensor& y = vs.get("y");
  EXPECT_EQ(y.str_at(0, 0), "a b c");
  // Runs collapse but are not removed — Trim is the operator that removes them.
  EXPECT_EQ(y.str_at(1, 0), " lead and trail ");
  EXPECT_EQ(y.str_at(2, 0), "single space");
}

TEST_F(OpTest, TextStringReplaceAll) {
  auto vs = make_vs();
  vs.put("x", str_col({"a-b-c", "no dashes"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "StringReplace";
  node.in_names = {"x"};
  node.out_names = {"y"};
  node.attrs["old"] = s_attr("-");
  node.attrs["new"] = s_attr("+");
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& y = vs.get("y");
  EXPECT_EQ(y.str_at(0, 0), "a+b+c");  // replace_all defaults to true
  EXPECT_EQ(y.str_at(1, 0), "no dashes");
}

TEST_F(OpTest, TextStringReplaceFirstOnly) {
  auto vs = make_vs();
  vs.put("x", str_col({"a-b-c"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "StringReplace";
  node.in_names = {"x"};
  node.out_names = {"y"};
  node.attrs["old"] = s_attr("-");
  node.attrs["new"] = s_attr("+");
  node.attrs["replace_all"] = b_attr(false);
  ASSERT_TRUE(node.execute(vs, 1).ok());

  EXPECT_EQ(vs.get("y").str_at(0, 0), "a+b-c");
}

TEST_F(OpTest, TextStringReplaceEmptyOldIsNoOp) {
  auto vs = make_vs();
  vs.put("x", str_col({"unchanged"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "StringReplace";
  node.in_names = {"x"};
  node.out_names = {"y"};
  node.attrs["old"] = s_attr("");
  node.attrs["new"] = s_attr("X");
  ASSERT_TRUE(node.execute(vs, 1).ok());

  EXPECT_EQ(vs.get("y").str_at(0, 0), "unchanged");
}

TEST_F(OpTest, TextRegexReplace) {
  auto vs = make_vs();
  vs.put("x", str_col({"abc123def456", "no digits"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "RegexReplace";
  node.in_names = {"x"};
  node.out_names = {"y"};
  node.attrs["pattern"] = s_attr("[0-9]+");
  node.attrs["replacement"] = s_attr("#");
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& y = vs.get("y");
  EXPECT_EQ(y.str_at(0, 0), "abc#def#");
  EXPECT_EQ(y.str_at(1, 0), "no digits");
}

TEST_F(OpTest, TextRegexReplaceInvalidPatternFails) {
  auto vs = make_vs();
  vs.put("x", str_col({"anything"}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "RegexReplace";
  node.in_names = {"x"};
  node.out_names = {"y"};
  node.attrs["pattern"] = s_attr("([unclosed");
  node.attrs["replacement"] = s_attr("");
  EXPECT_FALSE(node.execute(vs, 1).ok());
}

TEST_F(OpTest, TextTransformRejectsNumericInput) {
  auto vs = make_vs();
  vs.put("x", Tf(1, 1, {1.0f}));

  OperatorNode node;
  node.domain = "omle.text";
  node.op = "Lowercase";
  node.in_names = {"x"};
  node.out_names = {"y"};
  EXPECT_FALSE(node.execute(vs, 1).ok());
}

// -----------------------------------------------------------------------
// omle.core — SelectByPrimarySecondaryScore
// -----------------------------------------------------------------------

TEST_F(OpTest, SelectByPrimarySecondaryScore) {
  auto vs = make_vs();
  // row 0: primary picks class 2 outright.
  // row 1: classes 0 and 1 tie on primary; secondary favours 1.
  // row 2: classes 0 and 2 tie on both; the smallest index wins.
  vs.put("primary", Tf(3, 3, {1, 2, 5, 4, 4, 0, 7, 1, 7}));
  vs.put("secondary", Tf(3, 3, {9, 9, 0, 0.1f, 0.9f, 0, 0.5f, 9, 0.5f}));

  OperatorNode node;
  node.domain = "omle.core";
  node.op = "SelectByPrimarySecondaryScore";
  node.in_names = {"primary", "secondary"};
  node.out_names = {"y"};
  ASSERT_TRUE(node.execute(vs, 3).ok());

  const Tensor& y = vs.get("y");
  ASSERT_EQ(y.n_cols, 1);
  EXPECT_FLOAT_EQ(y.at(0, 0), 2.0f);
  EXPECT_FLOAT_EQ(y.at(1, 0), 1.0f);
  EXPECT_FLOAT_EQ(y.at(2, 0), 0.0f);
}

TEST_F(OpTest, SelectByPrimarySecondaryScoreShapeMismatchFails) {
  auto vs = make_vs();
  vs.put("primary", Tf(2, 3, {1, 2, 3, 4, 5, 6}));
  vs.put("secondary", Tf(2, 2, {1, 2, 3, 4}));

  OperatorNode node;
  node.domain = "omle.core";
  node.op = "SelectByPrimarySecondaryScore";
  node.in_names = {"primary", "secondary"};
  node.out_names = {"y"};
  EXPECT_FALSE(node.execute(vs, 2).ok());
}
