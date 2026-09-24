#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <limits>

#include "model_schema.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

static constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();

static bool is_nan(float v) {
  uint32_t u;
  std::memcpy(&u, &v, sizeof(u));
  return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

// -----------------------------------------------------------------------
// Helpers to build a schema node and run it
// -----------------------------------------------------------------------

class SchemaTest : public ::testing::Test {
 protected:
  ConstantStore cs;

  // Build a ValueStore with a source tensor and run the schema node.
  // source is [n_rows, n_cols]; features reference columns by index.
  Tensor run(ModelSchemaNode& node, const std::vector<float>& src_data,
             int n_rows, int n_cols, const std::string& src_name = "X") {
    ValueStore vs(cs);
    Tensor src(n_rows, n_cols);
    src.set_floats(src_data);
    vs.put(src_name, std::move(src));
    EXPECT_TRUE(node.execute(vs, n_rows).ok());
    return vs.get(node.features[0].name);  // return first feature's output
  }
};

// -----------------------------------------------------------------------
// Basic extraction
// -----------------------------------------------------------------------

TEST_F(SchemaTest, ExtractColumn) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "age";
  f.source = "X";
  f.col = 1;
  node.features.push_back(f);

  auto out = run(node, {10.f, 20.f, 30.f, 40.f}, 2, 2);
  EXPECT_FLOAT_EQ(out.at(0, 0), 20.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 40.f);
}

TEST_F(SchemaTest, MultipleFeatures) {
  ModelSchemaNode node;
  SchemaFeature f0;
  f0.name = "a";
  f0.source = "X";
  f0.col = 0;
  SchemaFeature f1;
  f1.name = "b";
  f1.source = "X";
  f1.col = 2;
  node.features.push_back(f0);
  node.features.push_back(f1);

  ValueStore vs(cs);
  Tensor src(2, 3);
  src.set_floats({1, 2, 3, 4, 5, 6});
  vs.put("X", std::move(src));
  ASSERT_TRUE(node.execute(vs, 2).ok());

  EXPECT_FLOAT_EQ(vs.get("a").at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(vs.get("a").at(1, 0), 4.f);
  EXPECT_FLOAT_EQ(vs.get("b").at(0, 0), 3.f);
  EXPECT_FLOAT_EQ(vs.get("b").at(1, 0), 6.f);
}

// -----------------------------------------------------------------------
// Missing value policies
// -----------------------------------------------------------------------

TEST_F(SchemaTest, MissingPropagate) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.missing_policy = MissingPolicy::Propagate;
  node.features.push_back(f);

  auto out = run(node, {1.f, kNaN, 3.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 1.f);
  EXPECT_TRUE(is_nan(out.at(1, 0)));
  EXPECT_FLOAT_EQ(out.at(2, 0), 3.f);
}

TEST_F(SchemaTest, MissingAsValue) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.missing_policy = MissingPolicy::AsValue;
  f.missing_replacement = 99.f;
  node.features.push_back(f);

  auto out = run(node, {1.f, kNaN, 3.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 99.f);
  EXPECT_FLOAT_EQ(out.at(2, 0), 3.f);
}

TEST_F(SchemaTest, MissingAsInvalid_ThenReturnInvalid) {
  // missing_policy = AsInvalid, invalid_policy = ReturnInvalid → NaN out
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.missing_policy = MissingPolicy::AsInvalid;
  f.invalid_policy = InvalidPolicy::ReturnInvalid;
  node.features.push_back(f);

  auto out = run(node, {kNaN}, 1, 1);
  EXPECT_TRUE(is_nan(out.at(0, 0)));
}

TEST_F(SchemaTest, MissingAsInvalid_ThenAsValue) {
  // missing_policy = AsInvalid, invalid_policy = AsValue → replacement out
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.missing_policy = MissingPolicy::AsInvalid;
  f.invalid_policy = InvalidPolicy::AsValue;
  f.invalid_replacement = 0.f;
  node.features.push_back(f);

  auto out = run(node, {kNaN}, 1, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 0.f);
}

// -----------------------------------------------------------------------
// Discrete domain: invalid / valid-set checks
// -----------------------------------------------------------------------

TEST_F(SchemaTest, DiscreteInvalidValue_ReturnInvalid) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.invalid_policy = InvalidPolicy::ReturnInvalid;
  f.domain.kind = CompiledDomain::Kind::Discrete;
  f.domain.invalid_set = {-1.f};
  node.features.push_back(f);

  auto out = run(node, {5.f, -1.f, 3.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 5.f);
  EXPECT_TRUE(is_nan(out.at(1, 0)));
  EXPECT_FLOAT_EQ(out.at(2, 0), 3.f);
}

TEST_F(SchemaTest, DiscreteInvalidValue_AsValue) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.invalid_policy = InvalidPolicy::AsValue;
  f.invalid_replacement = 0.f;
  f.domain.kind = CompiledDomain::Kind::Discrete;
  f.domain.invalid_set = {-1.f};
  node.features.push_back(f);

  auto out = run(node, {5.f, -1.f}, 2, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 5.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 0.f);
}

TEST_F(SchemaTest, DiscreteValidSet_ClosedWorld) {
  // valid_set = {1, 2, 3}; value 4 is not listed → invalid
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.invalid_policy = InvalidPolicy::AsValue;
  f.invalid_replacement = -1.f;
  f.domain.kind = CompiledDomain::Kind::Discrete;
  f.domain.valid_set = {1.f, 2.f, 3.f};
  node.features.push_back(f);

  auto out = run(node, {1.f, 3.f, 4.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 3.f);
  EXPECT_FLOAT_EQ(out.at(2, 0), -1.f);  // 4 not in valid_set
}

TEST_F(SchemaTest, DiscreteMissingValue_AsValue) {
  // Discrete domain declares 0 as a missing marker; replaced via
  // missing_policy.
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.missing_policy = MissingPolicy::AsValue;
  f.missing_replacement = 99.f;
  f.domain.kind = CompiledDomain::Kind::Discrete;
  f.domain.missing_set = {0.f};
  node.features.push_back(f);

  auto out = run(node, {1.f, 0.f, 2.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 99.f);
  EXPECT_FLOAT_EQ(out.at(2, 0), 2.f);
}

// -----------------------------------------------------------------------
// Continuous domain: outlier policies
// -----------------------------------------------------------------------

TEST_F(SchemaTest, ContinuousOutlier_AsIs) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.outlier_policy = OutlierPolicy::AsIs;
  f.domain.kind = CompiledDomain::Kind::Continuous;
  // interval [0, 10] closed-closed
  CompiledInterval iv;
  iv.left = 0.f;
  iv.right = 10.f;
  iv.left_open = false;
  iv.right_open = false;
  f.domain.intervals.push_back(iv);
  node.features.push_back(f);

  auto out = run(node, {5.f, -5.f, 15.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 5.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), -5.f);  // outlier but AsIs
  EXPECT_FLOAT_EQ(out.at(2, 0), 15.f);
}

TEST_F(SchemaTest, ContinuousOutlier_AsMissing) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.outlier_policy = OutlierPolicy::AsMissing;
  f.domain.kind = CompiledDomain::Kind::Continuous;
  CompiledInterval iv;
  iv.left = 0.f;
  iv.right = 10.f;
  iv.left_open = false;
  iv.right_open = false;
  f.domain.intervals.push_back(iv);
  node.features.push_back(f);

  auto out = run(node, {5.f, -5.f, 15.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 5.f);
  EXPECT_TRUE(is_nan(out.at(1, 0)));  // -5 is outlier → NaN
  EXPECT_TRUE(is_nan(out.at(2, 0)));  // 15 is outlier → NaN
}

TEST_F(SchemaTest, ContinuousOutlier_AsExtreme) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.outlier_policy = OutlierPolicy::AsExtreme;
  f.domain.kind = CompiledDomain::Kind::Continuous;
  CompiledInterval iv;
  iv.left = 0.f;
  iv.right = 10.f;
  iv.left_open = false;
  iv.right_open = false;
  f.domain.intervals.push_back(iv);
  node.features.push_back(f);

  auto out = run(node, {5.f, -5.f, 15.f}, 3, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), 5.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 0.f);   // clamped to left=0
  EXPECT_FLOAT_EQ(out.at(2, 0), 10.f);  // clamped to right=10
}

TEST_F(SchemaTest, ContinuousOpenInterval_BoundaryExcluded) {
  // interval (0, 10) open-open; value 0 and 10 are outliers
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.outlier_policy = OutlierPolicy::AsMissing;
  f.domain.kind = CompiledDomain::Kind::Continuous;
  CompiledInterval iv;
  iv.left = 0.f;
  iv.right = 10.f;
  iv.left_open = true;
  iv.right_open = true;
  f.domain.intervals.push_back(iv);
  node.features.push_back(f);

  auto out = run(node, {0.f, 5.f, 10.f}, 3, 1);
  EXPECT_TRUE(is_nan(out.at(0, 0)));  // 0 excluded by open left
  EXPECT_FLOAT_EQ(out.at(1, 0), 5.f);
  EXPECT_TRUE(is_nan(out.at(2, 0)));  // 10 excluded by open right
}

TEST_F(SchemaTest, MultipleIntervals_DisjointRange) {
  // Two valid intervals: [-∞,0] and [5,∞]; values in (0,5) are outliers.
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.outlier_policy = OutlierPolicy::AsMissing;
  f.domain.kind = CompiledDomain::Kind::Continuous;
  CompiledInterval left_iv;
  left_iv.left = -std::numeric_limits<float>::infinity();
  left_iv.left_open = true;
  left_iv.right = 0.f;
  left_iv.right_open = false;
  CompiledInterval right_iv;
  right_iv.left = 5.f;
  right_iv.left_open = false;
  right_iv.right = std::numeric_limits<float>::infinity();
  right_iv.right_open = true;
  f.domain.intervals.push_back(left_iv);
  f.domain.intervals.push_back(right_iv);
  node.features.push_back(f);

  auto out = run(node, {-10.f, 0.f, 2.5f, 5.f, 100.f}, 5, 1);
  EXPECT_FLOAT_EQ(out.at(0, 0), -10.f);
  EXPECT_FLOAT_EQ(out.at(1, 0), 0.f);
  EXPECT_TRUE(is_nan(out.at(2, 0)));  // 2.5 between 0 and 5 → outlier
  EXPECT_FLOAT_EQ(out.at(3, 0), 5.f);
  EXPECT_FLOAT_EQ(out.at(4, 0), 100.f);
}

// -----------------------------------------------------------------------
// Different source inputs per feature
// -----------------------------------------------------------------------

TEST_F(SchemaTest, FeaturesFromDifferentSources) {
  ModelSchemaNode node;
  SchemaFeature fa;
  fa.name = "a";
  fa.source = "col_a";
  fa.col = 0;
  fa.missing_policy = MissingPolicy::AsValue;
  fa.missing_replacement = -1.f;
  SchemaFeature fb;
  fb.name = "b";
  fb.source = "col_b";
  fb.col = 0;
  node.features.push_back(fa);
  node.features.push_back(fb);

  ValueStore vs(cs);
  Tensor ta(2, 1);
  ta.set_floats({kNaN, 5.f});
  Tensor tb(2, 1);
  tb.set_floats({10.f, 20.f});
  vs.put("col_a", std::move(ta));
  vs.put("col_b", std::move(tb));
  ASSERT_TRUE(node.execute(vs, 2).ok());

  EXPECT_FLOAT_EQ(vs.get("a").at(0, 0), -1.f);  // NaN replaced
  EXPECT_FLOAT_EQ(vs.get("a").at(1, 0), 5.f);
  EXPECT_FLOAT_EQ(vs.get("b").at(0, 0), 10.f);
  EXPECT_FLOAT_EQ(vs.get("b").at(1, 0), 20.f);
}

// -----------------------------------------------------------------------
// Float64 inputs
//
// The schema node is the first node to run and it touches every feature, so
// reading a Float64 source through the float32 accessors reinterprets double
// storage as float and corrupts the whole model. These cover both the
// per-feature path and the batch path, and pin the precision that survives:
// a value the schema leaves alone must come back undegraded, which is what
// keeps a scaled feature on the correct side of a tree split threshold.
// -----------------------------------------------------------------------

namespace {

Tensor f64_source(int n_rows, int n_cols, const std::vector<double>& vals) {
  Tensor t =
      omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows, n_cols);
  double* p = t.f64_ptr();
  for (int i = 0; i < n_rows * n_cols; ++i) p[i] = vals[i];
  return t;
}

}  // namespace

TEST_F(SchemaTest, Float64SourceIsNotReadAsFloat32) {
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  node.features.push_back(f);

  ValueStore vs(cs);
  vs.put("X", f64_source(2, 1, {1.0, 2.0}));
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& out = vs.get("x");
  ASSERT_EQ(out.dtype, omle::rt::DataType::Float64);
  EXPECT_DOUBLE_EQ(out.f64_at(0, 0), 1.0);
  EXPECT_DOUBLE_EQ(out.f64_at(1, 0), 2.0);
}

TEST_F(SchemaTest, Float64KeepsPrecisionFloat32WouldLose) {
  // 1.118033988749895 is sqrt(1.25) — a StandardScaler scale in the wild.
  // float32 rounds it to 1.1180340051651001, and that ulp is enough to put a
  // scaled value on the wrong side of a split threshold.
  const double exact = 1.118033988749895;
  ASSERT_NE(exact, static_cast<double>(static_cast<float>(exact)));

  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  node.features.push_back(f);

  ValueStore vs(cs);
  vs.put("X", f64_source(1, 1, {exact}));
  ASSERT_TRUE(node.execute(vs, 1).ok());

  const Tensor& out = vs.get("x");
  ASSERT_EQ(out.dtype, omle::rt::DataType::Float64);
  EXPECT_DOUBLE_EQ(out.f64_at(0, 0), exact);
}

TEST_F(SchemaTest, Float64BatchPathKeepsPrecision) {
  // batch_mode: every feature shares one source and sits in column order, so
  // the node emits a single wide tensor instead of one per feature.
  const double exact = 1.118033988749895;
  ModelSchemaNode node;
  for (int c = 0; c < 2; ++c) {
    SchemaFeature f;
    f.name = (c == 0) ? "a" : "b";
    f.source = "X";
    f.col = c;
    node.features.push_back(f);
  }
  node.batch_mode = true;
  node.batch_key = "X__batch__";

  ValueStore vs(cs);
  vs.put("X", f64_source(1, 2, {exact, 2.5}));
  ASSERT_TRUE(node.execute(vs, 1).ok());

  const Tensor& out = vs.get("X__batch__");
  ASSERT_EQ(out.dtype, omle::rt::DataType::Float64);
  EXPECT_DOUBLE_EQ(out.f64_at(0, 0), exact);
  EXPECT_DOUBLE_EQ(out.f64_at(0, 1), 2.5);
}

TEST_F(SchemaTest, Float64StillAppliesMissingPolicy) {
  // Precision preservation must not bypass the schema: a NaN still takes the
  // missing branch, and the replacement wins over the original value.
  ModelSchemaNode node;
  SchemaFeature f;
  f.name = "x";
  f.source = "X";
  f.col = 0;
  f.missing_policy = MissingPolicy::AsValue;
  f.missing_replacement = -1.f;
  node.features.push_back(f);

  ValueStore vs(cs);
  vs.put("X",
         f64_source(2, 1, {std::numeric_limits<double>::quiet_NaN(), 7.0}));
  ASSERT_TRUE(node.execute(vs, 2).ok());

  const Tensor& out = vs.get("x");
  ASSERT_EQ(out.dtype, omle::rt::DataType::Float64);
  EXPECT_DOUBLE_EQ(out.f64_at(0, 0), -1.0);
  EXPECT_DOUBLE_EQ(out.f64_at(1, 0), 7.0);
}
