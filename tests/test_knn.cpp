// Tests for the nearest-neighbour operators: omle.ml/KNN and
// omle.feature/KNNImputer.
//
// Training sets are deliberately 1-D and widely spaced wherever the expected
// answer depends on neighbour ordering, so that no two distances tie and the
// assertions stay exact rather than depending on sort stability.

#include "op_test_fixture.h"

using namespace omle_test;

namespace {

// KNN needs the training data as attribute-referenced constants; this keeps
// the per-test setup to one line.
AttrVal tensor_attr(std::string name) {
  AttrVal a;
  a.tensor_name = std::move(name);
  return a;
}

}  // namespace

class KnnTest : public OpTest {
 protected:
  // exec_n asserts success; error-path tests need the Status itself.
  omle::rt::Status try_exec(ValueStore& vs, int n_rows, const std::string& op,
                            std::vector<std::string> ins,
                            std::vector<std::string> outs,
                            AttributeMap attrs = {}) {
    OperatorNode node;
    node.domain = "omle.ml";
    node.op = op;
    node.in_names = std::move(ins);
    node.out_names = std::move(outs);
    node.attrs = std::move(attrs);
    return node.execute(vs, n_rows);
  }

  // Registers a 1-D training set as (features, targets) constants.
  void put_training(const std::string& prefix, std::vector<float> xs,
                    std::vector<float> ys) {
    const int n = static_cast<int>(xs.size());
    put_const(prefix + "_X", n, 1, std::move(xs));
    put_const(prefix + "_y", n, 1, std::move(ys));
  }

  AttributeMap knn_attrs(const std::string& prefix, int k,
                         const std::string& task) {
    return {{"train_features", tensor_attr(prefix + "_X")},
            {"train_targets", tensor_attr(prefix + "_y")},
            {"n_neighbors", aint({k})},
            {"task", astr(task)}};
  }
};

// =============================================================================
// omle.ml/KNN — classification
// =============================================================================

TEST_F(KnnTest, ClassificationUniform) {
  // 1D training to avoid tied distances: x=0,1 → class 0; x=5,6 → class 1
  put_training("knn", {0.0f, 1.0f, 5.0f, 6.0f}, {0.0f, 0.0f, 1.0f, 1.0f});

  auto vs = make_vs();
  vs.put("q",
         Tf(1, 1, {0.4f}));  // dist to x=0: 0.4, x=1: 0.6, x=5: 4.6, x=6: 5.6

  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"},
         knn_attrs("knn", 2, "classification"));
  // k=2 nearest: x=0 (class 0) and x=1 (class 0) → majority = 0
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 0.0f);
}

TEST_F(KnnTest, ClassificationDistanceWeighting) {
  put_const("kd_X", 3, 1, {0.0f, 1.0f, 100.0f});
  put_const("kd_y", 3, 1, {1.0f, 2.0f, 2.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.1f}));  // very close to x=0 (class 1)

  auto attrs = knn_attrs("kd", 2, "classification");
  attrs["weights"] = astr("distance");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"}, attrs);
  // k=2: x=0 (dist=0.1, class=1), x=1 (dist=0.9, class=2)
  // weighted: class1 weight=10, class2 weight=1.11 → class 1 wins
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 1.0f);
}

TEST_F(KnnTest, UniformWeightingWouldPickTheOtherClass) {
  // Same data as above: distance weighting is what flips the answer, so the
  // uniform case must genuinely disagree or that test proves nothing.
  put_const("kd_X", 3, 1, {0.0f, 1.0f, 1.5f});
  put_const("kd_y", 3, 1, {1.0f, 2.0f, 2.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.1f}));

  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y_uniform"},
         knn_attrs("kd", 3, "classification"));
  // Uniform: class 2 has two votes to class 1's one.
  EXPECT_FLOAT_EQ(vs.get("y_uniform").at(0, 0), 2.0f);

  auto weighted = knn_attrs("kd", 3, "classification");
  weighted["weights"] = astr("distance");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y_weighted"}, weighted);
  // Weighted: 1/0.1 = 10 beats 1/0.9 + 1/1.4 ≈ 1.82.
  EXPECT_FLOAT_EQ(vs.get("y_weighted").at(0, 0), 1.0f);
}

TEST_F(KnnTest, ClassificationEmitsProbabilities) {
  // Four neighbours, 3 of class 0 and 1 of class 1 → uniform probs 0.75/0.25.
  put_training("kp", {0.0f, 0.1f, 0.2f, 0.3f}, {0.0f, 0.0f, 0.0f, 1.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));

  auto attrs = knn_attrs("kp", 4, "classification");
  attrs["n_classes"] = aint({2});
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y", "prob"}, attrs);

  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 0.0f);
  const auto& p = vs.get("prob");
  ASSERT_EQ(p.n_cols, 2);
  EXPECT_NEAR(p.at(0, 0), 0.75f, 1e-5f);
  EXPECT_NEAR(p.at(0, 1), 0.25f, 1e-5f);
  EXPECT_NEAR(p.at(0, 0) + p.at(0, 1), 1.0f, 1e-5f);
}

TEST_F(KnnTest, ProbabilityColumnsFollowNClasses) {
  // A class index that never appears in the neighbour set still gets a column.
  put_training("kc", {0.0f, 1.0f}, {2.0f, 2.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));

  auto attrs = knn_attrs("kc", 2, "classification");
  attrs["n_classes"] = aint({4});
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y", "prob"}, attrs);

  const auto& p = vs.get("prob");
  ASSERT_EQ(p.n_cols, 4);
  EXPECT_FLOAT_EQ(p.at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(p.at(0, 1), 0.0f);
  EXPECT_FLOAT_EQ(p.at(0, 2), 1.0f);
  EXPECT_FLOAT_EQ(p.at(0, 3), 0.0f);
}

// =============================================================================
// omle.ml/KNN — regression
// =============================================================================

TEST_F(KnnTest, RegressionUniform) {
  // query=0.1: dist to x=0 is 0.1, x=2 is 1.9, x=10 is 9.9 — no ties
  put_training("kr", {0.0f, 2.0f, 10.0f}, {0.0f, 4.0f, 20.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.1f}));

  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"},
         knn_attrs("kr", 2, "regression"));
  // k=2 nearest: x=0 (y=0) and x=2 (y=4) → avg = 2.0
  EXPECT_NEAR(vs.get("y").at(0, 0), 2.0f, 1e-4f);
}

TEST_F(KnnTest, RegressionDistanceWeighting) {
  put_training("krd", {0.0f, 2.0f}, {0.0f, 4.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.5f}));  // dist 0.5 and 1.5 → weights 2 and 2/3

  auto attrs = knn_attrs("krd", 2, "regression");
  attrs["weights"] = astr("distance");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"}, attrs);
  // (2*0 + (2/3)*4) / (2 + 2/3) = (8/3) / (8/3) = 1.0
  EXPECT_NEAR(vs.get("y").at(0, 0), 1.0f, 1e-4f);
}

TEST_F(KnnTest, RegressionExactMatchDominatesDistanceWeighting) {
  // A zero distance would divide by zero; the kernel caps the weight instead,
  // which must make the exact match swamp every other neighbour.
  put_training("kz", {3.0f, 4.0f}, {7.0f, 100.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {3.0f}));

  auto attrs = knn_attrs("kz", 2, "regression");
  attrs["weights"] = astr("distance");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"}, attrs);
  EXPECT_NEAR(vs.get("y").at(0, 0), 7.0f, 1e-3f);
}

TEST_F(KnnTest, MultipleQueryRowsScoreIndependently) {
  put_training("km", {0.0f, 10.0f}, {1.0f, 100.0f});

  auto vs = make_vs();
  vs.put("q", Tf(3, 1, {0.0f, 9.9f, 4.0f}));

  exec_n(vs, 3, "omle.ml", "KNN", {"q"}, {"y"},
         knn_attrs("km", 1, "regression"));
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_rows, 3);
  EXPECT_NEAR(y.at(0, 0), 1.0f, 1e-4f);
  EXPECT_NEAR(y.at(1, 0), 100.0f, 1e-4f);
  EXPECT_NEAR(y.at(2, 0), 1.0f, 1e-4f);  // 4.0 is nearer 0 than 10
}

// =============================================================================
// omle.ml/KNN — distance metrics
// =============================================================================

TEST_F(KnnTest, ManhattanMetricPicksDifferentNeighbourThanEuclidean) {
  // a=(3,3): L1 = 6,  L2 = 4.243
  // b=(0,5): L1 = 5,  L2 = 5
  // Under L1 b is nearest; under L2 a is nearest.
  put_const("mm_X", 2, 2, {3.0f, 3.0f, 0.0f, 5.0f});
  put_const("mm_y", 2, 1, {10.0f, 20.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 2, {0.0f, 0.0f}));

  auto l2 = knn_attrs("mm", 1, "regression");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y_l2"}, l2);
  EXPECT_NEAR(vs.get("y_l2").at(0, 0), 10.0f, 1e-4f);

  auto l1 = knn_attrs("mm", 1, "regression");
  l1["metric"] = astr("manhattan");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y_l1"}, l1);
  EXPECT_NEAR(vs.get("y_l1").at(0, 0), 20.0f, 1e-4f);
}

TEST_F(KnnTest, CosineMetricIgnoresMagnitude) {
  // Same direction as the query but far away in Euclidean terms; the other
  // point is close but orthogonal.
  put_const("cos_X", 2, 2, {100.0f, 0.0f, 0.0f, 1.0f});
  put_const("cos_y", 2, 1, {10.0f, 20.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 2, {1.0f, 0.0f}));

  auto attrs = knn_attrs("cos", 1, "regression");
  attrs["metric"] = astr("cosine");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"}, attrs);
  // cosine distance 0 to the collinear point, 1 to the orthogonal one
  EXPECT_NEAR(vs.get("y").at(0, 0), 10.0f, 1e-4f);
}

TEST_F(KnnTest, MinkowskiRespectsP) {
  // p=1 reproduces Manhattan, so the same pair that flips between L1 and L2
  // must flip between minkowski p=1 and p=2.
  put_const("mk_X", 2, 2, {3.0f, 3.0f, 0.0f, 5.0f});
  put_const("mk_y", 2, 1, {10.0f, 20.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 2, {0.0f, 0.0f}));

  auto p1 = knn_attrs("mk", 1, "regression");
  p1["metric"] = astr("minkowski");
  p1["p"] = aflt(1.0);
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y_p1"}, p1);
  EXPECT_NEAR(vs.get("y_p1").at(0, 0), 20.0f, 1e-4f);

  auto p2 = knn_attrs("mk", 1, "regression");
  p2["metric"] = astr("minkowski");
  p2["p"] = aflt(2.0);
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y_p2"}, p2);
  EXPECT_NEAR(vs.get("y_p2").at(0, 0), 10.0f, 1e-4f);
}

// =============================================================================
// omle.ml/KNN — radius mode
// =============================================================================

TEST_F(KnnTest, RadiusModeUsesEveryNeighbourInRange) {
  // Three points inside radius 1.0, one far outside. n_neighbors is ignored.
  put_training("rad", {0.0f, 0.5f, 0.9f, 50.0f}, {0.0f, 0.0f, 0.0f, 1.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));

  auto attrs = knn_attrs("rad", 1, "classification");
  attrs["neighbor_mode"] = astr("radius");
  attrs["radius"] = aflt(1.0);
  attrs["n_classes"] = aint({2});
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y", "prob"}, attrs);

  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 0.0f);
  // All three in-radius neighbours are class 0, so it is unanimous even
  // though n_neighbors=1 would have looked at only one of them.
  EXPECT_NEAR(vs.get("prob").at(0, 0), 1.0f, 1e-5f);
}

TEST_F(KnnTest, RadiusModeFallsBackToOutlierLabel) {
  put_training("out", {100.0f, 200.0f}, {0.0f, 0.0f});
  put_const("outlier", 1, 1, {1.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));  // nothing within radius

  auto attrs = knn_attrs("out", 1, "classification");
  attrs["neighbor_mode"] = astr("radius");
  attrs["radius"] = aflt(1.0);
  attrs["n_classes"] = aint({2});
  attrs["outlier_label"] = tensor_attr("outlier");
  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y", "prob"}, attrs);

  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 1.0f);
  EXPECT_FLOAT_EQ(vs.get("prob").at(0, 0), 0.0f);
  EXPECT_FLOAT_EQ(vs.get("prob").at(0, 1), 1.0f);
}

// =============================================================================
// omle.ml/KNN — degenerate inputs
// =============================================================================

TEST_F(KnnTest, NeighbourCountIsClampedToTrainingSize) {
  // k=10 with only 2 training rows must average both, not read past the end.
  put_training("clamp", {0.0f, 2.0f}, {1.0f, 3.0f});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));

  exec_n(vs, 1, "omle.ml", "KNN", {"q"}, {"y"},
         knn_attrs("clamp", 10, "regression"));
  EXPECT_NEAR(vs.get("y").at(0, 0), 2.0f, 1e-4f);  // (1+3)/2
}

TEST_F(KnnTest, MissingTrainingAttributesIsAnError) {
  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));

  auto st = try_exec(vs, 1, "KNN", {"q"}, {"y"}, {{"n_neighbors", aint({1})}});
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), omle::rt::ErrorCode::InvalidArgument);
}

TEST_F(KnnTest, EmptyTrainingSetIsAnError) {
  put_const("empty_X", 0, 1, {});
  put_const("empty_y", 0, 1, {});

  auto vs = make_vs();
  vs.put("q", Tf(1, 1, {0.0f}));

  auto st = try_exec(vs, 1, "KNN", {"q"}, {"y"},
                     {{"train_features", tensor_attr("empty_X")},
                      {"train_targets", tensor_attr("empty_y")},
                      {"n_neighbors", aint({1})}});
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), omle::rt::ErrorCode::InvalidArgument);
}

// =============================================================================
// omle.feature/KNNImputer
// =============================================================================

TEST_F(KnnTest, Imputer) {
  // Training: 3 rows × 2 features; clear nearest to query by col 0 distance
  // row0=(0,5), row1=(1,5), row2=(10,1)
  put_const("train_x", 3, 2, {0.0f, 5.0f, 1.0f, 5.0f, 10.0f, 1.0f});
  auto vs = make_vs();
  vs.put("q", Tf(1, 2, {0.1f, kNaN}));  // missing col 1

  AttrVal ft;
  ft.tensor_name = "train_x";
  exec_n(vs, 1, "omle.feature", "KNNImputer", {"q"}, {"y"},
         {{"train_features", ft}, {"n_neighbors", aint({2})}});
  // Nearest (ignoring col 1): row0 dist≈0.14, row1 dist≈1.27, row2 large
  // k=2 → row0 (col1=5) and row1 (col1=5) → imputed = 5.0
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 0.1f);     // col 0 unchanged
  EXPECT_NEAR(vs.get("y").at(0, 1), 5.0f, 1e-4f);  // imputed from neighbors
}

TEST_F(KnnTest, ImputerLeavesCompleteRowsAlone) {
  put_const("ti_x", 2, 2, {0.0f, 5.0f, 10.0f, 1.0f});
  auto vs = make_vs();
  vs.put("q", Tf(1, 2, {7.0f, 8.0f}));  // nothing missing

  exec_n(vs, 1, "omle.feature", "KNNImputer", {"q"}, {"y"},
         {{"train_features", tensor_attr("ti_x")}, {"n_neighbors", aint({1})}});
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 0), 7.0f);
  EXPECT_FLOAT_EQ(vs.get("y").at(0, 1), 8.0f);
}

TEST_F(KnnTest, ImputerHandlesMultipleRows) {
  put_const("tm_x", 3, 2, {0.0f, 5.0f, 1.0f, 5.0f, 10.0f, 1.0f});
  auto vs = make_vs();
  vs.put("q", Tf(2, 2, {0.1f, kNaN, 9.5f, kNaN}));

  exec_n(vs, 2, "omle.feature", "KNNImputer", {"q"}, {"y"},
         {{"train_features", tensor_attr("tm_x")}, {"n_neighbors", aint({1})}});
  const auto& y = vs.get("y");
  ASSERT_EQ(y.n_rows, 2);
  EXPECT_NEAR(y.at(0, 1), 5.0f, 1e-4f);  // nearest row0
  EXPECT_NEAR(y.at(1, 1), 1.0f, 1e-4f);  // nearest row2
}
