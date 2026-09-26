#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <thread>
#include <vector>

#include "omle.pb.h"
#include "omle/runtime.h"

namespace P = omle;        // proto types
using namespace omle::rt;  // public API: Model, Session, Tensor, LoadOptions

// -----------------------------------------------------------------------
// Fixture helpers
// -----------------------------------------------------------------------

// Build a single-stump proto Tree: feature[feat] < threshold → lv, else rv.
//
// children_index is a flat pool of child node IDs.
// children_offset[i] = starting index in that pool for node i's children.
// children_count[i]  = how many children node i has (0 for leaves).
// Node 0 (BRANCH): children_offset=0, children_count=2 → left=pool[0]=1,
// right=pool[1]=2. Nodes 1,2 (LEAF): children_count=0, children_offset=-1
// (unused).
static P::Tree make_stump(int feat, float threshold, float lv, float rv) {
  P::Tree t;
  t.set_num_nodes(3);
  // node kinds
  t.add_node_kind(P::Tree::BRANCH);
  t.add_node_kind(P::Tree::LEAF);
  t.add_node_kind(P::Tree::LEAF);
  // split fields (unused on leaf nodes)
  t.add_split_feature(feat);
  t.add_split_feature(0);
  t.add_split_feature(0);
  {
    auto* st = t.mutable_split_threshold()->mutable_tensor();
    st->mutable_float32_data()->add_values(threshold);
    st->mutable_float32_data()->add_values(0.0f);
    st->mutable_float32_data()->add_values(0.0f);
  }
  t.add_split_op(P::Tree::LESS_THAN);
  t.add_split_op(P::Tree::SPLIT_OP_UNSPECIFIED);
  t.add_split_op(P::Tree::SPLIT_OP_UNSPECIFIED);
  // children pool: [left=1, right=2]
  t.add_children_index(1);  // pool[0] = left  child of node 0
  t.add_children_index(2);  // pool[1] = right child of node 0
  // per-node offset into the pool (-1 = no children)
  t.add_children_offset(0);   // node 0: starts at pool[0]
  t.add_children_offset(-1);  // node 1: leaf, no children
  t.add_children_offset(-1);  // node 2: leaf, no children
  // per-node child count
  t.add_children_count(2);  // node 0 has 2 children
  t.add_children_count(0);  // node 1 is a leaf
  t.add_children_count(0);  // node 2 is a leaf
  // default child for missing values (route to right leaf = node 2)
  t.add_default_child(2);
  t.add_default_child(0);
  t.add_default_child(0);
  // leaf values (index matches node index; node 0 is a branch so its value is
  // unused)
  {
    auto* lf = t.mutable_leaf_value()->mutable_tensor();
    lf->mutable_float32_data()->add_values(0.0f);
    lf->mutable_float32_data()->add_values(lv);
    lf->mutable_float32_data()->add_values(rv);
  }
  return t;
}

// Serialise a graph-based OMLE model with one TreeEnsemble node.
//   n_features : total input features
//   stumps     : {feature_index, threshold, left_val, right_val}
static std::string make_model_bytes(
    int n_features,
    const std::vector<std::tuple<int, float, float, float>>& stumps) {
  P::OMLEModel proto;

  for (int i = 0; i < n_features; ++i)
    proto.add_inputs()->set_name("f" + std::to_string(i));
  proto.add_outputs()->set_name("score");

  auto* node = proto.add_nodes();
  node->set_domain("omle.ml");
  node->set_op("TreeEnsemble");
  for (int i = 0; i < n_features; ++i)
    node->add_inputs()->mutable_name()->set_value("f" + std::to_string(i));
  node->add_outputs()->set_name("score");

  auto* te = node->mutable_tree_ensemble();
  te->set_aggregation(P::TreeEnsemble::SUM);
  te->set_post_transform(P::IDENTITY);

  for (auto& [fi, thr, lv, rv] : stumps)
    *te->add_trees() = make_stump(fi, thr, lv, rv);

  std::string out;
  EXPECT_TRUE(proto.SerializeToString(&out));
  return out;
}

// Hand-compute the sum of all stump outputs for one sample.
static float expected_score(
    const std::vector<std::tuple<int, float, float, float>>& stumps,
    const float* x) {
  float total = 0.0f;
  for (auto& [fi, thr, lv, rv] : stumps) total += (x[fi] < thr) ? lv : rv;
  return total;
}

// -----------------------------------------------------------------------
// Tensor
// -----------------------------------------------------------------------

TEST(Tensor, DefaultConstruct) {
  Tensor t;
  EXPECT_EQ(t.n_rows, 0);
  EXPECT_EQ(t.n_cols, 1);
  EXPECT_EQ(t.data_bytes(), static_cast<std::size_t>(0));
}

TEST(Tensor, ShapeAndConstAccess) {
  Tensor t = Tensor::from_floats(3, 2, {1, 2, 3, 4, 5, 6});
  EXPECT_EQ(t.n_rows, 3);
  EXPECT_EQ(t.n_cols, 2);
  EXPECT_EQ(t.row(0)[0], 1.0f);
  EXPECT_EQ(t.row(0)[1], 2.0f);
  EXPECT_EQ(t.row(1)[0], 3.0f);
  EXPECT_EQ(t.row(2)[1], 6.0f);
}

TEST(Tensor, MutableRow) {
  Tensor t = Tensor::f32(2, 2, 0.f);
  t.row(1)[0] = 9.0f;
  EXPECT_EQ(t.at(1, 0), 9.0f);
}

TEST(Tensor, SingleRowShape) {
  Tensor t = Tensor::from_floats(1, 4, {1, 2, 3, 4});
  EXPECT_EQ(t.n_rows, 1);
  EXPECT_EQ(t.n_cols, 4);
  EXPECT_EQ(t.row(0)[3], 4.0f);
}

// -----------------------------------------------------------------------
// Model — loading and schema
// -----------------------------------------------------------------------

TEST(Model, LoadFromBytes) {
  auto bytes = make_model_bytes(4, {{0, 0.5f, 1.0f, -1.0f}});
  auto m = Model::load(bytes.data(), bytes.size());
  ASSERT_TRUE(m.ok());
  EXPECT_EQ(m.value()->num_inputs(), 4);
  EXPECT_EQ(m.value()->num_outputs(), 1);
}

TEST(Model, LoadFromFile_MissingPath) {
  auto r = Model::load("/nonexistent/path.omle");
  EXPECT_FALSE(r.ok());
}

TEST(Model, InputsOutputsIntrospection) {
  auto bytes = make_model_bytes(3, {{0, 0.0f, 1.0f, -1.0f}});
  auto m = Model::load(bytes.data(), bytes.size()).value();
  ASSERT_EQ(m->inputs().size(), 3u);
  EXPECT_EQ(m->inputs()[0].name, "f0");
  EXPECT_EQ(m->inputs()[2].name, "f2");
  ASSERT_EQ(m->outputs().size(), 1u);
  EXPECT_EQ(m->outputs()[0].name, "score");
}

TEST(Model, PredictBatch_KnownValues) {
  // Stump: feature 0 < 0.5 → +2, else -2
  auto bytes = make_model_bytes(2, {{0, 0.5f, 2.0f, -2.0f}});
  auto m = Model::load(bytes.data(), bytes.size()).value();

  Tensor input = Tensor::from_floats(2, 2, {0.0f, 0.0f, 1.0f, 0.0f});
  auto res = m->predict({{"input", input}}).value();
  const Tensor& out = res.begin()->second;
  EXPECT_FLOAT_EQ(out.row(0)[0], 2.0f);
  EXPECT_FLOAT_EQ(out.row(1)[0], -2.0f);
}

TEST(Model, PredictConvenienceReturnsTensor) {
  auto bytes = make_model_bytes(2, {{0, 0.5f, 2.0f, -2.0f}});
  auto m = Model::load(bytes.data(), bytes.size()).value();

  Tensor input = Tensor::from_floats(2, 2, {0.0f, 0.0f, 1.0f, 0.0f});
  auto result_map = m->predict({{"input", input}}).value();
  ASSERT_EQ(result_map.size(), 1u);
  const Tensor& result = result_map.begin()->second;
  EXPECT_EQ(result.n_rows, 2);
  EXPECT_EQ(result.n_cols, 1);
  EXPECT_FLOAT_EQ(result.row(0)[0], 2.0f);
  EXPECT_FLOAT_EQ(result.row(1)[0], -2.0f);
}

TEST(Model, MultipleStumps_SumAggregation) {
  std::vector<std::tuple<int, float, float, float>> stumps = {
      {0, 0.5f, 1.0f, -1.0f},
      {1, 0.5f, 2.0f, -2.0f},
      {0, 0.0f, 0.5f, -0.5f},
  };
  auto bytes = make_model_bytes(2, stumps);
  auto m = Model::load(bytes.data(), bytes.size()).value();

  std::vector<float> features = {0.0f, 0.0f, 0.0f, 1.0f,
                                 1.0f, 0.0f, 1.0f, 1.0f};
  auto res =
      m->predict({{"input", Tensor::from_floats(4, 2, features)}}).value();
  const Tensor& out = res.begin()->second;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(out.row(i)[0],
                    expected_score(stumps, features.data() + i * 2))
        << "sample " << i;
}

TEST(Model, MultiThread_MatchesSingleThread) {
  std::vector<std::tuple<int, float, float, float>> stumps = {
      {0, 0.3f, 1.0f, -1.0f},
      {1, 0.7f, 0.5f, -0.5f},
      {2, 0.5f, 2.0f, -2.0f},
      {3, 0.4f, -1.0f, 1.0f},
  };
  auto bytes = make_model_bytes(4, stumps);

  LoadOptions opts1, opts4;
  opts1.n_threads = 1;
  opts4.n_threads = 4;
  opts4.min_parallel_rows = 1;  // force thread pool even for small batches

  auto m1 = Model::load(bytes.data(), bytes.size(), opts1).value();
  auto m4 = Model::load(bytes.data(), bytes.size(), opts4).value();

  const int n = 200;
  std::vector<float> features(n * 4);
  for (int i = 0; i < n * 4; ++i)
    features[i] = static_cast<float>(i % 10) / 10.0f;

  Tensor input = Tensor::from_floats(n, 4, features);
  auto res1 = m1->predict({{"input", input}}).value();
  auto res4 = m4->predict({{"input", input}}).value();
  const Tensor& t1 = res1.begin()->second;
  const Tensor& t4 = res4.begin()->second;
  for (int i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(t1.row(i)[0], t4.row(i)[0]) << "sample " << i;
}

// -----------------------------------------------------------------------
// Session
// -----------------------------------------------------------------------

class SessionTest : public ::testing::Test {
 protected:
  std::vector<std::tuple<int, float, float, float>> stumps = {
      {0, 0.5f, 1.0f, -1.0f},
      {1, 0.5f, 0.5f, -0.5f},
  };
  static constexpr int n_feat = 2;
  std::unique_ptr<Model> model;

  void SetUp() override {
    auto bytes = make_model_bytes(n_feat, stumps);
    model = Model::load(bytes.data(), bytes.size()).value();
  }
};

TEST_F(SessionTest, CreateAndSchema) {
  auto s = model->create_session();
  ASSERT_NE(s, nullptr);
  EXPECT_EQ(s->num_inputs(), n_feat);
  EXPECT_EQ(s->num_outputs(), 1);
}

TEST_F(SessionTest, RunOneSample) {
  auto s = model->create_session();
  std::vector<float> features = {0.0f, 0.0f};  // f0 < 0.5 → +1; f1 < 0.5 → +0.5
  s->bind_input("input", Tensor::from_floats(1, n_feat, features));
  ASSERT_TRUE(s->run().ok());
  EXPECT_FLOAT_EQ(s->results().begin()->second.row(0)[0],
                  expected_score(stumps, features.data()));
}

TEST_F(SessionTest, RunBatch_KnownValues) {
  auto s = model->create_session();
  std::vector<float> features = {
      0.0f, 0.0f,  // +1 + 0.5 = +1.5
      0.0f, 1.0f,  // +1 - 0.5 = +0.5
      1.0f, 0.0f,  // -1 + 0.5 = -0.5
      1.0f, 1.0f,  // -1 - 0.5 = -1.5
  };
  s->bind_input("input", Tensor::from_floats(4, n_feat, features));
  ASSERT_TRUE(s->run().ok());
  const Tensor& out = s->results().begin()->second;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(out.row(i)[0],
                    expected_score(stumps, features.data() + i * n_feat))
        << "sample " << i;
}

TEST_F(SessionTest, RunTensorInterface) {
  auto s = model->create_session();

  std::vector<float> raw = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f};
  s->bind_input("input", Tensor::from_floats(3, n_feat, raw));
  ASSERT_TRUE(s->run().ok());

  const Tensor& result = s->results().begin()->second;
  EXPECT_EQ(result.n_rows, 3);
  EXPECT_EQ(result.n_cols, 1);
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(result.row(i)[0],
                    expected_score(stumps, raw.data() + i * n_feat))
        << "row " << i;
}

TEST_F(SessionTest, RunTwice_GivesIdenticalResults) {
  // Calling run() twice with the same input must give identical results.
  auto s = model->create_session();
  s->bind_input("input",
                Tensor::from_floats(2, n_feat, {0.0f, 0.0f, 1.0f, 1.0f}));

  ASSERT_TRUE(s->run().ok());
  float r1_0 = s->results().begin()->second.row(0)[0];
  float r1_1 = s->results().begin()->second.row(1)[0];

  ASSERT_TRUE(s->run().ok());
  EXPECT_FLOAT_EQ(s->results().begin()->second.row(0)[0], r1_0);
  EXPECT_FLOAT_EQ(s->results().begin()->second.row(1)[0], r1_1);
}

TEST_F(SessionTest, MatchesModelPredict) {
  // Session::run() (serial) and Model::predict() (potentially pooled)
  // must produce bit-identical results.
  auto s = model->create_session();

  const int n = 50;
  std::vector<float> features(n * n_feat);
  for (int i = 0; i < n * n_feat; ++i) features[i] = (i % 3 == 0) ? 0.2f : 0.8f;

  Tensor input = Tensor::from_floats(n, n_feat, features);
  auto model_res = model->predict({{"input", input}}).value();

  s->bind_input("input", input);
  ASSERT_TRUE(s->run().ok());

  const Tensor& m_out = model_res.begin()->second;
  const Tensor& s_out = s->results().begin()->second;
  for (int i = 0; i < n; ++i)
    EXPECT_FLOAT_EQ(m_out.row(i)[0], s_out.row(i)[0]) << "sample " << i;
}

TEST_F(SessionTest, OutlivesModel) {
  // Session must remain fully functional after its originating Model is
  // destroyed.
  auto s = model->create_session();

  Tensor input = Tensor::from_floats(1, n_feat, {0.0f, 0.0f});
  s->bind_input("input", input);
  ASSERT_TRUE(s->run().ok());
  float before = s->results().begin()->second.row(0)[0];

  model.reset();  // destroy the Model; Session keeps the graph alive

  s->bind_input("input", input);
  EXPECT_TRUE(s->run().ok());
  EXPECT_FLOAT_EQ(s->results().begin()->second.row(0)[0], before);
}

TEST_F(SessionTest, MultipleSessionsIndependent) {
  // Two sessions must produce identical results and not share mutable state.
  auto s1 = model->create_session();
  auto s2 = model->create_session();

  Tensor input = Tensor::from_floats(1, n_feat, {0.0f, 0.0f});
  s1->bind_input("input", input);
  s2->bind_input("input", input);
  ASSERT_TRUE(s1->run().ok());
  ASSERT_TRUE(s2->run().ok());
  EXPECT_FLOAT_EQ(s1->results().begin()->second.row(0)[0],
                  s2->results().begin()->second.row(0)[0]);
}

TEST_F(SessionTest, ConcurrentSessionsOnSharedModel) {
  // One session per thread; all must agree with the single-thread reference.
  auto ref_s = model->create_session();
  Tensor ref_input = Tensor::from_floats(1, n_feat, {0.0f, 1.0f});
  ref_s->bind_input("input", ref_input);
  ASSERT_TRUE(ref_s->run().ok());
  const float ref = ref_s->results().begin()->second.row(0)[0];

  const int n_threads = 4;
  std::vector<float> results(n_threads, 0.0f);
  std::vector<std::thread> threads;

  for (int t = 0; t < n_threads; ++t) {
    threads.emplace_back([&, t] {
      auto s = model->create_session();
      s->bind_input("input", Tensor::from_floats(1, n_feat, {0.0f, 1.0f}));
      for (int rep = 0; rep < 50; ++rep) ASSERT_TRUE(s->run().ok());
      results[t] = s->results().begin()->second.row(0)[0];
    });
  }
  for (auto& th : threads) th.join();

  for (int t = 0; t < n_threads; ++t)
    EXPECT_FLOAT_EQ(results[t], ref) << "thread " << t;
}

TEST_F(SessionTest, BindOutput_FiltersResults) {
  // bind_output should restrict results to only the registered output names.
  auto s = model->create_session();
  s->bind_input("input", Tensor::from_floats(1, n_feat, {0.0f, 0.0f}));
  s->bind_output("score", Tensor{});
  ASSERT_TRUE(s->run().ok());
  EXPECT_EQ(s->results().size(), 1u);
  EXPECT_TRUE(s->get_output("score").ok());
}

TEST_F(SessionTest, ClearAndRebind) {
  // clear_inputs/clear_outputs then re-bind should work correctly.
  auto s = model->create_session();
  s->bind_input("input", Tensor::from_floats(1, n_feat, {0.0f, 0.0f}));
  ASSERT_TRUE(s->run().ok());
  float r1 = s->results().begin()->second.row(0)[0];

  s->clear_inputs();
  s->bind_input("input", Tensor::from_floats(1, n_feat, {0.0f, 0.0f}));
  ASSERT_TRUE(s->run().ok());
  EXPECT_FLOAT_EQ(s->results().begin()->second.row(0)[0], r1);
}

// -----------------------------------------------------------------------
// Multi-node graph tests — operator node + model node combinations
//
// Each test wires one or two omle.core / omle.feature operator nodes
// with a single TreeEnsemble model node.  Operators pre- or post-process
// the feature matrix; the model node does the actual prediction.
// -----------------------------------------------------------------------

// Helper: append a generic operator node (no structured body, uses attributes).
static P::Node* add_op_node(P::OMLEModel& proto, const std::string& domain,
                            const std::string& op,
                            const std::vector<std::string>& in_names,
                            const std::vector<std::string>& out_names) {
  auto* node = proto.add_nodes();
  node->set_domain(domain);
  node->set_op(op);
  for (const auto& n : in_names)
    node->add_inputs()->mutable_name()->set_value(n);
  for (const auto& n : out_names) node->add_outputs()->set_name(n);
  return node;
}

// Helper: append the TreeEnsemble model node.
static void add_te_node(
    P::OMLEModel& proto, const std::vector<std::string>& in_names,
    const std::string& out_name,
    const std::vector<std::tuple<int, float, float, float>>& stumps) {
  auto* node = proto.add_nodes();
  node->set_domain("omle.ml");
  node->set_op("TreeEnsemble");
  for (const auto& n : in_names)
    node->add_inputs()->mutable_name()->set_value(n);
  node->add_outputs()->set_name(out_name);
  auto* te = node->mutable_tree_ensemble();
  te->set_aggregation(P::TreeEnsemble::SUM);
  te->set_post_transform(P::IDENTITY);
  for (auto& [fi, thr, lv, rv] : stumps)
    *te->add_trees() = make_stump(fi, thr, lv, rv);
}

// ---- Test 1: Binarizer → TreeEnsemble -----------------------------------
//
// Binarizer converts raw features to 0/1 before the model sees them.
// Stump: binarized_f0 < 0.5 → +2, else -2
// With threshold=0 the binarizer maps any positive value → 1, non-positive → 0.
// Raw f0=0.3 (≤0) → bin=0 < 0.5 → score=+2
// Raw f0=1.0 (>0) → bin=1 ≥ 0.5 → score=-2

TEST(MultiNode, BinarizerThenTreeEnsemble) {
  P::OMLEModel proto;
  proto.add_inputs()->set_name("f0");
  proto.add_outputs()->set_name("score");

  auto* bin =
      add_op_node(proto, "omle.feature", "Binarizer", {"f0"}, {"f0_bin"});
  auto* attr = bin->add_attributes();
  attr->set_name("threshold");
  attr->set_f32(0.0);  // values > 0 → 1, else → 0

  add_te_node(proto, {"f0_bin"}, "score", {{0, 0.5f, 2.0f, -2.0f}});

  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  auto m = Model::load(bytes.data(), bytes.size()).value();
  ASSERT_EQ(m->num_inputs(), 1);
  ASSERT_EQ(m->num_outputs(), 1);

  // Binarizer: v > threshold (0) → 1, else → 0
  // f0=-1 → 0 < 0.5 → +2; f0=0 → 0 < 0.5 → +2; f0=1 → 1 ≥ 0.5 → -2
  std::vector<float> feat = {-1.0f, 0.0f, 1.0f};
  float expected[] = {2.0f, 2.0f, -2.0f};
  auto res = m->predict({{"input", Tensor::from_floats(3, 1, feat)}}).value();
  const Tensor& out = res.begin()->second;
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(out.row(i)[0], expected[i]) << "sample " << i;
}

// ---- Test 2: Clip → TreeEnsemble ----------------------------------------
//
// Clip clamps raw features to [0, 1] before the model.
// Stump: f0 < 0.5 → +3, else -3
// f0=-5 → clipped=0 < 0.5 → +3
// f0=0.3 → clipped=0.3 < 0.5 → +3
// f0=2   → clipped=1 ≥ 0.5 → -3

TEST(MultiNode, ClipThenTreeEnsemble) {
  P::OMLEModel proto;
  proto.add_inputs()->set_name("f0");
  proto.add_outputs()->set_name("score");

  auto* clip = add_op_node(proto, "omle.core", "Clip", {"f0"}, {"f0_clipped"});
  {
    auto* a = clip->add_attributes();
    a->set_name("min");
    a->set_f32(0.0);
    auto* b = clip->add_attributes();
    b->set_name("max");
    b->set_f32(1.0);
  }

  add_te_node(proto, {"f0_clipped"}, "score", {{0, 0.5f, 3.0f, -3.0f}});

  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  auto m = Model::load(bytes.data(), bytes.size()).value();

  std::vector<float> feat = {-5.0f, 0.3f, 2.0f};
  float expected[] = {3.0f, 3.0f, -3.0f};
  auto res = m->predict({{"input", Tensor::from_floats(3, 1, feat)}}).value();
  const Tensor& out = res.begin()->second;
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(out.row(i)[0], expected[i]) << "sample " << i;
}

// ---- Test 3: Gather (feature selection) → TreeEnsemble ------------------
//
// Gather selects a subset of columns before the model.
// Two raw features [f0, f1]; model only sees f1 (column index 1).
// Stump on gathered column 0 (= raw f1): f1 < 0.5 → +1, else -1

TEST(MultiNode, GatherThenTreeEnsemble) {
  P::OMLEModel proto;
  proto.add_inputs()->set_name("f0");
  proto.add_inputs()->set_name("f1");
  proto.add_outputs()->set_name("score");

  // Concat f0 and f1 into a 2-column tensor, then gather only column 1.
  add_op_node(proto, "omle.core", "Concat", {"f0", "f1"}, {"features"});

  auto* gather =
      add_op_node(proto, "omle.core", "Gather", {"features"}, {"selected"});
  {
    auto* a = gather->add_attributes();
    a->set_name("indices");
    a->mutable_ints()->add_values(1);  // keep only column 1 (= f1)
  }

  add_te_node(proto, {"selected"}, "score", {{0, 0.5f, 1.0f, -1.0f}});

  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  auto m = Model::load(bytes.data(), bytes.size()).value();
  ASSERT_EQ(m->num_inputs(), 2);

  // f0 is ignored; score depends only on f1
  std::vector<float> feat = {99.0f, 0.0f, 99.0f, 1.0f, 0.0f, 0.2f};
  float expected[] = {1.0f, -1.0f, 1.0f};
  auto res = m->predict({{"input", Tensor::from_floats(3, 2, feat)}}).value();
  const Tensor& out = res.begin()->second;
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(out.row(i)[0], expected[i]) << "sample " << i;
}

// ---- Test 4: TreeEnsemble → Clip (post-processing) ----------------------
//
// The model node runs first; its raw score is then clamped by Clip.
// Stumps (SUM of two):
//   stump A: f0 < 0.5 → +3, else -3
//   stump B: f1 < 0.5 → +3, else -3
// Raw score range: {-6, 0, +6}
// Clip to [-4, 4]:
//   raw=+6 → clipped=+4
//   raw= 0 → clipped= 0
//   raw=-6 → clipped=-4

TEST(MultiNode, TreeEnsembleThenClip) {
  P::OMLEModel proto;
  proto.add_inputs()->set_name("f0");
  proto.add_inputs()->set_name("f1");
  proto.add_outputs()->set_name("score");

  add_te_node(proto, {"f0", "f1"}, "raw_score",
              {{0, 0.5f, 3.0f, -3.0f}, {1, 0.5f, 3.0f, -3.0f}});

  auto* clip =
      add_op_node(proto, "omle.core", "Clip", {"raw_score"}, {"score"});
  {
    auto* a = clip->add_attributes();
    a->set_name("min");
    a->set_f32(-4.0);
    auto* b = clip->add_attributes();
    b->set_name("max");
    b->set_f32(4.0);
  }

  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  auto m = Model::load(bytes.data(), bytes.size()).value();

  std::vector<float> feat = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  float expected[] = {4.0f, 0.0f, 0.0f, -4.0f};
  auto res = m->predict({{"input", Tensor::from_floats(4, 2, feat)}}).value();
  const Tensor& out = res.begin()->second;
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(out.row(i)[0], expected[i]) << "sample " << i;
}

// ---- Test 5: Session runs operator + model graph correctly --------------
//
// Verifies Session::run(Tensor) on a Clip → TreeEnsemble graph.

TEST(MultiNode, SessionRunsOperatorAndModelGraph) {
  P::OMLEModel proto;
  proto.add_inputs()->set_name("f0");
  proto.add_outputs()->set_name("score");

  auto* clip = add_op_node(proto, "omle.core", "Clip", {"f0"}, {"f0_c"});
  {
    auto* a = clip->add_attributes();
    a->set_name("min");
    a->set_f32(0.0);
    auto* b = clip->add_attributes();
    b->set_name("max");
    b->set_f32(1.0);
  }

  add_te_node(proto, {"f0_c"}, "score", {{0, 0.5f, 5.0f, -5.0f}});

  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));
  auto m = Model::load(bytes.data(), bytes.size()).value();
  auto s = m->create_session();

  s->bind_input("input", Tensor::from_floats(4, 1,
                                             {-2.0f,     // clip→0 < 0.5 → +5
                                              0.3f,      // clip→0.3 < 0.5 → +5
                                              0.7f,      // clip→0.7 ≥ 0.5 → -5
                                              10.0f}));  // clip→1 ≥ 0.5 → -5
  float expected[] = {5.0f, 5.0f, -5.0f, -5.0f};

  ASSERT_TRUE(s->run().ok());
  const Tensor& result = s->results().begin()->second;
  EXPECT_EQ(result.n_rows, 4);
  EXPECT_EQ(result.n_cols, 1);
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(result.row(i)[0], expected[i]) << "row " << i;
}

// ---- Float64 verification inputs keep their precision -------------------
//
// build_tensor_entry_map() used to materialize every numeric tensor entry with
// extract_floats()/from_floats(), whatever dtype it declared. Verification and
// warmup feed those entries to the graph as inputs, so a recorded FLOAT64 value
// was narrowed to float32 before the first operator ran — even though
// run_graph()'s per-name branch takes care to preserve a Float64 input.
//
// Exposing that needs care, because run_verification() compares actual against
// expected in float32: the relative error from narrowing 0.1 is ~1.5e-8, below
// float32 resolution, so no amount of linear scaling survives the comparison.
// StandardScaler subtracting float32(0.1) from 0.1 cancels the leading digits
// instead, leaving exactly the quantization error as the output — ~1.5e-9 when
// the input arrives as float64, and exactly 0.0 when it has been narrowed.
//
// The narrowing goes through a volatile float rather than a constexpr cast.
// MSVC builds with /fp:fast (see the top-level CMakeLists), which permits
// eliding an intermediate rounding to float; folded at compile time, the
// quantization error becomes 0 and the premise of the test disappears — as a
// static_assert firing, or worse as a test that passes for the wrong reason.
// volatile forces the store to a float to actually happen.
//
// Tolerances leave three orders of magnitude of headroom for the same reason.
// The signal is the difference between ~1.5e-9 and 0, so rtol=1e-3 still fails
// the unfixed loader by a factor of ~1000 while tolerating any last-bit
// difference /fp:fast and /arch:AVX2 introduce.
//
// Deliberately not a tree split: P::Tree stores thresholds as float32, so a
// TreeEnsemble comparison narrows the feature by design, matching the source
// framework. That would pin the kernel's semantics rather than the loader's
// fidelity, which is what is at stake here.
TEST(Float64Fidelity, VerificationInputKeepsFloat64) {
  const double kInput = 0.1;
  volatile float narrowed = static_cast<float>(kInput);
  const double kMean = static_cast<double>(narrowed);
  const double kExpected = kInput - kMean;  // ≈ -1.49e-09

  ASSERT_NE(kExpected, 0.0)
      << "0.1 must not be representable in float32, or a narrowed input would "
         "produce the same answer and prove nothing";
  const double tol = std::fabs(kExpected) * 1e-3;

  P::OMLEModel proto;
  proto.add_inputs()->set_name("f0");
  proto.add_outputs()->set_name("y");

  auto add_f64_entry = [&proto](const char* id, const char* name, double v) {
    auto* e = proto.add_tensor_entries();
    e->set_id(id);
    auto* t = e->mutable_dense();
    t->set_name(name);
    t->mutable_type()->set_dtype(P::FLOAT64);
    t->mutable_type()->add_shape(1);
    t->mutable_float64_data()->add_values(v);
  };
  // kMean is exactly representable in float32, so narrowing it is harmless —
  // only the input entry's dtype decides the outcome.
  add_f64_entry("sc_mean", "sc_mean", kMean);
  add_f64_entry("sc_scale", "sc_scale", 1.0);
  add_f64_entry("verify_f0", "f0", kInput);
  add_f64_entry("verify_y", "y", kExpected);

  auto* node =
      add_op_node(proto, "omle.feature", "StandardScaler", {"f0"}, {"y"});
  node->add_attributes()->set_name("mean");
  node->mutable_attributes(0)->mutable_tensor_ref()->set_id("sc_mean");
  node->add_attributes()->set_name("scale");
  node->mutable_attributes(1)->mutable_tensor_ref()->set_id("sc_scale");

  auto* vc = proto.mutable_verification()->add_cases();
  vc->add_inputs()->set_id("verify_f0");
  vc->add_expected_outputs()->set_id("verify_y");
  auto* vtol = proto.mutable_verification()->mutable_tolerance();
  vtol->mutable_atol()->set_float_value(0.0f);
  vtol->mutable_rtol()->set_float_value(1e-3f);

  std::string bytes;
  ASSERT_TRUE(proto.SerializeToString(&bytes));

  // Loading runs the verification case; a narrowed entry makes this throw.
  auto loaded = Model::load(bytes.data(), bytes.size());
  ASSERT_TRUE(loaded.ok()) << loaded.status().message();

  // The same value through the public API, so the recorded path and the caller
  // path cannot drift apart again without one of these assertions failing.
  Tensor x = Tensor::dense(DataType::Float64, 1, 1);
  x.f64_at(0, 0) = kInput;
  auto res = loaded.value()->predict({{"f0", std::move(x)}});
  ASSERT_TRUE(res.ok()) << res.status().message();
  EXPECT_NEAR(res.value().begin()->second.get(0, 0), kExpected, tol);
}
