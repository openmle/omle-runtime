// Tests for CompositeNode — the sub-graph body.
//
// CompositeNode has no arithmetic of its own; what it owns is scoping and
// renaming. These tests therefore use trivial stand-in nodes rather than real
// operators, so a failure points at the composite's own behaviour and not at
// whichever operator happened to be used as a vehicle.

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "composite_node.h"
#include "runtime_tensor.h"

using namespace omle::rt::impl;

namespace {

// Multiplies its single input by `factor`.
class ScaleNode final : public GraphNode {
 public:
  float factor = 1.0f;

  omle::rt::Status execute(ValueStore& vs, int n_rows) const override {
    const omle::rt::Tensor& x = vs.get(in_names[0]);
    omle::rt::Tensor out =
        omle::rt::Tensor::dense(omle::rt::DataType::Float32, n_rows, x.n_cols);
    for (int r = 0; r < n_rows; ++r)
      for (int c = 0; c < x.n_cols; ++c) out.at(r, c) = x.at(r, c) * factor;
    vs.put(out_names[0], std::move(out));
    return {};
  }
};

// Copies its input through, recording whether `probe` was visible in scope.
class ProbeNode final : public GraphNode {
 public:
  std::string probe;
  bool* saw = nullptr;

  omle::rt::Status execute(ValueStore& vs, int /*n_rows*/) const override {
    if (saw) *saw = vs.has(probe);
    vs.put(out_names[0], vs.get(in_names[0]));
    return {};
  }
};

// Always fails, and records whether it ran at all.
class FailingNode final : public GraphNode {
 public:
  bool* ran = nullptr;

  omle::rt::Status execute(ValueStore& /*vs*/, int /*n_rows*/) const override {
    if (ran) *ran = true;
    return {omle::rt::ErrorCode::InvalidArgument, "composite child failed"};
  }
};

std::unique_ptr<ScaleNode> scale(float factor, std::string in,
                                 std::string out) {
  auto n = std::make_unique<ScaleNode>();
  n->factor = factor;
  n->in_names = {std::move(in)};
  n->out_names = {std::move(out)};
  return n;
}

class CompositeTest : public ::testing::Test {
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
};

}  // namespace

// =============================================================================
// Name plumbing
// =============================================================================

TEST_F(CompositeTest, RunsAnInnerNodeWithoutAliases) {
  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"y"};
  c.nodes.push_back(scale(2.0f, "x", "y"));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {3.0f, 4.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 6.0f, 1e-6f);
  EXPECT_NEAR(y.at(1, 0), 8.0f, 1e-6f);
}

TEST_F(CompositeTest, InputAliasRenamesOnEntry) {
  // The sub-graph is written against "inner_x"; the enclosing graph supplies
  // "outer_x".
  CompositeNode c;
  c.in_names = {"outer_x"};
  c.input_aliases = {{"outer_x", "inner_x"}};
  c.out_names = {"y"};
  c.nodes.push_back(scale(2.0f, "inner_x", "y"));

  auto vs = make_vs();
  vs.put("outer_x", F32(2, 1, {3.0f, 4.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  EXPECT_NEAR(vs.get("y").at(0, 0), 6.0f, 1e-6f);
}

TEST_F(CompositeTest, OutputAliasRenamesOnExit) {
  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"outer_y"};
  c.output_aliases = {{"inner_y", "outer_y"}};
  c.nodes.push_back(scale(3.0f, "x", "inner_y"));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 2.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  EXPECT_NEAR(vs.get("outer_y").at(0, 0), 3.0f, 1e-6f);
  EXPECT_NEAR(vs.get("outer_y").at(1, 0), 6.0f, 1e-6f);
  // The internal name is not published.
  EXPECT_FALSE(vs.has("inner_y"));
}

TEST_F(CompositeTest, HandlesSeveralInputsAndOutputs) {
  CompositeNode c;
  c.in_names = {"a", "b"};
  c.input_aliases = {{"a", "p"}, {"b", "q"}};
  c.out_names = {"out_p", "out_q"};
  c.output_aliases = {{"p2", "out_p"}, {"q2", "out_q"}};
  c.nodes.push_back(scale(2.0f, "p", "p2"));
  c.nodes.push_back(scale(10.0f, "q", "q2"));

  auto vs = make_vs();
  vs.put("a", F32(2, 1, {1.0f, 2.0f}));
  vs.put("b", F32(2, 1, {3.0f, 4.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  EXPECT_NEAR(vs.get("out_p").at(0, 0), 2.0f, 1e-6f);
  EXPECT_NEAR(vs.get("out_p").at(1, 0), 4.0f, 1e-6f);
  EXPECT_NEAR(vs.get("out_q").at(0, 0), 30.0f, 1e-6f);
  EXPECT_NEAR(vs.get("out_q").at(1, 0), 40.0f, 1e-6f);
}

// =============================================================================
// Execution order and scoping
// =============================================================================

TEST_F(CompositeTest, InnerNodesRunInOrderAndChainThroughLocals) {
  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"y"};
  c.nodes.push_back(scale(2.0f, "x", "t"));
  c.nodes.push_back(scale(3.0f, "t", "y"));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 5.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 6.0f, 1e-6f);   // 1 * 2 * 3
  EXPECT_NEAR(y.at(1, 0), 30.0f, 1e-6f);  // 5 * 2 * 3
}

TEST_F(CompositeTest, IntermediateValuesDoNotEscapeToTheOuterScope) {
  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"y"};
  c.nodes.push_back(scale(2.0f, "x", "t"));
  c.nodes.push_back(scale(3.0f, "t", "y"));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 5.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  EXPECT_TRUE(vs.has("y"));
  EXPECT_FALSE(vs.has("t"));
}

TEST_F(CompositeTest, OuterValuesAreInvisibleUnlessDeclaredAsInputs) {
  // The child scope is created with no parent, so only the declared inputs
  // and the shared constants cross the boundary. A sub-graph that reached a
  // value it never declared would be a scoping leak.
  bool saw_secret = true;

  auto probe = std::make_unique<ProbeNode>();
  probe->probe = "secret";
  probe->saw = &saw_secret;
  probe->in_names = {"x"};
  probe->out_names = {"y"};

  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"y"};
  c.nodes.push_back(std::move(probe));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 2.0f}));
  vs.put("secret", F32(2, 1, {9.0f, 9.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  EXPECT_FALSE(saw_secret);
  EXPECT_TRUE(vs.has("secret"));  // still there outside
}

TEST_F(CompositeTest, ConstantsRemainVisibleInsideTheSubGraph) {
  // Constants are model-level, not scope-level: a sub-graph must still be
  // able to reference the tensors its operators were compiled against.
  bool saw_constant = false;
  cs["shared_const"] =
      std::make_shared<omle::rt::Tensor>(F32(1, 2, {1.0f, 2.0f}));

  auto probe = std::make_unique<ProbeNode>();
  probe->probe = "shared_const";
  probe->saw = &saw_constant;
  probe->in_names = {"x"};
  probe->out_names = {"y"};

  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"y"};
  c.nodes.push_back(std::move(probe));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 2.0f}));
  ASSERT_TRUE(c.execute(vs, 2).ok());

  EXPECT_TRUE(saw_constant);
}

TEST_F(CompositeTest, NestedCompositesEachGetTheirOwnScope) {
  auto inner = std::make_unique<CompositeNode>();
  inner->in_names = {"mid"};
  inner->input_aliases = {{"mid", "in"}};
  inner->out_names = {"mid_out"};
  inner->output_aliases = {{"out", "mid_out"}};
  inner->nodes.push_back(scale(5.0f, "in", "out"));

  CompositeNode outer;
  outer.in_names = {"x"};
  outer.out_names = {"y"};
  outer.nodes.push_back(scale(2.0f, "x", "mid"));
  outer.nodes.push_back(std::move(inner));
  outer.nodes.push_back(scale(1.0f, "mid_out", "y"));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 3.0f}));
  ASSERT_TRUE(outer.execute(vs, 2).ok());

  const auto& y = vs.get("y");
  EXPECT_NEAR(y.at(0, 0), 10.0f, 1e-6f);  // 1 * 2 * 5
  EXPECT_NEAR(y.at(1, 0), 30.0f, 1e-6f);  // 3 * 2 * 5
  // Neither the outer local nor the inner local leaks out.
  EXPECT_FALSE(vs.has("mid"));
  EXPECT_FALSE(vs.has("mid_out"));
  EXPECT_FALSE(vs.has("out"));
}

// =============================================================================
// Error propagation
// =============================================================================

TEST_F(CompositeTest, AChildFailureAbortsTheCompositeAndPropagates) {
  bool third_ran = false;

  auto failing = std::make_unique<FailingNode>();

  auto never = std::make_unique<FailingNode>();
  never->ran = &third_ran;

  CompositeNode c;
  c.in_names = {"x"};
  c.out_names = {"y"};
  c.nodes.push_back(scale(2.0f, "x", "t"));
  c.nodes.push_back(std::move(failing));
  c.nodes.push_back(std::move(never));

  auto vs = make_vs();
  vs.put("x", F32(2, 1, {1.0f, 2.0f}));

  auto st = c.execute(vs, 2);
  EXPECT_FALSE(st.ok());
  EXPECT_EQ(st.code(), omle::rt::ErrorCode::InvalidArgument);
  EXPECT_EQ(st.message(), "composite child failed");

  // Execution stops at the first failure ...
  EXPECT_FALSE(third_ran);
  // ... and nothing is exported on the error path.
  EXPECT_FALSE(vs.has("y"));
}
