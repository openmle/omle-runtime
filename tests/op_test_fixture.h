#ifndef OMLE_TESTS_OP_TEST_FIXTURE_H_
#define OMLE_TESTS_OP_TEST_FIXTURE_H_

// Shared fixture for the operator-level test suites (test_operators_extended,
// test_knn). Operators are exercised through OperatorNode so that registry
// dispatch and attribute resolution are covered alongside the kernel itself.

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "operator_node.h"
#include "operator_registry.h"
#include "runtime_tensor.h"

namespace omle_test {

using namespace omle::rt::impl;

static const float kNaN = std::numeric_limits<float>::quiet_NaN();

class OpTest : public ::testing::Test {
 protected:
  void SetUp() override { register_builtin_operators(); }

  ConstantStore cs;
  ValueStore make_vs() { return ValueStore(cs); }

  Tensor Tf(int rows, int cols, std::vector<float> data) {
    Tensor t(rows, cols);
    t.set_floats(std::move(data));
    return t;
  }

  Tensor Tstr(int rows, int cols, std::vector<std::string> data) {
    return Tensor::strings(rows, cols, std::move(data));
  }

  void put_const(const std::string& name, int rows, int cols,
                 std::vector<float> data) {
    cs[name] = std::make_shared<Tensor>(Tf(rows, cols, std::move(data)));
  }

  void put_const_str(const std::string& name, std::vector<std::string> data) {
    // n_cols must be read before `data` is moved from: argument evaluation
    // order is unspecified, and GCC on x86-64 evaluates right-to-left, which
    // would otherwise hand Tensor::strings a column count of 0.
    const int n = static_cast<int>(data.size());
    cs[name] = std::make_shared<Tensor>(Tensor::strings(1, n, std::move(data)));
  }

  void exec(ValueStore& vs, const std::string& domain, const std::string& op,
            std::vector<std::string> ins, std::vector<std::string> outs,
            AttributeMap attrs = {}) {
    OperatorNode node;
    node.domain = domain;
    node.op = op;
    node.in_names = std::move(ins);
    node.out_names = std::move(outs);
    node.attrs = std::move(attrs);
    auto st = node.execute(vs, vs.has(ins.empty() ? "" : ins[0])
                                   ? vs.get(ins.empty() ? "" : ins[0]).n_rows
                                   : 1);
    ASSERT_TRUE(st.ok()) << st.message();
  }

  // Execute with explicit n_rows.
  void exec_n(ValueStore& vs, int n_rows, const std::string& domain,
              const std::string& op, std::vector<std::string> ins,
              std::vector<std::string> outs, AttributeMap attrs = {}) {
    OperatorNode node;
    node.domain = domain;
    node.op = op;
    node.in_names = std::move(ins);
    node.out_names = std::move(outs);
    node.attrs = std::move(attrs);
    auto st = node.execute(vs, n_rows);
    ASSERT_TRUE(st.ok()) << st.message();
  }

  AttrVal aint(std::vector<int64_t> v) {
    AttrVal a;
    a.kind = AttrVal::Kind::Ints;
    a.ints = std::move(v);
    return a;
  }
  AttrVal aflt(double v) {
    AttrVal a;
    a.kind = AttrVal::Kind::Float;
    a.f = v;
    return a;
  }
  AttrVal afloats(std::vector<double> v) {
    AttrVal a;
    a.kind = AttrVal::Kind::Floats;
    a.floats = std::move(v);
    return a;
  }
  AttrVal astr(std::string s) {
    AttrVal a;
    a.kind = AttrVal::Kind::String;
    a.s = std::move(s);
    return a;
  }
  AttrVal astrs(std::vector<std::string> v) {
    AttrVal a;
    a.kind = AttrVal::Kind::Strings;
    a.strings = std::move(v);
    return a;
  }
  AttrVal abool(bool b) {
    AttrVal a;
    a.kind = AttrVal::Kind::Bool;
    a.b = b;
    return a;
  }
  AttrVal atensor(std::string name) {
    AttrVal a;
    a.tensor_name = std::move(name);
    return a;
  }
};

}  // namespace omle_test

#endif  // OMLE_TESTS_OP_TEST_FIXTURE_H_
