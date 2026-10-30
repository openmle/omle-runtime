// Unit tests for the C++ public API (Tensor extended) and the C API
// (omle_tensor_*, omle_model_*, omle_session_*).
//
// Requires the proto-generated header to build fixture models (same as
// test_runtime.cpp).

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "omle.pb.h"
#include "omle/c_api.h"
#include "omle/runtime.h"

namespace P = omle;
using namespace omle::rt;

// ============================================================
// Shared fixture helpers (mirror of test_runtime.cpp)
// ============================================================

static P::Tree make_stump(int feat, float threshold, float lv, float rv) {
  P::Tree t;
  t.set_num_nodes(3);
  t.add_node_kind(P::Tree::BRANCH);
  t.add_node_kind(P::Tree::LEAF);
  t.add_node_kind(P::Tree::LEAF);
  t.add_split_feature(feat);
  t.add_split_feature(0);
  t.add_split_feature(0);
  {
    auto* st = t.mutable_split_threshold()->mutable_tensor();
    st->mutable_float32_data()->add_values(threshold);
    st->mutable_float32_data()->add_values(0.f);
    st->mutable_float32_data()->add_values(0.f);
  }
  t.add_split_op(P::Tree::LESS_THAN);
  t.add_split_op(P::Tree::SPLIT_OP_UNSPECIFIED);
  t.add_split_op(P::Tree::SPLIT_OP_UNSPECIFIED);
  t.add_children_index(1);
  t.add_children_index(2);
  t.add_children_offset(0);
  t.add_children_offset(-1);
  t.add_children_offset(-1);
  t.add_children_count(2);
  t.add_children_count(0);
  t.add_children_count(0);
  t.add_default_child(2);
  t.add_default_child(0);
  t.add_default_child(0);
  {
    auto* lf = t.mutable_leaf_value()->mutable_tensor();
    lf->mutable_float32_data()->add_values(0.f);
    lf->mutable_float32_data()->add_values(lv);
    lf->mutable_float32_data()->add_values(rv);
  }
  return t;
}

// Build a serialised model: n_features inputs, one TreeEnsemble node (SUM of
// stumps).
static std::string make_model_bytes(
    int n_features,
    const std::vector<std::tuple<int, float, float, float>>& stumps,
    const std::string& /*model_name*/ = "") {
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

// ============================================================
// Tensor — extended C++ API tests
// ============================================================

TEST(TensorExt, ScalarKind_F32) {
  Tensor t = Tensor::f32_scalar(3.14f);
  EXPECT_TRUE(t.is_scalar());
  EXPECT_FALSE(t.is_sparse());
  EXPECT_EQ(t.dtype, DataType::Float32);
  EXPECT_EQ(t.n_rows, 1);
  EXPECT_EQ(t.n_cols, 1);
  EXPECT_FLOAT_EQ(t.at(0, 0), 3.14f);
  EXPECT_FLOAT_EQ(static_cast<float>(t.get(0, 0)), 3.14);
  EXPECT_TRUE(t.is_scalar());  // scalar stores inline, not in data
}

TEST(TensorExt, ScalarKind_Int64) {
  Tensor t = Tensor::scalar(DataType::Int64, 42.0);
  EXPECT_TRUE(t.is_scalar());
  EXPECT_EQ(t.dtype, DataType::Int64);
  EXPECT_DOUBLE_EQ(t.get(0, 0), 42.0);
}

TEST(TensorExt, DenseNonFloat_Int32) {
  Tensor t = Tensor::dense(DataType::Int32, 2, 3);
  EXPECT_TRUE(t.is_dense());
  EXPECT_FALSE(t.is_scalar());
  EXPECT_EQ(t.dtype, DataType::Int32);
  EXPECT_EQ(t.n_rows, 2);
  EXPECT_EQ(t.n_cols, 3);
  // All elements zero-initialised
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 3; ++c) EXPECT_DOUBLE_EQ(t.get(r, c), 0.0);
}

TEST(TensorExt, DenseNonFloat_Float64) {
  Tensor t = Tensor::dense(DataType::Float64, 1, 2);
  EXPECT_EQ(t.dtype, DataType::Float64);
  EXPECT_EQ(t.numel(), 2);
  // Write via f64_ptr()
  auto* p = t.f64_ptr();
  p[0] = 1.5;
  p[1] = 2.5;
  EXPECT_DOUBLE_EQ(t.get(0, 0), 1.5);
  EXPECT_DOUBLE_EQ(t.get(0, 1), 2.5);
}

TEST(TensorExt, FromBuffer_NonOwning) {
  float buf[] = {1.f, 2.f, 3.f, 4.f};
  const Tensor t = Tensor::from_buffer(buf, DataType::Float32, 2, 2);
  EXPECT_TRUE(t.is_view());
  EXPECT_EQ(t.n_rows, 2);
  EXPECT_EQ(t.n_cols, 2);
  // Views are read-only: use const accessors (non-const f32_ptr bypasses
  // view_ptr_).
  EXPECT_FLOAT_EQ(t.at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(t.at(0, 1), 2.f);
  EXPECT_FLOAT_EQ(t.at(1, 1), 4.f);
  // is_view() already asserted above; no separate data.empty() needed
}

TEST(TensorExt, SparseCSR_BasicAccessors) {
  // Sparse 3×4 with 3 stored values at (0,1)=1.5, (1,3)=2.5, (2,0)=3.5
  std::vector<float> vals = {1.5f, 2.5f, 3.5f};
  std::vector<int32_t> indices = {1, 3, 0};
  std::vector<int32_t> indptr = {0, 1, 2, 3};
  Tensor t = Tensor::sparse_csr_f32(3, 4, vals, indices, indptr, 0.f);

  EXPECT_TRUE(t.is_sparse());
  EXPECT_FALSE(t.is_dense());
  EXPECT_EQ(t.dtype, DataType::Float32);
  EXPECT_EQ(t.n_rows, 3);
  EXPECT_EQ(t.n_cols, 4);
  EXPECT_EQ(t.nnz(), 3);
  EXPECT_DOUBLE_EQ(t.sp_default(), 0.0);

  // Stored values
  EXPECT_FLOAT_EQ(static_cast<float>(t.get(0, 1)), 1.5f);
  EXPECT_FLOAT_EQ(static_cast<float>(t.get(1, 3)), 2.5f);
  EXPECT_FLOAT_EQ(static_cast<float>(t.get(2, 0)), 3.5f);
  // Fill value for unstored cells
  EXPECT_FLOAT_EQ(static_cast<float>(t.get(0, 0)), 0.f);
  EXPECT_FLOAT_EQ(static_cast<float>(t.get(1, 0)), 0.f);

  // Row accessors
  EXPECT_EQ(t.row_nnz(0), 1);
  EXPECT_EQ(t.row_nnz(1), 1);
  EXPECT_EQ(t.row_nnz(2), 1);
  EXPECT_EQ(t.row_indices(0)[0], 1);
  EXPECT_FLOAT_EQ(t.row_values_f32(0)[0], 1.5f);
}

TEST(TensorExt, SparseAutoDowngrade_ToDense) {
  // 2×2 with 3 stored out of 4 (75% fill > 50% threshold) → downgraded to
  // Dense.
  std::vector<float> vals = {1.f, 2.f, 3.f};
  std::vector<int32_t> indices = {0, 1, 0};
  std::vector<int32_t> indptr = {0, 2, 3};
  Tensor t = Tensor::sparse_csr_f32(2, 2, vals, indices, indptr, 0.f);
  EXPECT_TRUE(t.is_dense());  // auto-downgraded
  EXPECT_FALSE(t.is_sparse());
  EXPECT_FLOAT_EQ(t.at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(t.at(0, 1), 2.f);
  EXPECT_FLOAT_EQ(t.at(1, 0), 3.f);
  EXPECT_FLOAT_EQ(t.at(1, 1), 0.f);
}

TEST(TensorExt, ToFloat32_FromInt32Dense) {
  Tensor src = Tensor::dense(DataType::Int32, 1, 3);
  auto* p = reinterpret_cast<int32_t*>(const_cast<void*>(src.raw_data()));
  p[0] = 7;
  p[1] = -3;
  p[2] = 100;
  Tensor f = src.to_float32();
  EXPECT_EQ(f.dtype, DataType::Float32);
  EXPECT_FLOAT_EQ(f.at(0, 0), 7.f);
  EXPECT_FLOAT_EQ(f.at(0, 1), -3.f);
  EXPECT_FLOAT_EQ(f.at(0, 2), 100.f);
}

TEST(TensorExt, ToFloat32_FromSparse) {
  std::vector<float> vals = {5.f, 9.f};
  std::vector<int32_t> indices = {0, 2};
  std::vector<int32_t> indptr = {0, 1, 2};
  Tensor s = Tensor::sparse_csr_f32(2, 3, vals, indices, indptr, 0.f);
  ASSERT_TRUE(s.is_sparse());
  Tensor d = s.to_float32();
  EXPECT_FALSE(d.is_sparse());
  EXPECT_EQ(d.dtype, DataType::Float32);
  EXPECT_FLOAT_EQ(d.at(0, 0), 5.f);
  EXPECT_FLOAT_EQ(d.at(0, 1), 0.f);
  EXPECT_FLOAT_EQ(d.at(0, 2), 0.f);
  EXPECT_FLOAT_EQ(d.at(1, 2), 9.f);
}

TEST(TensorExt, ToSparse_FromDense) {
  Tensor d = Tensor::from_floats(2, 3, {1.f, 0.f, 0.f, 0.f, 0.f, 2.f});
  Tensor s = Tensor::to_sparse(d, 0.f);
  EXPECT_TRUE(s.is_sparse());
  EXPECT_EQ(s.nnz(), 2);
  EXPECT_FLOAT_EQ(static_cast<float>(s.get(0, 0)), 1.f);
  EXPECT_FLOAT_EQ(static_cast<float>(s.get(1, 2)), 2.f);
  EXPECT_FLOAT_EQ(static_cast<float>(s.get(0, 1)), 0.f);  // fill
}

TEST(TensorExt, CopyDeepClone_Sparse) {
  std::vector<float> vals = {1.f};
  std::vector<int32_t> indices = {0};
  std::vector<int32_t> indptr = {0, 1};
  Tensor src = Tensor::sparse_csr_f32(1, 2, vals, indices, indptr, -1.f);
  Tensor cpy = src;  // copy ctor

  EXPECT_FLOAT_EQ(static_cast<float>(cpy.get(0, 0)), 1.f);
  EXPECT_DOUBLE_EQ(cpy.sp_default(), -1.0);

  // Mutating src's sp_values should not affect the copy — stores are
  // independent. (SparseStore is heap-allocated and deep-copied.)
  EXPECT_NE(&src.sp_values(), &cpy.sp_values());
}

TEST(TensorExt, CopyDeepClone_String) {
  Tensor src = Tensor::strings(2, 1, {"hello", "world"});
  Tensor cpy = src;
  EXPECT_EQ(cpy.str_at(0, 0), "hello");
  EXPECT_EQ(cpy.str_at(1, 0), "world");
  // Modify copy; source must be unchanged.
  cpy.str_at(0, 0) = "modified";
  EXPECT_EQ(src.str_at(0, 0), "hello");
}

TEST(TensorExt, MoveSemantics) {
  Tensor src = Tensor::from_floats(2, 2, {1.f, 2.f, 3.f, 4.f});
  Tensor dst = std::move(src);
  EXPECT_EQ(dst.n_rows, 2);
  EXPECT_FLOAT_EQ(dst.at(0, 0), 1.f);
  // src is in a valid-but-unspecified state after move; do not check its values
}

TEST(TensorExt, Predicates) {
  Tensor empty;
  EXPECT_TRUE(empty.empty());
  EXPECT_EQ(empty.numel(), 0);

  Tensor dense = Tensor::f32(2, 3);
  EXPECT_FALSE(dense.empty());
  EXPECT_TRUE(dense.is_dense());
  EXPECT_FALSE(dense.is_sparse());
  EXPECT_FALSE(dense.is_scalar());
  EXPECT_FALSE(dense.is_string());
  EXPECT_EQ(dense.numel(), 6);

  Tensor sc = Tensor::f32_scalar(1.f);
  EXPECT_TRUE(sc.is_scalar());
  EXPECT_TRUE(sc.is_dense());  // scalar is !Sparse → is_dense() true
  EXPECT_EQ(sc.numel(), 1);

  std::vector<float> vals = {1.f};
  std::vector<int32_t> indices = {0};
  std::vector<int32_t> indptr = {0, 1};
  Tensor sp = Tensor::sparse_csr_f32(1, 3, vals, indices, indptr, 0.f);
  EXPECT_TRUE(sp.is_sparse());
  EXPECT_FALSE(sp.is_dense());
  EXPECT_FALSE(sp.is_scalar());

  Tensor str = Tensor::strings(1, 2, {"a", "b"});
  EXPECT_TRUE(str.is_string());
  EXPECT_FALSE(str.is_scalar());
  EXPECT_FALSE(str.is_sparse());
}

TEST(TensorExt, SetFloats) {
  Tensor t = Tensor::f32(2, 2);
  t.set_floats({1.f, 2.f, 3.f, 4.f});
  EXPECT_FLOAT_EQ(t.at(0, 0), 1.f);
  EXPECT_FLOAT_EQ(t.at(1, 1), 4.f);
}

TEST(TensorExt, GetUniversal_Sparse) {
  std::vector<float> vals = {7.f};
  std::vector<int32_t> indices = {1};
  std::vector<int32_t> indptr = {0, 0, 1};  // row 0 empty, row 1 has col 1
  Tensor t = Tensor::sparse_csr_f32(2, 3, vals, indices, indptr, -1.f);
  EXPECT_DOUBLE_EQ(t.get(0, 0), -1.0);  // fill
  EXPECT_DOUBLE_EQ(t.get(1, 1), 7.0);
  EXPECT_DOUBLE_EQ(t.get(1, 0), -1.0);  // fill
}

TEST(TensorExt, DataBytes) {
  Tensor sc = Tensor::f32_scalar(1.f);
  EXPECT_EQ(sc.data_bytes(), sizeof(float));

  Tensor d = Tensor::f32(3, 2);
  EXPECT_EQ(d.data_bytes(), 3 * 2 * sizeof(float));

  std::vector<float> vals = {1.f};
  std::vector<int32_t> indices = {0};
  std::vector<int32_t> indptr = {0, 1};
  Tensor sp = Tensor::sparse_csr_f32(1, 2, vals, indices, indptr, 0.f);
  EXPECT_EQ(sp.data_bytes(), 0u);  // sparse: no dense buffer

  Tensor str = Tensor::strings(2, 1, {"a", "b"});
  EXPECT_EQ(str.data_bytes(), 0u);  // string: no byte buffer
}

TEST(TensorExt, StringTensor_Factory) {
  Tensor t = Tensor::strings(2, 2, {"a", "b", "c", "d"});
  EXPECT_TRUE(t.is_string());
  EXPECT_EQ(t.dtype, DataType::String);
  EXPECT_EQ(t.n_rows, 2);
  EXPECT_EQ(t.n_cols, 2);
  EXPECT_EQ(t.str_at(0, 0), "a");
  EXPECT_EQ(t.str_at(0, 1), "b");
  EXPECT_EQ(t.str_at(1, 0), "c");
  EXPECT_EQ(t.str_at(1, 1), "d");
}

TEST(TensorExt, StringTensor_MutableStrAt) {
  Tensor t = Tensor::strings(1, 3, {"x", "y", "z"});
  t.str_at(0, 1) = "Y";
  EXPECT_EQ(t.str_at(0, 0), "x");
  EXPECT_EQ(t.str_at(0, 1), "Y");
  EXPECT_EQ(t.str_at(0, 2), "z");
}

TEST(TensorExt, StringTensor_SetStrings) {
  Tensor t;
  t.set_strings({"alpha", "beta", "gamma"});
  EXPECT_TRUE(t.is_string());
  EXPECT_EQ(t.str_data().size(), 3u);
  EXPECT_EQ(t.str_at(0, 0), "alpha");
}

TEST(TensorExt, StringTensor_DefaultFill) {
  // Strings factory pads with empty string if vector is too short.
  Tensor t = Tensor::strings(2, 2);  // no values provided
  EXPECT_EQ(t.n_rows, 2);
  EXPECT_EQ(t.n_cols, 2);
  EXPECT_EQ(t.str_data().size(), 4u);
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 2; ++c) EXPECT_EQ(t.str_at(r, c), "");
}

// ============================================================
// C API — Tensor
// ============================================================

TEST(CApiTensor, CreateF32_Basic) {
  float data[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
  omle_tensor_t* t = omle_tensor_create_f32(2, 3, data);
  ASSERT_NE(t, nullptr);

  EXPECT_EQ(omle_tensor_dtype(t), OMLE_DTYPE_FLOAT32);
  EXPECT_EQ(omle_tensor_kind(t), OMLE_TENSOR_DENSE);
  EXPECT_EQ(omle_tensor_n_rows(t), 2);
  EXPECT_EQ(omle_tensor_n_cols(t), 3);
  EXPECT_EQ(omle_tensor_numel(t), 6);
  EXPECT_EQ(omle_tensor_data_bytes(t), 6 * sizeof(float));
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 1, 2), 6.0);

  const float* raw = static_cast<const float*>(omle_tensor_data(t));
  ASSERT_NE(raw, nullptr);
  EXPECT_FLOAT_EQ(raw[0], 1.f);
  EXPECT_FLOAT_EQ(raw[5], 6.f);

  omle_free_tensor(t);
}

TEST(CApiTensor, CreateF32_NullData_ZeroFill) {
  omle_tensor_t* t = omle_tensor_create_f32(3, 2, nullptr);
  ASSERT_NE(t, nullptr);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 2; ++c) EXPECT_DOUBLE_EQ(omle_tensor_get(t, r, c), 0.0);
  omle_free_tensor(t);
}

TEST(CApiTensor, CreateDense_Float64) {
  double data[] = {1.5, 2.5, 3.5};
  omle_tensor_t* t =
      omle_tensor_create_dense(OMLE_DTYPE_FLOAT64, 1, 3, data, sizeof(data));
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(omle_tensor_dtype(t), OMLE_DTYPE_FLOAT64);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 0, 0), 1.5);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 0, 2), 3.5);
  omle_free_tensor(t);
}

TEST(CApiTensor, CreateDense_Int64) {
  int64_t data[] = {10, 20, 30};
  omle_tensor_t* t =
      omle_tensor_create_dense(OMLE_DTYPE_INT64, 1, 3, data, sizeof(data));
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(omle_tensor_dtype(t), OMLE_DTYPE_INT64);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 0, 0), 10.0);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 0, 1), 20.0);
  omle_free_tensor(t);
}

TEST(CApiTensor, CreateSparseCSR) {
  // 3×4 sparse: (0,1)=1.5, (1,3)=2.5
  float vals[] = {1.5f, 2.5f};
  int32_t indices[] = {1, 3};
  int32_t indptr[] = {0, 1, 2, 2};
  omle_tensor_t* t = omle_tensor_create_sparse_csr(
      OMLE_DTYPE_FLOAT32, 3, 4, vals, sizeof(vals), indices, 2, indptr, 0.0);
  ASSERT_NE(t, nullptr);
  EXPECT_EQ(omle_tensor_kind(t), OMLE_TENSOR_SPARSE);
  EXPECT_EQ(omle_tensor_n_rows(t), 3);
  EXPECT_EQ(omle_tensor_n_cols(t), 4);
  EXPECT_EQ(omle_tensor_sp_nnz(t), 2);
  EXPECT_DOUBLE_EQ(omle_tensor_sp_fill(t), 0.0);

  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 0, 1), 1.5);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 1, 3), 2.5);
  EXPECT_DOUBLE_EQ(omle_tensor_get(t, 2, 0), 0.0);  // fill

  ASSERT_NE(omle_tensor_sp_values(t), nullptr);
  ASSERT_NE(omle_tensor_sp_indices(t), nullptr);
  ASSERT_NE(omle_tensor_sp_indptr(t), nullptr);
  EXPECT_GT(omle_tensor_sp_values_bytes(t), 0u);

  EXPECT_EQ(omle_tensor_data(t), nullptr);  // no dense buffer for sparse
  omle_free_tensor(t);
}

TEST(CApiTensor, DestroyNull_NocraSH) {
  omle_free_tensor(nullptr);  // must not crash
}

// ============================================================
// C API — Model
// ============================================================

class CApiModelTest : public ::testing::Test {
 protected:
  std::string bytes_;
  omle_model_t* model_ = nullptr;

  void SetUp() override {
    bytes_ = make_model_bytes(2, {{0, 0.5f, 1.0f, -1.0f}}, "test_model");
    ASSERT_EQ(
        omle_model_load_memory(bytes_.data(), bytes_.size(), nullptr, &model_),
        OMLE_OK);
  }
  void TearDown() override { omle_free_model(model_); }
};

TEST_F(CApiModelTest, NumInputsOutputs) {
  EXPECT_EQ(omle_model_num_inputs(model_), 2);
  EXPECT_EQ(omle_model_num_outputs(model_), 1);
}

TEST_F(CApiModelTest, InputSpec) {
  omle_tensor_spec_t spec;
  ASSERT_EQ(omle_model_input_spec(model_, 0, &spec), OMLE_OK);
  EXPECT_STREQ(spec.name, "f0");
  ASSERT_EQ(omle_model_input_spec(model_, 1, &spec), OMLE_OK);
  EXPECT_STREQ(spec.name, "f1");
}

TEST_F(CApiModelTest, InputSpec_OutOfRange) {
  omle_tensor_spec_t spec;
  EXPECT_NE(omle_model_input_spec(model_, 99, &spec), OMLE_OK);
}

TEST_F(CApiModelTest, OutputSpec) {
  omle_tensor_spec_t spec;
  ASSERT_EQ(omle_model_output_spec(model_, 0, &spec), OMLE_OK);
  EXPECT_STREQ(spec.name, "score");
}

TEST_F(CApiModelTest, TaskType) {
  // make_model_bytes doesn't set task_type → UNSPECIFIED
  EXPECT_EQ(omle_model_task_type(model_), OMLE_TASK_UNSPECIFIED);
}

TEST_F(CApiModelTest, Predict_KnownValues) {
  // Stump: f0 < 0.5 → +1, else -1; f1 ignored.
  // Two samples: [0,0] → +1, [1,0] → -1
  float feat[] = {0.f, 0.f, 1.f, 0.f};
  omle_tensor_t* in_tensor = omle_tensor_create_f32(2, 2, feat);
  ASSERT_NE(in_tensor, nullptr);

  const char* in_names[] = {"input"};
  omle_tensor_t* in_arr[] = {in_tensor};
  int n_out = 0;
  char** out_names = nullptr;
  omle_tensor_t** out_tensors = nullptr;

  auto st = omle_model_predict(model_, 1, in_names, in_arr, 0, nullptr, &n_out,
                               &out_names, &out_tensors);
  ASSERT_EQ(st, OMLE_OK);
  ASSERT_EQ(n_out, 1);
  ASSERT_NE(out_tensors[0], nullptr);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out_tensors[0], 0, 0), 1.0);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out_tensors[0], 1, 0), -1.0);

  omle_free_strings(out_names, n_out);
  omle_free_tensors(out_tensors, n_out);
  omle_free_tensor(in_tensor);
}

TEST_F(CApiModelTest, Predict_OutputFilter) {
  float feat[] = {0.f, 0.f};
  omle_tensor_t* in_tensor = omle_tensor_create_f32(1, 2, feat);
  ASSERT_NE(in_tensor, nullptr);

  const char* in_names[] = {"input"};
  omle_tensor_t* in_arr[] = {in_tensor};
  const char* req_names[] = {"score"};
  int n_out = 0;
  char** out_names = nullptr;
  omle_tensor_t** out_tensors = nullptr;

  ASSERT_EQ(omle_model_predict(model_, 1, in_names, in_arr, 1, req_names,
                               &n_out, &out_names, &out_tensors),
            OMLE_OK);
  EXPECT_EQ(n_out, 1);
  EXPECT_STREQ(out_names[0], "score");

  omle_free_strings(out_names, n_out);
  omle_free_tensors(out_tensors, n_out);
  omle_free_tensor(in_tensor);
}

TEST(CApiModel, LoadFile_Error_SetsLastError) {
  omle_model_t* model = nullptr;
  auto st = omle_model_load_file("/nonexistent/model.omle", nullptr, &model);
  EXPECT_NE(st, OMLE_OK);
  EXPECT_EQ(model, nullptr);
  const char* msg = omle_last_error();
  EXPECT_NE(msg, nullptr);
  EXPECT_GT(strlen(msg), 0u);
}

TEST(CApiModel, LoadMemory_InvalidData) {
  const char bad[] = "not a valid proto";
  omle_model_t* model = nullptr;
  auto st = omle_model_load_memory(bad, sizeof(bad), nullptr, &model);
  EXPECT_NE(st, OMLE_OK);
  EXPECT_EQ(model, nullptr);
}

TEST(CApiModel, DestroyNull_NoCrash) { omle_free_model(nullptr); }

// ============================================================
// C API — Session
// ============================================================

class CApiSessionTest : public ::testing::Test {
 protected:
  std::string bytes_;
  omle_model_t* model_ = nullptr;
  omle_session_t* session_ = nullptr;

  static constexpr int n_feat = 2;
  // Stump: f0 < 0.5 → +2, else -2
  static constexpr float LV = 2.f, RV = -2.f;

  void SetUp() override {
    bytes_ = make_model_bytes(n_feat, {{0, 0.5f, LV, RV}});
    ASSERT_EQ(
        omle_model_load_memory(bytes_.data(), bytes_.size(), nullptr, &model_),
        OMLE_OK);
    ASSERT_EQ(omle_session_create(model_, &session_), OMLE_OK);
  }
  void TearDown() override {
    omle_free_session(session_);
    omle_free_model(model_);
  }
};

TEST_F(CApiSessionTest, NumInputsOutputs) {
  EXPECT_EQ(omle_session_num_inputs(session_), n_feat);
  EXPECT_EQ(omle_session_num_outputs(session_), 1);
}

TEST_F(CApiSessionTest, BindAndRun_OneSample) {
  float feat[] = {0.f, 0.f};  // f0 < 0.5 → +2
  omle_tensor_t* t = omle_tensor_create_f32(1, n_feat, feat);
  ASSERT_NE(t, nullptr);
  ASSERT_EQ(omle_session_bind_input(session_, "input", t), OMLE_OK);
  ASSERT_EQ(omle_session_run(session_), OMLE_OK);

  omle_tensor_t* out = nullptr;
  ASSERT_EQ(omle_session_get_output(session_, "score", &out), OMLE_OK);
  ASSERT_NE(out, nullptr);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out, 0, 0), LV);

  omle_free_tensor(t);
}

TEST_F(CApiSessionTest, BindAndRun_Batch) {
  // Two samples: [0,0]→+2, [1,0]→-2
  float feat[] = {0.f, 0.f, 1.f, 0.f};
  omle_tensor_t* t = omle_tensor_create_f32(2, n_feat, feat);
  ASSERT_NE(t, nullptr);
  ASSERT_EQ(omle_session_bind_input(session_, "input", t), OMLE_OK);
  ASSERT_EQ(omle_session_run(session_), OMLE_OK);

  omle_tensor_t* out = nullptr;
  ASSERT_EQ(omle_session_get_output(session_, "score", &out), OMLE_OK);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out, 0, 0), LV);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out, 1, 0), RV);

  omle_free_tensor(t);
}

TEST_F(CApiSessionTest, GetOutput_NotFound) {
  float feat[] = {0.f, 0.f};
  omle_tensor_t* t = omle_tensor_create_f32(1, n_feat, feat);
  ASSERT_NE(t, nullptr);
  omle_session_bind_input(session_, "input", t);
  omle_session_run(session_);

  omle_tensor_t* out = nullptr;
  auto st = omle_session_get_output(session_, "nonexistent", &out);
  EXPECT_NE(st, OMLE_OK);
  EXPECT_EQ(out, nullptr);

  omle_free_tensor(t);
}

TEST_F(CApiSessionTest, ClearInputs_ThenRebind) {
  float feat[] = {0.f, 0.f};
  omle_tensor_t* t = omle_tensor_create_f32(1, n_feat, feat);
  ASSERT_NE(t, nullptr);

  omle_session_bind_input(session_, "input", t);
  ASSERT_EQ(omle_session_run(session_), OMLE_OK);

  omle_session_clear_inputs(session_);
  omle_session_bind_input(session_, "input", t);
  ASSERT_EQ(omle_session_run(session_), OMLE_OK);

  omle_tensor_t* out = nullptr;
  ASSERT_EQ(omle_session_get_output(session_, "score", &out), OMLE_OK);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out, 0, 0), LV);

  omle_free_tensor(t);
}

TEST_F(CApiSessionTest, BindOutput_FiltersResults) {
  float feat[] = {0.f, 0.f};
  omle_tensor_t* t = omle_tensor_create_f32(1, n_feat, feat);
  ASSERT_NE(t, nullptr);

  omle_tensor_t* out_slot = omle_tensor_create_f32(1, 1, nullptr);
  ASSERT_NE(out_slot, nullptr);

  omle_session_bind_input(session_, "input", t);
  omle_session_bind_output(session_, "score", out_slot);
  ASSERT_EQ(omle_session_run(session_), OMLE_OK);

  omle_tensor_t* out = nullptr;
  ASSERT_EQ(omle_session_get_output(session_, "score", &out), OMLE_OK);
  ASSERT_NE(out, nullptr);

  omle_free_tensor(t);
  omle_free_tensor(out_slot);
}

TEST_F(CApiSessionTest, RunTwice_SameResult) {
  float feat[] = {0.f, 0.f};
  omle_tensor_t* t = omle_tensor_create_f32(1, n_feat, feat);
  ASSERT_NE(t, nullptr);
  omle_session_bind_input(session_, "input", t);

  ASSERT_EQ(omle_session_run(session_), OMLE_OK);
  double first = omle_tensor_get(
      [&] {
        omle_tensor_t* o = nullptr;
        omle_session_get_output(session_, "score", &o);
        return o;
      }(),
      0, 0);

  ASSERT_EQ(omle_session_run(session_), OMLE_OK);
  omle_tensor_t* out = nullptr;
  ASSERT_EQ(omle_session_get_output(session_, "score", &out), OMLE_OK);
  EXPECT_DOUBLE_EQ(omle_tensor_get(out, 0, 0), first);

  omle_free_tensor(t);
}

TEST(CApiSession, DestroyNull_NoCrash) { omle_free_session(nullptr); }

TEST(CApiSession, CreateFromNull_Error) {
  omle_session_t* s = nullptr;
  // Passing a null model must not crash and must fail gracefully.
  auto st = omle_session_create(nullptr, &s);
  EXPECT_NE(st, OMLE_OK);
  EXPECT_EQ(s, nullptr);
}
