#include "omle/c_api.h"

#include <cstring>
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include "omle/runtime.h"
#include "omle/status.h"
#include "omle/tensor.h"

// ---------------------------------------------------------------------------
// Thread-local error message
// ---------------------------------------------------------------------------

static thread_local std::string tl_last_error;

static omle_status_t set_error(omle_status_t code, std::string_view msg) {
  tl_last_error = msg;
  return code;
}

static omle_status_t translate_status(const omle::rt::Status& s) noexcept {
  tl_last_error = s.message();
  switch (s.code()) {
    case omle::rt::ErrorCode::FileNotFound:
      return OMLE_ERR_FILE_NOT_FOUND;
    case omle::rt::ErrorCode::FileReadError:
      return OMLE_ERR_FILE_READ;
    case omle::rt::ErrorCode::ParseError:
      return OMLE_ERR_PARSE;
    case omle::rt::ErrorCode::InvalidGraph:
      return OMLE_ERR_INVALID_GRAPH;
    case omle::rt::ErrorCode::VerificationFailed:
      return OMLE_ERR_VERIFICATION_FAILED;
    case omle::rt::ErrorCode::MissingInput:
      return OMLE_ERR_MISSING_INPUT;
    case omle::rt::ErrorCode::OutputNotProduced:
      return OMLE_ERR_OUTPUT_NOT_PRODUCED;
    case omle::rt::ErrorCode::OutputNotFound:
      return OMLE_ERR_OUTPUT_NOT_FOUND;
    case omle::rt::ErrorCode::UnknownOperator:
      return OMLE_ERR_UNKNOWN_OPERATOR;
    case omle::rt::ErrorCode::InvalidArgument:
      return OMLE_ERR_INVALID_ARGUMENT;
    default:
      return OMLE_ERR_UNKNOWN;
  }
}

// Used by constructors and other paths that still propagate exceptions
// directly.
static omle_status_t translate_exception() noexcept {
  try {
    throw;
  } catch (const std::exception& e) {
    tl_last_error = e.what();
    return OMLE_ERR_UNKNOWN;
  } catch (...) {
    tl_last_error = "unknown error";
    return OMLE_ERR_UNKNOWN;
  }
}

// ---------------------------------------------------------------------------
// Opaque handle definitions
// ---------------------------------------------------------------------------

struct omle_model_s {
  std::unique_ptr<omle::rt::Model> model;

  // Cached string vectors for spec accessors — stable pointers for the
  // model's lifetime.
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<std::vector<int64_t>> output_shapes;
};

struct omle_tensor_s {
  omle::rt::Tensor tensor;
};

struct RegisteredColumns {
  std::vector<std::string> names;
  std::vector<const char*> name_ptrs;  // stable: points into names[i]
  std::vector<int> types;
  std::string out_name;
  // Per-call reusable input tensors — pre-allocated at register_columns time,
  // updated in-place on each predict call to avoid per-call heap allocation.
  std::unordered_map<std::string, omle::rt::Tensor> cached_inputs;
  // Direct pointers into cached_inputs values — stable after initial build
  // (unordered_map uses node-based storage; refs survive rehash).
  std::vector<omle::rt::Tensor*> cached_ptrs;
  int cached_n_rows = 0;
};

struct omle_session_s {
  std::unique_ptr<omle::rt::Session> session;
  // Flat vector — output count is tiny (1–5 entries); cache-friendly linear
  // scan beats hash-map overhead.  Pointer stability is guaranteed because
  // reserve() at creation time prevents any reallocation.
  std::vector<std::pair<std::string, omle_tensor_s>> owned_outputs;
  // Optional pre-registered column schema for the fast registered-predict path.
  std::unique_ptr<RegisteredColumns> reg_cols;
};

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

extern "C" const char* omle_last_error(void) { return tl_last_error.c_str(); }

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

static omle::rt::LoadOptions to_cpp_opts(const omle_load_options_t* opts) {
  omle::rt::LoadOptions o;
  if (opts) {
    o.n_threads = opts->n_threads;
    o.min_parallel_rows = opts->min_parallel_rows;
  }
  return o;
}

static omle_model_s* make_model_handle(std::unique_ptr<omle::rt::Model> m) {
  auto* h = new omle_model_s;
  h->model = std::move(m);

  // Pre-cache shape vectors so InputSpec / OutputSpec shape pointers are
  // stable.
  const auto& inputs = h->model->inputs();
  h->input_shapes.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i)
    h->input_shapes[i] = inputs[i].shape;

  const auto& outputs = h->model->outputs();
  h->output_shapes.resize(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i)
    h->output_shapes[i] = outputs[i].shape;

  return h;
}

extern "C" omle_status_t omle_model_load_file(const char* path,
                                              const omle_load_options_t* opts,
                                              omle_model_t** out_model) {
  if (!path || !out_model)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  auto result = omle::rt::Model::load(std::string(path), to_cpp_opts(opts));
  if (!result.ok()) return translate_status(result.status());
  *out_model = make_model_handle(std::move(*result));
  return OMLE_OK;
}

extern "C" omle_status_t omle_model_load_memory(const void* data, size_t size,
                                                const omle_load_options_t* opts,
                                                omle_model_t** out_model) {
  if (!data || !out_model)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  auto result = omle::rt::Model::load(data, size, to_cpp_opts(opts));
  if (!result.ok()) return translate_status(result.status());
  *out_model = make_model_handle(std::move(*result));
  return OMLE_OK;
}

extern "C" void omle_free_model(omle_model_t* model) { delete model; }

// ---------------------------------------------------------------------------
// Model introspection
// ---------------------------------------------------------------------------

extern "C" int omle_model_num_inputs(const omle_model_t* model) {
  return model ? model->model->num_inputs() : 0;
}

extern "C" int omle_model_num_outputs(const omle_model_t* model) {
  return model ? model->model->num_outputs() : 0;
}

static omle_dtype_t to_c_dtype(omle::rt::DataType dt) {
  return static_cast<omle_dtype_t>(static_cast<uint8_t>(dt));
}

extern "C" omle_status_t omle_model_input_spec(const omle_model_t* model,
                                               int idx,
                                               omle_tensor_spec_t* out_spec) {
  if (!model || !out_spec)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  const auto& inputs = model->model->inputs();
  if (idx < 0 || static_cast<size_t>(idx) >= inputs.size())
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "index out of range");
  const auto& s = inputs[idx];
  out_spec->name = s.name.c_str();
  out_spec->dtype = to_c_dtype(s.dtype);
  out_spec->shape = model->input_shapes[idx].data();
  out_spec->shape_rank = static_cast<int>(model->input_shapes[idx].size());
  out_spec->role = OMLE_ROLE_UNSPECIFIED;
  return OMLE_OK;
}

extern "C" omle_status_t omle_model_output_spec(const omle_model_t* model,
                                                int idx,
                                                omle_tensor_spec_t* out_spec) {
  if (!model || !out_spec)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  const auto& outputs = model->model->outputs();
  if (idx < 0 || static_cast<size_t>(idx) >= outputs.size())
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "index out of range");
  const auto& s = outputs[idx];
  out_spec->name = s.name.c_str();
  out_spec->dtype = to_c_dtype(s.dtype);
  out_spec->shape = model->output_shapes[idx].data();
  out_spec->shape_rank = static_cast<int>(model->output_shapes[idx].size());
  out_spec->role =
      static_cast<omle_output_role_t>(static_cast<uint8_t>(s.role));
  return OMLE_OK;
}

extern "C" omle_task_type_t omle_model_task_type(
    const omle_model_t* /*model*/) {
  return OMLE_TASK_UNSPECIFIED;
}

// ---------------------------------------------------------------------------
// Stateless prediction
// ---------------------------------------------------------------------------

extern "C" omle_status_t omle_model_predict(
    const omle_model_t* model, int n_inputs, const char* const* input_names,
    omle_tensor_t* const* input_tensors, int n_req_outputs,
    const char* const* req_output_names, int* n_out, char*** out_names,
    omle_tensor_t*** out_tensors) {
  if (!model || !n_out || !out_names || !out_tensors)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");

  std::unordered_map<std::string, omle::rt::Tensor> inputs_map;
  inputs_map.reserve(n_inputs);
  for (int i = 0; i < n_inputs; ++i) {
    const auto& t = input_tensors[i]->tensor;
    if (t.is_dense() && !t.is_string() && !t.is_scalar())
      inputs_map.emplace(input_names[i],
                         omle::rt::Tensor::from_buffer(t.raw_data(), t.dtype,
                                                       t.n_rows, t.n_cols));
    else
      inputs_map.emplace(input_names[i], t);
  }

  std::vector<std::string> req;
  req.reserve(n_req_outputs);
  for (int i = 0; i < n_req_outputs; ++i) req.emplace_back(req_output_names[i]);

  auto result = model->model->predict(inputs_map, req);
  if (!result.ok()) return translate_status(result.status());

  const int count = static_cast<int>(result->size());
  *n_out = count;

  char** names_arr = new char*[count];
  omle_tensor_t** tensors_arr = new omle_tensor_t*[count];

  int idx = 0;
  for (auto& [name, tensor] : *result) {
    names_arr[idx] = new char[name.size() + 1];
    std::memcpy(names_arr[idx], name.c_str(), name.size() + 1);

    tensors_arr[idx] = new omle_tensor_t;
    tensors_arr[idx]->tensor = std::move(tensor);
    ++idx;
  }

  *out_names = names_arr;
  *out_tensors = tensors_arr;
  return OMLE_OK;
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

extern "C" omle_status_t omle_session_create(const omle_model_t* model,
                                             omle_session_t** out_session) {
  if (!model || !out_session)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  try {
    auto* h = new omle_session_s;
    h->session = model->model->create_session();
    h->owned_outputs.reserve(
        static_cast<std::size_t>(model->model->num_outputs()));
    *out_session = h;
    return OMLE_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" void omle_free_session(omle_session_t* session) { delete session; }

extern "C" omle_status_t omle_session_bind_input(omle_session_t* session,
                                                 const char* name,
                                                 omle_tensor_t* tensor) {
  if (!session || !name || !tensor)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  try {
    const auto& t = tensor->tensor;
    if (t.is_dense() && !t.is_string() && !t.is_scalar())
      session->session->bind_input(
          std::string(name), omle::rt::Tensor::from_buffer(
                                 t.raw_data(), t.dtype, t.n_rows, t.n_cols));
    else
      session->session->bind_input(std::string(name), t);
    return OMLE_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" omle_status_t omle_session_bind_output(omle_session_t* session,
                                                  const char* name,
                                                  omle_tensor_t* tensor) {
  if (!session || !name || !tensor)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  try {
    const auto& t = tensor->tensor;
    if (t.is_dense() && !t.is_string() && !t.is_scalar())
      session->session->bind_output(
          std::string(name), omle::rt::Tensor::from_buffer(
                                 t.raw_data(), t.dtype, t.n_rows, t.n_cols));
    else
      session->session->bind_output(std::string(name), t);
    return OMLE_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" void omle_session_clear_inputs(omle_session_t* session) {
  if (session) session->session->clear_inputs();
}

extern "C" void omle_session_clear_outputs(omle_session_t* session) {
  if (session) session->session->clear_outputs();
}

extern "C" omle_status_t omle_session_run(omle_session_t* session) {
  if (!session) return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");

  auto st = session->session->run();
  if (!st.ok()) return translate_status(st);

  // Move results into owned handles — no Tensor data copies.
  // Update existing entries in-place so pointers returned by earlier
  // get_output calls remain valid; push new entries otherwise.
  auto results = session->session->take_results();
  for (auto& [name, tensor] : results) {
    bool found = false;
    for (auto& p : session->owned_outputs) {
      if (p.first == name) {
        p.second = omle_tensor_s{std::move(tensor)};
        found = true;
        break;
      }
    }
    if (!found)
      session->owned_outputs.emplace_back(name,
                                          omle_tensor_s{std::move(tensor)});
  }
  return OMLE_OK;
}

extern "C" omle_status_t omle_session_get_output(omle_session_t* session,
                                                 const char* name,
                                                 omle_tensor_t** out_tensor) {
  if (!session || !name || !out_tensor)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  for (auto& p : session->owned_outputs) {
    if (p.first == name) {
      *out_tensor = &p.second;
      return OMLE_OK;
    }
  }
  tl_last_error = "output not found: ";
  tl_last_error += name;
  return OMLE_ERR_OUTPUT_NOT_FOUND;
}

extern "C" int omle_session_num_inputs(const omle_session_t* session) {
  return session ? session->session->num_inputs() : 0;
}

extern "C" int omle_session_num_outputs(const omle_session_t* session) {
  return session ? session->session->num_outputs() : 0;
}

extern "C" omle_status_t omle_session_predict_f32(
    omle_session_t* session, const char* input_name, int n_rows, int n_cols,
    const float* input, const char* output_name, float* output) {
  if (!session || !input_name || !input || !output_name || !output)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");

  // Zero-copy view: input data is owned by the caller (numpy array lifetime)
  session->session->clear_inputs();
  session->session->bind_input(
      std::string(input_name),
      omle::rt::Tensor::from_buffer(input, omle::rt::DataType::Float32, n_rows,
                                    n_cols));

  auto st = session->session->run();
  if (!st.ok()) return translate_status(st);

  // Move results into owned_outputs (same layout as omle_session_run)
  auto results = session->session->take_results();
  for (auto& [name, tensor] : results) {
    bool found = false;
    for (auto& p : session->owned_outputs) {
      if (p.first == name) {
        p.second = omle_tensor_s{std::move(tensor)};
        found = true;
        break;
      }
    }
    if (!found)
      session->owned_outputs.emplace_back(name,
                                          omle_tensor_s{std::move(tensor)});
  }

  // Write to caller's buffer, downcasting float64 → float32 if needed
  for (const auto& p : session->owned_outputs) {
    if (p.first != output_name) continue;
    const auto& t = p.second.tensor;
    const std::size_t n = static_cast<std::size_t>(t.n_rows) * t.n_cols;
    if (t.dtype == omle::rt::DataType::Float64) {
      const double* src = static_cast<const double*>(t.raw_data());
      for (std::size_t i = 0; i < n; ++i)
        output[i] = static_cast<float>(src[i]);
    } else {
      std::memcpy(output, t.raw_data(), n * sizeof(float));
    }
    return OMLE_OK;
  }
  tl_last_error = "output not found: ";
  tl_last_error += output_name;
  return OMLE_ERR_OUTPUT_NOT_FOUND;
}

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

static omle::rt::DataType from_c_dtype(omle_dtype_t dt) {
  return static_cast<omle::rt::DataType>(static_cast<uint8_t>(dt));
}

extern "C" omle_tensor_t* omle_tensor_create_dense(omle_dtype_t dtype,
                                                   int n_rows, int n_cols,
                                                   const void* data,
                                                   size_t data_bytes) {
  try {
    omle::rt::DataType dt = from_c_dtype(dtype);
    std::vector<uint8_t> raw;
    if (data && data_bytes > 0) {
      const auto* src = static_cast<const uint8_t*>(data);
      raw = std::vector<uint8_t>(src,
                                 src + data_bytes);  // range ctor: no zero-init
    } else {
      raw.assign(data_bytes, 0);
    }
    auto* h = new omle_tensor_s;
    h->tensor = omle::rt::Tensor::from_raw(dt, n_rows, n_cols, std::move(raw));
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" omle_tensor_t* omle_tensor_create_f32(int n_rows, int n_cols,
                                                 const float* data) {
  try {
    auto* h = new omle_tensor_s;
    if (n_rows == 1 && n_cols == 1) {
      h->tensor = omle::rt::Tensor::f32_scalar(data ? *data : 0.f);
    } else if (data) {
      const std::size_t bytes =
          static_cast<std::size_t>(n_rows) * n_cols * sizeof(float);
      h->tensor = omle::rt::Tensor::from_ptr(omle::rt::DataType::Float32,
                                             n_rows, n_cols, data, bytes);
    } else {
      h->tensor = omle::rt::Tensor::f32(n_rows, n_cols);
    }
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" omle_tensor_t* omle_tensor_create_f64(int n_rows, int n_cols,
                                                 const double* data) {
  try {
    auto* h = new omle_tensor_s;
    if (n_rows == 1 && n_cols == 1) {
      h->tensor = omle::rt::Tensor::scalar(omle::rt::DataType::Float64,
                                           data ? *data : 0.0);
    } else if (data) {
      const std::size_t bytes =
          static_cast<std::size_t>(n_rows) * n_cols * sizeof(double);
      h->tensor = omle::rt::Tensor::from_ptr(omle::rt::DataType::Float64,
                                             n_rows, n_cols, data, bytes);
    } else {
      h->tensor =
          omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows, n_cols);
    }
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" omle_tensor_t* omle_tensor_view_dense_f32(int n_rows, int n_cols,
                                                     const float* data) {
  if (!data) {
    set_error(OMLE_ERR_INVALID_ARGUMENT, "null data pointer");
    return nullptr;
  }
  try {
    auto* h = new omle_tensor_s;
    h->tensor = omle::rt::Tensor::from_buffer(data, omle::rt::DataType::Float32,
                                              n_rows, n_cols);
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" omle_tensor_t* omle_tensor_view_dense_f64(int n_rows, int n_cols,
                                                     const double* data) {
  if (!data) {
    set_error(OMLE_ERR_INVALID_ARGUMENT, "null data pointer");
    return nullptr;
  }
  try {
    auto* h = new omle_tensor_s;
    h->tensor = omle::rt::Tensor::from_buffer(data, omle::rt::DataType::Float64,
                                              n_rows, n_cols);
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" omle_tensor_t* omle_tensor_create_strings(
    int n_rows, int n_cols, const char* const* strings) {
  try {
    int count = n_rows * n_cols;
    std::vector<std::string> sv(static_cast<size_t>(count));
    if (strings) {
      for (int i = 0; i < count; ++i) sv[i] = strings[i] ? strings[i] : "";
    }
    auto* h = new omle_tensor_s;
    h->tensor = omle::rt::Tensor::strings(n_rows, n_cols, std::move(sv));
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" omle_tensor_t* omle_tensor_create_sparse_csr(
    omle_dtype_t dtype, int n_rows, int n_cols, const void* values,
    size_t values_bytes, const int32_t* col_indices, int nnz,
    const int32_t* row_indptr, double fill_value) {
  if (!col_indices || !row_indptr) {
    set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
    return nullptr;
  }
  try {
    std::vector<uint8_t> vals;
    if (values && values_bytes > 0) {
      const auto* src = static_cast<const uint8_t*>(values);
      vals = std::vector<uint8_t>(
          src, src + values_bytes);  // range ctor: no zero-init
    } else {
      vals.assign(values_bytes, 0);
    }

    std::vector<int32_t> idx(col_indices, col_indices + nnz);
    std::vector<int32_t> ptr(row_indptr, row_indptr + n_rows + 1);

    auto* h = new omle_tensor_s;
    h->tensor = omle::rt::Tensor::sparse_csr(
        from_c_dtype(dtype), n_rows, n_cols, std::move(vals), std::move(idx),
        std::move(ptr), fill_value);
    return h;
  } catch (...) {
    translate_exception();
    return nullptr;
  }
}

extern "C" void omle_free_tensor(omle_tensor_t* tensor) { delete tensor; }

// ---------------------------------------------------------------------------
// Tensor accessors
// ---------------------------------------------------------------------------

extern "C" omle_dtype_t omle_tensor_dtype(const omle_tensor_t* t) {
  return t ? to_c_dtype(t->tensor.dtype) : OMLE_DTYPE_UNSPECIFIED;
}

extern "C" omle_tensor_kind_t omle_tensor_kind(const omle_tensor_t* t) {
  if (!t) return OMLE_TENSOR_DENSE;
  return t->tensor.is_sparse() ? OMLE_TENSOR_SPARSE : OMLE_TENSOR_DENSE;
}

extern "C" int omle_tensor_n_rows(const omle_tensor_t* t) {
  return t ? t->tensor.n_rows : 0;
}

extern "C" int omle_tensor_n_cols(const omle_tensor_t* t) {
  return t ? t->tensor.n_cols : 0;
}

extern "C" int omle_tensor_numel(const omle_tensor_t* t) {
  return t ? static_cast<int>(t->tensor.numel()) : 0;
}

extern "C" const void* omle_tensor_data(const omle_tensor_t* t) {
  return t ? t->tensor.raw_data() : nullptr;
}

extern "C" size_t omle_tensor_data_bytes(const omle_tensor_t* t) {
  if (!t) return 0;
  return t->tensor.data_bytes();
}

extern "C" const void* omle_tensor_sp_values(const omle_tensor_t* t) {
  return (t && t->tensor.is_sparse()) ? t->tensor.sp_values().data() : nullptr;
}

extern "C" size_t omle_tensor_sp_values_bytes(const omle_tensor_t* t) {
  return (t && t->tensor.is_sparse()) ? t->tensor.sp_values().size() : 0;
}

extern "C" const int32_t* omle_tensor_sp_indices(const omle_tensor_t* t) {
  return (t && t->tensor.is_sparse()) ? t->tensor.sp_indices().data() : nullptr;
}

extern "C" const int32_t* omle_tensor_sp_indptr(const omle_tensor_t* t) {
  return (t && t->tensor.is_sparse()) ? t->tensor.sp_indptr().data() : nullptr;
}

extern "C" int omle_tensor_sp_nnz(const omle_tensor_t* t) {
  return (t && t->tensor.is_sparse()) ? t->tensor.nnz() : 0;
}

extern "C" double omle_tensor_sp_fill(const omle_tensor_t* t) {
  return (t && t->tensor.is_sparse()) ? t->tensor.sp_default() : 0.0;
}

extern "C" double omle_tensor_get(const omle_tensor_t* t, int row, int col) {
  return t ? t->tensor.get(row, col) : 0.0;
}

// ---------------------------------------------------------------------------
// Memory helpers
// ---------------------------------------------------------------------------

extern "C" void omle_free_strings(char** names, int count) {
  if (!names) return;
  for (int i = 0; i < count; ++i) delete[] names[i];
  delete[] names;
}

extern "C" void omle_free_tensors(omle_tensor_t** tensors, int count) {
  if (!tensors) return;
  for (int i = 0; i < count; ++i) delete tensors[i];
  delete[] tensors;
}

// ---------------------------------------------------------------------------
// Bulk column prediction
// ---------------------------------------------------------------------------

static std::unordered_map<std::string, omle::rt::Tensor> build_columns_map(
    int n_rows, int n_cols, const char* const* col_names, const int* col_types,
    const void* const* col_data) {
  std::unordered_map<std::string, omle::rt::Tensor> inputs;
  inputs.reserve(n_cols);
  for (int c = 0; c < n_cols; ++c) {
    if (col_types[c] == OMLE_COL_STRING) {
      const char* const* strs = static_cast<const char* const*>(col_data[c]);
      omle::rt::Tensor t = omle::rt::Tensor::strings(n_rows, 1);
      for (int r = 0; r < n_rows; ++r)
        t.str_at(r, 0) = (strs && strs[r]) ? strs[r] : "";
      inputs.emplace(col_names[c], std::move(t));
    } else if (col_types[c] == OMLE_COL_FLOAT64) {
      const double* data = static_cast<const double*>(col_data[c]);
      omle::rt::Tensor t =
          omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows, 1);
      if (data)
        std::memcpy(t.f64_ptr(), data,
                    static_cast<std::size_t>(n_rows) * sizeof(double));
      inputs.emplace(col_names[c], std::move(t));
    } else {
      const float* data = static_cast<const float*>(col_data[c]);
      omle::rt::Tensor t(n_rows, 1);
      if (data)
        std::memcpy(t.f32_ptr(), data,
                    static_cast<std::size_t>(n_rows) * sizeof(float));
      inputs.emplace(col_names[c], std::move(t));
    }
  }
  return inputs;
}

extern "C" omle_status_t omle_model_predict_columns(
    const omle_model_t* model, int n_rows, int n_cols,
    const char* const* col_names, const int* col_types,
    const void* const* col_data, int n_filter, const char* const* filter_names,
    int* n_out, char*** out_names, omle_tensor_t*** out_tensors) {
  if (!model || !n_out || !out_names || !out_tensors)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");

  try {
    auto inputs =
        build_columns_map(n_rows, n_cols, col_names, col_types, col_data);

    std::vector<std::string> req;
    req.reserve(n_filter);
    for (int i = 0; i < n_filter; ++i) req.emplace_back(filter_names[i]);

    auto result = model->model->predict(inputs, req);
    if (!result.ok()) return translate_status(result.status());

    const int count = static_cast<int>(result->size());
    *n_out = count;
    char** names_arr = new char*[count];
    omle_tensor_t** tens_arr = new omle_tensor_t*[count];
    int idx = 0;
    for (auto& [name, tensor] : *result) {
      names_arr[idx] = new char[name.size() + 1];
      std::memcpy(names_arr[idx], name.c_str(), name.size() + 1);
      tens_arr[idx] = new omle_tensor_t;
      tens_arr[idx]->tensor = std::move(tensor);
      ++idx;
    }
    *out_names = names_arr;
    *out_tensors = tens_arr;
    return OMLE_OK;
  } catch (...) {
    return translate_exception();
  }
}

static void bind_columns_impl(omle_session_s* session, int n_rows, int n_cols,
                              const char* const* col_names,
                              const int* col_types,
                              const void* const* col_data) {
  session->session->clear_inputs();
  for (int c = 0; c < n_cols; ++c) {
    omle::rt::Tensor t;
    if (col_types[c] == OMLE_COL_STRING) {
      const char* const* strs = static_cast<const char* const*>(col_data[c]);
      t = omle::rt::Tensor::strings(n_rows, 1);
      for (int r = 0; r < n_rows; ++r)
        t.str_at(r, 0) = (strs && strs[r]) ? strs[r] : "";
    } else if (col_types[c] == OMLE_COL_FLOAT64) {
      const double* data = static_cast<const double*>(col_data[c]);
      t = omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows, 1);
      if (data)
        std::memcpy(t.f64_ptr(), data,
                    static_cast<std::size_t>(n_rows) * sizeof(double));
    } else {
      const float* data = static_cast<const float*>(col_data[c]);
      t = omle::rt::Tensor(n_rows, 1);
      if (data)
        std::memcpy(t.f32_ptr(), data,
                    static_cast<std::size_t>(n_rows) * sizeof(float));
    }
    session->session->bind_input(col_names[c], std::move(t));
  }
}

static omle_status_t write_output_f32(omle_session_s* session,
                                      const char* output_name, float* output) {
  for (const auto& p : session->owned_outputs) {
    if (p.first != output_name) continue;
    const auto& t = p.second.tensor;
    const std::size_t n = static_cast<std::size_t>(t.n_rows) * t.n_cols;
    if (t.dtype == omle::rt::DataType::Float64) {
      const double* src = static_cast<const double*>(t.raw_data());
      for (std::size_t i = 0; i < n; ++i)
        output[i] = static_cast<float>(src[i]);
    } else {
      std::memcpy(output, t.raw_data(), n * sizeof(float));
    }
    return OMLE_OK;
  }
  tl_last_error = "output not found: ";
  tl_last_error += output_name;
  return OMLE_ERR_OUTPUT_NOT_FOUND;
}

extern "C" omle_status_t omle_session_bind_columns(
    omle_session_t* session, int n_rows, int n_cols,
    const char* const* col_names, const int* col_types,
    const void* const* col_data) {
  if (!session) return set_error(OMLE_ERR_INVALID_ARGUMENT, "null session");
  try {
    bind_columns_impl(session, n_rows, n_cols, col_names, col_types, col_data);
    return OMLE_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" omle_status_t omle_session_predict_columns_f32(
    omle_session_t* session, int n_rows, int n_cols,
    const char* const* col_names, const int* col_types,
    const void* const* col_data, const char* output_name, float* output) {
  if (!session || !output_name || !output)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  try {
    bind_columns_impl(session, n_rows, n_cols, col_names, col_types, col_data);

    auto st = session->session->run();
    if (!st.ok()) return translate_status(st);

    auto results = session->session->take_results();
    for (auto& [name, tensor] : results) {
      bool found = false;
      for (auto& p : session->owned_outputs) {
        if (p.first == name) {
          p.second = omle_tensor_s{std::move(tensor)};
          found = true;
          break;
        }
      }
      if (!found)
        session->owned_outputs.emplace_back(name,
                                            omle_tensor_s{std::move(tensor)});
    }
    return write_output_f32(session, output_name, output);
  } catch (...) {
    return translate_exception();
  }
}

extern "C" omle_status_t omle_session_register_columns(
    omle_session_t* session, int n_cols, const char* const* col_names,
    const int* col_types, const char* out_name) {
  if (!session) return set_error(OMLE_ERR_INVALID_ARGUMENT, "null session");
  try {
    auto rc = std::make_unique<RegisteredColumns>();
    rc->names.resize(n_cols);
    rc->name_ptrs.resize(n_cols);
    rc->types.assign(col_types, col_types + n_cols);
    for (int i = 0; i < n_cols; ++i) {
      rc->names[i] = col_names[i] ? col_names[i] : "";
      rc->name_ptrs[i] = rc->names[i].c_str();
    }
    rc->out_name = out_name ? out_name : "";

    // Pre-allocate input tensors (sized for batch=1); updated in-place each
    // call.
    rc->cached_inputs.reserve(static_cast<std::size_t>(n_cols) * 2);
    for (int i = 0; i < n_cols; ++i) {
      if (col_types[i] == OMLE_COL_STRING)
        rc->cached_inputs[rc->names[i]] = omle::rt::Tensor::strings(1, 1);
      else if (col_types[i] == OMLE_COL_FLOAT64)
        // Allocate at the declared width. These tensors are filled in place on
        // every call, so a float32 cell here would either narrow the column or
        // hold double bytes under a Float32 dtype -- which every later read
        // would then misinterpret.
        rc->cached_inputs[rc->names[i]] =
            omle::rt::Tensor::scalar(omle::rt::DataType::Float64, 0.0);
      else
        rc->cached_inputs[rc->names[i]] = omle::rt::Tensor::f32_scalar(0.f);
    }
    rc->cached_n_rows = 1;
    // Build stable pointer vector into the map values (node-based storage,
    // safe since we never insert/erase after this point).
    rc->cached_ptrs.resize(n_cols);
    for (int i = 0; i < n_cols; ++i)
      rc->cached_ptrs[i] = &rc->cached_inputs.at(rc->names[i]);

    session->reg_cols = std::move(rc);
    return OMLE_OK;
  } catch (...) {
    return translate_exception();
  }
}

extern "C" omle_status_t omle_session_predict_registered_f32(
    omle_session_t* session, int n_rows, const void* const* col_data,
    float* output) {
  if (!session || !output)
    return set_error(OMLE_ERR_INVALID_ARGUMENT, "null argument");
  if (!session->reg_cols)
    return set_error(
        OMLE_ERR_INVALID_ARGUMENT,
        "no columns registered; call omle_session_register_columns first");
  try {
    auto& rc = *session->reg_cols;
    const int n_cols = static_cast<int>(rc.cached_ptrs.size());

    if (n_rows == rc.cached_n_rows) {
      // Fast path: update pre-allocated tensors in-place — zero heap alloc.
      for (int c = 0; c < n_cols; ++c) {
        omle::rt::Tensor& t = *rc.cached_ptrs[c];
        if (rc.types[c] == OMLE_COL_STRING) {
          const char* const* strs =
              static_cast<const char* const*>(col_data[c]);
          for (int r = 0; r < n_rows; ++r)
            t.str_at(r, 0) = (strs && strs[r]) ? strs[r] : "";
        } else if (rc.types[c] == OMLE_COL_FLOAT64) {
          // The caller declared this column float64, so its buffer holds
          // doubles. Reading it as const float* did not narrow the values, it
          // reinterpreted the bytes -- every other column entry point already
          // honours the declared type, and this one silently did not.
          const double* d = static_cast<const double*>(col_data[c]);
          if (d)
            std::memcpy(t.f64_ptr(), d,
                        static_cast<std::size_t>(n_rows) * sizeof(double));
        } else {
          const float* d = static_cast<const float*>(col_data[c]);
          if (d) {
            if (n_rows == 1)
              t.f32_ptr()[0] = d[0];
            else
              std::memcpy(t.f32_ptr(), d,
                          static_cast<std::size_t>(n_rows) * sizeof(float));
          }
        }
      }
    } else {
      // n_rows changed: rebuild cached tensors to new size.
      rc.cached_n_rows = n_rows;
      for (int c = 0; c < n_cols; ++c) {
        omle::rt::Tensor& t = *rc.cached_ptrs[c];
        if (rc.types[c] == OMLE_COL_STRING) {
          t = omle::rt::Tensor::strings(n_rows, 1);
          const char* const* strs =
              static_cast<const char* const*>(col_data[c]);
          for (int r = 0; r < n_rows; ++r)
            t.str_at(r, 0) = (strs && strs[r]) ? strs[r] : "";
        } else if (rc.types[c] == OMLE_COL_FLOAT64) {
          t = omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows, 1);
          const double* d = static_cast<const double*>(col_data[c]);
          if (d)
            std::memcpy(t.f64_ptr(), d,
                        static_cast<std::size_t>(n_rows) * sizeof(double));
        } else {
          t = omle::rt::Tensor(n_rows, 1);
          const float* d = static_cast<const float*>(col_data[c]);
          if (d)
            std::memcpy(t.f32_ptr(), d,
                        static_cast<std::size_t>(n_rows) * sizeof(float));
        }
      }
    }

    auto st = session->session->run_with_inputs(rc.cached_inputs);
    if (!st.ok()) return translate_status(st);

    auto results = session->session->take_results();
    for (auto& [name, tensor] : results) {
      bool found = false;
      for (auto& p : session->owned_outputs) {
        if (p.first == name) {
          p.second = omle_tensor_s{std::move(tensor)};
          found = true;
          break;
        }
      }
      if (!found)
        session->owned_outputs.emplace_back(name,
                                            omle_tensor_s{std::move(tensor)});
    }
    return write_output_f32(session, rc.out_name.c_str(), output);
  } catch (...) {
    return translate_exception();
  }
}
