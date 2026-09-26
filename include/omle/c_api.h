// SPDX-License-Identifier: Apache-2.0
// C ABI for the omle runtime.
//
// All functions return omle_status_t.  On failure, call
// omle_last_error() on the same thread to retrieve a human-readable
// message.  All opaque handles are heap-allocated and must be freed
// by the corresponding omle_free_*() function.
#ifndef OMLE_C_API_H_
#define OMLE_C_API_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---------------------------------------------------------------------------
// Opaque handle types
// ---------------------------------------------------------------------------

typedef struct omle_model_s omle_model_t;
typedef struct omle_session_s omle_session_t;
typedef struct omle_tensor_s omle_tensor_t;

// ---------------------------------------------------------------------------
// Status codes  (mirrors omle::rt::ErrorCode; 0 = success)
// ---------------------------------------------------------------------------

typedef enum omle_status {
  OMLE_OK = 0,
  // Load-time
  OMLE_ERR_FILE_NOT_FOUND = 1,
  OMLE_ERR_FILE_READ = 2,
  OMLE_ERR_PARSE = 3,
  OMLE_ERR_INVALID_GRAPH = 4,
  OMLE_ERR_VERIFICATION_FAILED = 5,
  // Inference-time
  OMLE_ERR_MISSING_INPUT = 10,
  OMLE_ERR_OUTPUT_NOT_PRODUCED = 11,
  OMLE_ERR_OUTPUT_NOT_FOUND = 12,
  OMLE_ERR_UNKNOWN_OPERATOR = 13,
  OMLE_ERR_INVALID_ARGUMENT = 14,
  // Generic / unexpected
  OMLE_ERR_UNKNOWN = 99,
} omle_status_t;

// ---------------------------------------------------------------------------
// Data type enum  (mirrors omle::rt::DataType)
// ---------------------------------------------------------------------------

typedef enum omle_dtype {
  OMLE_DTYPE_UNSPECIFIED = 0,
  OMLE_DTYPE_BOOL = 1,
  OMLE_DTYPE_INT8 = 2,
  OMLE_DTYPE_INT16 = 3,
  OMLE_DTYPE_INT32 = 4,
  OMLE_DTYPE_INT64 = 5,
  OMLE_DTYPE_UINT8 = 6,
  OMLE_DTYPE_UINT16 = 7,
  OMLE_DTYPE_UINT32 = 8,
  OMLE_DTYPE_UINT64 = 9,
  OMLE_DTYPE_FLOAT16 = 10,
  OMLE_DTYPE_FLOAT32 = 11,
  OMLE_DTYPE_FLOAT64 = 12,
  OMLE_DTYPE_STRING = 13,
  OMLE_DTYPE_BYTES = 14,
  OMLE_DTYPE_DATE = 15,
  OMLE_DTYPE_TIME = 16,
  OMLE_DTYPE_TIMESTAMP = 17,
} omle_dtype_t;

// ---------------------------------------------------------------------------
// Task type enum  (mirrors omle::rt::TaskType)
// ---------------------------------------------------------------------------

typedef enum omle_task_type {
  OMLE_TASK_UNSPECIFIED = 0,
  OMLE_TASK_REGRESSION = 1,
  OMLE_TASK_BINARY = 2,
  OMLE_TASK_MULTICLASS = 3,
  OMLE_TASK_CLUSTERING = 4,
  OMLE_TASK_ANOMALY_DETECTION = 5,
} omle_task_type_t;

// ---------------------------------------------------------------------------
// Tensor kind
// ---------------------------------------------------------------------------

typedef enum omle_tensor_kind {
  OMLE_TENSOR_DENSE = 0,
  OMLE_TENSOR_SPARSE = 1,
} omle_tensor_kind_t;

// ---------------------------------------------------------------------------
// Load options
// ---------------------------------------------------------------------------

typedef struct omle_load_options {
  int n_threads;          // 0 = auto, 1 = single-threaded (default)
  int min_parallel_rows;  // rows per thread; a batch parallelises at
                          // n_threads * this. default 64
} omle_load_options_t;

// Default-initialised load options.
static inline omle_load_options_t omle_default_load_options(void) {
  omle_load_options_t o;
  o.n_threads = 1;
  o.min_parallel_rows = 64;
  return o;
}

// ---------------------------------------------------------------------------
// Output role enum  (mirrors omle::rt::OutputRole)
// ---------------------------------------------------------------------------

typedef enum omle_output_role {
  OMLE_ROLE_UNSPECIFIED = 0,
  OMLE_ROLE_PREDICTION = 1,
  OMLE_ROLE_PROBABILITY = 2,
  OMLE_ROLE_SCORE = 3,
  OMLE_ROLE_CONFIDENCE = 4,
  OMLE_ROLE_STANDARD_ERROR = 5,
  OMLE_ROLE_STANDARD_DEV = 6,
  OMLE_ROLE_RESIDUAL = 7,
  OMLE_ROLE_TRANSFORMED_VALUE = 8,
  OMLE_ROLE_ENTITY_ID = 9,
  OMLE_ROLE_AFFINITY = 10,
  OMLE_ROLE_CONTRIBUTION = 11,
  OMLE_ROLE_INTERMEDIATE = 12,
} omle_output_role_t;

// ---------------------------------------------------------------------------
// Input / output spec (flattened, caller does not own the strings)
// ---------------------------------------------------------------------------

typedef struct omle_tensor_spec {
  const char* name;  // points into model memory — do not free
  omle_dtype_t dtype;
  const int64_t* shape;  // points into model memory — do not free
  int shape_rank;
  omle_output_role_t role;  // OMLE_ROLE_UNSPECIFIED for inputs
} omle_tensor_spec_t;

// ---------------------------------------------------------------------------
// Error reporting
// ---------------------------------------------------------------------------

// Returns the error message from the most recent failed call on this thread.
// The pointer is valid until the next omle call on this thread.
const char* omle_last_error(void);

// ---------------------------------------------------------------------------
// Model lifecycle
// ---------------------------------------------------------------------------

// Load a model from a file path.
// On success *out_model is set to a newly allocated handle; caller must call
// omle_free_model() when done.
omle_status_t omle_model_load_file(
    const char* path,
    const omle_load_options_t* opts,  // NULL → defaults
    omle_model_t** out_model);

// Load a model from an in-memory buffer.  The buffer is not retained after
// this call returns.
omle_status_t omle_model_load_memory(
    const void* data, size_t size,
    const omle_load_options_t* opts,  // NULL → defaults
    omle_model_t** out_model);

void omle_free_model(omle_model_t* model);

// ---------------------------------------------------------------------------
// Model introspection
// ---------------------------------------------------------------------------

int omle_model_num_inputs(const omle_model_t* model);

/* Non-fatal advisories raised while loading, e.g. a schema version this build
 * does not recognise on a model carrying no verification cases to prove it runs
 * correctly. Usually zero. The returned pointer is owned by the model and valid
 * for its lifetime; NULL for an out-of-range index. */
int omle_model_num_warnings(const omle_model_t* model);
const char* omle_model_warning(const omle_model_t* model, int index);
int omle_model_num_outputs(const omle_model_t* model);

// Returns OMLE_OK and fills *spec, or OMLE_ERR_INVALID_ARGUMENT if idx
// is out of range.  The pointers inside *spec are valid for the model lifetime.
omle_status_t omle_model_input_spec(const omle_model_t* model, int idx,
                                    omle_tensor_spec_t* out_spec);

omle_status_t omle_model_output_spec(const omle_model_t* model, int idx,
                                     omle_tensor_spec_t* out_spec);

omle_task_type_t omle_model_task_type(const omle_model_t* model);

// ---------------------------------------------------------------------------
// Stateless prediction (thread-safe, uses the model's thread pool)
//
// n_inputs        : number of entries in input_names / input_tensors
// input_names     : array of C strings naming each input tensor
// input_tensors   : array of borrowed tensor handles (not destroyed by this
// call) n_req_outputs   : number of entries in req_output_names (0 = return all
// outputs) req_output_names: requested output names (may be NULL if
// n_req_outputs == 0) n_out           : [out] number of outputs produced
// out_names       : [out] array of C strings; caller must call
// omle_free_strings() out_tensors     : [out] array of tensor handles;
// caller must call
//                   omle_free_tensors()
// ---------------------------------------------------------------------------

omle_status_t omle_model_predict(const omle_model_t* model, int n_inputs,
                                 const char* const* input_names,
                                 omle_tensor_t* const* input_tensors,
                                 int n_req_outputs,
                                 const char* const* req_output_names,
                                 int* n_out, char*** out_names,
                                 omle_tensor_t*** out_tensors);

// ---------------------------------------------------------------------------
// Session (per-thread execution context, not thread-safe)
// ---------------------------------------------------------------------------

omle_status_t omle_session_create(const omle_model_t* model,
                                  omle_session_t** out_session);

void omle_free_session(omle_session_t* session);

// Bind a named input tensor.  The session holds a reference; the tensor
// must remain valid until omle_session_run() returns.
omle_status_t omle_session_bind_input(omle_session_t* session, const char* name,
                                      omle_tensor_t* tensor);

// Pre-allocate an output slot.  If no outputs are bound, all graph outputs
// are collected automatically after run.
omle_status_t omle_session_bind_output(omle_session_t* session,
                                       const char* name, omle_tensor_t* tensor);

void omle_session_clear_inputs(omle_session_t* session);
void omle_session_clear_outputs(omle_session_t* session);

omle_status_t omle_session_run(omle_session_t* session);

// Retrieve a result tensor produced by the last run().  The returned handle
// is owned by the session and is valid until the next run() or
// omle_free_session().
omle_status_t omle_session_get_output(omle_session_t* session, const char* name,
                                      omle_tensor_t** out_tensor);

int omle_session_num_inputs(const omle_session_t* session);
int omle_session_num_outputs(const omle_session_t* session);

// Fast-path predict: zero-copy input view, output written directly to caller's
// buffer. For float64 models the output is downcast to float32. `output` must
// hold at least n_rows * out_cols floats where out_cols matches the model's
// output column count.
omle_status_t omle_session_predict_f32(omle_session_t* session,
                                       const char* input_name, int n_rows,
                                       int n_cols, const float* input,
                                       const char* output_name, float* output);

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

// Dense tensor.  `data` is copied into the handle; passing NULL
// zero-initialises. Returns NULL on failure; call omle_last_error() for
// details.
omle_tensor_t* omle_tensor_create_dense(omle_dtype_t dtype, int n_rows,
                                        int n_cols,
                                        const void* data,  // NULL = zero-fill
                                        size_t data_bytes);

// Dense Float32 convenience constructor.
// Returns NULL on failure; call omle_last_error() for details.
omle_tensor_t* omle_tensor_create_f32(int n_rows, int n_cols,
                                      const float* data  // NULL = zero-fill
);

// Dense Float64 convenience constructor.
// Returns NULL on failure; call omle_last_error() for details.
omle_tensor_t* omle_tensor_create_f64(int n_rows, int n_cols,
                                      const double* data  // NULL = zero-fill
);

// Non-owning Float32 view.  Zero-copy: `data` must remain valid for the
// lifetime of the tensor. Returns NULL on failure; call omle_last_error()
// for details.
omle_tensor_t* omle_tensor_view_dense_f32(int n_rows, int n_cols,
                                          const float* data);

// Non-owning Float64 view.  Zero-copy: `data` must remain valid for the
// lifetime of the tensor. Returns NULL on failure; call omle_last_error()
// for details.
omle_tensor_t* omle_tensor_view_dense_f64(int n_rows, int n_cols,
                                          const double* data);

// String tensor constructor.  `strings` is an array of n_rows*n_cols
// null-terminated C strings (may be NULL to create an all-empty string tensor).
// All strings are copied. Returns NULL on failure; call omle_last_error()
// for details.
omle_tensor_t* omle_tensor_create_strings(
    int n_rows, int n_cols,
    const char* const* strings  // NULL = all empty
);

// Sparse CSR tensor.  All arrays are copied.
// Returns NULL on failure; call omle_last_error() for details.
omle_tensor_t* omle_tensor_create_sparse_csr(
    omle_dtype_t dtype, int n_rows, int n_cols,
    const void* values,  // stored values, nnz * dtype_size bytes
    size_t values_bytes,
    const int32_t* col_indices,  // length nnz
    int nnz,
    const int32_t* row_indptr,  // length n_rows + 1
    double fill_value);

void omle_free_tensor(omle_tensor_t* tensor);

// ---------------------------------------------------------------------------
// Tensor accessors
// ---------------------------------------------------------------------------

omle_dtype_t omle_tensor_dtype(const omle_tensor_t* tensor);
omle_tensor_kind_t omle_tensor_kind(const omle_tensor_t* tensor);
int omle_tensor_n_rows(const omle_tensor_t* tensor);
int omle_tensor_n_cols(const omle_tensor_t* tensor);
int omle_tensor_numel(const omle_tensor_t* tensor);

// Dense tensor raw data pointer (NULL for sparse tensors).
const void* omle_tensor_data(const omle_tensor_t* tensor);
size_t omle_tensor_data_bytes(const omle_tensor_t* tensor);

// Sparse CSR accessors (valid only for OMLE_TENSOR_SPARSE).
const void* omle_tensor_sp_values(const omle_tensor_t* tensor);
size_t omle_tensor_sp_values_bytes(const omle_tensor_t* tensor);
const int32_t* omle_tensor_sp_indices(const omle_tensor_t* tensor);
const int32_t* omle_tensor_sp_indptr(const omle_tensor_t* tensor);
int omle_tensor_sp_nnz(const omle_tensor_t* tensor);
double omle_tensor_sp_fill(const omle_tensor_t* tensor);

// Element access — works for both kinds and all numeric dtypes; returns double.
double omle_tensor_get(const omle_tensor_t* tensor, int row, int col);

// ---------------------------------------------------------------------------
// Memory helpers for arrays returned by omle_model_predict()
// ---------------------------------------------------------------------------

// Free the name array returned by omle_model_predict().
void omle_free_strings(char** names, int count);

// Free the tensor array returned by omle_model_predict().
void omle_free_tensors(omle_tensor_t** tensors, int count);

// ---------------------------------------------------------------------------
// Bulk column prediction — avoids per-column tensor handle overhead
// ---------------------------------------------------------------------------

// Column type constants for omle_model_predict_columns /
// omle_session_run_columns.
//
// FLOAT64 keeps the caller's precision through preprocessing: narrowing at
// ingest and scaling afterwards is not the same as scaling first and narrowing
// once, and the difference is enough to move a value across a tree split
// threshold. It is also how integer columns are passed, since float32 carries
// only 24 bits of mantissa and silently alters values above 2**24.
#define OMLE_COL_FLOAT32 0
#define OMLE_COL_FLOAT64 1
#define OMLE_COL_STRING 2

// Stateless prediction from column arrays.
// col_data[c]: for FLOAT32 columns, points to float[n_rows];
//              for FLOAT64 columns, points to double[n_rows];
//              for STRING columns,  points to const char*[n_rows].
// All column data is read but not retained after the call returns.
// Output handling is identical to omle_model_predict().
omle_status_t omle_model_predict_columns(
    const omle_model_t* model, int n_rows, int n_cols,
    const char* const* col_names, const int* col_types,
    const void* const* col_data, int n_filter, const char* const* filter_names,
    int* n_out, char*** out_names, omle_tensor_t*** out_tensors);

// Session version: bind column arrays as inputs, then call
// omle_session_run(). Does NOT run inference — call omle_session_run()
// after this.
omle_status_t omle_session_bind_columns(omle_session_t* session, int n_rows,
                                        int n_cols,
                                        const char* const* col_names,
                                        const int* col_types,
                                        const void* const* col_data);

// Combined bind-columns + run + write-output in one call.
// Equivalent to bind_columns → session_run → get_output → memcpy, but without
// intermediate Python round-trips.  output must hold n_rows * out_cols floats.
omle_status_t omle_session_predict_columns_f32(
    omle_session_t* session, int n_rows, int n_cols,
    const char* const* col_names, const int* col_types,
    const void* const* col_data, const char* output_name, float* output);

// Pre-register column schema once at session creation to avoid per-call name
// building. col_types uses the OMLE_COL_* constants above.  out_name is the
// output tensor name. Not thread-safe; call once after
// omle_session_create().
omle_status_t omle_session_register_columns(omle_session_t* session, int n_cols,
                                            const char* const* col_names,
                                            const int* col_types,
                                            const char* out_name);

// Predict using pre-registered column schema (no per-call name overhead).
// col_data[c] must match the registered type for column c.
// output must hold n_rows * out_cols floats.
omle_status_t omle_session_predict_registered_f32(omle_session_t* session,
                                                  int n_rows,
                                                  const void* const* col_data,
                                                  float* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // OMLE_C_API_H_
