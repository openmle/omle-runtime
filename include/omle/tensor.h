#ifndef OMLE_TENSOR_H_
#define OMLE_TENSOR_H_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "omle/port.h"

namespace omle::rt {

// -----------------------------------------------------------------------
// DataType — element type of a tensor (mirrors omle.v0_1.DataType).
// -----------------------------------------------------------------------
enum class DataType : uint8_t {
  Unknown = 0,
  Bool = 1,
  Int8 = 2,
  Int16 = 3,
  Int32 = 4,
  Int64 = 5,
  UInt8 = 6,
  UInt16 = 7,
  UInt32 = 8,
  UInt64 = 9,
  Float16 = 10,
  Float32 = 11,
  Float64 = 12,
  String = 13,
  Bytes = 14,
  Date = 15,
  Time = 16,
  Timestamp = 17,
};

// Bytes per element for numeric dtypes (1 for Bool/String/Bytes/unknown).
inline int dtype_size(DataType dt) noexcept {
  switch (dt) {
    case DataType::Bool:
    case DataType::Int8:
    case DataType::UInt8:
      return 1;
    case DataType::Int16:
    case DataType::UInt16:
    case DataType::Float16:
      return 2;
    case DataType::Int32:
    case DataType::UInt32:
    case DataType::Float32:
      return 4;
    case DataType::Int64:
    case DataType::UInt64:
    case DataType::Float64:
      return 8;
    default:
      return 1;
  }
}

// ============================================================
// Tensor — unified dense / sparse / scalar rank-2 tensor.
//
// Three storage kinds:
//
//   Scalar  [1×1, any numeric dtype]
//     The single value is stored inline in scalar_buf_ (8 raw bytes,
//     interpreted per dtype) — zero heap allocation.  f32_ptr() / row(0)
//     / at(0,0) are valid for Float32 only; use get(0,0) for other dtypes.
//
//   Dense   [n_rows × n_cols, any dtype]
//     `data` (DenseStore) holds n_rows*n_cols*dtype_size(dtype) bytes,
//     row-major. Rare attributes (string row data, sparse CSR fields,
//     non-owning view ptr) live behind extras_ and only allocate when
//     actually used.
//
//   Sparse  [CSR, Float32]
//     SparseStore (held inside extras_) carries sp_values / sp_indices /
//     sp_indptr.  sparse_csr() auto-downgrades to Dense when fill-rate
//     > DENSE_THRESHOLD (50%).
//
// Approximate sizeof(Tensor) on 64-bit:
//   ~64 B for the common dense-numeric / scalar case (header + DenseStore
//   + 8B inline scalar buffer + 8B extras pointer).  Tensors that need
//   strings, views, or sparse CSR allocate a small TensorExtras on demand.
// ============================================================
struct Tensor {
  DataType dtype = DataType::Float32;
  int n_rows = 0;
  int n_cols = 1;

  enum class Kind : uint8_t { Dense = 0, Sparse = 1, Scalar = 2 };
  Kind kind = Kind::Dense;

  // ================================================================
  // Constructors
  // ================================================================

  Tensor() = default;

  // Legacy constructor: Float32, row-major.
  // Uses Kind::Scalar (inline, no heap) when rows==1 && cols==1.
  Tensor(int rows, int cols, float fill = 0.f)
      : dtype(DataType::Float32), n_rows(rows), n_cols(cols) {
    if (rows == 1 && cols == 1) {
      kind = Kind::Scalar;
      std::memcpy(scalar_buf_, &fill, sizeof(float));
      return;
    }
    kind = Kind::Dense;
    const std::size_t bytes =
        static_cast<std::size_t>(rows) * cols * sizeof(float);
    if (fill == 0.f) {
      data = DenseStore::alloc_zeroed(bytes);
    } else {
      data = DenseStore::alloc(bytes);
      std::fill(f32_ptr(), f32_ptr() + static_cast<std::size_t>(rows) * cols,
                fill);
    }
  }

  // ================================================================
  // Copy / move
  // ================================================================
  Tensor(const Tensor& o)
      : dtype(o.dtype),
        n_rows(o.n_rows),
        n_cols(o.n_cols),
        kind(o.kind),
        data(o.data.clone()),
        extras_(o.extras_ ? std::make_unique<Extras>(*o.extras_) : nullptr) {
    std::memcpy(scalar_buf_, o.scalar_buf_, sizeof(scalar_buf_));
  }

  Tensor& operator=(const Tensor& o) {
    if (this != &o) {
      Tensor tmp(o);
      *this = std::move(tmp);
    }
    return *this;
  }

  Tensor(Tensor&&) = default;
  Tensor& operator=(Tensor&&) = default;
  ~Tensor() = default;

  // ================================================================
  // Factory functions
  // ================================================================

  // Dense Float32 [rows × cols] filled with `fill`.
  // [1×1] uses Scalar kind (no heap).
  static Tensor f32(int rows, int cols, float fill = 0.f) {
    return Tensor(rows, cols, fill);
  }

  // Dense Float32 [rows × 1] column vector.
  static Tensor f32_column(int rows, float fill = 0.f) {
    return Tensor(rows, 1, fill);
  }

  // [1×1] scalar of any numeric dtype — stored inline, zero heap allocation.
  static Tensor scalar(DataType dt, double v = 0.0) {
    Tensor t;
    t.dtype = dt;
    t.n_rows = 1;
    t.n_cols = 1;
    t.kind = Kind::Scalar;
    _write_scalar_bytes(t.scalar_buf_, dt, v);
    return t;
  }

  // [1×1] Float32 scalar convenience wrapper.
  static Tensor f32_scalar(float v) { return scalar(DataType::Float32, v); }

  // Dense tensor of the given dtype, zero-initialised.
  // Any [1×1] uses Scalar kind (inline, no heap).
  static Tensor dense(DataType dt, int rows, int cols) {
    if (rows == 1 && cols == 1) return scalar(dt, 0.0);
    Tensor t;
    t.dtype = dt;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    const std::size_t bytes =
        static_cast<std::size_t>(rows) * cols * dtype_size(dt);
    t.data = DenseStore::alloc_zeroed(bytes);
    return t;
  }

  // Dense Float32 from an existing float vector — zero-copy: adopts the
  // vector's heap buffer instead of memcpying it.
  // [1×1] uses Scalar kind.
  static Tensor from_floats(int rows, int cols, std::vector<float> v) {
    if (rows == 1 && cols == 1 && v.size() >= 1) return f32_scalar(v[0]);
    Tensor t;
    t.dtype = DataType::Float32;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    t.data = DenseStore::adopt(std::move(v));
    return t;
  }

  // Dense Float64 from an existing double vector — zero-copy.
  static Tensor from_doubles(int rows, int cols, std::vector<double> v) {
    if (rows == 1 && cols == 1 && v.size() >= 1)
      return scalar(DataType::Float64, v[0]);
    Tensor t;
    t.dtype = DataType::Float64;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    t.data = DenseStore::adopt(std::move(v));
    return t;
  }

  // Non-owning view over an external buffer. Zero-copy: caller must ensure
  // the buffer outlives this Tensor (and any copies of it).
  static Tensor from_buffer(const void* buf, DataType dt, int rows, int cols) {
    Tensor t;
    t.dtype = dt;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    t._ensure_extras()->view_ptr = buf;
    return t;
  }

  bool is_view() const noexcept {
    return extras_ && extras_->view_ptr != nullptr;
  }

  // Dense tensor from raw bytes (bytes are moved in — zero-copy adoption).
  static Tensor from_raw(DataType dt, int rows, int cols,
                         std::vector<uint8_t> raw) {
    const int esz = dtype_size(dt);
    if (rows == 1 && cols == 1 && static_cast<int>(raw.size()) >= esz) {
      Tensor t;
      t.dtype = dt;
      t.n_rows = 1;
      t.n_cols = 1;
      t.kind = Kind::Scalar;
      std::memcpy(t.scalar_buf_, raw.data(), esz);
      return t;
    }
    Tensor t;
    t.dtype = dt;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    t.data = DenseStore::adopt(std::move(raw));
    return t;
  }

  // Dense tensor from a raw pointer — single alloc + memcpy, no zero-init.
  // [1×1] uses Scalar kind (inline, no heap).  src==nullptr → zero-fill.
  static Tensor from_ptr(DataType dt, int rows, int cols, const void* src,
                         std::size_t bytes) {
    if (rows == 1 && cols == 1) {
      Tensor t;
      t.dtype = dt;
      t.n_rows = 1;
      t.n_cols = 1;
      t.kind = Kind::Scalar;
      if (src && bytes > 0)
        std::memcpy(t.scalar_buf_, src, std::min(bytes, std::size_t(8)));
      return t;
    }
    Tensor t;
    t.dtype = dt;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    if (src && bytes > 0) {
      t.data = DenseStore::alloc(bytes);
      std::memcpy(t.data.data(), src, bytes);
    } else {
      t.data = DenseStore::alloc_zeroed(bytes);
    }
    return t;
  }

  // Dense String tensor [rows × cols].
  static Tensor strings(int rows, int cols, std::vector<std::string> v = {}) {
    Tensor t;
    t.dtype = DataType::String;
    t.n_rows = rows;
    t.n_cols = cols;
    t.kind = Kind::Dense;
    auto* x = t._ensure_extras();
    x->str_data = std::move(v);
    x->str_data.resize(static_cast<std::size_t>(rows) * cols);
    return t;
  }

  static constexpr float DENSE_THRESHOLD = 0.5f;

  static Tensor sparse_csr(DataType dt, int rows, int cols,
                           std::vector<uint8_t> sp_values_bytes,
                           std::vector<int32_t> indices,
                           std::vector<int32_t> indptr, double fill = 0.0) {
    const int nnz_count = static_cast<int>(indices.size());
    const int numel_ = rows * cols;
    if (numel_ > 0 && dt == DataType::Float32 &&
        static_cast<float>(nnz_count) / static_cast<float>(numel_) >
            DENSE_THRESHOLD) {
      return _expand_csr_to_dense(rows, cols, sp_values_bytes, indices, indptr,
                                  static_cast<float>(fill));
    }

    Tensor t;
    t.dtype = dt;
    t.kind = Kind::Sparse;
    t.n_rows = rows;
    t.n_cols = cols;
    auto* x = t._ensure_extras();
    x->sparse = std::make_unique<SparseStore>();
    x->sparse->sp_values = std::move(sp_values_bytes);
    x->sparse->sp_indices = std::move(indices);
    x->sparse->sp_indptr = std::move(indptr);
    x->sparse->sp_default = fill;
    return t;
  }

  static Tensor sparse_csr_f32(int rows, int cols, std::vector<float> values,
                               std::vector<int32_t> indices,
                               std::vector<int32_t> indptr, float fill = 0.f) {
    std::vector<uint8_t> bytes(values.size() * sizeof(float));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return sparse_csr(DataType::Float32, rows, cols, std::move(bytes),
                      std::move(indices), std::move(indptr),
                      static_cast<double>(fill));
  }

  // Compress a dense Float32 tensor to sparse CSR.
  static Tensor to_sparse(const Tensor& src, float threshold = 0.f) {
    assert(src.dtype == DataType::Float32 && !src.is_sparse());
    const std::size_t cap = src.numel();
    std::vector<uint8_t> vals;
    std::vector<int32_t> idx;
    vals.reserve(cap * sizeof(float));
    idx.reserve(cap);
    std::vector<int32_t> ptr(src.n_rows + 1, 0);
    const float* p = src.f32_ptr();
    for (int r = 0; r < src.n_rows; ++r) {
      for (int c = 0; c < src.n_cols; ++c) {
        float v = p[r * src.n_cols + c];
        if (std::fabs(v) > threshold) {
          const auto* vp = reinterpret_cast<const uint8_t*>(&v);
          vals.insert(vals.end(), vp, vp + sizeof(float));
          idx.push_back(c);
        }
      }
      ptr[r + 1] = static_cast<int32_t>(idx.size());
    }
    Tensor t;
    t.dtype = DataType::Float32;
    t.kind = Kind::Sparse;
    t.n_rows = src.n_rows;
    t.n_cols = src.n_cols;
    auto* x = t._ensure_extras();
    x->sparse = std::make_unique<SparseStore>();
    x->sparse->sp_values = std::move(vals);
    x->sparse->sp_indices = std::move(idx);
    x->sparse->sp_indptr = std::move(ptr);
    x->sparse->sp_default = 0.0;
    return t;
  }

  // ================================================================
  // Kind / dtype predicates
  // ================================================================
  bool is_dense() const noexcept { return kind != Kind::Sparse; }
  bool is_sparse() const noexcept { return kind == Kind::Sparse; }
  bool is_scalar() const noexcept { return kind == Kind::Scalar; }
  bool is_string() const noexcept { return dtype == DataType::String; }
  bool empty() const noexcept { return n_rows == 0; }

  // Pointer to the raw byte buffer for dense / scalar tensors (nullptr for
  // sparse).
  const void* raw_data() const noexcept {
    if (kind == Kind::Sparse) return nullptr;
    if (kind == Kind::Scalar) return scalar_buf_;
    return _raw_bytes();
  }
  std::size_t numel() const noexcept {
    return static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(n_cols);
  }

  // Byte count of the dense data buffer (0 for sparse or string tensors).
  std::size_t data_bytes() const noexcept {
    if (kind == Kind::Scalar)
      return static_cast<std::size_t>(dtype_size(dtype));
    if (kind == Kind::Sparse) return 0;
    if (dtype == DataType::String) return 0;
    return data.size();
  }

  // ================================================================
  // String accessors (valid when is_string()).
  // ================================================================
  const std::string& str_at(int r, int c) const {
    return extras_->str_data[r * n_cols + c];
  }
  std::string& str_at(int r, int c) {
    return _ensure_extras()->str_data[r * n_cols + c];
  }

  const std::vector<std::string>& str_data() const noexcept {
    static const std::vector<std::string> kEmpty;
    return extras_ ? extras_->str_data : kEmpty;
  }

  void set_strings(std::vector<std::string> v) {
    dtype = DataType::String;
    kind = Kind::Dense;
    data.clear();
    _ensure_extras()->str_data = std::move(v);
  }

  // ================================================================
  // Float32 accessors (Dense and Scalar paths).
  // ================================================================
  float* f32_ptr() noexcept {
    assert(dtype == DataType::Float32 && "f32_ptr requires Float32 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<float*>(scalar_buf_);
    return reinterpret_cast<float*>(data.data());
  }
  const float* f32_ptr() const noexcept {
    assert(dtype == DataType::Float32 && "f32_ptr requires Float32 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<const float*>(scalar_buf_);
    return reinterpret_cast<const float*>(_raw_bytes());
  }
  float* row(int r) noexcept { return f32_ptr() + r * n_cols; }
  const float* row(int r) const noexcept { return f32_ptr() + r * n_cols; }
  float& at(int r, int c) noexcept { return f32_ptr()[r * n_cols + c]; }
  const float& at(int r, int c) const noexcept {
    return f32_ptr()[r * n_cols + c];
  }

  // Assign float32 content from a vector<float> — zero-copy adoption.
  void set_floats(std::vector<float> v) {
    if (kind == Kind::Scalar && v.size() == 1) {
      std::memcpy(scalar_buf_, &v[0], sizeof(float));
      return;
    }
    kind = Kind::Dense;
    data = DenseStore::adopt(std::move(v));
  }

  // ================================================================
  // Float64 accessors (Dense and Scalar paths).
  // ================================================================
  double* f64_ptr() noexcept {
    assert(dtype == DataType::Float64 && "f64_ptr requires Float64 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<double*>(scalar_buf_);
    return reinterpret_cast<double*>(data.data());
  }
  const double* f64_ptr() const noexcept {
    assert(dtype == DataType::Float64 && "f64_ptr requires Float64 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<const double*>(scalar_buf_);
    return reinterpret_cast<const double*>(_raw_bytes());
  }
  double* f64_row(int r) noexcept { return f64_ptr() + r * n_cols; }
  const double* f64_row(int r) const noexcept { return f64_ptr() + r * n_cols; }
  double& f64_at(int r, int c) noexcept { return f64_ptr()[r * n_cols + c]; }
  const double& f64_at(int r, int c) const noexcept {
    return f64_ptr()[r * n_cols + c];
  }

  // ================================================================
  // Int32 accessors (Dense and Scalar paths).
  // ================================================================
  int32_t* i32_ptr() noexcept {
    assert(dtype == DataType::Int32 && "i32_ptr requires Int32 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<int32_t*>(scalar_buf_);
    return reinterpret_cast<int32_t*>(data.data());
  }
  const int32_t* i32_ptr() const noexcept {
    assert(dtype == DataType::Int32 && "i32_ptr requires Int32 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<const int32_t*>(scalar_buf_);
    return reinterpret_cast<const int32_t*>(_raw_bytes());
  }
  int32_t i32_at(int r, int c) const noexcept {
    return i32_ptr()[r * n_cols + c];
  }

  // ================================================================
  // Int64 accessors (Dense and Scalar paths).
  // ================================================================
  int64_t* i64_ptr() noexcept {
    assert(dtype == DataType::Int64 && "i64_ptr requires Int64 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<int64_t*>(scalar_buf_);
    return reinterpret_cast<int64_t*>(data.data());
  }
  const int64_t* i64_ptr() const noexcept {
    assert(dtype == DataType::Int64 && "i64_ptr requires Int64 dtype");
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return reinterpret_cast<const int64_t*>(scalar_buf_);
    return reinterpret_cast<const int64_t*>(_raw_bytes());
  }

  // ================================================================
  // Sparse CSR field accessors (valid when is_sparse()).
  // ================================================================
  const std::vector<uint8_t>& sp_values() const noexcept {
    return extras_->sparse->sp_values;
  }
  const std::vector<int32_t>& sp_indices() const noexcept {
    return extras_->sparse->sp_indices;
  }
  const std::vector<int32_t>& sp_indptr() const noexcept {
    return extras_->sparse->sp_indptr;
  }
  double sp_default() const noexcept { return extras_->sparse->sp_default; }

  int row_nnz(int r) const noexcept {
    const auto& sp = *extras_->sparse;
    return sp.sp_indptr[r + 1] - sp.sp_indptr[r];
  }
  const float* row_values_f32(int r) const noexcept {
    assert(dtype == DataType::Float32 &&
           "row_values_f32 requires Float32 sparse");
    const auto& sp = *extras_->sparse;
    return reinterpret_cast<const float*>(sp.sp_values.data()) +
           sp.sp_indptr[r];
  }
  const int32_t* row_indices(int r) const noexcept {
    const auto& sp = *extras_->sparse;
    return sp.sp_indices.data() + sp.sp_indptr[r];
  }
  int nnz() const noexcept {
    if (!extras_ || !extras_->sparse) return 0;
    return static_cast<int>(extras_->sparse->sp_indices.size());
  }

  // ================================================================
  // Universal element access — works for all kinds and dtypes.
  // ================================================================
  double get(int r, int c) const noexcept {
    if (kind == Kind::Scalar) OMLE_UNLIKELY
    return _read_elem(scalar_buf_, 0);
    if (kind == Kind::Sparse) OMLE_UNLIKELY {
        const auto& sp = *extras_->sparse;
        int32_t lo = sp.sp_indptr[r], hi = sp.sp_indptr[r + 1];
        while (lo < hi) {
          int32_t mid = lo + (hi - lo) / 2;
          if (sp.sp_indices[mid] == c)
            return _read_elem(sp.sp_values.data(), mid);
          if (sp.sp_indices[mid] < c)
            lo = mid + 1;
          else
            hi = mid;
        }
        return sp.sp_default;
      }
    return _read_elem(_raw_bytes(), r * n_cols + c);
  }

  // ================================================================
  // Conversion
  // ================================================================

  // Convert to Float32 dense/scalar; one-time dtype dispatch outside the loop.
  Tensor to_float32() const {
    if (dtype == DataType::Float32 && kind == Kind::Dense) return *this;

    Tensor out(n_rows, n_cols);
    float* dst = out.f32_ptr();
    const std::size_t n = numel();

    if (kind == Kind::Sparse) {
      const auto& sp = *extras_->sparse;
      const float fill_f = static_cast<float>(sp.sp_default);
      std::fill(dst, dst + n, fill_f);
      const auto* vals_f32 =
          (dtype == DataType::Float32)
              ? reinterpret_cast<const float*>(sp.sp_values.data())
              : nullptr;
      for (int r = 0; r < n_rows; ++r) {
        for (int32_t k = sp.sp_indptr[r]; k < sp.sp_indptr[r + 1]; ++k) {
          const int col = sp.sp_indices[k];
          dst[static_cast<std::size_t>(r) * n_cols + col] =
              vals_f32 ? vals_f32[k]
                       : static_cast<float>(_read_elem(sp.sp_values.data(), k));
        }
      }
      return out;
    }

    if (kind == Kind::Scalar) {
      dst[0] = static_cast<float>(_read_elem(scalar_buf_, 0));
      return out;
    }

    const uint8_t* src = _raw_bytes();
    switch (dtype) {
      case DataType::Float32:
        std::memcpy(dst, src, n * sizeof(float));
        break;
      case DataType::Float64: {
        const auto* p = reinterpret_cast<const double*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::Int8: {
        const auto* p = reinterpret_cast<const int8_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::Int16: {
        const auto* p = reinterpret_cast<const int16_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::Int32: {
        const auto* p = reinterpret_cast<const int32_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::Int64: {
        const auto* p = reinterpret_cast<const int64_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::UInt8:
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
        break;
      case DataType::UInt16: {
        const auto* p = reinterpret_cast<const uint16_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::UInt32: {
        const auto* p = reinterpret_cast<const uint32_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::UInt64: {
        const auto* p = reinterpret_cast<const uint64_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(p[i]);
        break;
      }
      case DataType::Bool:
        for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] ? 1.f : 0.f;
        break;
      default:
        std::fill(dst, dst + n, 0.f);
        break;
    }
    return out;
  }

  // Read-only Float32 view: for already-Float32-dense tensors, returns a
  // non-owning view over the same buffer (no copy). Otherwise falls back to
  // a converted (owning) tensor.
  Tensor to_float32_view() const {
    if (dtype == DataType::Float32 && kind == Kind::Dense)
      return Tensor::from_buffer(_raw_bytes(), DataType::Float32, n_rows,
                                 n_cols);
    return to_float32();
  }

  // Convert to a Float64 dense (or scalar) tensor.
  Tensor to_float64() const {
    if (dtype == DataType::Float64 && kind == Kind::Dense) return *this;

    Tensor out = dense(DataType::Float64, n_rows, n_cols);
    double* dst = out.f64_ptr();
    const std::size_t n = numel();

    if (kind == Kind::Sparse) {
      const auto& sp = *extras_->sparse;
      const double fill_d = sp.sp_default;
      std::fill(dst, dst + n, fill_d);
      for (int r = 0; r < n_rows; ++r) {
        for (int32_t k = sp.sp_indptr[r]; k < sp.sp_indptr[r + 1]; ++k) {
          const int col = sp.sp_indices[k];
          dst[static_cast<std::size_t>(r) * n_cols + col] =
              _read_elem(sp.sp_values.data(), k);
        }
      }
      return out;
    }

    if (kind == Kind::Scalar) {
      dst[0] = _read_elem(scalar_buf_, 0);
      return out;
    }

    const uint8_t* src = _raw_bytes();
    switch (dtype) {
      case DataType::Float64:
        std::memcpy(dst, src, n * sizeof(double));
        break;
      case DataType::Float32: {
        const auto* p = reinterpret_cast<const float*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::Int8: {
        const auto* p = reinterpret_cast<const int8_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::Int16: {
        const auto* p = reinterpret_cast<const int16_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::Int32: {
        const auto* p = reinterpret_cast<const int32_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::Int64: {
        const auto* p = reinterpret_cast<const int64_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::UInt8:
        for (std::size_t i = 0; i < n; ++i)
          dst[i] = static_cast<double>(src[i]);
        break;
      case DataType::UInt16: {
        const auto* p = reinterpret_cast<const uint16_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::UInt32: {
        const auto* p = reinterpret_cast<const uint32_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::UInt64: {
        const auto* p = reinterpret_cast<const uint64_t*>(src);
        for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<double>(p[i]);
        break;
      }
      case DataType::Bool:
        for (std::size_t i = 0; i < n; ++i) dst[i] = src[i] ? 1.0 : 0.0;
        break;
      default:
        std::fill(dst, dst + n, 0.0);
        break;
    }
    return out;
  }

  Tensor to_dense() const {
    if (kind != Kind::Sparse) return *this;
    return to_float32();
  }

 private:
  // -----------------------------------------------------------------------
  // DenseStore — owning byte buffer with type-erased deleter.
  //
  // Three ownership flavors, all freed correctly via the tag:
  //   * alloc(n)        — `new uint8_t[n]`, NO zero-init.
  //   * alloc_zeroed(n) — `new uint8_t[n]{}` zero-init.
  //   * adopt<T>(vec)   — zero-copy: moves the vector onto the heap.
  //
  // Move-only. Use clone() for explicit deep-copy.
  // -----------------------------------------------------------------------
  class DenseStore {
   public:
    DenseStore() noexcept = default;

    DenseStore(const DenseStore&) = delete;
    DenseStore& operator=(const DenseStore&) = delete;

    DenseStore(DenseStore&& o) noexcept
        : ptr_(o.ptr_), size_(o.size_), holder_(o.holder_), tag_(o.tag_) {
      o.ptr_ = nullptr;
      o.size_ = 0;
      o.holder_ = nullptr;
      o.tag_ = Tag::Empty;
    }

    DenseStore& operator=(DenseStore&& o) noexcept {
      if (this != &o) {
        release();
        ptr_ = o.ptr_;
        size_ = o.size_;
        holder_ = o.holder_;
        tag_ = o.tag_;
        o.ptr_ = nullptr;
        o.size_ = 0;
        o.holder_ = nullptr;
        o.tag_ = Tag::Empty;
      }
      return *this;
    }

    ~DenseStore() { release(); }

    static DenseStore alloc(std::size_t n) {
      DenseStore s;
      if (n == 0) return s;
      s.ptr_ = new uint8_t[n];
      s.size_ = n;
      s.tag_ = Tag::OwnedNew;
      return s;
    }

    static DenseStore alloc_zeroed(std::size_t n) {
      DenseStore s;
      if (n == 0) return s;
      s.ptr_ = new uint8_t[n]();
      s.size_ = n;
      s.tag_ = Tag::OwnedNew;
      return s;
    }

    template <typename T>
    static DenseStore adopt(std::vector<T> v) {
      static_assert(std::is_trivially_copyable_v<T>,
                    "DenseStore::adopt requires trivially-copyable element");
      DenseStore s;
      if (v.empty()) return s;
      if constexpr (std::is_same_v<T, uint8_t>) {
        auto* h = new std::vector<uint8_t>(std::move(v));
        s.ptr_ = h->data();
        s.size_ = h->size();
        s.holder_ = h;
        s.tag_ = Tag::OwnedVecBytes;
      } else if constexpr (std::is_same_v<T, float>) {
        auto* h = new std::vector<float>(std::move(v));
        s.ptr_ = reinterpret_cast<uint8_t*>(h->data());
        s.size_ = h->size() * sizeof(float);
        s.holder_ = h;
        s.tag_ = Tag::OwnedVecF32;
      } else if constexpr (std::is_same_v<T, double>) {
        auto* h = new std::vector<double>(std::move(v));
        s.ptr_ = reinterpret_cast<uint8_t*>(h->data());
        s.size_ = h->size() * sizeof(double);
        s.holder_ = h;
        s.tag_ = Tag::OwnedVecF64;
      } else {
        const std::size_t bytes = v.size() * sizeof(T);
        s = alloc(bytes);
        std::memcpy(s.ptr_, v.data(), bytes);
      }
      return s;
    }

    DenseStore clone() const {
      if (!ptr_ || size_ == 0) return {};
      DenseStore s = alloc(size_);
      std::memcpy(s.ptr_, ptr_, size_);
      return s;
    }

    uint8_t* data() noexcept { return ptr_; }
    const uint8_t* data() const noexcept { return ptr_; }
    std::size_t size() const noexcept { return size_; }
    bool empty() const noexcept { return size_ == 0; }

    void clear() noexcept { release(); }

   private:
    enum class Tag : uint8_t {
      Empty = 0,
      OwnedNew,       // new uint8_t[n]
      OwnedVecBytes,  // holder_ = std::vector<uint8_t>*
      OwnedVecF32,    // holder_ = std::vector<float>*
      OwnedVecF64,    // holder_ = std::vector<double>*
    };

    uint8_t* ptr_ = nullptr;
    std::size_t size_ = 0;
    void* holder_ = nullptr;
    Tag tag_ = Tag::Empty;

    void release() noexcept {
      switch (tag_) {
        case Tag::OwnedNew:
          delete[] ptr_;
          break;
        case Tag::OwnedVecBytes:
          delete static_cast<std::vector<uint8_t>*>(holder_);
          break;
        case Tag::OwnedVecF32:
          delete static_cast<std::vector<float>*>(holder_);
          break;
        case Tag::OwnedVecF64:
          delete static_cast<std::vector<double>*>(holder_);
          break;
        case Tag::Empty:
          break;
      }
      ptr_ = nullptr;
      size_ = 0;
      holder_ = nullptr;
      tag_ = Tag::Empty;
    }
  };

  // Owning byte buffer — empty when kind == Scalar or kind == Sparse.
  DenseStore data;

  // Inline scalar value — interpreted as `dtype` when kind == Kind::Scalar.
  alignas(8) uint8_t scalar_buf_[8] = {0, 0, 0, 0, 0, 0, 0, 0};

  // Sparse CSR storage — only allocated when kind == Kind::Sparse.
  struct SparseStore {
    std::vector<uint8_t> sp_values;
    std::vector<int32_t> sp_indices;
    std::vector<int32_t> sp_indptr;
    double sp_default = 0.0;
  };

  // Heap-allocated only when the tensor uses any of these rare features.
  // Common dense-numeric tensors carry a single 8-byte null pointer here
  // instead of a string vector + view ptr + sparse unique_ptr (~40 B).
  struct Extras {
    const void* view_ptr = nullptr;
    std::vector<std::string> str_data;
    std::unique_ptr<SparseStore> sparse;

    Extras() = default;
    Extras(const Extras& o)
        : view_ptr(o.view_ptr),
          str_data(o.str_data),
          sparse(o.sparse ? std::make_unique<SparseStore>(*o.sparse)
                          : nullptr) {}
    Extras& operator=(const Extras&) = delete;
    Extras(Extras&&) noexcept = default;
    Extras& operator=(Extras&&) noexcept = default;
  };
  std::unique_ptr<Extras> extras_;

  Extras* _ensure_extras() {
    if (!extras_) extras_ = std::make_unique<Extras>();
    return extras_.get();
  }

  static void _write_scalar_bytes(uint8_t* buf, DataType dt,
                                  double v) noexcept {
    switch (dt) {
      case DataType::Bool: {
        uint8_t x = (v != 0.0) ? 1 : 0;
        std::memcpy(buf, &x, 1);
        break;
      }
      case DataType::Int8: {
        int8_t x = static_cast<int8_t>(v);
        std::memcpy(buf, &x, 1);
        break;
      }
      case DataType::Int16: {
        int16_t x = static_cast<int16_t>(v);
        std::memcpy(buf, &x, 2);
        break;
      }
      case DataType::Int32: {
        int32_t x = static_cast<int32_t>(v);
        std::memcpy(buf, &x, 4);
        break;
      }
      case DataType::Int64: {
        int64_t x = static_cast<int64_t>(v);
        std::memcpy(buf, &x, 8);
        break;
      }
      case DataType::UInt8: {
        uint8_t x = static_cast<uint8_t>(v);
        std::memcpy(buf, &x, 1);
        break;
      }
      case DataType::UInt16: {
        uint16_t x = static_cast<uint16_t>(v);
        std::memcpy(buf, &x, 2);
        break;
      }
      case DataType::UInt32: {
        uint32_t x = static_cast<uint32_t>(v);
        std::memcpy(buf, &x, 4);
        break;
      }
      case DataType::UInt64: {
        uint64_t x = static_cast<uint64_t>(v);
        std::memcpy(buf, &x, 8);
        break;
      }
      case DataType::Float64: {
        double x = v;
        std::memcpy(buf, &x, 8);
        break;
      }
      default: {
        float x = static_cast<float>(v);
        std::memcpy(buf, &x, 4);
        break;
      }
    }
  }

  const uint8_t* _raw_bytes() const noexcept {
    if (extras_ && extras_->view_ptr) OMLE_UNLIKELY
    return static_cast<const uint8_t*>(extras_->view_ptr);
    return data.data();
  }

  double _read_elem(const uint8_t* buf, std::size_t idx) const noexcept {
    const uint8_t* ep = buf + idx * static_cast<std::size_t>(dtype_size(dtype));
    switch (dtype) {
      case DataType::Float32:
        return *reinterpret_cast<const float*>(ep);
      case DataType::Float64:
        return *reinterpret_cast<const double*>(ep);
      case DataType::Int8:
        return *reinterpret_cast<const int8_t*>(ep);
      case DataType::Int16:
        return *reinterpret_cast<const int16_t*>(ep);
      case DataType::Int32:
        return *reinterpret_cast<const int32_t*>(ep);
      case DataType::Int64:
        return static_cast<double>(*reinterpret_cast<const int64_t*>(ep));
      case DataType::UInt8:
        return *ep;
      case DataType::UInt16:
        return *reinterpret_cast<const uint16_t*>(ep);
      case DataType::UInt32:
        return *reinterpret_cast<const uint32_t*>(ep);
      case DataType::UInt64:
        return static_cast<double>(*reinterpret_cast<const uint64_t*>(ep));
      case DataType::Bool:
        return *ep ? 1.0 : 0.0;
      default:
        return 0.0;
    }
  }

  static Tensor _expand_csr_to_dense(int rows, int cols,
                                     const std::vector<uint8_t>& vals_bytes,
                                     const std::vector<int32_t>& indices,
                                     const std::vector<int32_t>& indptr,
                                     float fill) {
    Tensor out(rows, cols, fill);
    float* dst = out.f32_ptr();
    const float* src = reinterpret_cast<const float*>(vals_bytes.data());
    for (int r = 0; r < rows; ++r)
      for (int32_t k = indptr[r]; k < indptr[r + 1]; ++k)
        dst[static_cast<std::size_t>(r) * cols + indices[k]] = src[k];
    return out;
  }
};

}  // namespace omle::rt

#endif  // OMLE_TENSOR_H_
