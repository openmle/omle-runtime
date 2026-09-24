#ifndef OMLE_WIDTH_DISPATCH_H_
#define OMLE_WIDTH_DISPATCH_H_

// Element width as a template parameter.
//
// An operator or node that honours float64 used to be written twice: one arm
// reading f64_ptr() and writing f64_at(), another reading row() and writing
// at(). Both arms are the same algorithm, so they drift -- a bounds check or a
// correction applied to one silently misses the other -- and the branch often
// sat inside the innermost loop, costing a test per element for a decision
// that is fixed for the whole call.
//
// Width<T> names the per-type pieces and with_width() runs a generic lambda
// once for whichever type the data has, so the algorithm is written once and
// the compiler emits both instantiations.

#include <cstddef>
#include <vector>

#include "runtime_tensor.h"

namespace omle::rt::impl {

template <typename T>
struct Width;

template <>
struct Width<float> {
  static constexpr omle::rt::DataType kDtype = omle::rt::DataType::Float32;
  static float* data(Tensor& t) { return t.f32_ptr(); }
  static const float* data(const Tensor& t) { return t.f32_ptr(); }
  static Tensor convert(const Tensor& t) { return t.to_float32(); }
};

template <>
struct Width<double> {
  static constexpr omle::rt::DataType kDtype = omle::rt::DataType::Float64;
  static double* data(Tensor& t) { return t.f64_ptr(); }
  static const double* data(const Tensor& t) { return t.f64_ptr(); }
  static Tensor convert(const Tensor& t) { return t.to_float64(); }
};

// Run `body` with its parameter bound to float or double. The lambda takes a
// value it never reads -- `[&](auto tag) { using T = decltype(tag); ... }` --
// which is the least ceremony that still gives the body a type to work with.
template <typename F>
inline auto with_width(bool f64, F&& body) -> decltype(body(float{})) {
  if (f64) return body(double{});
  return body(float{});
}

// Dense result tensor of width T, zero-filled.
template <typename T>
inline Tensor dense_w(int rows, int cols) {
  return Tensor::dense(Width<T>::kDtype, rows, cols);
}

template <typename T>
inline T* data_w(Tensor& t) {
  return Width<T>::data(t);
}
template <typename T>
inline const T* data_w(const Tensor& t) {
  return Width<T>::data(t);
}
template <typename T>
inline T* row_w(Tensor& t, int r) {
  return Width<T>::data(t) + static_cast<std::size_t>(r) * t.n_cols;
}
template <typename T>
inline const T* row_w(const Tensor& t, int r) {
  return Width<T>::data(t) + static_cast<std::size_t>(r) * t.n_cols;
}

// Read-only view of a tensor at width T: no copy when it already has that
// dtype, a converted copy owned by `scratch` otherwise. Also the safe way to
// read a source whose dtype is neither float -- an Int32 column read straight
// through at() would be reinterpreted rather than converted.
template <typename T>
inline const T* view_w(const Tensor* t, Tensor& scratch) {
  if (!t) return nullptr;
  // A non-owning view is fine to read: the const accessor goes through
  // _raw_bytes(), which knows about view_ptr. Only writes would need it
  // materialised, and this view is read-only.
  if (t->dtype == Width<T>::kDtype && !t->is_sparse())
    return Width<T>::data(*t);
  scratch = Width<T>::convert(*t);
  return Width<T>::data(scratch);
}

template <typename T>
inline const T* view_w(const Tensor& t, Tensor& scratch) {
  return view_w<T>(&t, scratch);
}

// True when a kernel should produce float64: the data arrived that way.
inline bool wants_f64(const Tensor& src) {
  return src.dtype == omle::rt::DataType::Float64;
}

// Write one element at the tensor's own width -- the counterpart to get().
// Saves repeating a three-line if/else at every in-place write-back.
inline void set_at(Tensor& t, int r, int c, double v) {
  if (t.dtype == omle::rt::DataType::Float64)
    t.f64_at(r, c) = v;
  else
    t.at(r, c) = static_cast<float>(v);
}

// Result buffer whose element width follows the input's: float64 in, float64
// out, so a pipeline that entered in double stays in double all the way to
// the model. Kernels compute in double regardless and store through set(),
// which rounds once on the way out when the result is float32 -- accumulating
// in float instead rounds twice and can land an ulp away, which is invisible
// to a linear head but flips a tree split that sits on the boundary.
class OutBuf {
 public:
  OutBuf(bool want_f64, int rows, int cols)
      : f64_(want_f64),
        t_(want_f64 ? Tensor::dense(omle::rt::DataType::Float64, rows, cols)
                    : Tensor(rows, cols, 0.0f)) {}

  void set(int r, int c, double v) {
    if (f64_)
      t_.f64_at(r, c) = v;
    else
      t_.at(r, c) = static_cast<float>(v);
  }
  double get(int r, int c) const { return t_.get(r, c); }
  bool is_f64() const { return f64_; }
  Tensor& tensor() { return t_; }
  Tensor take() { return std::move(t_); }

 private:
  bool f64_;
  Tensor t_;
};

}  // namespace omle::rt::impl

#endif  // OMLE_WIDTH_DISPATCH_H_
