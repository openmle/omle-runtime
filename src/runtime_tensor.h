#ifndef OMLE_RUNTIME_TENSOR_H_
#define OMLE_RUNTIME_TENSOR_H_

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "omle/tensor.h"

namespace omle::rt::impl {

using Tensor = omle::rt::Tensor;

// Named constants loaded once at model-load time.
using ConstantStore =
    std::unordered_map<std::string, std::shared_ptr<const Tensor>>;

// ================================================================
// ValueStore — per-call mutable namespace.
// Lookup chain: local values → parent scope → constants.
// ================================================================
class ValueStore {
 public:
  explicit ValueStore(const ConstantStore& consts,
                      const ValueStore* parent = nullptr)
      : constants_(&consts), parent_(parent) {}

  ValueStore child_scope(const ValueStore* parent = nullptr) const {
    return ValueStore(*constants_, parent);
  }

  void put(const std::string& name, Tensor t) { values_[name] = std::move(t); }

  const Tensor& get(const std::string& name) const {
    auto it = values_.find(name);
    if (it != values_.end()) return it->second;
    if (parent_) return parent_->get(name);
    if (constants_) {
      auto ci = constants_->find(name);
      if (ci != constants_->end()) return *ci->second;
    }
    throw std::runtime_error("omle: graph value '" + name + "' not found");
  }

  bool has(const std::string& name) const {
    if (values_.count(name)) return true;
    if (parent_ && parent_->has(name)) return true;
    return constants_ && constants_->count(name);
  }

  const std::unordered_map<std::string, Tensor>& local() const {
    return values_;
  }

 private:
  std::unordered_map<std::string, Tensor> values_;
  const ConstantStore* constants_ = nullptr;
  const ValueStore* parent_ = nullptr;
};

// ================================================================
// gather_slots
// Concatenates named slots from the ValueStore into a contiguous
// dense [n_rows, total_cols] tensor.
//
// dtype controls the output element type:
//   Unknown  → infer from the first input slot (Float32 or Float64)
//   Float32  → always produce a Float32 tensor (legacy behaviour)
//   Float64  → always produce a Float64 tensor
//
// Sparse slots are expanded before gathering.
// ================================================================
inline Tensor gather_slots(
    const ValueStore& vs, const std::vector<std::string>& slot_names,
    int n_rows, omle::rt::DataType dtype = omle::rt::DataType::Unknown) {
  int total_cols = 0;
  for (const auto& name : slot_names) total_cols += vs.get(name).n_cols;

  // Infer output dtype from first slot when not specified.
  if (dtype == omle::rt::DataType::Unknown) {
    if (!slot_names.empty()) dtype = vs.get(slot_names[0]).dtype;
    if (dtype != omle::rt::DataType::Float64)
      dtype = omle::rt::DataType::Float32;
  }

  if (dtype == omle::rt::DataType::Float64) {
    Tensor out = omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows,
                                         total_cols);
    int col_off = 0;
    for (const auto& name : slot_names) {
      const Tensor& src = vs.get(name);
      const int w = src.n_cols;
      if (src.is_dense() && src.dtype == omle::rt::DataType::Float64) {
        for (int r = 0; r < n_rows; ++r)
          std::memcpy(out.f64_row(r) + col_off, src.f64_row(r),
                      w * sizeof(double));
      } else {
        Tensor d = src.to_float64();
        for (int r = 0; r < n_rows; ++r)
          std::memcpy(out.f64_row(r) + col_off, d.f64_row(r),
                      w * sizeof(double));
      }
      col_off += w;
    }
    return out;
  }

  // Float32 path (original behaviour).
  Tensor out(n_rows, total_cols);
  int col_off = 0;
  for (const auto& name : slot_names) {
    const Tensor& src = vs.get(name);
    const int w = src.n_cols;
    if (src.is_dense() && src.dtype == omle::rt::DataType::Float32) {
      for (int r = 0; r < n_rows; ++r)
        std::memcpy(out.row(r) + col_off, src.row(r), w * sizeof(float));
    } else {
      Tensor dense_src = src.to_float32();
      for (int r = 0; r < n_rows; ++r)
        std::memcpy(out.row(r) + col_off, dense_src.row(r), w * sizeof(float));
    }
    col_off += w;
  }
  return out;
}

// ================================================================
// scatter_outputs
// Writes a raw [n_rows × total_cols] buffer (float or double) into
// the ValueStore under the given output names.
// ================================================================
template <typename T>
inline void scatter_outputs(ValueStore& vs,
                            const std::vector<std::string>& output_names,
                            const T* data, int n_rows, int total_cols) {
  constexpr auto dt = std::is_same<T, double>::value
                          ? omle::rt::DataType::Float64
                          : omle::rt::DataType::Float32;

  if (output_names.size() == 1) {
    Tensor t = omle::rt::Tensor::dense(dt, n_rows, total_cols);
    T* dst = std::is_same<T, double>::value ? reinterpret_cast<T*>(t.f64_ptr())
                                            : reinterpret_cast<T*>(t.f32_ptr());
    std::memcpy(dst, data,
                static_cast<std::size_t>(n_rows) * total_cols * sizeof(T));
    vs.put(output_names[0], std::move(t));
    return;
  }
  for (int j = 0; j < static_cast<int>(output_names.size()) && j < total_cols;
       ++j) {
    Tensor col = omle::rt::Tensor::dense(dt, n_rows, 1);
    T* dst = std::is_same<T, double>::value
                 ? reinterpret_cast<T*>(col.f64_ptr())
                 : reinterpret_cast<T*>(col.f32_ptr());
    for (int r = 0; r < n_rows; ++r) dst[r] = data[r * total_cols + j];
    vs.put(output_names[j], std::move(col));
  }
}

// Convenience overload keeping the old float-only call sites working.
inline void scatter_outputs(ValueStore& vs,
                            const std::vector<std::string>& output_names,
                            const float* data, int n_rows, int total_cols) {
  scatter_outputs<float>(vs, output_names, data, n_rows, total_cols);
}

}  // namespace omle::rt::impl

#endif  // OMLE_RUNTIME_TENSOR_H_
