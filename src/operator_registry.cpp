#include "operator_registry.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "expression_eval.h"
#include "math_utils.h"
#include "omle/status.h"
#include "status_macros.h"
#include "width_dispatch.h"

namespace omle::rt::impl {

OperatorRegistry& OperatorRegistry::instance() {
  static OperatorRegistry inst;
  return inst;
}

void OperatorRegistry::reg(const std::string& domain, const std::string& op,
                           OperatorFn fn) {
  table_[domain + "/" + op] = std::move(fn);
}

const OperatorFn* OperatorRegistry::find(const std::string& domain,
                                         const std::string& op) const {
  auto it = table_.find(domain + "/" + op);
  return (it != table_.end()) ? &it->second : nullptr;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int attr_i(const AttributeMap& attrs, const std::string& key,
                  int def = 0) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  if (it->second.kind == AttrVal::Kind::Int)
    return static_cast<int>(it->second.i);
  if (it->second.kind == AttrVal::Kind::Float)
    return static_cast<int>(it->second.f);
  if (it->second.kind == AttrVal::Kind::Ints && !it->second.ints.empty())
    return static_cast<int>(it->second.ints[0]);
  return def;
}

static std::string attr_s(const AttributeMap& attrs, const std::string& key,
                          const std::string& def = {}) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  if (it->second.kind == AttrVal::Kind::String) return it->second.s;
  return def;
}

static bool attr_b(const AttributeMap& attrs, const std::string& key,
                   bool def = false) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  if (it->second.kind == AttrVal::Kind::Bool) return it->second.b;
  if (it->second.kind == AttrVal::Kind::Int) return it->second.i != 0;
  return def;
}

// Resolve a tensor_ref: try slot index first, then attribute name.
static const Tensor* resolve_t(const ValueStore& vs,
                               const std::vector<std::string>& in, size_t slot,
                               const AttributeMap& attrs, const char* aname) {
  if (slot < in.size() && vs.has(in[slot])) return &vs.get(in[slot]);
  const auto& tn = attr_tensor(attrs, aname);
  return (!tn.empty() && vs.has(tn)) ? &vs.get(tn) : nullptr;
}

// A FLOAT64 constant is stored twice by the loader: narrowed to float32 under
// its own name (what every kernel here reads), and undegraded under this
// suffix. Operators whose output feeds an exact comparison -- a tree split
// threshold, say -- resolve the twin so the model's real parameters are used.
static const Tensor* resolve_t_exact(const ValueStore& vs,
                                     const AttributeMap& attrs,
                                     const char* aname) {
  const auto& tn = attr_tensor(attrs, aname);
  if (tn.empty()) return nullptr;
  const std::string exact = tn + "\001f64";
  return vs.has(exact) ? &vs.get(exact) : nullptr;
}

// Resolve a tensor_ref preferring the undegraded float64 twin. resolve_t()
// hands back the float32 narrowing the loader stores under the constant's own
// name; a kernel that accumulates in double wants the real coefficients, or
// the extra width buys it nothing.
static const Tensor* resolve_t64(const ValueStore& vs,
                                 const std::vector<std::string>& in,
                                 size_t slot, const AttributeMap& attrs,
                                 const char* aname) {
  if (const Tensor* x = resolve_t_exact(vs, attrs, aname)) return x;
  return resolve_t(vs, in, slot, attrs, aname);
}

// Flat, dtype-generic element read for 1-D parameter vectors, whose stored
// shape is [1, n] or [n, 1] depending on the producer. Going through get()
// with the index decomposed against the real n_cols keeps both shapes -- and
// the sparse and scalar kinds -- reading correctly.
static inline double flat_get(const Tensor& t, int i) {
  const int nc = t.n_cols > 0 ? t.n_cols : 1;
  return t.get(i / nc, i % nc);
}

static inline double flat_get_or(const Tensor* t, int i, double def) {
  return t ? flat_get(*t, i) : def;
}

// Same question for a variadic operator: float64 out if any input carried it.
static inline bool any_f64(const ValueStore& vs,
                           const std::vector<std::string>& in) {
  for (const auto& nm : in)
    if (vs.has(nm) && vs.get(nm).dtype == omle::rt::DataType::Float64)
      return true;
  return false;
}

// An owned tensor of the given width, safe to write through.
//
// to_float32()/to_float64() return the original whenever it already has the
// requested dtype and dense kind -- and a non-owning view satisfies both.
// Reads on a view go through _raw_bytes(), which knows about view_ptr, but
// f32_ptr()/f64_ptr() write through data.data(), which for a view is null.
// A kernel that materialises and then mutates needs the copy forced.
static Tensor owned_of(const Tensor& t, omle::rt::DataType dt) {
  if (t.dtype == dt && !t.is_sparse() && !t.is_view()) return t;
  Tensor out = Tensor::dense(dt, t.n_rows, t.n_cols);
  const int nc = t.n_cols;
  for (int r = 0; r < t.n_rows; ++r)
    for (int c = 0; c < nc; ++c) {
      if (dt == omle::rt::DataType::Float64)
        out.f64_at(r, c) = t.get(r, c);
      else
        out.at(r, c) = static_cast<float>(t.get(r, c));
    }
  return out;
}

// Gather all variadic inputs into one wide dense float32 tensor.
// (inline gather_slots from runtime_tensor.h already covers this, this wrapper
// handles
//  the "no inputs" edge case cleanly.)
static Tensor concat_inputs(const ValueStore& vs,
                            const std::vector<std::string>& in, int n_rows) {
  if (in.empty()) return Tensor(n_rows, 0);
  int total = 0;
  // Widen to float64 when any input carries it. Operators below branch on
  // src.dtype to pick a double kernel, but narrowing here destroyed that
  // information before they ever saw it, so the double paths could not fire
  // and a float64 column silently lost precision on its way in.
  bool any_f64 = false;
  for (const auto& nm : in) {
    const Tensor& t = vs.get(nm);
    total += t.n_cols;
    if (t.dtype == omle::rt::DataType::Float64) any_f64 = true;
  }

  if (any_f64) {
    Tensor out =
        omle::rt::Tensor::dense(omle::rt::DataType::Float64, n_rows, total);
    int off = 0;
    for (const auto& nm : in) {
      const Tensor& src = vs.get(nm);
      const int w = src.n_cols;
      if (src.is_dense() && src.dtype == omle::rt::DataType::Float64) {
        for (int r = 0; r < n_rows; ++r)
          std::memcpy(out.f64_row(r) + off, src.f64_row(r), w * sizeof(double));
      } else {
        const Tensor d = src.to_float64();
        for (int r = 0; r < n_rows; ++r)
          std::memcpy(out.f64_row(r) + off, d.f64_row(r), w * sizeof(double));
      }
      off += w;
    }
    return out;
  }

  Tensor out(n_rows, total);
  int off = 0;
  for (const auto& nm : in) {
    const Tensor& src = vs.get(nm);
    const int w = src.n_cols;
    if (src.is_dense() && src.dtype == omle::rt::DataType::Float32) {
      for (int r = 0; r < n_rows; ++r)
        std::memcpy(out.row(r) + off, src.row(r), w * sizeof(float));
    } else {
      Tensor d = src.to_float32();
      for (int r = 0; r < n_rows; ++r)
        std::memcpy(out.row(r) + off, d.row(r), w * sizeof(float));
    }
    off += w;
  }
  return out;
}

// Simple dense matrix-vector product (row of A times columns of B).
// A: [n_rows, k]   B: [k, n_cols]   out: [n_rows, n_cols]
template <typename T>
static Tensor matmul(const Tensor& A, const Tensor& B, int n_rows) {
  const int k = A.n_cols, n_cols = B.n_cols;
  constexpr bool is64 = std::is_same<T, double>::value;
  Tensor out = Tensor::dense(
      is64 ? omle::rt::DataType::Float64 : omle::rt::DataType::Float32, n_rows,
      n_cols);
  T* op = is64 ? reinterpret_cast<T*>(out.f64_ptr())
               : reinterpret_cast<T*>(out.f32_ptr());
  const T* ap = is64 ? reinterpret_cast<const T*>(A.f64_ptr())
                     : reinterpret_cast<const T*>(A.f32_ptr());
  const T* bp = is64 ? reinterpret_cast<const T*>(B.f64_ptr())
                     : reinterpret_cast<const T*>(B.f32_ptr());
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
      for (int i = 0; i < k; ++i)
        op[r * n_cols + c] += ap[r * k + i] * bp[i * n_cols + c];
  return out;
}

// Clamp a probability to [eps, 1-eps] for log/probit stability.
static constexpr double kProbitEps = 1e-7;

// Rational approximation for the probit function (inverse normal CDF).
// Abramowitz and Stegun approximation, max error ~4.5e-4.
template <typename T>
static T probit(T p) {
  p = std::clamp(p, T(kProbitEps), T(1.0) - T(kProbitEps));
  if (p < T(0.5)) return -probit(T(1.0) - p);
  const T t = std::sqrt(T(-2.0) * std::log(T(1.0) - p));
  const T c0 = T(2.515517), c1 = T(0.802853), c2 = T(0.010328);
  const T d1 = T(1.432788), d2 = T(0.189269), d3 = T(0.001308);
  return t - (c0 + c1 * t + c2 * t * t) /
                 (T(1.0) + d1 * t + d2 * t * t + d3 * t * t * t);
}

// ---------------------------------------------------------------------------
// omle.core operators
// ---------------------------------------------------------------------------

static omle::rt::Status op_identity(ValueStore& vs, int /*n_rows*/,
                                    const std::vector<std::string>& in,
                                    const std::vector<std::string>& out,
                                    const AttributeMap&) {
  for (size_t i = 0; i < in.size() && i < out.size(); ++i)
    vs.put(out[i], vs.get(in[i]));
  return {};
}

static omle::rt::Status op_concat(ValueStore& vs, int n_rows,
                                  const std::vector<std::string>& in,
                                  const std::vector<std::string>& out,
                                  const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0], concat_inputs(vs, in, n_rows));
  return {};
}

static omle::rt::Status op_gather(ValueStore& vs, int n_rows,
                                  const std::vector<std::string>& in,
                                  const std::vector<std::string>& out,
                                  const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  std::vector<int> cols;
  auto it = attrs.find("indices");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Ints)
    for (auto v : it->second.ints) cols.push_back(static_cast<int>(v));

  if (cols.empty()) {
    vs.put(out[0], src);
    return {};
  }

  // Gather picks columns; it computes nothing. Reading through at() retyped
  // a float64 source to float32 in passing, so a column selection in the
  // middle of a pipeline silently narrowed everything downstream of it.
  OutBuf result(wants_f64(src), n_rows, static_cast<int>(cols.size()));
  for (int r = 0; r < n_rows; ++r)
    for (int j = 0; j < (int)cols.size(); ++j)
      result.set(r, j, src.get(r, cols[j]));
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_clip(ValueStore& vs, int n_rows,
                                const std::vector<std::string>& in,
                                const std::vector<std::string>& out,
                                const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  Tensor src = vs.get(in[0]);
  const double lo =
      attr_float(attrs, "min", -std::numeric_limits<double>::infinity());
  const double hi =
      attr_float(attrs, "max", std::numeric_limits<double>::infinity());
  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    // The bounds are compared at the storage width, so a float32 tensor
    // clamps against the rounded bound exactly as it did before.
    const T tlo = static_cast<T>(lo), thi = static_cast<T>(hi);
    T* p = data_w<T>(src);
    for (int i = 0, n = static_cast<int>(src.numel()); i < n; ++i)
      p[i] = std::clamp(p[i], tlo, thi);
  });
  vs.put(out[0], std::move(src));
  return {};
}

static omle::rt::Status op_argmax(ValueStore& vs, int n_rows,
                                  const std::vector<std::string>& in,
                                  const std::vector<std::string>& out,
                                  const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);
  const int nf = src.n_cols;
  // The output is a column index -- exact in float32 -- but the comparison
  // that picks it is not. at() reinterpreted float64 scores as float, and
  // even converted correctly, two near-tied class scores can order
  // differently once narrowed, which changes the predicted class rather than
  // nudging a number. Strict > keeps sklearn's first-wins tie-break.
  Tensor result(n_rows, 1);
  for (int r = 0; r < n_rows; ++r) {
    double best = -std::numeric_limits<double>::infinity();
    int best_idx = 0;
    for (int c = 0; c < nf; ++c) {
      const double v = src.get(r, c);
      if (v > best) {
        best = v;
        best_idx = c;
      }
    }
    result.at(r, 0) = static_cast<float>(best_idx);
  }
  vs.put(out[0], std::move(result));
  return {};
}

// Select one class index per row: highest primary score, ties broken by the
// highest secondary score, and any remaining tie by the smallest class index.
// Shares ArgMax's output contract (int64 class index in a single column).
// Used to decode sklearn OneVsOneClassifier, where the primary scores are vote
// counts and the secondary scores are summed confidences.
static omle::rt::Status op_select_by_primary_secondary_score(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap&) {
  if (in.size() < 2 || out.empty())
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.core/SelectByPrimarySecondaryScore: primary_scores and "
            "secondary_scores are required"};
  const Tensor& primary = vs.get(in[0]);
  const Tensor& secondary = vs.get(in[1]);
  if (primary.n_rows != secondary.n_rows || primary.n_cols != secondary.n_cols)
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.core/SelectByPrimarySecondaryScore: primary_scores and "
            "secondary_scores must have the same shape"};
  const int nc = primary.n_cols;
  Tensor result(n_rows, 1);
  for (int r = 0; r < n_rows; ++r) {
    int best = 0;
    // Ascending scan replacing only on a strict improvement, so an exact tie on
    // both scores keeps the earlier (smaller) index, as the spec requires.
    for (int c = 1; c < nc; ++c) {
      const double p = primary.get(r, c);
      const double pb = primary.get(r, best);
      if (p > pb) {
        best = c;
      } else if (p == pb && secondary.get(r, c) > secondary.get(r, best)) {
        best = c;
      }
    }
    result.at(r, 0) = static_cast<float>(best);
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_cast(ValueStore& vs, int n_rows,
                                const std::vector<std::string>& in,
                                const std::vector<std::string>& out,
                                const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const std::string dtype = attr_s(attrs, "output_dtype", "float32");
  const Tensor& src = vs.get(in[0]);

  // Cast used to funnel every request through to_float32(), so the one thing
  // the operator exists to do -- change width -- it could not do. Asking for
  // float64 gave float32, and asking for an integer type rounded in float32:
  // above 2^24 consecutive integers are not representable, so the rounding
  // landed on a neighbour of the intended value.
  if (dtype == "float64") {
    vs.put(out[0], src.to_float64());
    return {};
  }
  if (dtype == "int32" || dtype == "int64") {
    // Quantise the value, keep the width: narrowing here would undo the
    // exactness that made the integer cast meaningful.
    Tensor result = with_width(wants_f64(src), [&](auto tag) {
      using T = decltype(tag);
      Tensor res = owned_of(src, Width<T>::kDtype);
      T* p = data_w<T>(res);
      for (int i = 0, n = static_cast<int>(res.numel()); i < n; ++i)
        p[i] = std::round(p[i]);
      return res;
    });
    vs.put(out[0], std::move(result));
    return {};
  }
  vs.put(out[0], src.to_float32());
  return {};
}

static omle::rt::Status op_reshape(ValueStore& vs, int n_rows,
                                   const std::vector<std::string>& in,
                                   const std::vector<std::string>& out,
                                   const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  Tensor src = vs.get(in[0]);
  const int total = static_cast<int>(src.numel());

  auto it = attrs.find("new_shape");
  if (it == attrs.end() || it->second.kind != AttrVal::Kind::Ints) {
    vs.put(out[0], std::move(src));
    return {};
  }
  const auto& sh = it->second.ints;
  int new_rows = n_rows, new_cols = 1;
  if (sh.size() == 1) {
    new_cols = (sh[0] == -1) ? total / n_rows : static_cast<int>(sh[0]);
  } else if (sh.size() >= 2) {
    new_rows = (sh[0] == -1) ? total / static_cast<int>(sh[1])
                             : static_cast<int>(sh[0]);
    new_cols = (sh[1] == -1) ? total / new_rows : static_cast<int>(sh[1]);
  }
  // Reshape only relabels the element grid -- the bytes are unchanged, so the
  // dtype must survive. Copying out through f32_ptr() reinterpreted float64
  // storage as float outright, turning a shape change into a data change.
  // Sparse is materialised first (its CSR indptr is tied to the old shape),
  // at the source's own width.
  Tensor result = src.is_sparse() ? (src.dtype == omle::rt::DataType::Float64
                                         ? src.to_float64()
                                         : src.to_float32())
                                  : std::move(src);
  result.n_rows = new_rows;
  result.n_cols = new_cols;
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_split(ValueStore& vs, int n_rows,
                                 const std::vector<std::string>& in,
                                 const std::vector<std::string>& out,
                                 const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  std::vector<int> secs;
  auto it = attrs.find("sections");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Ints)
    for (auto v : it->second.ints) secs.push_back(static_cast<int>(v));

  if (secs.empty()) {
    vs.put(out[0], src);
    return {};
  }

  // Splitting is slicing, not arithmetic: each segment keeps the source's
  // width. The byte-wise memcpy through row() read float64 storage as float.
  const bool f64 = wants_f64(src);
  int off = 0;
  for (size_t s = 0; s < secs.size() && s < out.size(); ++s) {
    const int w = secs[s];
    OutBuf seg(f64, n_rows, w);
    for (int r = 0; r < n_rows; ++r)
      for (int c = 0; c < w; ++c) seg.set(r, c, src.get(r, off + c));
    vs.put(out[s], seg.take());
    off += w;
  }
  return {};
}

static omle::rt::Status op_take_slots(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  Tensor wide = concat_inputs(vs, in, n_rows);

  std::vector<int> cols;
  auto it = attrs.find("indices");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Ints)
    for (auto v : it->second.ints) cols.push_back(static_cast<int>(v));

  if (cols.empty()) {
    vs.put(out[0], std::move(wide));
    return {};
  }

  // TakeSlots only selects columns, so it must hand the incoming precision
  // through. Allocating float32 unconditionally silently narrowed a float64
  // pipeline at the first selection.
  const int n_out = static_cast<int>(cols.size());
  Tensor result = with_width(wants_f64(wide), [&](auto tag) {
    using T = decltype(tag);
    Tensor wview;
    const T* wp = view_w<T>(wide, wview);
    const int wc = wide.n_cols;
    Tensor res = dense_w<T>(n_rows, n_out);
    T* dp = data_w<T>(res);
    for (int r = 0; r < n_rows; ++r)
      for (int j = 0; j < n_out; ++j) dp[r * n_out + j] = wp[r * wc + cols[j]];
    return res;
  });
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_derive(ValueStore& vs, int n_rows,
                                  const std::vector<std::string>& in,
                                  const std::vector<std::string>& out,
                                  const AttributeMap& attrs) {
  auto it = attrs.find("expr");
  if (it == attrs.end() || !it->second.expr)
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.core/Derive: missing 'expr' attribute"};

  // Multi-column fast path: evaluate expr once per input column via a
  // child scope that shadows the input name with a 1-column slice.
  if (!in.empty() && vs.get(in[0]).n_cols > 1) {
    const int n_cols = vs.get(in[0]).n_cols;
    OutBuf result(any_f64(vs, in), n_rows, n_cols);
    for (int c = 0; c < n_cols; ++c) {
      ValueStore child = vs.child_scope(&vs);
      const Tensor& src = vs.get(in[0]);
      // The slice is always float64: the evaluator reads it back through
      // get() either way, so widening here costs nothing and copying a
      // float64 column through a float32 slice would cost everything.
      std::vector<double> col(n_rows);
      for (int r = 0; r < n_rows; ++r) col[r] = src.get(r, c);
      child.put(in[0], Tensor::from_doubles(n_rows, 1, std::move(col)));
      ASSIGN_OR_RETURN(auto col_vals,
                       eval_expr(*it->second.expr, child, n_rows));
      for (int r = 0; r < n_rows; ++r) result.set(r, c, col_vals[r]);
    }
    if (!out.empty()) vs.put(out[0], result.take());
    return {};
  }

  ASSIGN_OR_RETURN(auto values, eval_expr(*it->second.expr, vs, n_rows));
  Tensor t(n_rows, 1);
  if (any_f64(vs, in)) {
    t.set_doubles(std::move(values));
  } else {
    std::vector<float> narrowed(values.begin(), values.end());
    t.set_floats(std::move(narrowed));
  }
  if (!out.empty()) vs.put(out[0], std::move(t));
  return {};
}

static omle::rt::Status op_select(ValueStore& vs, int n_rows,
                                  const std::vector<std::string>& in,
                                  const std::vector<std::string>& out,
                                  const AttributeMap&) {
  // in[0]: selector (int column), in[1..]: candidate tensors
  if (in.size() < 2 || out.empty()) return {};
  const Tensor& sel = vs.get(in[0]);
  const int n_cands = static_cast<int>(in.size()) - 1;
  const int n_cols = vs.get(in[1]).n_cols;
  // The row-wise memcpy assumed every candidate was float32 and so was the
  // result. Candidates can differ in width -- one branch of a pipeline may
  // have stayed in double while another did not -- and a raw byte copy out of
  // a float64 candidate produced garbage rather than a narrowed value.
  // Selecting is a copy, so the result keeps float64 if any candidate has it.
  OutBuf result(any_f64(vs, {in.begin() + 1, in.end()}), n_rows, n_cols);
  for (int r = 0; r < n_rows; ++r) {
    int idx = static_cast<int>(sel.get(r, 0));
    idx = std::clamp(idx, 0, n_cands - 1);
    const Tensor& cand = vs.get(in[1 + idx]);
    for (int c = 0; c < n_cols; ++c) result.set(r, c, cand.get(r, c));
  }
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_sparse_to_dense(ValueStore& vs, int /*n_rows*/,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0], vs.get(in[0]).to_float32());
  return {};
}

static omle::rt::Status op_dense_to_sparse(ValueStore& vs, int /*n_rows*/,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0], Tensor::to_sparse(vs.get(in[0])));
  return {};
}

// ---- Variadic reduction operators -----------------------------------------

// Collect one per-element list across all inputs for a given (row, col).
// All inputs must have the same shape [n_rows, n_cols].
// The element list is always double. Every input is read through the
// dtype-generic get(), which returns double whatever the storage is, so a
// float reducer would only round the values on the way in and again on the
// way out -- one reducer serves both widths. Each operator below used to pass
// two lambdas that were transcriptions of each other.
//
// The output width follows the inputs: float64 in, float64 out.
using Reducer = std::function<double(std::vector<double>&)>;

static Tensor variadic_reduce(ValueStore& vs,
                              const std::vector<std::string>& in, int n_rows,
                              bool f64_out, const Reducer& reducer) {
  if (in.empty()) return Tensor(n_rows, 0);
  const int n_cols = vs.get(in[0]).n_cols;
  OutBuf result(f64_out, n_rows, n_cols);
  std::vector<double> vals(in.size());
  for (int r = 0; r < n_rows; ++r) {
    for (int c = 0; c < n_cols; ++c) {
      for (size_t k = 0; k < in.size(); ++k) vals[k] = vs.get(in[k]).get(r, c);
      result.set(r, c, reducer(vals));
    }
  }
  return result.take();
}

// Widen if any input is float64, not just the first: a reduction that mixes
// widths should not be decided by argument order.
static Tensor variadic_reduce_dispatch(ValueStore& vs,
                                       const std::vector<std::string>& in,
                                       int n_rows, const Reducer& reducer) {
  return variadic_reduce(vs, in, n_rows, any_f64(vs, in), reducer);
}

static omle::rt::Status op_sum(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0],
         variadic_reduce_dispatch(vs, in, n_rows, [](std::vector<double>& v) {
           return std::accumulate(v.begin(), v.end(), 0.0);
         }));
  return {};
}

static omle::rt::Status op_average(ValueStore& vs, int n_rows,
                                   const std::vector<std::string>& in,
                                   const std::vector<std::string>& out,
                                   const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  const double n = static_cast<double>(in.size());
  vs.put(out[0],
         variadic_reduce_dispatch(vs, in, n_rows, [n](std::vector<double>& v) {
           return std::accumulate(v.begin(), v.end(), 0.0) / n;
         }));
  return {};
}

static omle::rt::Status op_min(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0],
         variadic_reduce_dispatch(vs, in, n_rows, [](std::vector<double>& v) {
           return *std::min_element(v.begin(), v.end());
         }));
  return {};
}

static omle::rt::Status op_max(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0],
         variadic_reduce_dispatch(vs, in, n_rows, [](std::vector<double>& v) {
           return *std::max_element(v.begin(), v.end());
         }));
  return {};
}

static omle::rt::Status op_median(ValueStore& vs, int n_rows,
                                  const std::vector<std::string>& in,
                                  const std::vector<std::string>& out,
                                  const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  vs.put(out[0],
         variadic_reduce_dispatch(vs, in, n_rows, [](std::vector<double>& v) {
           std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
           return v[v.size() / 2];
         }));
  return {};
}

static omle::rt::Status op_weighted_sum(ValueStore& vs, int n_rows,
                                        const std::vector<std::string>& in,
                                        const std::vector<std::string>& out,
                                        const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  std::vector<double> weights;
  {
    auto it = attrs.find("weights");
    if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats) {
      for (double w : it->second.floats) weights.push_back(w);
    } else {
      // Weights may be stored as a tensor attr (e.g. from ElementwiseProduct).
      const Tensor* wt = resolve_t(vs, in, in.size(), attrs, "weights");
      if (wt)
        for (int i = 0, n = (int)wt->numel(); i < n; ++i)
          weights.push_back(wt->get(0, i));
    }
  }

  // ElementwiseProduct pattern: single multi-column input, per-column weights.
  if (in.size() == 1) {
    const Tensor& src = vs.get(in[0]);
    const int nf = src.n_cols;
    while ((int)weights.size() < nf) weights.push_back(1.0);
    Tensor result = with_width(wants_f64(src), [&](auto tag) {
      using T = decltype(tag);
      Tensor sview;
      const T* sp = view_w<T>(src, sview);
      Tensor res = dense_w<T>(n_rows, nf);
      T* dp = data_w<T>(res);
      for (int r = 0; r < n_rows; ++r)
        for (int c = 0; c < nf; ++c)
          dp[r * nf + c] =
              static_cast<T>(static_cast<double>(sp[r * nf + c]) * weights[c]);
      return res;
    });
    vs.put(out[0], std::move(result));
    return {};
  }

  while (weights.size() < in.size()) weights.push_back(1.0);
  vs.put(out[0], variadic_reduce_dispatch(
                     vs, in, n_rows, [&weights](std::vector<double>& v) {
                       double s = 0.0;
                       for (size_t k = 0; k < v.size(); ++k)
                         s += weights[k] * v[k];
                       return s;
                     }));
  return {};
}

static omle::rt::Status op_weighted_average(ValueStore& vs, int n_rows,
                                            const std::vector<std::string>& in,
                                            const std::vector<std::string>& out,
                                            const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  std::vector<double> weights;
  auto it = attrs.find("weights");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats)
    for (double w : it->second.floats) weights.push_back(w);
  while (weights.size() < in.size()) weights.push_back(1.0);
  double wsum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (wsum == 0.0) wsum = 1.0;

  vs.put(out[0], variadic_reduce_dispatch(
                     vs, in, n_rows, [&weights, wsum](std::vector<double>& v) {
                       double s = 0.0;
                       for (size_t k = 0; k < v.size(); ++k)
                         s += weights[k] * v[k];
                       return s / wsum;
                     }));
  return {};
}

static omle::rt::Status op_weighted_median(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  std::vector<double> weights;
  auto it = attrs.find("weights");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats)
    for (double w : it->second.floats) weights.push_back(w);
  while (weights.size() < in.size()) weights.push_back(1.0);

  // Sort by value, accumulate weights until past 50%.
  vs.put(out[0], variadic_reduce_dispatch(
                     vs, in, n_rows, [&weights](std::vector<double>& v) {
                       const double total =
                           std::accumulate(weights.begin(), weights.end(), 0.0);
                       std::vector<size_t> idx(v.size());
                       std::iota(idx.begin(), idx.end(), 0);
                       std::sort(
                           idx.begin(), idx.end(),
                           [&v](size_t a, size_t b) { return v[a] < v[b]; });
                       double cum = 0.0;
                       for (size_t i : idx) {
                         cum += weights[i];
                         if (cum >= total * 0.5) return v[i];
                       }
                       return v[idx.back()];
                     }));
  return {};
}

// MajorityVote: each input is a [n_rows, 1] integer label column.
static omle::rt::Status op_majority_vote(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  Tensor result(n_rows, 1);
  std::unordered_map<int, int> votes;
  for (int r = 0; r < n_rows; ++r) {
    votes.clear();
    for (const auto& nm : in) votes[static_cast<int>(vs.get(nm).get(r, 0))]++;
    int best = 0, best_count = -1;
    for (const auto& kv : votes)
      if (kv.second > best_count) {
        best_count = kv.second;
        best = kv.first;
      }
    result.at(r, 0) = static_cast<float>(best);
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_weighted_majority_vote(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  std::vector<double> weights;
  auto it = attrs.find("weights");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats)
    for (double w : it->second.floats) weights.push_back(w);
  while (weights.size() < in.size()) weights.push_back(1.0);

  Tensor result(n_rows, 1);
  std::unordered_map<int, double> wvotes;
  for (int r = 0; r < n_rows; ++r) {
    wvotes.clear();
    for (size_t k = 0; k < in.size(); ++k)
      wvotes[static_cast<int>(vs.get(in[k]).get(r, 0))] += weights[k];
    int best = 0;
    double best_w = -1.0;
    for (const auto& kv : wvotes)
      if (kv.second > best_w) {
        best_w = kv.second;
        best = kv.first;
      }
    result.at(r, 0) = static_cast<float>(best);
  }
  vs.put(out[0], std::move(result));
  return {};
}

// SoftVote: weighted average (or column-stack) of probability inputs.
//
// Column-stacking mode (all inputs are [n,1]): each input becomes one column of
// the [n, n_inputs] output.  This lets OVR and similar patterns pass per-class
// (n,1) columns directly without an explicit Concat node.
//
// Averaging mode (inputs are [n, n_classes]): standard weighted average.
//
// normalize_rows=true divides each row by its sum after assembly.
static omle::rt::Status op_soft_vote(ValueStore& vs, int n_rows,
                                     const std::vector<std::string>& in,
                                     const std::vector<std::string>& out,
                                     const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  std::vector<double> weights;
  auto it = attrs.find("weights");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats)
    for (double w : it->second.floats) weights.push_back(w);
  while (weights.size() < in.size()) weights.push_back(1.0);
  double wsum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (wsum == 0.0) wsum = 1.0;
  const bool normalize_rows = attr_b(attrs, "normalize_rows", false);

  // Column-stacking mode: several single-column inputs assembled column-wise.
  // Otherwise averaging mode: every input is [n_rows, n_classes].
  const bool stacking = in.size() > 1 && vs.get(in[0]).n_cols == 1;
  const int n_cols =
      stacking ? static_cast<int>(in.size()) : vs.get(in[0]).n_cols;

  // One accumulator in double, whatever the storage width. This is a sum over
  // every estimator in the ensemble followed by a division by a total formed
  // the same way, so accumulating at float32 width would round at each of
  // those steps; here the result is rounded once, on write-back.
  std::vector<double> acc(static_cast<std::size_t>(n_rows) * n_cols, 0.0);
  for (size_t k = 0; k < in.size(); ++k) {
    const Tensor& t = vs.get(in[k]);
    const double wk = weights[k];
    if (stacking) {
      if (static_cast<int>(k) >= n_cols) break;
      for (int r = 0; r < n_rows; ++r)
        acc[static_cast<std::size_t>(r) * n_cols + k] = wk * t.get(r, 0);
    } else {
      for (int r = 0; r < n_rows; ++r)
        for (int c = 0; c < n_cols; ++c)
          acc[static_cast<std::size_t>(r) * n_cols + c] += wk * t.get(r, c);
    }
  }

  for (int r = 0; r < n_rows; ++r) {
    double* row = acc.data() + static_cast<std::size_t>(r) * n_cols;
    double denom = wsum;
    if (normalize_rows) {
      denom = 0.0;
      for (int c = 0; c < n_cols; ++c) denom += row[c];
      if (!(denom > 0.0)) continue;
    }
    for (int c = 0; c < n_cols; ++c) row[c] /= denom;
  }

  OutBuf result(any_f64(vs, in), n_rows, n_cols);
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
      result.set(r, c, acc[static_cast<std::size_t>(r) * n_cols + c]);
  vs.put(out[0], result.take());

  // Emit y_pred when a second output slot is requested. The label is exact in
  // float32, but the argmax that picks it reads the probabilities at their
  // own width -- strict > keeps the first maximum, matching np.argmax.
  if (out.size() > 1) {
    const Tensor& prob = vs.get(out[0]);
    const int n_prob_cols = prob.n_cols;
    Tensor pred(n_rows, 1);
    float* pp = pred.f32_ptr();
    for (int r = 0; r < n_rows; ++r) {
      if (n_prob_cols == 1) {
        pp[r] = prob.get(r, 0) >= 0.5 ? 1.f : 0.f;
        continue;
      }
      int best = 0;
      double best_v = prob.get(r, 0);
      for (int c = 1; c < n_prob_cols; ++c) {
        const double v = prob.get(r, c);
        if (v > best_v) {
          best_v = v;
          best = c;
        }
      }
      pp[r] = static_cast<float>(best);
    }
    vs.put(out[1], std::move(pred));
  }
  return {};
}

// SAMMEVote: AdaBoost SAMME decision function + softmax.
// Inputs: n_estimators hard-prediction tensors, each [n_rows, 1].
// Attributes: weights (n_estimators floats), n_classes (int).
// Outputs[0]: y_prob — [n_rows, 1] for binary (P(class=1)), [n_rows, K] for
// multiclass. Outputs[1]: y_pred — [n_rows, 1] argmax.
static omle::rt::Status op_samme_vote(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};

  std::vector<double> weights;
  auto it = attrs.find("weights");
  if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats)
    for (double w : it->second.floats) weights.push_back(w);
  while (weights.size() < in.size()) weights.push_back(1.0);
  double wsum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (wsum == 0.0) wsum = 1.0;

  const int K = attr_i(attrs, "n_classes", 2);
  const double Km1 = static_cast<double>(K - 1 < 1 ? 1 : K - 1);

  // Compute SAMME decision function: decision[r,c] = sum_t
  // w_t*(pred_t==c?1:-1/(K-1)) / wsum
  std::vector<double> decision(static_cast<size_t>(n_rows) * K, 0.0);
  for (size_t t = 0; t < in.size(); ++t) {
    const Tensor& pred_t = vs.get(in[t]);
    double wt = weights[t] / wsum;
    double neg = -wt / Km1;
    for (int r = 0; r < n_rows; ++r) {
      int cls = static_cast<int>(pred_t.get(r, 0));
      double* dec = decision.data() + r * K;
      for (int c = 0; c < K; ++c) dec[c] += (c == cls) ? wt : neg;
    }
  }

  // Scale decision by 1/(K-1), then softmax to get probabilities.
  Tensor result = Tensor::dense(omle::rt::DataType::Float32, n_rows, K);
  float* rp = result.f32_ptr();

  Tensor pred_out(n_rows, 1);
  float* pp = pred_out.f32_ptr();

  for (int r = 0; r < n_rows; ++r) {
    double* dec = decision.data() + r * K;
    for (int c = 0; c < K; ++c) dec[c] /= Km1;

    // Softmax with numerical stability.
    double max_d = *std::max_element(dec, dec + K);
    double expsum = 0.0;
    for (int c = 0; c < K; ++c) expsum += std::exp(dec[c] - max_d);

    int best = 0;
    double best_p = -1.0;
    for (int c = 0; c < K; ++c) {
      double pc = std::exp(dec[c] - max_d) / expsum;
      rp[r * K + c] = static_cast<float>(pc);
      if (pc > best_p) {
        best_p = pc;
        best = c;
      }
    }
    pp[r] = static_cast<float>(best);
  }

  vs.put(out[0], std::move(result));
  if (out.size() > 1) vs.put(out[1], std::move(pred_out));
  return {};
}

// ---------------------------------------------------------------------------
// omle.feature operators
// ---------------------------------------------------------------------------

template <typename T>
// T is the storage width; the arithmetic is double either way.
//
// Training frameworks scale in float64 and only the ingest into a float32
// model rounds. Repeating the arithmetic in float32 rounds a second time and
// lands up to an ulp away -- invisible to a linear head, but a tree split
// threshold sits exactly on such a value often enough that one ulp low takes
// the other branch and returns the sibling leaf. So the parameters arrive as
// double and the row is widened, adjusted, and rounded once on write-back.
static void standard_scaler_impl(T* row, const double* mean,
                                 const double* scale, int nf) {
  for (int f = 0; f < nf; ++f) {
    double v = static_cast<double>(row[f]);
    if (mean) v -= mean[f];
    if (scale && scale[f] != 0.0) v /= scale[f];
    row[f] = static_cast<T>(v);
  }
}

static omle::rt::Status op_standard_scaler(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;

  // Prefer the undegraded parameters when the model declared them float64.
  const Tensor* mean_t =
      resolve_t64(vs, in, multi ? in.size() : 1, attrs, "mean");
  const Tensor* scale_t =
      resolve_t64(vs, in, multi ? in.size() : 2, attrs, "scale");

  Tensor m64, s64;
  const double* mp = view_w<double>(mean_t, m64);
  const double* sp = view_w<double>(scale_t, s64);

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r)
      standard_scaler_impl<T>(row_w<T>(src, r), mp, sp, nf);
  });
  vs.put(out[0], std::move(src));
  return {};
}

template <typename T>
static void minmax_scaler_impl(T* row, const double* data_min,
                               const double* data_scale, double feat_min,
                               double feat_max, int nf) {
  for (int f = 0; f < nf; ++f) {
    double v = static_cast<double>(row[f]);
    if (data_min) v -= data_min[f];
    if (data_scale && data_scale[f] != 0.0) v /= data_scale[f];
    row[f] = static_cast<T>(v * (feat_max - feat_min) + feat_min);
  }
}

static omle::rt::Status op_minmax_scaler(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;

  auto rt = [&](size_t slot, const char* aname) {
    return resolve_t(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* data_min = rt(1, "data_min");
  const Tensor* data_scale =
      rt(2, "data_max");  // stored as data_scale = data_max - data_min

  const double feat_min = attr_float(attrs, "feature_range_min", 0.0);
  const double feat_max = attr_float(attrs, "feature_range_max", 1.0);
  Tensor dm64, ds64;
  const double* dmin = view_w<double>(data_min, dm64);
  const double* dscale = view_w<double>(data_scale, ds64);

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r)
      minmax_scaler_impl<T>(row_w<T>(src, r), dmin, dscale, feat_min, feat_max,
                            nf);
  });
  vs.put(out[0], std::move(src));
  return {};
}

static omle::rt::Status op_robust_scaler(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;

  auto rt = [&](size_t slot, const char* aname) {
    return resolve_t(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* center_t = rt(1, "center");
  const Tensor* scale_t = rt(2, "scale");

  Tensor c64, s64;
  const double* cp = view_w<double>(center_t, c64);
  const double* sp = view_w<double>(scale_t, s64);

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r)
      standard_scaler_impl<T>(row_w<T>(src, r), cp, sp, nf);
  });
  vs.put(out[0], std::move(src));
  return {};
}

static omle::rt::Status op_max_abs_scaler(ValueStore& vs, int n_rows,
                                          const std::vector<std::string>& in,
                                          const std::vector<std::string>& out,
                                          const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;

  auto rt = [&](size_t slot, const char* aname) {
    return resolve_t(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* scale_t = rt(1, "scale");

  Tensor s64;
  const double* sp = view_w<double>(scale_t, s64);

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r) {
      T* row = row_w<T>(src, r);
      for (int f = 0; f < nf; ++f)
        if (sp && sp[f] != 0.0)
          row[f] = static_cast<T>(static_cast<double>(row[f]) / sp[f]);
    }
  });
  vs.put(out[0], std::move(src));
  return {};
}

static omle::rt::Status op_binarizer(ValueStore& vs, int n_rows,
                                     const std::vector<std::string>& in,
                                     const std::vector<std::string>& out,
                                     const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor* thr_t = resolve_t(vs, in, in.size(), attrs, "thresholds");

  // Multi-output: one output per input column, one threshold per column.
  if (out.size() > 1 && in.size() > 1) {
    for (int i = 0; i < (int)out.size() && i < (int)in.size(); ++i) {
      const Tensor& src = vs.get(in[i]);
      const double thr = (thr_t && i < (int)thr_t->numel())
                             ? thr_t->get(0, i)
                             : attr_float(attrs, "threshold", 0.0);
      Tensor result = with_width(wants_f64(src), [&](auto tag) {
        using T = decltype(tag);
        Tensor sview;
        const T* sp = view_w<T>(src, sview);
        Tensor res = dense_w<T>(n_rows, src.n_cols);
        T* dp = data_w<T>(res);
        const T thr_t2 = static_cast<T>(thr);
        for (int k = 0, n = static_cast<int>(src.numel()); k < n; ++k)
          dp[k] = sp[k] > thr_t2 ? T(1) : T(0);
        return res;
      });
      vs.put(out[i], std::move(result));
    }
    return {};
  }

  // Single output: use first threshold from tensor attr (or scalar
  // "threshold").
  Tensor src = (in.size() > 1) ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const double threshold =
      thr_t ? thr_t->get(0, 0) : attr_float(attrs, "threshold", 0.0);
  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    const T thr = static_cast<T>(threshold);
    T* bp = data_w<T>(src);
    for (int i = 0, n = static_cast<int>(src.numel()); i < n; ++i)
      bp[i] = (bp[i] > thr) ? T(1) : T(0);
  });
  vs.put(out[0], std::move(src));
  return {};
}

template <typename T>
static void normalizer_impl(T* row, int nf, const std::string& norm) {
  double n = 0.0;
  if (norm == "l1") {
    for (int f = 0; f < nf; ++f) n += std::fabs(static_cast<double>(row[f]));
  } else if (norm == "max") {
    for (int f = 0; f < nf; ++f)
      n = std::max(n, std::fabs(static_cast<double>(row[f])));
  } else {  // l2
    // The sum of squares is where a float32 norm loses most: every term is
    // squared before it is added, so the accumulator spans twice the dynamic
    // range of the data.
    for (int f = 0; f < nf; ++f) {
      const double v = static_cast<double>(row[f]);
      n += v * v;
    }
    n = std::sqrt(n);
  }
  if (n > 1e-12)
    for (int f = 0; f < nf; ++f)
      row[f] = static_cast<T>(static_cast<double>(row[f]) / n);
}

static omle::rt::Status op_normalizer(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  Tensor src = (in.size() > 1) ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;
  std::string norm = attr_s(attrs, "norm", "l2");

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r)
      normalizer_impl<T>(row_w<T>(src, r), nf, norm);
  });
  vs.put(out[0], std::move(src));
  return {};
}

template <typename T>
static void power_transformer_impl(T* row, const double* lambdas, int nf,
                                   const std::string& method) {
  for (int f = 0; f < nf; ++f) {
    double x = static_cast<double>(row[f]);
    const double lam = lambdas ? lambdas[f] : 0.0;
    double y;
    if (method == "box_cox") {
      x = std::max(x, 1e-10);
      y = (std::fabs(lam) < 1e-8) ? std::log(x)
                                  : (std::pow(x, lam) - 1.0) / lam;
    } else {  // yeo_johnson
      if (x >= 0.0) {
        y = (std::fabs(lam) < 1e-8) ? std::log1p(x)
                                    : (std::pow(x + 1.0, lam) - 1.0) / lam;
      } else {
        const double lam2 = 2.0 - lam;
        y = (std::fabs(lam2) < 1e-8) ? -std::log1p(-x)
                                     : -(std::pow(1.0 - x, lam2) - 1.0) / lam2;
      }
    }
    row[f] = static_cast<T>(y);
  }
}

static omle::rt::Status op_power_transformer(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;

  std::string method = attr_s(attrs, "method", "yeo_johnson");
  bool standardize = attr_b(attrs, "standardize", true);

  auto rt = [&](size_t slot, const char* aname) {
    return resolve_t(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* lambdas_t = rt(1, "lambdas");
  const Tensor* mean_t = rt(2, "mean");
  const Tensor* scale_t = rt(3, "scale");

  Tensor l64, m64, s64;
  const double* lam = view_w<double>(lambdas_t, l64);
  const double* mp = view_w<double>(mean_t, m64);
  const double* sp = view_w<double>(scale_t, s64);
  const bool post_scale = standardize && mean_t && scale_t;

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r)
      power_transformer_impl<T>(row_w<T>(src, r), lam, nf, method);
    if (post_scale)
      for (int r = 0; r < n_rows; ++r)
        standard_scaler_impl<T>(row_w<T>(src, r), mp, sp, nf);
  });
  vs.put(out[0], std::move(src));
  return {};
}

static omle::rt::Status op_quantile_transformer(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;

  // quantiles: [n_quantile_levels] - the percentile levels (0..1)
  // references: [n_quantile_levels, n_features] - feature values at each level
  auto rt = [&](size_t slot, const char* aname) {
    return resolve_t(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* quant_t = rt(1, "quantiles");
  const Tensor* ref_t = rt(2, "references");
  std::string output_dist = attr_s(attrs, "output_distribution", "uniform");

  if (!quant_t || !ref_t) {
    vs.put(out[0], std::move(src));
    return {};
  }

  const int nq = static_cast<int>(quant_t->numel());

  // The quantile levels and reference values are model parameters, so they
  // are read at full width and the interpolation runs in double; only the
  // write-back follows the tensor.
  Tensor q64, r64;
  const double* qlevs = view_w<double>(quant_t, q64);
  const double* refs_all = view_w<double>(ref_t, r64);
  const bool to_normal = (output_dist == "normal");

  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r) {
      T* row = row_w<T>(src, r);
      for (int f = 0; f < nf; ++f) {
        const double x = static_cast<double>(row[f]);
        const double* refs = refs_all + f * nq;
        double q;
        if (x <= refs[0]) {
          q = qlevs[0];
        } else if (x >= refs[nq - 1]) {
          q = qlevs[nq - 1];
        } else {
          int lo = 0, hi = nq - 1;
          while (lo + 1 < hi) {
            int mid = (lo + hi) / 2;
            (x < refs[mid] ? hi : lo) = mid;
          }
          const double t = (x - refs[lo]) / (refs[hi] - refs[lo] + 1e-12);
          q = qlevs[lo] + t * (qlevs[hi] - qlevs[lo]);
        }
        q = std::clamp(q, 0.0, 1.0);
        row[f] = static_cast<T>(to_normal ? probit<double>(q) : q);
      }
    }
  });
  vs.put(out[0], std::move(src));
  return {};
}

// B-spline basis evaluation using the Cox-de Boor recurrence.
// Accepts the FULL augmented knot vector (length n_t); n_basis = n_t - degree
// - 1.
template <typename T>
static void bspline_basis(const T* t, int n_t, int degree, T x, T* out_basis,
                          int n_basis) {
  const int n_intervals = n_t - 1;
  std::vector<T> d(n_intervals, T(0));
  T xlast = t[n_t - 1];
  for (int i = 0; i < n_intervals; ++i)
    d[i] = (x >= t[i] && (x < t[i + 1] || (x == xlast && i == n_intervals - 1)))
               ? T(1)
               : T(0);

  for (int p = 1; p <= degree; ++p) {
    for (int i = 0; i < n_intervals - p; ++i) {
      T left = (t[i + p] - t[i] > T(1e-10))
                   ? (x - t[i]) / (t[i + p] - t[i]) * d[i]
                   : T(0);
      T right = (t[i + p + 1] - t[i + 1] > T(1e-10))
                    ? (t[i + p + 1] - x) / (t[i + p + 1] - t[i + 1]) * d[i + 1]
                    : T(0);
      d[i] = left + right;
    }
  }
  for (int i = 0; i < n_basis; ++i)
    out_basis[i] = (i < n_intervals - degree) ? d[i] : T(0);
}

static omle::rt::Status op_spline_transformer(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  const int nf = src.n_cols;

  // knots: [n_aug, n_features] — augmented knot vector per feature (from bs.t).
  const Tensor* knots_t =
      resolve_t(vs, in, multi ? in.size() : 1, attrs, "knots");
  int degree = attr_i(attrs, "degree", 3);
  bool include_bias = attr_b(attrs, "include_bias", true);

  if (!knots_t) {
    vs.put(out[0], src);
    return {};
  }

  // n_t = number of augmented knots per feature; n_cols should equal nf.
  const int n_t = (knots_t->n_cols == nf)
                      ? knots_t->n_rows
                      : static_cast<int>(knots_t->numel() / nf);
  const int n_basis = n_t - degree - 1;  // standard: length - degree - 1
  const int n_out = include_bias ? n_basis : n_basis - 1;

  // The de Boor recurrence divides by differences of neighbouring knots,
  // which are small and nearly equal near a repeated knot, so it is evaluated
  // in double at either storage width and rounded once on write-back.
  Tensor kn64;
  const double* kp = view_w<double>(knots_t, kn64);
  const int start = include_bias ? 0 : 1;

  Tensor result = with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    Tensor sview;
    const T* sp = view_w<T>(src, sview);
    Tensor res = dense_w<T>(n_rows, nf * n_out);
    T* dp = data_w<T>(res);
    std::vector<double> basis(n_basis), kk(n_t);
    for (int r = 0; r < n_rows; ++r) {
      for (int f = 0; f < nf; ++f) {
        const double x = static_cast<double>(sp[r * nf + f]);
        for (int k = 0; k < n_t; ++k) kk[k] = kp[k * nf + f];
        bspline_basis<double>(kk.data(), n_t, degree, x, basis.data(), n_basis);
        for (int b = start; b < n_basis; ++b)
          dp[r * nf * n_out + f * n_out + (b - start)] =
              static_cast<T>(basis[b]);
      }
    }
    return res;
  });
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_polynomial_features(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  const int nf = src.n_cols;
  const bool is64 = (src.dtype == omle::rt::DataType::Float64);

  // If the powers matrix is stored (sklearn's t.powers_), use it for exact
  // feature order.
  const Tensor* powers_t =
      resolve_t(vs, in, multi ? in.size() : 1, attrs, "powers");
  if (powers_t && powers_t->n_rows > 0) {
    const int n_out = powers_t->n_rows;
    // The exponents do not depend on the row, so lift them out of the loop
    // rather than calling get() once per element per row.
    std::vector<int> pw(static_cast<std::size_t>(n_out) * nf);
    for (int c = 0; c < n_out; ++c)
      for (int f = 0; f < nf; ++f)
        pw[static_cast<std::size_t>(c) * nf + f] =
            static_cast<int>(powers_t->get(c, f));

    Tensor result = with_width(is64, [&](auto tag) {
      using T = decltype(tag);
      Tensor sview;
      const T* sp = view_w<T>(src, sview);
      Tensor res = dense_w<T>(n_rows, n_out);
      T* dp = data_w<T>(res);
      for (int r = 0; r < n_rows; ++r) {
        for (int c = 0; c < n_out; ++c) {
          double val = 1.0;
          for (int f = 0; f < nf; ++f) {
            const int e = pw[static_cast<std::size_t>(c) * nf + f];
            if (e > 0)
              val *= std::pow(static_cast<double>(sp[r * nf + f]),
                              static_cast<double>(e));
          }
          dp[r * n_out + c] = static_cast<T>(val);
        }
      }
      return res;
    });
    vs.put(out[0], std::move(result));
    return {};
  }

  // Fallback: enumerate all valid multi-indices (e0, e1, ...) with sum in
  // [min_degree, max_degree].
  int min_degree = attr_i(attrs, "min_degree", 1);
  int max_degree = attr_i(attrs, "max_degree", 2);
  bool interaction_only = attr_b(attrs, "interaction_only", false);
  bool include_bias = attr_b(attrs, "include_bias", true);

  std::vector<std::vector<int>> combos;
  std::function<void(int, int, std::vector<int>&)> gen =
      [&](int feat, int remain, std::vector<int>& cur) {
        if ((int)cur.size() == nf) {
          int total = std::accumulate(cur.begin(), cur.end(), 0);
          if (total >= min_degree) combos.push_back(cur);
          return;
        }
        for (int e = 0; e <= remain; ++e) {
          if (interaction_only && e > 1) break;
          cur.push_back(e);
          gen(feat + 1, remain - e, cur);
          cur.pop_back();
        }
      };

  if (include_bias) combos.push_back(std::vector<int>(nf, 0));
  std::vector<int> tmp;
  gen(0, max_degree, tmp);

  const int n_out = static_cast<int>(combos.size());
  Tensor result = with_width(is64, [&](auto tag) {
    using T = decltype(tag);
    Tensor sview;
    const T* sp = view_w<T>(src, sview);
    Tensor res = dense_w<T>(n_rows, n_out);
    T* dp = data_w<T>(res);
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < n_out; ++c) {
        double val = 1.0;
        for (int f = 0; f < nf; ++f)
          val *= std::pow(static_cast<double>(sp[r * nf + f]),
                          static_cast<double>(combos[c][f]));
        dp[r * n_out + c] = static_cast<T>(val);
      }
    }
    return res;
  });
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_bucketizer(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;

  // boundaries and boundary_offsets: tensor-name attr or inline floats attr.
  // Use slot=in.size() so we never accidentally pick up a data column as a
  // parameter.
  // A bucket boundary is compared for exact ordering against the data, so it
  // has to be the boundary the model declared -- not the float32 narrowing
  // the loader also stores. A value sitting on a boundary falls in the wrong
  // bucket when the boundary has drifted by an ulp.
  const Tensor* bnd_t = resolve_t64(vs, in, in.size(), attrs, "boundaries");
  const Tensor* off_t = resolve_t(vs, in, in.size(), attrs, "boundary_offsets");

  std::optional<Tensor> bnd_tmp;
  if (!bnd_t) {
    auto it = attrs.find("boundaries");
    if (it != attrs.end() && it->second.kind == AttrVal::Kind::Floats) {
      const auto& fv = it->second.floats;
      // The inline attribute already holds doubles; materialising it into a
      // float32 tensor threw that away before the first comparison.
      bnd_tmp.emplace(
          Tensor::from_doubles(1, static_cast<int>(fv.size()),
                               std::vector<double>(fv.begin(), fv.end())));
      bnd_t = &*bnd_tmp;
    }
  }
  if (!bnd_t) return {};

  auto bucket_for = [&](double x, int col_idx) -> double {
    int bstart = 0, bend = static_cast<int>(bnd_t->numel());
    if (off_t && off_t->numel() > static_cast<std::size_t>(col_idx + 1)) {
      bstart = static_cast<int>(off_t->get(0, col_idx));
      bend = static_cast<int>(off_t->get(0, col_idx + 1));
    }
    int bucket = 0;
    for (int k = bstart; k < bend; ++k) {
      if (bnd_t->get(0, k) <= x)
        ++bucket;
      else
        break;
    }
    return static_cast<double>(bucket);
  };

  if (multi) {
    for (int i = 0; i < (int)in.size() && i < (int)out.size(); ++i) {
      const Tensor& col_src = vs.get(in[i]);
      Tensor col_out(n_rows, 1);
      for (int r = 0; r < n_rows; ++r)
        col_out.at(r, 0) = static_cast<float>(bucket_for(col_src.get(r, 0), i));
      vs.put(out[i], std::move(col_out));
    }
    return {};
  }

  const Tensor& src = vs.get(in[0]);
  const int nf = src.n_cols;
  Tensor result(n_rows, nf);
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < nf; ++c)
      result.at(r, c) = static_cast<float>(bucket_for(src.get(r, c), c));
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_simple_imputer(ValueStore& vs, int n_rows,
                                          const std::vector<std::string>& in,
                                          const std::vector<std::string>& out,
                                          const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src = multi ? concat_inputs(vs, in, n_rows) : vs.get(in[0]);
  const int nf = src.n_cols;
  const Tensor* stats =
      resolve_t(vs, in, multi ? in.size() : 1, attrs, "fill_tensor");

  const double fill_value = attr_float(attrs, "fill_value", 0.0);
  Tensor s64;
  const double* sp = view_w<double>(stats, s64);
  with_width(wants_f64(src), [&](auto tag) {
    using T = decltype(tag);
    for (int r = 0; r < n_rows; ++r) {
      T* row = row_w<T>(src, r);
      for (int f = 0; f < nf; ++f)
        if (is_nan_safe(row[f]))
          row[f] = static_cast<T>(sp ? sp[f] : fill_value);
    }
  });
  // When there are multiple outputs (e.g. Spark Imputer with per-column
  // outputCols), split the wide imputed tensor back into one scalar column per
  // output.
  if (out.size() > 1) {
    const int n_out = static_cast<int>(out.size());
    with_width(wants_f64(src), [&](auto tag) {
      using T = decltype(tag);
      for (int i = 0; i < n_out && i < nf; ++i) {
        Tensor col = dense_w<T>(n_rows, 1);
        T* dst = data_w<T>(col);
        for (int r = 0; r < n_rows; ++r) dst[r] = row_w<T>(src, r)[i];
        vs.put(out[i], std::move(col));
      }
    });
  } else {
    vs.put(out[0], std::move(src));
  }
  return {};
}

static omle::rt::Status op_missing_indicator(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  const int nf = src.n_cols;

  // feature_indices: which columns to indicate; empty = all
  std::vector<int> indices;
  const Tensor* fi_t =
      resolve_t(vs, in, multi ? in.size() : 1, attrs, "feature_indices");
  if (fi_t)
    for (std::size_t i = 0, n = static_cast<int>(fi_t->numel()); i < n; ++i)
      indices.push_back(static_cast<int>(flat_get(*fi_t, static_cast<int>(i))));

  if (indices.empty())
    for (int f = 0; f < nf; ++f) indices.push_back(f);

  Tensor result(n_rows, static_cast<int>(indices.size()));
  for (int r = 0; r < n_rows; ++r)
    for (int j = 0; j < (int)indices.size(); ++j)
      result.at(r, j) = std::isnan(src.get(r, indices[j])) ? 1.0f : 0.0f;
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_label_encoder(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};

  // Per-column mode: multiple inputs, one per feature column.
  // Labels/offsets must come from attrs (in-slots are all data).
  // Single-tensor mode: in[0] is a wide string tensor; in[1]/in[2] may be
  // labels/offsets.
  const bool per_col = in.size() > 1;
  const size_t labels_slot =
      per_col ? in.size() : 1;  // out-of-range → attrs only
  const size_t offsets_slot = per_col ? in.size() : 2;

  const int nf = per_col ? static_cast<int>(in.size()) : vs.get(in[0]).n_cols;
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  // Encoded indices are small integers, exact either way, but the no-label
  // branch below degenerates to a pass-through -- and that has to carry the
  // input's width through untouched.
  OutBuf result(any_f64(vs, in), n_rows, nf);

  const Tensor* labels_t = resolve_t64(vs, in, labels_slot, attrs, "labels");
  const Tensor* offsets_t =
      resolve_t(vs, in, offsets_slot, attrs, "label_offsets");

  // Build per-feature label spans from the flat labels tensor + offsets.
  // offsets[f] and offsets[f+1] delimit labels for feature f.
  // Fall back to treating the whole labels tensor as a single-feature list.
  auto feat_begin = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f))
      return static_cast<int>(offsets_t->get(0, f));
    return 0;
  };
  auto feat_end = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f + 1))
      return static_cast<int>(offsets_t->get(0, f + 1));
    return labels_t ? static_cast<int>(labels_t->numel()) : 0;
  };

  const bool is_str = vs.get(in[0]).is_string();
  if (is_str) {
    if (!labels_t || !labels_t->is_string()) {
      for (int r = 0; r < n_rows; ++r)
        for (int f = 0; f < nf; ++f) result.set(r, f, kNaN);
    } else {
      for (int r = 0; r < n_rows; ++r) {
        for (int f = 0; f < nf; ++f) {
          const std::string& x =
              per_col ? vs.get(in[f]).str_at(r, 0) : vs.get(in[0]).str_at(r, f);
          double idx = kNaN;
          const int lo = feat_begin(f), hi = feat_end(f);
          for (int k = lo; k < hi; ++k) {
            if (labels_t->str_at(0, k) == x) {
              idx = static_cast<double>(k - lo);
              break;
            }
          }
          result.set(r, f, idx);
        }
      }
    }
  } else {
    // Numeric input: parse labels to numbers and match by value.
    if (!labels_t) {
      for (int r = 0; r < n_rows; ++r)
        for (int f = 0; f < nf; ++f)
          result.set(
              r, f,
              per_col ? vs.get(in[f]).get(r, 0) : vs.get(in[0]).get(r, f));
    } else {
      // The match below is exact equality, so probe and table must agree bit
      // for bit. Both were float: a label id past 2^24 shared its float32
      // representation with its neighbour, and reading the probe through at()
      // reinterpreted float64 storage as float so nothing matched at all.
      const int n_total = static_cast<int>(labels_t->numel());
      std::vector<double> label_vals(n_total, kNaN);
      if (labels_t->is_string()) {
        for (int k = 0; k < n_total; ++k) {
          try {
            label_vals[k] = std::stod(labels_t->str_at(0, k));
          } catch (...) {
            label_vals[k] = kNaN;
          }
        }
      } else {
        for (int k = 0; k < n_total; ++k) label_vals[k] = labels_t->get(0, k);
      }
      for (int r = 0; r < n_rows; ++r) {
        for (int f = 0; f < nf; ++f) {
          const double x =
              per_col ? vs.get(in[f]).get(r, 0) : vs.get(in[0]).get(r, f);
          double idx = kNaN;
          const int lo = feat_begin(f), hi = feat_end(f);
          for (int k = lo; k < hi; ++k) {
            const double lv = label_vals[k];
            const double match =
                std::isnan(lv) ? static_cast<double>(k - lo) : lv;
            if (match == x) {
              idx = static_cast<double>(k - lo);
              break;
            }
          }
          result.set(r, f, idx);
        }
      }
    }
  }
  if (out.size() > 1) {
    for (int i = 0; i < (int)out.size() && i < nf; ++i) {
      OutBuf col(result.is_f64(), n_rows, 1);
      for (int r = 0; r < n_rows; ++r) col.set(r, 0, result.get(r, i));
      vs.put(out[i], col.take());
    }
  } else {
    vs.put(out[0], result.take());
  }
  return {};
}

static omle::rt::Status op_one_hot_encoder(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};

  // Per-column mode: multiple inputs, one per feature column.
  // Categories/offsets must come from attrs (in-slots are all data).
  // Single-tensor mode: in[0] is a wide tensor; in[1]/in[2] may be
  // cats/offsets.
  const bool per_col = in.size() > 1;
  const size_t cats_slot =
      per_col ? in.size() : 1;  // out-of-range → attrs only
  const size_t off_slot = per_col ? in.size() : 2;
  const Tensor* cats_t = resolve_t(vs, in, cats_slot, attrs, "categories");
  const Tensor* offsets_t =
      resolve_t(vs, in, off_slot, attrs, "category_offsets");

  const int nf = per_col ? static_cast<int>(in.size()) : vs.get(in[0]).n_cols;

  // Wide mode keeps every column in one tensor, so one dtype governs all of
  // them. Per-column mode keeps one tensor per feature and they need not
  // agree — a ColumnTransformer can one-hot a string column beside an integer
  // one. Deciding once from in[0] and applying it to every feature called
  // str_at() on numeric tensors, which dereferences absent string storage.
  std::vector<char> feat_is_str(nf, 0);
  if (per_col) {
    for (int f = 0; f < nf; ++f)
      if (f < static_cast<int>(in.size()))
        feat_is_str[f] = vs.get(in[f]).is_string() ? 1 : 0;
  } else {
    std::fill(feat_is_str.begin(), feat_is_str.end(),
              vs.get(in[0]).is_string() ? 1 : 0);
  }

  // category_offsets comes from the model file and is not guaranteed to agree
  // with the length of the categories tensor, so every bound derived from it is
  // clamped: both branches below index the category block directly by k.
  const int n_cats = cats_t ? static_cast<int>(cats_t->numel()) : 0;
  auto feat_begin = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f))
      return std::min(static_cast<int>(offsets_t->get(0, f)), n_cats);
    return 0;
  };
  auto feat_end = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f + 1))
      return std::min(static_cast<int>(offsets_t->get(0, f + 1)), n_cats);
    return n_cats;
  };
  int total = 0;
  for (int f = 0; f < nf; ++f) total += feat_end(f) - feat_begin(f);

  Tensor result(n_rows, total, 0.0f);
  const bool cats_are_str = cats_t && cats_t->is_string();
  for (int r = 0; r < n_rows; ++r) {
    int out_off = 0;
    for (int f = 0; f < nf; ++f) {
      const int lo = feat_begin(f), hi = feat_end(f);
      if (feat_is_str[f]) {
        // String input: match each cell against its feature's categories.
        if (cats_are_str) {
          const std::string& x =
              per_col ? vs.get(in[f]).str_at(r, 0) : vs.get(in[0]).str_at(r, f);
          for (int k = lo; k < hi; ++k) {
            if (cats_t->str_at(0, k) == x) {
              result.at(r, out_off + (k - lo)) = 1.0f;
              break;
            }
          }
        }
      } else {
        // get() is the dtype-generic reader; at() is float32-only and would
        // reinterpret a float64 column's storage.
        const double x =
            per_col ? vs.get(in[f]).get(r, 0) : vs.get(in[0]).get(r, f);
        if (cats_t && !cats_are_str) {
          // Numeric input against numeric categories: match by value. The
          // operator spec defines this as "fitted one-hot category lookup",
          // so a value of 3 selects the category 3, not the fourth column.
          for (int k = lo; k < hi; ++k) {
            if (cats_t->get(0, k) == x) {
              result.at(r, out_off + (k - lo)) = 1.0f;
              break;
            }
          }
        } else {
          // Numeric input against string categories cannot be matched. Keep
          // the historical ordinal-index reading rather than silently
          // changing what existing models of this shape already compute.
          const int idx = static_cast<int>(x);
          if (idx >= 0 && idx < (hi - lo)) result.at(r, out_off + idx) = 1.0f;
        }
      }
      out_off += hi - lo;
    }
  }
  if (out.size() > 1) {
    int out_off = 0;
    for (int f = 0; f < (int)out.size() && f < nf; ++f) {
      const int lo = feat_begin(f), hi = feat_end(f);
      const int width = hi - lo;
      Tensor col(n_rows, width, 0.0f);
      for (int r = 0; r < n_rows; ++r)
        for (int k = 0; k < width; ++k)
          col.at(r, k) = result.at(r, out_off + k);
      out_off += width;
      vs.put(out[f], std::move(col));
    }
  } else {
    vs.put(out[0], std::move(result));
  }
  return {};
}

static omle::rt::Status op_ordinal_encoder(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};

  // Per-column mode: multiple inputs, one per feature column.
  // Categories/offsets must come from attrs (in-slots are all data).
  // Single-tensor mode: in[0] is a wide string tensor; in[1]/in[2] may be
  // cats/offsets.
  const bool per_col = in.size() > 1;
  const size_t cats_slot =
      per_col ? in.size() : 1;  // out-of-range → attrs only
  const size_t offsets_slot = per_col ? in.size() : 2;

  const int nf = per_col ? static_cast<int>(in.size()) : vs.get(in[0]).n_cols;
  const double kNaN = std::numeric_limits<double>::quiet_NaN();

  const Tensor* cats_t = resolve_t64(vs, in, cats_slot, attrs, "categories");
  const Tensor* offsets_t =
      resolve_t(vs, in, offsets_slot, attrs, "category_offsets");
  // Optional: float64 lookup values (one per category). When present, return
  // values[k-lo] instead of the ordinal index k-lo. Used by target/count
  // encoders.
  const Tensor* values_t =
      resolve_t64(vs, in, in.size(), attrs, "encoded_values");
  // Optional: per-feature fallback for unseen categories (one float per
  // feature).
  const Tensor* unknown_t =
      resolve_t64(vs, in, in.size(), attrs, "default_values");

  // Without encoded_values the output is an ordinal index and the operator is
  // a pass-through when there are no categories, so it follows the input's
  // width. With them it is a target or count encoder: the values are means,
  // genuinely continuous, and rounding them to float32 before a downstream
  // tree compares them against a float64 split threshold is how a row ends up
  // on the wrong side of the branch.
  const bool f64_out =
      any_f64(vs, in) ||
      (values_t && values_t->dtype == omle::rt::DataType::Float64);
  OutBuf result(f64_out, n_rows, nf);

  // category_offsets comes from the model file and is not guaranteed to agree
  // with the length of the categories tensor, so every bound derived from it is
  // clamped: both branches below index the category block directly by k.
  const int n_cats = cats_t ? static_cast<int>(cats_t->numel()) : 0;
  auto feat_begin = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f))
      return std::min(static_cast<int>(offsets_t->get(0, f)), n_cats);
    return 0;
  };
  auto feat_end = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f + 1))
      return std::min(static_cast<int>(offsets_t->get(0, f + 1)), n_cats);
    return n_cats;
  };
  // Return the encoded value for position pos within feature f's category
  // block.
  auto encode_val = [&](int f, int pos) -> double {
    if (values_t &&
        values_t->numel() > static_cast<std::size_t>(feat_begin(f) + pos))
      return values_t->get(0, feat_begin(f) + pos);
    return static_cast<double>(pos);
  };
  // Return the fallback (unseen category) value for feature f.
  auto unseen_val = [&](int f) -> double {
    if (unknown_t && unknown_t->numel() > static_cast<std::size_t>(f))
      return unknown_t->get(0, f);
    return kNaN;
  };

  const bool is_str = vs.get(in[0]).is_string();
  if (is_str) {
    if (!cats_t || !cats_t->is_string()) {
      for (int r = 0; r < n_rows; ++r)
        for (int f = 0; f < nf; ++f) result.set(r, f, kNaN);
    } else {
      for (int r = 0; r < n_rows; ++r) {
        for (int f = 0; f < nf; ++f) {
          const std::string& x =
              per_col ? vs.get(in[f]).str_at(r, 0) : vs.get(in[0]).str_at(r, f);
          const int lo = feat_begin(f), hi = feat_end(f);
          double ord = unseen_val(f);
          for (int k = lo; k < hi; ++k) {
            if (cats_t->str_at(0, k) == x) {
              ord = encode_val(f, k - lo);
              break;
            }
          }
          result.set(r, f, ord);
        }
      }
    }
  } else {
    if (!cats_t) {
      for (int r = 0; r < n_rows; ++r)
        for (int f = 0; f < nf; ++f)
          result.set(
              r, f,
              per_col ? vs.get(in[f]).get(r, 0) : vs.get(in[0]).get(r, f));
    } else {
      // Exact-equality lookup: probe and table have to agree bit for bit, so
      // neither may pass through float on the way here.
      const int n_total = n_cats;
      std::vector<double> cat_vals(n_total, kNaN);
      if (cats_t->is_string()) {
        for (int k = 0; k < n_total; ++k) {
          try {
            cat_vals[k] = std::stod(cats_t->str_at(0, k));
          } catch (...) {
            cat_vals[k] = kNaN;
          }
        }
      } else {
        for (int k = 0; k < n_total; ++k) cat_vals[k] = cats_t->get(0, k);
      }
      for (int r = 0; r < n_rows; ++r) {
        for (int f = 0; f < nf; ++f) {
          const double x =
              per_col ? vs.get(in[f]).get(r, 0) : vs.get(in[0]).get(r, f);
          const int lo = feat_begin(f), hi = feat_end(f);
          double ord = unseen_val(f);
          for (int k = lo; k < hi; ++k) {
            const double cv = cat_vals[k];
            const double match =
                std::isnan(cv) ? static_cast<double>(k - lo) : cv;
            if (match == x) {
              ord = encode_val(f, k - lo);
              break;
            }
          }
          result.set(r, f, ord);
        }
      }
    }
  }
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_label_binarizer(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  const Tensor* classes_t = resolve_t64(vs, in, 1, attrs, "classes");
  const float neg = static_cast<float>(attr_i(attrs, "neg_label", 0));
  const float pos = static_cast<float>(attr_i(attrs, "pos_label", 1));

  // The output is one of two label constants, exact in float32 whatever the
  // input width. Only the reads need widening -- at() on a float64 source
  // reinterprets its storage instead of converting it.
  if (!classes_t) {
    // Binary case: just threshold at 0.5
    Tensor result(n_rows, 1);
    for (int r = 0; r < n_rows; ++r)
      result.at(r, 0) = (src.get(r, 0) >= 0.5) ? pos : neg;
    vs.put(out[0], std::move(result));
    return {};
  }

  const int n_classes = static_cast<int>(classes_t->numel());
  Tensor result(n_rows, n_classes, neg);
  for (int r = 0; r < n_rows; ++r) {
    const double x = src.get(r, 0);
    for (int k = 0; k < n_classes; ++k)
      if (flat_get(*classes_t, k) == x) {
        result.at(r, k) = pos;
        break;
      }
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_multi_label_binarizer(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  const Tensor* classes_t = resolve_t64(vs, in, 1, attrs, "classes");
  if (!classes_t) {
    vs.put(out[0], src);
    return {};
  }

  const int n_classes = static_cast<int>(classes_t->numel());
  const int n_labels_in = src.n_cols;

  // Indicator output: 0 or 1, exact in float32. The class match is exact
  // equality though, so probe and table both read at full width.
  Tensor result(n_rows, n_classes, 0.0f);
  for (int r = 0; r < n_rows; ++r) {
    for (int l = 0; l < n_labels_in; ++l) {
      const double x = src.get(r, l);
      for (int k = 0; k < n_classes; ++k)
        if (flat_get(*classes_t, k) == x) {
          result.at(r, k) = 1.0f;
          break;
        }
    }
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_target_encoder(ValueStore& vs, int n_rows,
                                          const std::vector<std::string>& in,
                                          const std::vector<std::string>& out,
                                          const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};

  // Per-column mode: multiple inputs, one per feature column.
  const bool per_col = in.size() > 1;
  const int nf = per_col ? static_cast<int>(in.size()) : vs.get(in[0]).n_cols;

  // Attribute tensors. For per-column mode all data slots are taken, so
  // cats/enc/offsets/default come exclusively from attrs.
  const size_t data_slots = per_col ? in.size() : 1;
  const Tensor* cats_t = resolve_t64(vs, in, data_slots, attrs, "categories");
  // The encodings are target means -- genuinely continuous, and the whole
  // point of the operator. Read the undegraded constants so a float32 result
  // rounds once here rather than twice.
  const Tensor* enc_t =
      resolve_t64(vs, in, data_slots + 1, attrs, "encoded_values");
  const Tensor* offsets_t =
      resolve_t(vs, in, data_slots + 2, attrs, "category_offsets");
  const Tensor* default_t =
      resolve_t64(vs, in, data_slots + 3, attrs, "default_values");

  if (!cats_t || !enc_t) {
    vs.put(out[0], vs.get(in[0]));
    return {};
  }

  // Offset helpers: feat_begin/feat_end give the [lo, hi) index range in the
  // flat cats/enc arrays for feature f. Falls back to equal-split when no
  // category_offsets attribute is present (backward compat with old models).
  auto feat_begin = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f))
      return static_cast<int>(offsets_t->get(0, f));
    return f * (static_cast<int>(cats_t->numel()) / nf);
  };
  auto feat_end = [&](int f) -> int {
    if (offsets_t && offsets_t->numel() > static_cast<std::size_t>(f + 1))
      return static_cast<int>(offsets_t->get(0, f + 1));
    return (f + 1) * (static_cast<int>(cats_t->numel()) / nf);
  };

  // Target encodings are means: continuous, and typically float64 in the
  // source model. Keep that width when the table has it, for the same reason
  // as OrdinalEncoder's encoded_values -- these feed tree splits.
  const bool f64_out =
      any_f64(vs, in) || (enc_t->dtype == omle::rt::DataType::Float64);
  OutBuf result(f64_out, n_rows, nf);
  const bool is_str = vs.get(in[0]).is_string();

  if (is_str && cats_t->is_string()) {
    // String categorical input path (e.g. category_encoders).
    for (int r = 0; r < n_rows; ++r) {
      for (int f = 0; f < nf; ++f) {
        const std::string& x =
            per_col ? vs.get(in[f]).str_at(r, 0) : vs.get(in[0]).str_at(r, f);
        double y = default_t ? default_t->get(0, f) : 0.0;
        const int lo = feat_begin(f), hi = feat_end(f);
        for (int k = lo; k < hi; ++k) {
          if (cats_t->str_at(0, k) == x) {
            y = enc_t->get(0, k);
            break;
          }
        }
        result.set(r, f, y);
      }
    }
  } else {
    // Numeric (ordinal-code) input path (e.g. sklearn TargetEncoder).
    //
    // The lookup below is an exact equality against the category value, so
    // the table and the probe have to agree bit for bit. Holding the table in
    // float means a category id above 2^24 collides with its neighbour -- and
    // reading the probe through at() reinterpreted a float64 column as float
    // outright, so nothing matched and every row fell back to the default.
    const int n_total = static_cast<int>(cats_t->numel());
    std::vector<double> cat_vals(n_total);
    if (cats_t->is_string()) {
      for (int k = 0; k < n_total; ++k) {
        try {
          cat_vals[k] = std::stod(cats_t->str_at(0, k));
        } catch (...) {
          cat_vals[k] = std::numeric_limits<double>::quiet_NaN();
        }
      }
    } else {
      for (int k = 0; k < n_total; ++k) cat_vals[k] = cats_t->get(0, k);
    }
    for (int r = 0; r < n_rows; ++r) {
      for (int f = 0; f < nf; ++f) {
        const double x =
            per_col ? vs.get(in[f]).get(r, 0) : vs.get(in[0]).get(r, f);
        double y = default_t ? default_t->get(0, f) : 0.0;
        const int lo = feat_begin(f), hi = feat_end(f);
        for (int k = lo; k < hi; ++k) {
          if (cat_vals[k] == x) {
            y = enc_t->get(0, k);
            break;
          }
        }
        result.set(r, f, y);
      }
    }
  }
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_discretize(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  // Bin edges decide which side of a boundary a value lands on, so they are
  // resolved at their declared width. Widening the loader's float32 twin back
  // to double, as this used to, restores the type but not the value.
  const Tensor* bin_left_t = resolve_t64(vs, in, 1, attrs, "bin_left");
  const Tensor* bin_right_t = resolve_t64(vs, in, 2, attrs, "bin_right");
  const Tensor* bin_vals_t = resolve_t64(vs, in, 3, attrs, "bin_values");
  if (!bin_left_t || !bin_right_t || !bin_vals_t) return {};

  const int K = static_cast<int>(bin_left_t->numel());

  std::vector<bool> lc_flags(K, true), rc_flags(K, true);
  auto lc_it = attrs.find("left_closed");
  if (lc_it != attrs.end() && lc_it->second.kind == AttrVal::Kind::Ints)
    for (int k = 0; k < K && k < (int)lc_it->second.ints.size(); ++k)
      lc_flags[k] = (lc_it->second.ints[k] != 0);
  auto rc_it = attrs.find("right_closed");
  if (rc_it != attrs.end() && rc_it->second.kind == AttrVal::Kind::Ints)
    for (int k = 0; k < K && k < (int)rc_it->second.ints.size(); ++k)
      rc_flags[k] = (rc_it->second.ints[k] != 0);

  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  float def_val = kNaN, miss_val = kNaN;
  bool has_def = false, has_miss = false;
  {
    auto it = attrs.find("default_value");
    if (it != attrs.end()) {
      if (it->second.kind == AttrVal::Kind::Float) {
        def_val = static_cast<float>(it->second.f);
        has_def = true;
      } else if (it->second.kind == AttrVal::Kind::Int) {
        def_val = static_cast<float>(it->second.i);
        has_def = true;
      }
    }
  }
  {
    auto it = attrs.find("map_missing_to");
    if (it != attrs.end()) {
      if (it->second.kind == AttrVal::Kind::Float) {
        miss_val = static_cast<float>(it->second.f);
        has_miss = true;
      } else if (it->second.kind == AttrVal::Kind::Int) {
        miss_val = static_cast<float>(it->second.i);
        has_miss = true;
      }
    }
  }

  // Use double precision for boundary comparisons to avoid precision loss
  // regardless of src dtype; output is a bin value (stays float32).
  Tensor bl64 = bin_left_t->to_float64();
  Tensor br64 = bin_right_t->to_float64();
  Tensor bv64 = bin_vals_t->to_float64();
  const double* lp = bl64.f64_ptr();
  const double* rp = br64.f64_ptr();
  const double* vp = bv64.f64_ptr();

  Tensor result(n_rows, src.n_cols);
  for (int r = 0; r < n_rows; ++r) {
    for (int c = 0; c < src.n_cols; ++c) {
      double x = src.get(r, c);
      if (std::isnan(x)) {
        result.at(r, c) = has_miss ? miss_val : kNaN;
        continue;
      }
      float out_val = has_def ? def_val : kNaN;
      for (int k = 0; k < K; ++k) {
        bool in_left = lc_flags[k] ? (x >= lp[k]) : (x > lp[k]);
        bool in_right = rc_flags[k] ? (x <= rp[k]) : (x < rp[k]);
        if (in_left && in_right) {
          out_val = static_cast<float>(vp[k]);
          break;
        }
      }
      result.at(r, c) = out_val;
    }
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_norm_continuous(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);
  const int nf = src.n_cols;

  const Tensor* orig_t = resolve_t(vs, in, 1, attrs, "orig_points");
  const Tensor* norm_t = resolve_t(vs, in, 2, attrs, "norm_points");
  const Tensor* off_t = resolve_t(vs, in, 3, attrs, "point_offsets");

  if (!orig_t || !norm_t) return {};

  // Number of output features is determined by point_offsets (n+1 values → n
  // features). Falls back to input column count when point_offsets is absent.
  const int nf_out = (off_t && static_cast<int>(off_t->numel()) >= 2)
                         ? static_cast<int>(off_t->numel()) - 1
                         : nf;

  const std::string treatment =
      attr_s(attrs, "outlier_treatment", "as_extreme_values");
  const bool clamp_outliers = (treatment == "as_extreme_values");
  const double kNaN = std::numeric_limits<double>::quiet_NaN();

  // The interpolation below already runs in double -- every operand comes
  // through get(). Only the write-back was float32, which threw that away
  // for a float64 input; OutBuf keeps the input's width instead.
  OutBuf result(wants_f64(src), n_rows, nf_out);
  for (int f = 0; f < nf_out; ++f) {
    int bstart = 0, bend = static_cast<int>(orig_t->numel());
    if (off_t && static_cast<int>(off_t->numel()) > f)
      bstart = static_cast<int>(off_t->get(0, f));
    if (off_t && static_cast<int>(off_t->numel()) > f + 1)
      bend = static_cast<int>(off_t->get(0, f + 1));
    const int n_pts = bend - bstart;

    for (int r = 0; r < n_rows; ++r) {
      const double x = src.get(r, f);
      double y;
      if (n_pts == 0) {
        y = x;
      } else if (n_pts == 1) {
        y = norm_t->get(0, bstart);
      } else {
        const double x0 = orig_t->get(0, bstart);
        const double xN = orig_t->get(0, bend - 1);
        if (x <= x0) {
          y = clamp_outliers ? norm_t->get(0, bstart) : kNaN;
        } else if (x >= xN) {
          y = clamp_outliers ? norm_t->get(0, bend - 1) : kNaN;
        } else {
          // Binary search for bracket [lo, lo+1]
          int lo = bstart, hi = bend - 1;
          while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (orig_t->get(0, mid) <= x)
              lo = mid;
            else
              hi = mid;
          }
          const double x_lo = orig_t->get(0, lo), x_hi = orig_t->get(0, hi);
          const double y_lo = norm_t->get(0, lo), y_hi = norm_t->get(0, hi);
          const double t = (x - x_lo) / (x_hi - x_lo);
          y = y_lo + t * (y_hi - y_lo);
        }
      }
      result.set(r, f, y);
    }
  }
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_norm_discrete(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  const double target_val = attr_float(attrs, "value", 0.0);
  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  double miss_val = kNaN;
  bool has_miss = false;
  {
    auto it = attrs.find("map_missing_to");
    if (it != attrs.end()) {
      if (it->second.kind == AttrVal::Kind::Float) {
        miss_val = it->second.f;
        has_miss = true;
      } else if (it->second.kind == AttrVal::Kind::Int) {
        miss_val = static_cast<double>(it->second.i);
        has_miss = true;
      }
    }
  }

  // Indicator output, but the test is exact equality against the target
  // value. Reading the source through f32_ptr() reinterpreted float64
  // storage, and narrowing the target to float made the comparison agree
  // only by luck.
  const int nc = src.n_cols > 0 ? src.n_cols : 1;
  OutBuf result(wants_f64(src), n_rows, src.n_cols);
  for (int i = 0, n = static_cast<int>(src.numel()); i < n; ++i) {
    const int r = i / nc, c = i % nc;
    const double x = src.get(r, c);
    if (is_nan_safe(x))
      result.set(r, c, has_miss ? miss_val : kNaN);
    else
      result.set(r, c, (x == target_val) ? 1.0 : 0.0);
  }
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_map_values(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);

  const Tensor* keys_t = resolve_t64(vs, in, 1, attrs, "keys");
  const Tensor* values_t = resolve_t64(vs, in, 2, attrs, "values");
  if (!keys_t || !values_t) return {};

  const int K = static_cast<int>(keys_t->numel());

  const double kNaN = std::numeric_limits<double>::quiet_NaN();
  double def_val = kNaN, miss_val = kNaN;
  bool has_def = false, has_miss = false;
  {
    auto it = attrs.find("default_value");
    if (it != attrs.end()) {
      if (it->second.kind == AttrVal::Kind::Float) {
        def_val = it->second.f;
        has_def = true;
      } else if (it->second.kind == AttrVal::Kind::Int) {
        def_val = static_cast<double>(it->second.i);
        has_def = true;
      }
    }
  }
  {
    auto it = attrs.find("map_missing_to");
    if (it != attrs.end()) {
      if (it->second.kind == AttrVal::Kind::Float) {
        miss_val = it->second.f;
        has_miss = true;
      } else if (it->second.kind == AttrVal::Kind::Int) {
        miss_val = static_cast<double>(it->second.i);
        has_miss = true;
      }
    }
  }

  // The key match is exact equality. Both probe and key table used to be
  // float -- the probe read through f32_ptr(), which reinterprets float64
  // storage rather than converting it, so a float64 column matched nothing
  // and every row took the default.
  const int nc = src.n_cols > 0 ? src.n_cols : 1;
  OutBuf result(wants_f64(src), n_rows, src.n_cols);
  for (int i = 0, n = static_cast<int>(src.numel()); i < n; ++i) {
    const int r = i / nc, c = i % nc;
    const double x = src.get(r, c);
    if (is_nan_safe(x)) {
      result.set(r, c, has_miss ? miss_val : kNaN);
      continue;
    }
    double out_val = has_def ? def_val : kNaN;
    for (int k = 0; k < K; ++k) {
      if (x == flat_get(*keys_t, k)) {
        out_val = flat_get(*values_t, k);
        break;
      }
    }
    result.set(r, c, out_val);
  }
  vs.put(out[0], result.take());
  return {};
}

// Linear projection helpers used by TruncatedSVD, FastICA, FactorAnalysis, etc.
// components: [n_components, n_features], result: [n_rows, n_components]
// Y = (X - mean) @ comp.T, accumulated in double.
//
// Every operand is read through the dtype-generic get(), so a float64 input --
// or an undegraded float64 components matrix -- contributes its real value
// instead of being reinterpreted as float. The six decomposition operators
// that delegate here (PCA, SparsePCA, TruncatedSVD, FactorAnalysis, FastICA,
// KernelPCA) look dtype-neutral in their own bodies; this helper is where
// their precision is actually decided.
static Tensor linear_project(const Tensor& src, const Tensor& comp,
                             const Tensor* mean_t, int n_rows) {
  const int nf = comp.n_cols;
  const int nc = comp.n_rows;
  OutBuf result(wants_f64(src), n_rows, nc);
  for (int r = 0; r < n_rows; ++r) {
    for (int c = 0; c < nc; ++c) {
      double s = 0.0;
      for (int f = 0; f < nf; ++f) {
        const double x = src.get(r, f) - flat_get_or(mean_t, f, 0.0);
        s += x * comp.get(c, f);
      }
      result.set(r, c, s);
    }
  }
  return result.take();
}

static omle::rt::Status op_truncated_svd(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  const Tensor* comp_t =
      resolve_t64(vs, in, multi ? in.size() : 1, attrs, "components");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }
  vs.put(out[0], linear_project(src, *comp_t, nullptr, n_rows));
  return {};
}

static omle::rt::Status op_fast_ica(ValueStore& vs, int n_rows,
                                    const std::vector<std::string>& in,
                                    const std::vector<std::string>& out,
                                    const AttributeMap& attrs) {
  // Y = (X - mean) @ whitening.T @ components.T
  // Equivalently: Y = ((X - mean) @ whitening.T) @ components.T
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  auto rt64 = [&](size_t slot, const char* aname) {
    return resolve_t64(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* mean_t = rt64(1, "mean");
  const Tensor* white_t = rt64(2, "whitening");
  const Tensor* comp_t = rt64(3, "components");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }

  // Intermediate: center and apply whitening (if available).
  Tensor intermediate;
  if (white_t) {
    // white_t: [n_features, n_features], apply as X_w = (X - mean) @ white_t.T
    const int nf = white_t->n_rows;
    OutBuf mid(wants_f64(src), n_rows, nf);
    for (int r = 0; r < n_rows; ++r) {
      for (int c = 0; c < nf; ++c) {
        double s = 0.0;
        for (int f = 0; f < src.n_cols; ++f) {
          const double x = src.get(r, f) - flat_get_or(mean_t, f, 0.0);
          s += x * white_t->get(c, f);  // white_t is [nf_out, nf_in]
        }
        mid.set(r, c, s);
      }
    }
    intermediate = mid.take();
  } else {
    intermediate = src;
  }
  vs.put(out[0], linear_project(intermediate, *comp_t,
                                (white_t ? nullptr : mean_t), n_rows));
  return {};
}

static omle::rt::Status op_factor_analysis(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  // Y = (X - mean) @ components.T  (components = [n_components, n_features])
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  auto rt64 = [&](size_t slot, const char* aname) {
    return resolve_t64(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* mean_t = rt64(1, "mean");
  const Tensor* comp_t = rt64(2, "components");
  const Tensor* noise_t = rt64(3, "noise_variance");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }

  // FactorAnalysis.transform is not a projection onto components_. sklearn
  // computes the posterior mean of the latent variables:
  //
  //   Wpsi  = components_ / noise_variance_
  //   cov_z = inv(I + Wpsi @ components_.T)
  //   Z     = (X - mean_) @ Wpsi.T @ cov_z
  //
  // This used to project straight onto components_.T, ignoring the stored
  // noise_variance entirely, which is wrong by a wide margin (~0.27 on
  // standard-normal data) at any precision. Fold the two parameter matrices
  // into one effective [k, n_features] matrix here -- it depends only on the
  // model, not the data -- and hand that to the shared projection.
  if (noise_t && noise_t->numel() > 0) {
    const int k = comp_t->n_rows;
    const int nf = comp_t->n_cols;

    std::vector<double> wpsi(static_cast<std::size_t>(k) * nf);
    for (int i = 0; i < k; ++i)
      for (int f = 0; f < nf; ++f) {
        const double nv = flat_get(*noise_t, f);
        wpsi[static_cast<std::size_t>(i) * nf + f] =
            (nv != 0.0) ? comp_t->get(i, f) / nv : 0.0;
      }

    // M = I + Wpsi @ components.T, then invert in place by Gauss-Jordan with
    // partial pivoting. k is the number of latent factors -- single digits in
    // practice -- so an explicit inverse is cheaper than a factorisation.
    const std::size_t kk = static_cast<std::size_t>(k);
    std::vector<double> m(kk * kk, 0.0), inv(kk * kk, 0.0);
    for (int i = 0; i < k; ++i) {
      inv[static_cast<std::size_t>(i) * kk + i] = 1.0;
      for (int j = 0; j < k; ++j) {
        double s = (i == j) ? 1.0 : 0.0;
        for (int f = 0; f < nf; ++f)
          s += wpsi[static_cast<std::size_t>(i) * nf + f] * comp_t->get(j, f);
        m[static_cast<std::size_t>(i) * kk + j] = s;
      }
    }
    bool ok = true;
    for (int c = 0; c < k && ok; ++c) {
      int piv = c;
      for (int r = c + 1; r < k; ++r)
        if (std::fabs(m[static_cast<std::size_t>(r) * kk + c]) >
            std::fabs(m[static_cast<std::size_t>(piv) * kk + c]))
          piv = r;
      if (std::fabs(m[static_cast<std::size_t>(piv) * kk + c]) < 1e-300) {
        ok = false;
        break;
      }
      if (piv != c)
        for (std::size_t j = 0; j < kk; ++j) {
          std::swap(m[static_cast<std::size_t>(c) * kk + j],
                    m[static_cast<std::size_t>(piv) * kk + j]);
          std::swap(inv[static_cast<std::size_t>(c) * kk + j],
                    inv[static_cast<std::size_t>(piv) * kk + j]);
        }
      const double d = m[static_cast<std::size_t>(c) * kk + c];
      for (std::size_t j = 0; j < kk; ++j) {
        m[static_cast<std::size_t>(c) * kk + j] /= d;
        inv[static_cast<std::size_t>(c) * kk + j] /= d;
      }
      for (int r = 0; r < k; ++r) {
        if (r == c) continue;
        const double f2 = m[static_cast<std::size_t>(r) * kk + c];
        if (f2 == 0.0) continue;
        for (std::size_t j = 0; j < kk; ++j) {
          m[static_cast<std::size_t>(r) * kk + j] -=
              f2 * m[static_cast<std::size_t>(c) * kk + j];
          inv[static_cast<std::size_t>(r) * kk + j] -=
              f2 * inv[static_cast<std::size_t>(c) * kk + j];
        }
      }
    }

    if (ok) {
      // effective[i][f] = sum_j cov_z[i][j] * Wpsi[j][f]  (cov_z is symmetric,
      // so this is already the transpose the projection wants).
      std::vector<double> eff(static_cast<std::size_t>(k) * nf, 0.0);
      for (int i = 0; i < k; ++i)
        for (int f = 0; f < nf; ++f) {
          double s = 0.0;
          for (int j = 0; j < k; ++j)
            s += inv[static_cast<std::size_t>(i) * kk + j] *
                 wpsi[static_cast<std::size_t>(j) * nf + f];
          eff[static_cast<std::size_t>(i) * nf + f] = s;
        }
      const Tensor eff_t = Tensor::from_doubles(k, nf, std::move(eff));
      vs.put(out[0], linear_project(src, eff_t, mean_t, n_rows));
      return {};
    }
  }

  vs.put(out[0], linear_project(src, *comp_t, mean_t, n_rows));
  return {};
}

static omle::rt::Status op_pca(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap& attrs) {
  // Y = (X - mean) @ components.T  (components: [k, n_features])
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  auto rt64 = [&](size_t slot, const char* aname) {
    return resolve_t64(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* mean_t = rt64(1, "mean");
  const Tensor* comp_t = rt64(2, "components");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }
  vs.put(out[0], linear_project(src, *comp_t, mean_t, n_rows));
  return {};
}

static omle::rt::Status op_sparse_pca(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  // At inference: linear projection Y = (X - mean) @ components.T
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  auto rt64 = [&](size_t slot, const char* aname) {
    return resolve_t64(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* mean_t = rt64(1, "mean");
  const Tensor* comp_t = rt64(2, "components");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }
  vs.put(out[0], linear_project(src, *comp_t, mean_t, n_rows));
  return {};
}

static omle::rt::Status op_nmf(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap& attrs) {
  // At inference: linear projection Y = max(0, X @ components.T)
  // components (H): [n_components, n_features]
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  const Tensor* comp_t =
      resolve_t64(vs, in, multi ? in.size() : 1, attrs, "components");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }
  Tensor result = linear_project(src, *comp_t, nullptr, n_rows);
  // NMF coefficients are non-negative. linear_project() hands back float64
  // when the input carried it, so clamp at whichever width came back.
  with_width(wants_f64(result), [&](auto tag) {
    using T = decltype(tag);
    T* p = data_w<T>(result);
    for (int i = 0, n = static_cast<int>(result.numel()); i < n; ++i)
      p[i] = std::max(T(0), p[i]);
  });
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_lda(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap& attrs) {
  // Approximate LDA transform: Y = softmax(X @ components.T)
  // components: [n_topics, n_features] (word-topic matrix)
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor src_tmp;
  if (multi) src_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& src = multi ? src_tmp : vs.get(in[0]);
  const Tensor* comp_t =
      resolve_t64(vs, in, multi ? in.size() : 1, attrs, "components");
  if (!comp_t) {
    vs.put(out[0], src);
    return {};
  }

  Tensor result = linear_project(src, *comp_t, nullptr, n_rows);
  // Apply softmax row-wise to get topic distributions. The exponential is
  // taken in double whatever the result's width -- exp() of a float32 arg
  // already rounds the input, and the row sum that follows divides every
  // entry, so an early rounding spreads across the whole distribution.
  const int nc = result.n_cols;
  std::vector<double> buf(static_cast<std::size_t>(nc));
  for (int r = 0; r < n_rows; ++r) {
    double max_v = -std::numeric_limits<double>::infinity();
    for (int c = 0; c < nc; ++c) {
      buf[c] = result.get(r, c);
      if (buf[c] > max_v) max_v = buf[c];
    }
    double sum = 0.0;
    for (int c = 0; c < nc; ++c) {
      buf[c] = std::exp(buf[c] - max_v);
      sum += buf[c];
    }
    if (sum > 1e-12)
      for (int c = 0; c < nc; ++c) buf[c] /= sum;
    for (int c = 0; c < nc; ++c) set_at(result, r, c, buf[c]);
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_kernel_pca(ValueStore& vs, int n_rows,
                                      const std::vector<std::string>& in,
                                      const std::vector<std::string>& out,
                                      const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor X_tmp;
  if (multi) X_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& X = multi ? X_tmp : vs.get(in[0]);
  auto rt64 = [&](size_t slot, const char* aname) {
    return resolve_t64(vs, in, multi ? in.size() : slot, attrs, aname);
  };
  const Tensor* Xt_t = rt64(1, "fit_samples");     // [n_train, n_features]
  const Tensor* A_t = rt64(2, "dual_components");  // [n_components, n_train]
  // Kernel centering means
  const Tensor* kr_t = rt64(3, "kernel_center_row_mean");  // [n_train]
  const Tensor* ka_t = rt64(4, "kernel_center_all_mean");  // scalar

  if (!Xt_t || !A_t) {
    vs.put(out[0], X);
    return {};
  }

  std::string kernel = attr_s(attrs, "kernel", "rbf");
  const double gamma =
      attr_float(attrs, "gamma", 1.0 / static_cast<double>(X.n_cols));
  int degree = attr_i(attrs, "degree", 3);
  const double coef0 = attr_float(attrs, "coef0", 1.0);

  const int n_train = Xt_t->n_rows;
  const int nf = Xt_t->n_cols;

  // The kernel matrix is a pure intermediate -- no caller ever sees it -- so
  // it is held in double unconditionally. It is also where the precision is
  // won or lost: an RBF kernel exponentiates a sum of squared differences,
  // and centering then subtracts two means of comparable size from every
  // entry, so float32 rounding here shows up amplified in the projection.
  std::vector<double> K(static_cast<std::size_t>(n_rows) * n_train, 0.0);
  auto Kat = [&](int i, int j) -> double& {
    return K[static_cast<std::size_t>(i) * n_train + j];
  };
  for (int i = 0; i < n_rows; ++i) {
    for (int j = 0; j < n_train; ++j) {
      double val = 0.0;
      if (kernel == "linear") {
        for (int f = 0; f < nf; ++f) val += X.get(i, f) * Xt_t->get(j, f);
      } else if (kernel == "poly" || kernel == "polynomial") {
        double dot = 0.0;
        for (int f = 0; f < nf; ++f) dot += X.get(i, f) * Xt_t->get(j, f);
        val = std::pow(gamma * dot + coef0, static_cast<double>(degree));
      } else if (kernel == "sigmoid") {
        double dot = 0.0;
        for (int f = 0; f < nf; ++f) dot += X.get(i, f) * Xt_t->get(j, f);
        val = std::tanh(gamma * dot + coef0);
      } else {  // rbf (default)
        double sq = 0.0;
        for (int f = 0; f < nf; ++f) {
          const double d = X.get(i, f) - Xt_t->get(j, f);
          sq += d * d;
        }
        val = std::exp(-gamma * sq);
      }
      Kat(i, j) = val;
    }
  }

  // Center K: K_c = K - row_mean - col_mean + all_mean
  if (kr_t) {
    const double all_mean = ka_t ? flat_get(*ka_t, 0) : 0.0;
    for (int i = 0; i < n_rows; ++i) {
      double row_mean = 0.0;
      for (int j = 0; j < n_train; ++j) row_mean += Kat(i, j);
      row_mean /= n_train;
      for (int j = 0; j < n_train; ++j)
        Kat(i, j) += -row_mean - flat_get_or(kr_t, j, 0.0) + all_mean;
    }
  }

  // Project: Y = K @ A.T  where A: [n_components, n_train]
  const int nc = A_t->n_rows;
  OutBuf result(wants_f64(X), n_rows, nc);
  for (int i = 0; i < n_rows; ++i)
    for (int c = 0; c < nc; ++c) {
      double s = 0.0;
      for (int j = 0; j < n_train; ++j) s += Kat(i, j) * A_t->get(c, j);
      result.set(i, c, s);
    }
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_knn_imputer(ValueStore& vs, int n_rows,
                                       const std::vector<std::string>& in,
                                       const std::vector<std::string>& out,
                                       const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const bool multi = in.size() > 1;
  Tensor raw_query_tmp;
  if (multi) raw_query_tmp = concat_inputs(vs, in, n_rows);
  const Tensor& raw_query = multi ? raw_query_tmp : vs.get(in[0]);

  const std::string feat_name = attr_tensor(attrs, "train_features");
  if (feat_name.empty() || !vs.has(feat_name))
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.feature/KNNImputer: train_features is required"};

  const int k = attr_i(attrs, "n_neighbors", 5);
  const std::string weights = attr_s(attrs, "weights", "uniform");
  const std::string metric = attr_s(attrs, "metric", "nan_euclidean");
  const bool use_distance = (weights == "distance");

  const Tensor& train_x = vs.get(feat_name);
  const int M = train_x.n_rows;
  const int D = train_x.n_cols;
  const int kk = std::min(k, M);

  Tensor result = raw_query;  // copy; we'll fill in missing values
  std::vector<std::pair<double, int>> dist_buf(M);

  for (int s = 0; s < n_rows; ++s) {
    // Identify missing features in this row (works for both f32 and f64).
    std::vector<int> missing_cols;
    for (int f = 0; f < D; ++f) {
      double v = result.get(s, f);
      if (std::isnan(v)) missing_cols.push_back(f);
    }
    if (missing_cols.empty()) continue;

    // Compute distance ignoring missing features in query or training.
    for (int i = 0; i < M; ++i) {
      double sq = 0.0;
      int cnt = 0;
      for (int f = 0; f < D; ++f) {
        double qv = result.get(s, f), tv = train_x.get(i, f);
        if (std::isnan(qv) || std::isnan(tv)) continue;
        double d = qv - tv;
        sq += d * d;
        ++cnt;
      }
      double dist = (cnt > 0) ? std::sqrt(sq * D / cnt)
                              : std::numeric_limits<double>::infinity();
      dist_buf[i] = {dist, i};
    }
    std::partial_sort(
        dist_buf.begin(), dist_buf.begin() + kk, dist_buf.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
          return a.first < b.first;
        });

    // Impute each missing feature.
    for (int f : missing_cols) {
      double num = 0.0, denom = 0.0;
      for (int j = 0; j < kk; ++j) {
        double tv = train_x.get(dist_buf[j].second, f);
        if (std::isnan(tv)) continue;
        double w =
            use_distance
                ? ((dist_buf[j].first > 1e-12) ? 1.0 / dist_buf[j].first : 1e12)
                : 1.0;
        num += w * tv;
        denom += w;
      }
      if (denom > 0.0) set_at(result, s, f, num / denom);
    }
  }
  vs.put(out[0], std::move(result));
  return {};
}

// ---------------------------------------------------------------------------
// omle.ml operators
// ---------------------------------------------------------------------------

static omle::rt::Status op_knn(ValueStore& vs, int n_rows,
                               const std::vector<std::string>& in,
                               const std::vector<std::string>& out,
                               const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};

  const std::string feat_name = attr_tensor(attrs, "train_features");
  const std::string target_name = attr_tensor(attrs, "train_targets");
  if (feat_name.empty() || target_name.empty())
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.ml/KNN: train_features and train_targets are required"};

  const int k = attr_i(attrs, "n_neighbors", 5);
  const std::string task = attr_s(attrs, "task", "classification");
  const std::string metric = attr_s(attrs, "metric", "euclidean");
  const std::string wt = attr_s(attrs, "weights", "uniform");
  const double p = attr_float(attrs, "p", 2.0);
  const bool use_distance = (wt == "distance");
  const std::string neighbor_mode = attr_s(attrs, "neighbor_mode", "knn");
  const bool is_radius = (neighbor_mode == "radius");
  const double radius = attr_float(attrs, "radius", 1.0);

  // Gather all variadic inputs into one dense query matrix.
  Tensor query_in = concat_inputs(vs, in, n_rows);
  const bool f64_out = wants_f64(query_in);

  // Both matrices are materialised in float64 once, up front. Distances
  // decide the neighbour set, and two training points can sit a float32 ulp
  // apart, so ranking them from narrowed coordinates picks a different set of
  // neighbours entirely -- not a small numeric wobble in the output but a
  // different answer. The copy is one pass over data that to_float32() was
  // already copying anyway.
  Tensor query = query_in.to_float64();
  Tensor train_x = vs.get(feat_name).to_float64();
  const Tensor& train_y = vs.get(target_name);

  const int M = train_x.n_rows;
  const int D = train_x.n_cols;
  if (M < 1 || D < 1)
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.ml/KNN: empty training set"};
  const int kk = std::min(k, M);

  auto compute_dist = [&](const double* q, const double* t) -> double {
    if (metric == "manhattan") {
      double s = 0.0;
      for (int d = 0; d < D; ++d) s += std::fabs(q[d] - t[d]);
      return s;
    } else if (metric == "cosine") {
      double dot = 0.0, nq = 0.0, nt = 0.0;
      for (int d = 0; d < D; ++d) {
        dot += q[d] * t[d];
        nq += q[d] * q[d];
        nt += t[d] * t[d];
      }
      double denom = std::sqrt(nq) * std::sqrt(nt);
      return denom > 1e-12 ? 1.0 - dot / denom : 1.0;
    } else if (metric == "minkowski") {
      double s = 0.0;
      for (int d = 0; d < D; ++d) s += std::pow(std::fabs(q[d] - t[d]), p);
      return std::pow(s, 1.0 / p);
    } else {
      double s = 0.0;
      for (int d = 0; d < D; ++d) {
        const double diff = q[d] - t[d];
        s += diff * diff;
      }
      return std::sqrt(s);
    }
  };

  const int n_classes = attr_i(attrs, "n_classes", 2);
  const bool emit_prob = (out.size() > 1 && task != "regression");
  const int prob_cols = n_classes;

  OutBuf result(f64_out, n_rows, 1);
  OutBuf probs(f64_out, emit_prob ? n_rows : 0, emit_prob ? prob_cols : 1);
  std::vector<std::pair<double, int>> dist_buf(M);

  // outlier_label for radius mode (classification): class to use when no
  // neighbor found
  const std::string outlier_ref = attr_tensor(attrs, "outlier_label");
  double outlier_val = 0.0;
  if (!outlier_ref.empty()) {
    const Tensor& ot = vs.get(outlier_ref);
    if (ot.n_rows > 0) outlier_val = ot.get(0, 0);
  }

  for (int s = 0; s < n_rows; ++s) {
    const double* q = query.f64_row(s);
    for (int i = 0; i < M; ++i)
      dist_buf[i] = {compute_dist(q, train_x.f64_row(i)), i};

    // Determine neighbor set: k nearest OR all within radius.
    int n_neighbors_used;
    if (is_radius) {
      std::sort(
          dist_buf.begin(), dist_buf.end(),
          [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
          });
      n_neighbors_used = 0;
      for (int i = 0; i < M; ++i) {
        if (dist_buf[i].first <= radius)
          ++n_neighbors_used;
        else
          break;
      }
    } else {
      std::partial_sort(
          dist_buf.begin(), dist_buf.begin() + kk, dist_buf.end(),
          [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first < b.first;
          });
      n_neighbors_used = kk;
    }

    if (task == "regression") {
      double num = 0.0, denom = 0.0;
      for (int j = 0; j < n_neighbors_used; ++j) {
        const double dist = dist_buf[j].first;
        const double yval = train_y.get(dist_buf[j].second, 0);
        const double w =
            use_distance ? ((dist > 1e-12) ? 1.0 / dist : 1e12) : 1.0;
        num += w * yval;
        denom += w;
      }
      result.set(s, 0, (denom > 0.0) ? num / denom : 0.0);
    } else {
      if (n_neighbors_used == 0) {
        // radius mode with no neighbors: use outlier label
        result.set(s, 0, outlier_val);
        if (emit_prob) {
          for (int c = 0; c < prob_cols; ++c) probs.set(s, c, 0.0);
          int outlier_cls = static_cast<int>(outlier_val);
          if (outlier_cls >= 0 && outlier_cls < prob_cols)
            probs.set(s, outlier_cls, 1.0);
        }
        continue;
      }
      std::unordered_map<int, double> votes;
      double total_w = 0.0;
      for (int j = 0; j < n_neighbors_used; ++j) {
        int cls = static_cast<int>(train_y.get(dist_buf[j].second, 0));
        const double dist = dist_buf[j].first;
        const double w =
            use_distance ? ((dist > 1e-12) ? 1.0 / dist : 1e12) : 1.0;
        votes[cls] += w;
        total_w += w;
      }
      // Ties broken by lowest class index, matching sklearn's np.argmax
      // behavior.
      int best_cls = -1;
      double best_score = -1.0;
      for (const auto& kv : votes)
        if (kv.second > best_score ||
            (kv.second == best_score && kv.first < best_cls)) {
          best_score = kv.second;
          best_cls = kv.first;
        }
      result.set(s, 0, static_cast<double>(best_cls));

      if (emit_prob) {
        for (int c = 0; c < n_classes; ++c) {
          auto it = votes.find(c);
          probs.set(s, c,
                    (it != votes.end() && total_w > 0.0) ? it->second / total_w
                                                         : 0.0);
        }
      }
    }
  }
  vs.put(out[0], result.take());
  if (emit_prob) vs.put(out[1], probs.take());
  return {};
}

// ---------------------------------------------------------------------------
// omle.text operators
// ---------------------------------------------------------------------------

// MurmurHash3_x86_32 on UTF-8 bytes, seed=42 (matches Spark's HashingTF).
static int32_t murmur3_x86_32(const char* data, int len, int32_t seed = 42) {
  auto rotl32 = [](uint32_t v, int n) -> uint32_t {
    return (v << n) | (v >> (32 - n));
  };
  const uint32_t C1 = 0xcc9e2d51u;
  const uint32_t C2 = 0x1b873593u;
  const uint32_t E = 0xe6546b64u;
  uint32_t h = static_cast<uint32_t>(seed);
  const int n_blocks = len / 4;
  const auto* blocks = reinterpret_cast<const uint32_t*>(data);
  for (int i = 0; i < n_blocks; ++i) {
    uint32_t k;
    std::memcpy(&k, blocks + i, 4);  // safe unaligned read, little-endian
    k *= C1;
    k = rotl32(k, 15);
    k *= C2;
    h ^= k;
    h = rotl32(h, 13);
    h = h * 5u + E;
  }
  const auto* tail = reinterpret_cast<const uint8_t*>(data + n_blocks * 4);
  uint32_t k = 0;
  switch (len & 3) {
    case 3:
      k ^= static_cast<uint32_t>(tail[2]) << 16;
      [[fallthrough]];
    case 2:
      k ^= static_cast<uint32_t>(tail[1]) << 8;
      [[fallthrough]];
    case 1:
      k ^= static_cast<uint32_t>(tail[0]);
      k *= C1;
      k = rotl32(k, 15);
      k *= C2;
      h ^= k;
  }
  h ^= static_cast<uint32_t>(len);
  h ^= h >> 16;
  h *= 0x85ebca6bu;
  h ^= h >> 13;
  h *= 0xc2b2ae35u;
  h ^= h >> 16;
  return static_cast<int32_t>(h);
}

static int hash_bucket(const std::string& word, int num_features) {
  int32_t h = murmur3_x86_32(word.data(), static_cast<int>(word.size()));
  int raw = static_cast<int>(h % num_features);
  return raw < 0 ? raw + num_features : raw;
}

// Extract non-empty tokens from a row of a 2-D string tensor.
static std::vector<std::string> row_tokens(const Tensor& t, int row) {
  std::vector<std::string> toks;
  for (int c = 0; c < t.n_cols; ++c) {
    const std::string& s = t.str_at(row, c);
    if (!s.empty()) toks.push_back(s);
  }
  return toks;
}

// Get the single text string from a row (Tokenizer / RegexTokenizer input).
// Input shape is [n_rows, 1] after the 1-D reshape fix.
static const std::string& row_text(const Tensor& t, int row) {
  return t.str_at(row, 0);
}

// Build a 2-D string tensor from per-row token lists, padded with "".
static Tensor tokens_to_tensor(
    const std::vector<std::vector<std::string>>& rows) {
  int n = static_cast<int>(rows.size());
  int w = 0;
  for (const auto& r : rows) w = std::max(w, static_cast<int>(r.size()));
  Tensor out = Tensor::strings(n, w);
  for (int r = 0; r < n; ++r)
    for (int c = 0; c < static_cast<int>(rows[r].size()); ++c)
      out.str_at(r, c) = rows[r][c];
  return out;
}

// ── Elementwise string transforms ────────────────────────────────────────────
// Lowercase, Trim, NormalizeWhitespace, StringReplace and RegexReplace each map
// every string cell through a function and preserve the input shape, which is
// what the registry's same_shape(x) output rule requires.
//
// Case folding is ASCII-only, matching the rest of the runtime's text path
// (Tokenizer and RegexTokenizer fold with std::tolower). Bytes outside ASCII
// pass through untouched, so UTF-8 text survives but non-ASCII letters are not
// case-folded.
static omle::rt::Status string_map(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const char* op_name,
    const std::function<std::string(const std::string&)>& fn) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);
  if (!src.is_string())
    return {omle::rt::ErrorCode::InvalidArgument,
            std::string("omle.text/") + op_name +
                ": input must be a string tensor"};
  const int n_cols = src.n_cols;
  Tensor dst = Tensor::strings(n_rows, n_cols);
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c) dst.str_at(r, c) = fn(src.str_at(r, c));
  vs.put(out[0], std::move(dst));
  return {};
}

static std::string str_lowercase(const std::string& s) {
  std::string t = s;
  std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return t;
}

static std::string str_trim(const std::string& s) {
  std::size_t b = 0, e = s.size();
  while (b < e && std::isspace(static_cast<unsigned char>(s[b]))) ++b;
  while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) --e;
  return s.substr(b, e - b);
}

// Collapse each run of whitespace to a single space. Leading and trailing runs
// are collapsed but not removed: the registry says to pair this with Trim for
// full normalization, so removing them here would make Trim unobservable.
static std::string str_normalize_ws(const std::string& s) {
  std::string t;
  t.reserve(s.size());
  bool in_ws = false;
  for (unsigned char c : s) {
    if (std::isspace(c)) {
      if (!in_ws) t.push_back(' ');
      in_ws = true;
    } else {
      t.push_back(static_cast<char>(c));
      in_ws = false;
    }
  }
  return t;
}

// Literal (non-regex) substring replacement. An empty `old` would match at
// every position, so it is treated as a no-op rather than looping forever.
static std::string str_replace_literal(const std::string& s,
                                       const std::string& old_s,
                                       const std::string& new_s,
                                       bool replace_all) {
  if (old_s.empty()) return s;
  std::string t;
  t.reserve(s.size());
  std::size_t pos = 0;
  while (true) {
    const std::size_t hit = s.find(old_s, pos);
    if (hit == std::string::npos) break;
    t.append(s, pos, hit - pos);
    t.append(new_s);
    pos = hit + old_s.size();
    if (!replace_all) break;
  }
  t.append(s, pos, std::string::npos);
  return t;
}

static omle::rt::Status op_lowercase(ValueStore& vs, int n_rows,
                                     const std::vector<std::string>& in,
                                     const std::vector<std::string>& out,
                                     const AttributeMap&) {
  return string_map(vs, n_rows, in, out, "Lowercase", str_lowercase);
}

static omle::rt::Status op_trim(ValueStore& vs, int n_rows,
                                const std::vector<std::string>& in,
                                const std::vector<std::string>& out,
                                const AttributeMap&) {
  return string_map(vs, n_rows, in, out, "Trim", str_trim);
}

static omle::rt::Status op_normalize_whitespace(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap&) {
  return string_map(vs, n_rows, in, out, "NormalizeWhitespace",
                    str_normalize_ws);
}

static omle::rt::Status op_string_replace(ValueStore& vs, int n_rows,
                                          const std::vector<std::string>& in,
                                          const std::vector<std::string>& out,
                                          const AttributeMap& attrs) {
  const std::string old_s = attr_s(attrs, "old", "");
  const std::string new_s = attr_s(attrs, "new", "");
  const bool all = attr_b(attrs, "replace_all", true);
  return string_map(vs, n_rows, in, out, "StringReplace",
                    [&](const std::string& v) {
                      return str_replace_literal(v, old_s, new_s, all);
                    });
}

static omle::rt::Status op_regex_replace(ValueStore& vs, int n_rows,
                                         const std::vector<std::string>& in,
                                         const std::vector<std::string>& out,
                                         const AttributeMap& attrs) {
  const std::string pat = attr_s(attrs, "pattern", "");
  const std::string rep = attr_s(attrs, "replacement", "");
  if (pat.empty())
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.text/RegexReplace: pattern is required"};
  std::regex re;
  try {
    re = std::regex(pat);
  } catch (const std::regex_error& e) {
    return {
        omle::rt::ErrorCode::InvalidArgument,
        "omle.text/RegexReplace: invalid pattern: " + std::string(e.what())};
  }
  return string_map(
      vs, n_rows, in, out, "RegexReplace",
      [&](const std::string& v) { return std::regex_replace(v, re, rep); });
}

static omle::rt::Status op_tokenizer(ValueStore& vs, int n_rows,
                                     const std::vector<std::string>& in,
                                     const std::vector<std::string>& out,
                                     const AttributeMap&) {
  if (in.empty() || out.empty()) return {};
  const Tensor& src = vs.get(in[0]);
  std::vector<std::vector<std::string>> rows(n_rows);
  for (int r = 0; r < n_rows; ++r) {
    std::string text = src.is_string() ? row_text(src, r) : std::string{};
    // lowercase
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    std::istringstream iss(text);
    std::string tok;
    while (iss >> tok) rows[r].push_back(tok);
  }
  vs.put(out[0], tokens_to_tensor(rows));
  return {};
}

static omle::rt::Status op_regex_tokenizer(ValueStore& vs, int n_rows,
                                           const std::vector<std::string>& in,
                                           const std::vector<std::string>& out,
                                           const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  std::string pat = attr_s(attrs, "pattern", "\\s+");
  bool gaps = attr_b(attrs, "gaps", true);
  int min_len = attr_i(attrs, "min_token_length", 1);
  const Tensor& src = vs.get(in[0]);

  std::regex re(pat);
  std::vector<std::vector<std::string>> rows(n_rows);
  for (int r = 0; r < n_rows; ++r) {
    std::string text = src.is_string() ? row_text(src, r) : std::string{};
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (gaps) {
      // pattern is a delimiter — split on it
      auto it = std::sregex_token_iterator(text.begin(), text.end(), re, -1);
      auto end = std::sregex_token_iterator{};
      for (; it != end; ++it) {
        std::string tok = it->str();
        if (static_cast<int>(tok.size()) >= min_len) rows[r].push_back(tok);
      }
    } else {
      // pattern is a token — find all matches
      auto it = std::sregex_iterator(text.begin(), text.end(), re);
      auto end = std::sregex_iterator{};
      for (; it != end; ++it) {
        std::string tok = (*it)[0].str();
        if (static_cast<int>(tok.size()) >= min_len) rows[r].push_back(tok);
      }
    }
  }
  vs.put(out[0], tokens_to_tensor(rows));
  return {};
}

static omle::rt::Status op_ngram(ValueStore& vs, int n_rows,
                                 const std::vector<std::string>& in,
                                 const std::vector<std::string>& out,
                                 const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  int n_min = attr_i(attrs, "n_min", 2);
  int n_max = attr_i(attrs, "n_max", 2);
  const Tensor& src = vs.get(in[0]);
  std::vector<std::vector<std::string>> rows(n_rows);
  for (int r = 0; r < n_rows; ++r) {
    auto toks = row_tokens(src, r);
    int nt = static_cast<int>(toks.size());
    for (int n = n_min; n <= n_max; ++n) {
      for (int i = 0; i + n <= nt; ++i) {
        std::string gram = toks[i];
        for (int j = 1; j < n; ++j) gram += " " + toks[i + j];
        rows[r].push_back(gram);
      }
    }
  }
  vs.put(out[0], tokens_to_tensor(rows));
  return {};
}

static omle::rt::Status op_stop_words_remover(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  bool case_sensitive = attr_b(attrs, "case_sensitive", false);
  const Tensor* sw_t = resolve_t(vs, in, in.size(), attrs, "stop_words");
  std::unordered_set<std::string> stop_set;
  if (sw_t && sw_t->is_string()) {
    for (int i = 0; i < sw_t->n_rows * sw_t->n_cols; ++i) {
      std::string w = sw_t->str_data()[i];
      if (!case_sensitive) {
        std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) {
          return static_cast<char>(std::tolower(c));
        });
      }
      if (!w.empty()) stop_set.insert(w);
    }
  }
  const Tensor& src = vs.get(in[0]);
  std::vector<std::vector<std::string>> rows(n_rows);
  for (int r = 0; r < n_rows; ++r) {
    auto toks = row_tokens(src, r);
    for (auto& tok : toks) {
      std::string key = tok;
      if (!case_sensitive) {
        std::transform(
            key.begin(), key.end(), key.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
      }
      if (!stop_set.count(key)) rows[r].push_back(tok);
    }
  }
  vs.put(out[0], tokens_to_tensor(rows));
  return {};
}

static omle::rt::Status op_count_vectorizer(ValueStore& vs, int n_rows,
                                            const std::vector<std::string>& in,
                                            const std::vector<std::string>& out,
                                            const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  bool binary = attr_b(attrs, "binary", false);
  const Tensor* vocab_t = resolve_t(vs, in, in.size(), attrs, "vocabulary");
  if (!vocab_t || !vocab_t->is_string())
    return {omle::rt::ErrorCode::InvalidArgument,
            "CountVectorizer: missing vocabulary"};
  int vocab_size = vocab_t->n_rows * vocab_t->n_cols;
  std::unordered_map<std::string, int> vocab_index;
  for (int i = 0; i < vocab_size; ++i) vocab_index[vocab_t->str_data()[i]] = i;
  const Tensor& src = vs.get(in[0]);
  Tensor result(n_rows, vocab_size);
  float* dp = result.f32_ptr();
  std::fill(dp, dp + n_rows * vocab_size, 0.f);
  for (int r = 0; r < n_rows; ++r) {
    auto toks = row_tokens(src, r);
    for (const auto& tok : toks) {
      auto it = vocab_index.find(tok);
      if (it != vocab_index.end()) {
        float& cell = dp[r * vocab_size + it->second];
        if (binary)
          cell = 1.f;
        else
          cell += 1.f;
      }
    }
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_hashing_vectorizer(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  int num_features = attr_i(attrs, "num_features", 1 << 18);
  bool binary = attr_b(attrs, "binary", false);
  bool alternate_sign = attr_b(attrs, "alternate_sign", false);
  const Tensor& src = vs.get(in[0]);
  Tensor result(n_rows, num_features);
  float* dp = result.f32_ptr();
  std::fill(dp, dp + (std::size_t)n_rows * num_features, 0.f);
  for (int r = 0; r < n_rows; ++r) {
    auto toks = row_tokens(src, r);
    for (const auto& tok : toks) {
      int bucket = hash_bucket(tok, num_features);
      float& cell = dp[r * num_features + bucket];
      if (binary) {
        cell = 1.f;
      } else if (alternate_sign) {
        int32_t h = murmur3_x86_32(tok.data(), static_cast<int>(tok.size()));
        float sign = (h >= 0) ? 1.f : -1.f;
        cell += sign;
      } else {
        cell += 1.f;
      }
    }
  }
  vs.put(out[0], std::move(result));
  return {};
}

static omle::rt::Status op_tfidf_transformer(
    ValueStore& vs, int n_rows, const std::vector<std::string>& in,
    const std::vector<std::string>& out, const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor* idf_t = resolve_t64(vs, in, in.size(), attrs, "idf");
  if (!idf_t)
    return {omle::rt::ErrorCode::InvalidArgument,
            "TfIdfTransformer: missing idf"};
  const Tensor& src = vs.get(in[0]);
  int n_cols = src.n_cols;
  // src was read through f32_ptr(), which reinterprets a float64 term-count
  // matrix as float outright. get() reads whatever is actually stored, and
  // the idf weights -- log ratios, never exact in binary -- multiply in
  // double before the result is rounded to the input's width.
  OutBuf result(wants_f64(src), n_rows, n_cols);
  for (int r = 0; r < n_rows; ++r)
    for (int c = 0; c < n_cols; ++c)
      result.set(r, c, src.get(r, c) * idf_t->get(0, c));
  vs.put(out[0], result.take());
  return {};
}

static omle::rt::Status op_word2vec(ValueStore& vs, int n_rows,
                                    const std::vector<std::string>& in,
                                    const std::vector<std::string>& out,
                                    const AttributeMap& attrs) {
  if (in.empty() || out.empty()) return {};
  const Tensor* vocab_t = resolve_t(vs, in, in.size(), attrs, "vocabulary");
  const Tensor* emb_t = resolve_t64(vs, in, in.size(), attrs, "embeddings");
  if (!vocab_t || !vocab_t->is_string() || !emb_t)
    return {omle::rt::ErrorCode::InvalidArgument,
            "Word2Vec: missing vocabulary or embeddings"};
  int vocab_size = vocab_t->n_rows * vocab_t->n_cols;
  int embed_dim = emb_t->n_cols;
  std::unordered_map<std::string, int> vocab_index;
  for (int i = 0; i < vocab_size; ++i) vocab_index[vocab_t->str_data()[i]] = i;
  const Tensor& src = vs.get(in[0]);
  // The input is tokenised text, so there is no input width to follow -- the
  // document vector is float32 as before. The sum is accumulated in double
  // and divided once: a long document adds many embedding rows, and a running
  // float32 total loses low bits on every one of them before the mean is
  // taken.
  Tensor result(n_rows, embed_dim);
  float* dp = result.f32_ptr();
  std::vector<double> acc(static_cast<std::size_t>(embed_dim));
  for (int r = 0; r < n_rows; ++r) {
    auto toks = row_tokens(src, r);
    std::fill(acc.begin(), acc.end(), 0.0);
    int count = 0;
    for (const auto& tok : toks) {
      auto it = vocab_index.find(tok);
      if (it == vocab_index.end()) continue;
      int vocab_idx = it->second;
      for (int d = 0; d < embed_dim; ++d) acc[d] += emb_t->get(vocab_idx, d);
      ++count;
    }
    const double inv = (count > 1) ? 1.0 / count : 1.0;
    for (int d = 0; d < embed_dim; ++d)
      dp[r * embed_dim + d] = static_cast<float>(acc[d] * inv);
  }
  vs.put(out[0], std::move(result));
  return {};
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

void register_builtin_operators() {
  auto& r = OperatorRegistry::instance();

  // ── omle.core ─────────────────────────────────────────────────────────
  r.reg("omle.core", "Identity", op_identity);
  r.reg("omle.core", "Concat", op_concat);
  r.reg("omle.core", "Gather",
        op_gather);  // used internally, no registry entry
  r.reg("omle.core", "Clip", op_clip);
  r.reg("omle.core", "ArgMax", op_argmax);
  r.reg("omle.core", "SelectByPrimarySecondaryScore",
        op_select_by_primary_secondary_score);
  r.reg("omle.core", "Cast", op_cast);
  r.reg("omle.core", "Reshape", op_reshape);
  r.reg("omle.core", "Split", op_split);
  r.reg("omle.core", "TakeSlots", op_take_slots);
  r.reg("omle.core", "Derive", op_derive);
  r.reg("omle.core", "Select", op_select);
  r.reg("omle.core", "SparseToDense", op_sparse_to_dense);
  r.reg("omle.core", "DenseToSparse", op_dense_to_sparse);
  r.reg("omle.core", "Sum", op_sum);
  r.reg("omle.core", "Average", op_average);
  r.reg("omle.core", "Min", op_min);
  r.reg("omle.core", "Max", op_max);
  r.reg("omle.core", "Median", op_median);
  r.reg("omle.core", "WeightedSum", op_weighted_sum);
  r.reg("omle.core", "WeightedAverage", op_weighted_average);
  r.reg("omle.core", "WeightedMedian", op_weighted_median);
  r.reg("omle.core", "MajorityVote", op_majority_vote);
  r.reg("omle.core", "WeightedMajorityVote", op_weighted_majority_vote);
  r.reg("omle.core", "SoftVote", op_soft_vote);
  r.reg("omle.core", "SAMMEVote", op_samme_vote);

  // ── omle.feature ──────────────────────────────────────────────────────
  r.reg("omle.feature", "StandardScaler", op_standard_scaler);
  r.reg("omle.feature", "MinMaxScaler", op_minmax_scaler);
  r.reg("omle.feature", "RobustScaler", op_robust_scaler);
  r.reg("omle.feature", "MaxAbsScaler", op_max_abs_scaler);
  r.reg("omle.feature", "Normalizer", op_normalizer);
  r.reg("omle.feature", "PowerTransformer", op_power_transformer);
  r.reg("omle.feature", "QuantileTransformer", op_quantile_transformer);
  r.reg("omle.feature", "SplineTransformer", op_spline_transformer);
  r.reg("omle.feature", "PolynomialFeatures", op_polynomial_features);
  r.reg("omle.feature", "Binarizer", op_binarizer);
  r.reg("omle.feature", "Bucketizer", op_bucketizer);
  r.reg("omle.feature", "Discretizer", op_discretize);
  r.reg("omle.feature", "Imputer", op_simple_imputer);
  r.reg("omle.feature", "KNNImputer", op_knn_imputer);
  r.reg("omle.feature", "MissingIndicator", op_missing_indicator);
  r.reg("omle.feature", "OneHotEncoder", op_one_hot_encoder);
  r.reg("omle.feature", "OrdinalEncoder", op_ordinal_encoder);
  r.reg("omle.feature", "LabelEncoder", op_label_encoder);
  r.reg("omle.feature", "LabelBinarizer", op_label_binarizer);
  r.reg("omle.feature", "MultiLabelBinarizer", op_multi_label_binarizer);
  r.reg("omle.feature", "TargetEncoder", op_target_encoder);
  r.reg("omle.feature", "NormContinuous", op_norm_continuous);
  r.reg("omle.feature", "NormDiscrete", op_norm_discrete);
  r.reg("omle.feature", "MapValues", op_map_values);
  r.reg("omle.feature", "TruncatedSVD", op_truncated_svd);
  r.reg("omle.feature", "FastICA", op_fast_ica);
  r.reg("omle.feature", "FactorAnalysis", op_factor_analysis);
  r.reg("omle.feature", "PCA", op_pca);
  r.reg("omle.feature", "KernelPCA", op_kernel_pca);
  r.reg("omle.feature", "SparsePCA", op_sparse_pca);
  r.reg("omle.feature", "NMF", op_nmf);
  r.reg("omle.feature", "LatentDirichletAllocation", op_lda);

  // ── backward-compat aliases (old names before registry refactor) ─────────
  r.reg("omle.feature", "Binarize", op_binarizer);
  r.reg("omle.feature", "Discretize", op_discretize);
  r.reg("omle.feature", "Impute", op_simple_imputer);
  r.reg("omle.feature", "SimpleImputer", op_simple_imputer);
  r.reg("omle.feature", "LabelEncode", op_label_encoder);
  r.reg("omle.feature", "OneHotEncode", op_one_hot_encoder);

  // ── omle.ml ───────────────────────────────────────────────────────────
  r.reg("omle.ml", "KNN", op_knn);

  // ── omle.text ─────────────────────────────────────────────────────────
  r.reg("omle.text", "Lowercase", op_lowercase);
  r.reg("omle.text", "Trim", op_trim);
  r.reg("omle.text", "NormalizeWhitespace", op_normalize_whitespace);
  r.reg("omle.text", "StringReplace", op_string_replace);
  r.reg("omle.text", "RegexReplace", op_regex_replace);
  r.reg("omle.text", "Tokenizer", op_tokenizer);
  r.reg("omle.text", "RegexTokenizer", op_regex_tokenizer);
  r.reg("omle.text", "NGram", op_ngram);
  r.reg("omle.text", "StopWordsRemover", op_stop_words_remover);
  r.reg("omle.text", "CountVectorizer", op_count_vectorizer);
  r.reg("omle.text", "HashingVectorizer", op_hashing_vectorizer);
  r.reg("omle.text", "TfIdfTransformer", op_tfidf_transformer);
  r.reg("omle.text", "Word2Vec", op_word2vec);

  // ── omle.functions ────────────────────────────────────────────────────
  r.reg("omle.functions", "Compute",
        [](ValueStore& vs, int n_rows, const std::vector<std::string>& in_names,
           const std::vector<std::string>& out_names,
           const AttributeMap& attrs) -> omle::rt::Status {
          auto it = attrs.find("expr");
          if (it == attrs.end() || !it->second.expr)
            return {omle::rt::ErrorCode::InvalidArgument,
                    "omle.functions/Compute: missing 'expr' attribute"};
          ASSIGN_OR_RETURN(auto values,
                           eval_expr(*it->second.expr, vs, n_rows));
          Tensor t(n_rows, 1);
          if (any_f64(vs, in_names)) {
            t.set_doubles(std::move(values));
          } else {
            std::vector<float> narrowed(values.begin(), values.end());
            t.set_floats(std::move(narrowed));
          }
          if (!out_names.empty()) vs.put(out_names[0], std::move(t));
          return {};
        });
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template void standard_scaler_impl<float>(float*, const double*, const double*,
                                          int);
template void standard_scaler_impl<double>(double*, const double*,
                                           const double*, int);

template void minmax_scaler_impl<float>(float*, const double*, const double*,
                                        double, double, int);
template void minmax_scaler_impl<double>(double*, const double*, const double*,
                                         double, double, int);

template void normalizer_impl<float>(float*, int, const std::string&);
template void normalizer_impl<double>(double*, int, const std::string&);

template void power_transformer_impl<float>(float*, const double*, int,
                                            const std::string&);
template void power_transformer_impl<double>(double*, const double*, int,
                                             const std::string&);

template void bspline_basis<double>(const double*, int, int, double, double*,
                                    int);

template double probit<double>(double);

template Tensor matmul<float>(const Tensor&, const Tensor&, int);
template Tensor matmul<double>(const Tensor&, const Tensor&, int);

}  // namespace omle::rt::impl
