#include "svm_node.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "../post_transform.h"
#include "../simd_traits.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Abstract Impl base
// -----------------------------------------------------------------------
struct SVMNode::Impl {
  virtual int n_features() const noexcept = 0;
  virtual int n_outputs() const noexcept = 0;
  virtual omle::rt::DataType dtype() const noexcept = 0;
  virtual bool has_platt() const noexcept = 0;
  virtual double platt_a(int idx) const noexcept { return 0.0; }
  virtual double platt_b(int idx) const noexcept { return 0.0; }
  virtual void compute(const void* features, int n_rows,
                       void* output) const = 0;
  virtual ~Impl() = default;
};

// -----------------------------------------------------------------------
// Typed model storage
// -----------------------------------------------------------------------
template <typename T>
struct SVMModelT {
  SVMKind kind = SVMKind::Linear;
  KernelType kernel_type = KernelType::RBF;

  // Linear SVM
  std::vector<T> coefficients;  // [n_outputs, n_features] row-major
  std::vector<T> intercept;     // [n_outputs]

  // Kernel SVM
  std::vector<T> support_vectors;    // [n_sv, n_features]
  std::vector<T> dual_coefficients;  // [n_outputs, n_sv]
  double gamma = 1.0;
  int degree = 3;
  double coef0 = 0.0;
  int n_sv = 0;

  // Platt scaling (probability=True in sklearn)
  std::vector<T> prob_a;  // A params: p = 1/(1+exp(A*f+B))
  std::vector<T> prob_b;  // B params

  // Per-class support vector counts for OVO multiclass.
  // If non-empty, multiclass OVO Platt path is used.
  std::vector<int> n_support;

  PostTransform post_transform = PostTransform::Identity;
  int n_features = 0;
  int n_outputs = 1;
};

// -----------------------------------------------------------------------
// Templated compute kernels
// -----------------------------------------------------------------------
template <typename T>
static T dot_t(const T* a, const T* b, int n) noexcept {
  T s = T(0);
  for (int k = 0; k < n; ++k) s += a[k] * b[k];
  return s;
}

template <typename T>
static void compute_kernel_row(const SVMModelT<T>& m, const T* x,
                               std::vector<T>& krow)  // [n_sv]
{
  const int nf = m.n_features;
  const int nsv = m.n_sv;
  const T* sv = m.support_vectors.data();

  switch (m.kernel_type) {
    case KernelType::Linear:
      for (int i = 0; i < nsv; ++i) krow[i] = dot_t<T>(x, sv + i * nf, nf);
      break;

    case KernelType::RBF: {
      const T g = static_cast<T>(m.gamma);
      for (int i = 0; i < nsv; ++i) {
        T d = T(0);
        const T* svi = sv + i * nf;
        for (int k = 0; k < nf; ++k) {
          T diff = x[k] - svi[k];
          d += diff * diff;
        }
        krow[i] = std::exp(-g * d);
      }
      break;
    }

    case KernelType::Poly: {
      const T g = static_cast<T>(m.gamma);
      const T c0 = static_cast<T>(m.coef0);
      for (int i = 0; i < nsv; ++i) {
        T base = g * dot_t<T>(x, sv + i * nf, nf) + c0;
        krow[i] = std::pow(base, static_cast<T>(m.degree));
      }
      break;
    }

    case KernelType::Sigmoid: {
      const T g = static_cast<T>(m.gamma);
      const T c0 = static_cast<T>(m.coef0);
      for (int i = 0; i < nsv; ++i)
        krow[i] = std::tanh(g * dot_t<T>(x, sv + i * nf, nf) + c0);
      break;
    }
  }
}

template <typename T>
static void predict_linear_svm(const SVMModelT<T>& m, const T* features,
                               int n_samples, T* output) {
  const int nf = m.n_features;
  const int nout = m.n_outputs;
  std::fill(output, output + static_cast<std::size_t>(n_samples) * nout, T(0));
  for (int j = 0; j < nout; ++j) {
    const T* w = m.coefficients.data() + j * nf;
    const T b = j < (int)m.intercept.size() ? m.intercept[j] : T(0);
    for (int s = 0; s < n_samples; ++s)
      output[s * nout + j] = dot_t<T>(features + s * nf, w, nf) + b;
  }
}

// Numerically stable Platt sigmoid: 1/(1+exp(A*f+B)) — libsvm convention.
template <typename T>
static T platt_sigmoid(T f, T a, T b) noexcept {
  T fApB = a * f + b;
  if (fApB >= T(0)) return std::exp(-fApB) / (T(1) + std::exp(-fApB));
  return T(1) / (T(1) + std::exp(fApB));
}

// Hastie-Tibshirani multiclass probability from K*(K-1)/2 pairwise probs.
// r[i*K+j] = P(class i | pair (i,j)).  Writes K probs into p[].
// Algorithm: "Method 2" from Wu, Lin & Weng (2004), as in libsvm.
static void multiclass_probability(int K, const double* r_flat, double* p) {
  const int max_iter = std::max(100, K);
  const double eps = 0.005 / K;
  const double min_prob = 1e-7;

  std::vector<std::vector<double>> Q(K, std::vector<double>(K, 0.0));
  for (int t = 0; t < K; ++t) {
    for (int j = 0; j < t; ++j) {
      Q[t][t] += r_flat[j * K + t] * r_flat[j * K + t];
      Q[t][j] = Q[j][t];
    }
    for (int j = t + 1; j < K; ++j) {
      Q[t][t] += r_flat[j * K + t] * r_flat[j * K + t];
      Q[t][j] = -r_flat[j * K + t] * r_flat[t * K + j];
    }
    p[t] = 1.0 / K;
  }

  std::vector<double> Qp(K);
  for (int iter = 0; iter < max_iter; ++iter) {
    double pQp = 0.0;
    for (int t = 0; t < K; ++t) {
      Qp[t] = 0.0;
      for (int j = 0; j < K; ++j) Qp[t] += Q[t][j] * p[j];
      pQp += p[t] * Qp[t];
    }
    double max_err = 0.0;
    for (int t = 0; t < K; ++t) {
      double e = std::abs(Qp[t] - pQp);
      if (e > max_err) max_err = e;
    }
    if (max_err < eps) break;

    for (int t = 0; t < K; ++t) {
      double diff = (-Qp[t] + pQp) / Q[t][t];
      p[t] += diff;
      pQp = (pQp + diff * (diff * Q[t][t] + 2.0 * Qp[t])) / (1.0 + diff) /
            (1.0 + diff);
      for (int j = 0; j < K; ++j) {
        Qp[j] = (Qp[j] + diff * Q[t][j]) / (1.0 + diff);
        p[j] /= (1.0 + diff);
      }
      if (p[t] < min_prob) p[t] = min_prob;
    }
  }
}

// Multiclass OVO Platt-scaled probability prediction.
// Uses n_support to determine SV ranges per class.
template <typename T>
static void predict_kernel_svm_multiclass(
    const SVMModelT<T>& m, const T* features, int n_samples,
    T* output)  // [n_samples, K] class probabilities
{
  const int K = static_cast<int>(m.n_support.size());
  const int nsv = m.n_sv;
  const int nf = m.n_features;

  // Precompute sv_start[c] = first SV index for class c
  std::vector<int> sv_start(K + 1, 0);
  for (int c = 0; c < K; ++c) sv_start[c + 1] = sv_start[c] + m.n_support[c];

  std::vector<T> krow(nsv);
  std::vector<double> r_flat(static_cast<std::size_t>(K) * K, 0.0);
  std::vector<double> p(K);

  for (int s = 0; s < n_samples; ++s) {
    compute_kernel_row<T>(m, features + s * nf, krow);

    // Compute K*(K-1)/2 pairwise decision values and Platt probs
    int pair_idx = 0;
    for (int ci = 0; ci < K; ++ci) {
      for (int cj = ci + 1; cj < K; ++cj) {
        // d = sum_{sv in ci} dc[cj-1][sv] * K(x,sv)
        //   + sum_{sv in cj} dc[ci][sv] * K(x,sv) + intercept
        T d = pair_idx < (int)m.intercept.size() ? m.intercept[pair_idx] : T(0);
        const T* dc_ci =
            m.dual_coefficients.data() + (cj - 1) * nsv + sv_start[ci];
        const T* dc_cj = m.dual_coefficients.data() + ci * nsv + sv_start[cj];
        for (int k = 0; k < m.n_support[ci]; ++k)
          d += dc_ci[k] * krow[sv_start[ci] + k];
        for (int k = 0; k < m.n_support[cj]; ++k)
          d += dc_cj[k] * krow[sv_start[cj] + k];

        // r[ci][cj] = sigmoid_predict(d, probA[pair], probB[pair])
        double rij = pair_idx < (int)m.prob_a.size()
                         ? static_cast<double>(platt_sigmoid<T>(
                               d, m.prob_a[pair_idx], m.prob_b[pair_idx]))
                         : (d > 0 ? 1.0 : 0.0);
        rij = std::min(std::max(rij, 1e-7), 1.0 - 1e-7);
        r_flat[ci * K + cj] = rij;
        r_flat[cj * K + ci] = 1.0 - rij;
        ++pair_idx;
      }
    }
    multiclass_probability(K, r_flat.data(), p.data());
    for (int c = 0; c < K; ++c) output[s * K + c] = static_cast<T>(p[c]);
  }
}

template <typename T>
static void predict_kernel_svm(const SVMModelT<T>& m, const T* features,
                               int n_samples, T* output) {
  const int nout = m.n_outputs;
  const int nsv = m.n_sv;
  std::vector<T> krow(nsv);
  std::fill(output, output + static_cast<std::size_t>(n_samples) * nout, T(0));
  for (int s = 0; s < n_samples; ++s) {
    compute_kernel_row<T>(m, features + s * m.n_features, krow);
    for (int j = 0; j < nout; ++j) {
      const T* dc = m.dual_coefficients.data() + j * nsv;
      T val = j < (int)m.intercept.size() ? m.intercept[j] : T(0);
      for (int i = 0; i < nsv; ++i) val += dc[i] * krow[i];
      output[s * nout + j] =
          val;  // raw decision value; Platt applied in execute()
    }
  }
}

// -----------------------------------------------------------------------
// Typed Impl
// -----------------------------------------------------------------------
template <typename T>
struct SVMImplT final : SVMNode::Impl {
  SVMModelT<T> model;

  int n_features() const noexcept override { return model.n_features; }
  int n_outputs() const noexcept override { return model.n_outputs; }
  bool has_platt() const noexcept override { return !model.prob_a.empty(); }
  double platt_a(int idx) const noexcept override {
    return idx < (int)model.prob_a.size()
               ? static_cast<double>(model.prob_a[idx])
               : 0.0;
  }
  double platt_b(int idx) const noexcept override {
    return idx < (int)model.prob_b.size()
               ? static_cast<double>(model.prob_b[idx])
               : 0.0;
  }
  omle::rt::DataType dtype() const noexcept override {
    return std::is_same<T, double>::value ? omle::rt::DataType::Float64
                                          : omle::rt::DataType::Float32;
  }

  void compute(const void* features_raw, int n_rows,
               void* output_raw) const override {
    const T* features = static_cast<const T*>(features_raw);
    T* output = static_cast<T*>(output_raw);

    if (model.kind == SVMKind::Linear) {
      predict_linear_svm<T>(model, features, n_rows, output);
      apply_post_transform(output, n_rows, model.n_outputs,
                           model.post_transform);
    } else if (model.n_support.size() > 2 && !model.prob_a.empty()) {
      // Multiclass OVO Platt: outputs class probabilities directly.
      predict_kernel_svm_multiclass<T>(model, features, n_rows, output);
    } else {
      predict_kernel_svm<T>(model, features, n_rows, output);
      apply_post_transform(output, n_rows, model.n_outputs,
                           model.post_transform);
    }
  }
};

// Explicit instantiations — linker finds these; no other TU instantiates them.
template struct SVMImplT<float>;
template struct SVMImplT<double>;

// -----------------------------------------------------------------------
// Factory functions
// -----------------------------------------------------------------------
std::unique_ptr<SVMNode::Impl> make_svm_impl(
    SVMKind kind, KernelType kernel_type, std::vector<float> coefficients,
    std::vector<float> intercept, std::vector<float> support_vectors,
    std::vector<float> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<float> prob_a,
    std::vector<float> prob_b, std::vector<int> n_support) {
  auto p = std::make_unique<SVMImplT<float>>();
  p->model.kind = kind;
  p->model.kernel_type = kernel_type;
  p->model.coefficients = std::move(coefficients);
  p->model.intercept = std::move(intercept);
  p->model.support_vectors = std::move(support_vectors);
  p->model.dual_coefficients = std::move(dual_coefficients);
  p->model.gamma = gamma;
  p->model.degree = degree;
  p->model.coef0 = coef0;
  p->model.n_sv = n_sv;
  p->model.n_features = n_features;
  p->model.post_transform = post_transform;
  p->model.prob_a = std::move(prob_a);
  p->model.prob_b = std::move(prob_b);
  p->model.n_support = std::move(n_support);
  // For multiclass OVO, n_outputs = K (number of classes); otherwise use
  // provided value.
  p->model.n_outputs =
      (!p->model.n_support.empty() && p->model.n_support.size() > 2 &&
       !p->model.prob_a.empty())
          ? static_cast<int>(p->model.n_support.size())
          : n_outputs;
  return p;
}

std::unique_ptr<SVMNode::Impl> make_svm_impl_f64(
    SVMKind kind, KernelType kernel_type, std::vector<double> coefficients,
    std::vector<double> intercept, std::vector<double> support_vectors,
    std::vector<double> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<double> prob_a,
    std::vector<double> prob_b, std::vector<int> n_support) {
  auto p = std::make_unique<SVMImplT<double>>();
  p->model.kind = kind;
  p->model.kernel_type = kernel_type;
  p->model.coefficients = std::move(coefficients);
  p->model.intercept = std::move(intercept);
  p->model.support_vectors = std::move(support_vectors);
  p->model.dual_coefficients = std::move(dual_coefficients);
  p->model.gamma = gamma;
  p->model.degree = degree;
  p->model.coef0 = coef0;
  p->model.n_sv = n_sv;
  p->model.n_features = n_features;
  p->model.post_transform = post_transform;
  p->model.prob_a = std::move(prob_a);
  p->model.prob_b = std::move(prob_b);
  p->model.n_support = std::move(n_support);
  p->model.n_outputs =
      (!p->model.n_support.empty() && p->model.n_support.size() > 2 &&
       !p->model.prob_a.empty())
          ? static_cast<int>(p->model.n_support.size())
          : n_outputs;
  return p;
}

// Node-level wrappers — defined here where Impl is complete.
std::unique_ptr<SVMNode> make_svm_node(
    SVMKind kind, KernelType kernel_type, std::vector<float> coefficients,
    std::vector<float> intercept, std::vector<float> support_vectors,
    std::vector<float> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<float> prob_a,
    std::vector<float> prob_b, std::vector<int> n_support) {
  return std::make_unique<SVMNode>(make_svm_impl(
      kind, kernel_type, std::move(coefficients), std::move(intercept),
      std::move(support_vectors), std::move(dual_coefficients), gamma, degree,
      coef0, n_sv, n_features, n_outputs, post_transform, std::move(prob_a),
      std::move(prob_b), std::move(n_support)));
}

std::unique_ptr<SVMNode> make_svm_node_f64(
    SVMKind kind, KernelType kernel_type, std::vector<double> coefficients,
    std::vector<double> intercept, std::vector<double> support_vectors,
    std::vector<double> dual_coefficients, double gamma, int degree,
    double coef0, int n_sv, int n_features, int n_outputs,
    PostTransform post_transform, std::vector<double> prob_a,
    std::vector<double> prob_b, std::vector<int> n_support) {
  return std::make_unique<SVMNode>(make_svm_impl_f64(
      kind, kernel_type, std::move(coefficients), std::move(intercept),
      std::move(support_vectors), std::move(dual_coefficients), gamma, degree,
      coef0, n_sv, n_features, n_outputs, post_transform, std::move(prob_a),
      std::move(prob_b), std::move(n_support)));
}

// -----------------------------------------------------------------------
// SVMNode
// -----------------------------------------------------------------------
SVMNode::SVMNode(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
SVMNode::~SVMNode() = default;

omle::rt::Status SVMNode::execute(ValueStore& vs, int n_rows) const {
  const omle::rt::DataType dt = impl_->dtype();
  auto features = gather_slots(vs, in_names, n_rows, dt);

  const int n_out = impl_->n_outputs();
  omle::rt::Tensor scores = omle::rt::Tensor::dense(dt, n_rows, n_out);
  impl_->compute(dt == omle::rt::DataType::Float64
                     ? (const void*)features.f64_ptr()
                     : (const void*)features.f32_ptr(),
                 n_rows,
                 dt == omle::rt::DataType::Float64 ? (void*)scores.f64_ptr()
                                                   : (void*)scores.f32_ptr());

  if (out_names.size() == 2) {
    if (n_out == 1) {
      omle::rt::Tensor pred =
          omle::rt::Tensor::dense(omle::rt::DataType::Int64, n_rows, 1);
      int64_t* pred_ptr = pred.i64_ptr();
      if (impl_->has_platt()) {
        // Binary Platt: scores hold raw decision values (d).
        // p(class_1) = 1 - sigmoid(A*(-d)+B) — sklearn's calibration
        // convention. y_pred mirrors sklearn's predict(): sign of raw decision
        // value (d >= 0).
        const double A = impl_->platt_a(0);
        const double B = impl_->platt_b(0);
        omle::rt::Tensor prob2 = omle::rt::Tensor::dense(dt, n_rows, 2);
        if (dt == omle::rt::DataType::Float64) {
          const double* sp = scores.f64_ptr();
          double* dp = prob2.f64_ptr();
          for (int r = 0; r < n_rows; ++r) {
            double d = sp[r];
            double p = 1.0 - platt_sigmoid<double>(-d, A, B);
            dp[r * 2] = 1.0 - p;
            dp[r * 2 + 1] = p;
            pred_ptr[r] = d >= 0.0 ? 1 : 0;
          }
        } else {
          const float* sp = scores.f32_ptr();
          float* dp = prob2.f32_ptr();
          const float Af = static_cast<float>(A);
          const float Bf = static_cast<float>(B);
          for (int r = 0; r < n_rows; ++r) {
            float d = sp[r];
            float p = 1.f - platt_sigmoid<float>(-d, Af, Bf);
            dp[r * 2] = 1.f - p;
            dp[r * 2 + 1] = p;
            pred_ptr[r] = d >= 0.f ? 1 : 0;
          }
        }
        vs.put(out_names[1], std::move(prob2));
      } else {
        // Binary without Platt (LinearSVC): raw decision function, threshold 0
        vs.put(out_names[1], scores);
        if (dt == omle::rt::DataType::Float64) {
          const double* sp = scores.f64_ptr();
          for (int r = 0; r < n_rows; ++r) pred_ptr[r] = sp[r] >= 0.0 ? 1 : 0;
        } else {
          const float* sp = scores.f32_ptr();
          for (int r = 0; r < n_rows; ++r) pred_ptr[r] = sp[r] >= 0.0f ? 1 : 0;
        }
      }
      vs.put(out_names[0], std::move(pred));
    } else {
      // Multiclass: y_prob = full score matrix, y_pred = argmax class index.
      omle::rt::Tensor prob = scores;
      vs.put(out_names[1], std::move(prob));

      omle::rt::Tensor pred =
          omle::rt::Tensor::dense(omle::rt::DataType::Int64, n_rows, 1);
      int64_t* pred_ptr = pred.i64_ptr();
      if (dt == omle::rt::DataType::Float64) {
        const double* sp = scores.f64_ptr();
        for (int r = 0; r < n_rows; ++r) {
          const double* row = sp + r * n_out;
          pred_ptr[r] = static_cast<int64_t>(
              std::distance(row, std::max_element(row, row + n_out)));
        }
      } else {
        const float* sp = scores.f32_ptr();
        for (int r = 0; r < n_rows; ++r) {
          const float* row = sp + r * n_out;
          pred_ptr[r] = static_cast<int64_t>(
              std::distance(row, std::max_element(row, row + n_out)));
        }
      }
      vs.put(out_names[0], std::move(pred));
    }
    return {};
  }

  if (dt == omle::rt::DataType::Float64)
    scatter_outputs(vs, out_names, scores.f64_ptr(), n_rows, n_out);
  else
    scatter_outputs(vs, out_names, scores.f32_ptr(), n_rows, n_out);
  return {};
}

}  // namespace omle::rt::impl
