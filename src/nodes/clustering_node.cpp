#include "clustering_node.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "../post_transform.h"
#include "../simd_traits.h"
#include "../width_dispatch.h"

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Abstract Impl base
// -----------------------------------------------------------------------
struct ClusteringNode::Impl {
  virtual int n_features() const noexcept = 0;
  virtual int n_clusters() const noexcept = 0;
  // dtype() is the input feature dtype; outputs are always Float32.
  virtual omle::rt::DataType dtype() const noexcept = 0;
  // features: [n_rows, n_features] array of T; pred_out: [n_rows, 1] float32;
  // dist_out: [n_rows, n_clusters] float32.
  virtual void compute(const void* features, int n_rows, float* pred_out,
                       float* dist_out) const = 0;
  virtual ~Impl() = default;
};

// -----------------------------------------------------------------------
// Typed model storage
// -----------------------------------------------------------------------
template <typename T>
struct ClusteringModelT {
  ClusterKind kind = ClusterKind::Prototype;
  ClusterDistance distance = ClusterDistance::Euclidean;
  ClusterCovType cov_type = ClusterCovType::Full;

  std::vector<T> centers;           // [n_clusters, n_features]
  std::vector<T> weights;           // [n_clusters] (GMM)
  std::vector<T> means;             // [n_clusters, n_features] (GMM)
  std::vector<T> covariances;       // layout depends on cov_type (GMM)
  std::vector<std::string> labels;  // optional cluster labels

  int n_clusters = 0;
  int n_features = 0;
};

// -----------------------------------------------------------------------
// Templated compute kernels
// -----------------------------------------------------------------------
template <typename T>
static T sq_t(T x) noexcept {
  return x * x;
}

template <typename T>
static int nearest_prototype_t(const T* x, const T* centers, int n_clusters,
                               int nf, ClusterDistance dist) {
  int best = 0;
  T best_d = std::numeric_limits<T>::infinity();

  for (int c = 0; c < n_clusters; ++c) {
    const T* ctr = centers + c * nf;
    T d = T(0);
    switch (dist) {
      case ClusterDistance::SquaredEuclidean:
        for (int f = 0; f < nf; ++f) d += sq_t<T>(x[f] - ctr[f]);
        break;
      case ClusterDistance::Euclidean:
        for (int f = 0; f < nf; ++f) d += sq_t<T>(x[f] - ctr[f]);
        d = std::sqrt(d);
        break;
      case ClusterDistance::Manhattan:
        for (int f = 0; f < nf; ++f) d += std::abs(x[f] - ctr[f]);
        break;
      case ClusterDistance::Cosine: {
        T dot = T(0), nx = T(0), nc2 = T(0);
        for (int f = 0; f < nf; ++f) {
          dot += x[f] * ctr[f];
          nx += sq_t<T>(x[f]);
          nc2 += sq_t<T>(ctr[f]);
        }
        T denom = std::sqrt(nx * nc2) + T(1e-12);
        d = T(1) - dot / denom;
        break;
      }
    }
    if (d < best_d) {
      best_d = d;
      best = c;
    }
  }
  return best;
}

template <typename T>
static T compute_distance_t(const T* x, const T* ctr, int nf,
                            ClusterDistance dist) {
  T d = T(0);
  switch (dist) {
    case ClusterDistance::SquaredEuclidean:
      for (int f = 0; f < nf; ++f) d += sq_t<T>(x[f] - ctr[f]);
      break;
    case ClusterDistance::Euclidean:
      for (int f = 0; f < nf; ++f) d += sq_t<T>(x[f] - ctr[f]);
      d = std::sqrt(d);
      break;
    case ClusterDistance::Manhattan:
      for (int f = 0; f < nf; ++f) d += std::abs(x[f] - ctr[f]);
      break;
    case ClusterDistance::Cosine: {
      T dot = T(0), nx = T(0), nc2 = T(0);
      for (int f = 0; f < nf; ++f) {
        dot += x[f] * ctr[f];
        nx += sq_t<T>(x[f]);
        nc2 += sq_t<T>(ctr[f]);
      }
      T denom = std::sqrt(nx * nc2) + T(1e-12);
      d = T(1) - dot / denom;
      break;
    }
  }
  return d;
}

// Log of multivariate Gaussian: -0.5 * (mahal^2 + log_det + k*log(2pi))
template <typename T>
static T gmm_log_prob_t(const T* x, const T* mu, const T* cov, int nf,
                        ClusterCovType ct) {
  static constexpr T log2pi = T(1.8378770664);  // log(2*pi)
  T mahal = T(0), log_det = T(0);

  switch (ct) {
    case ClusterCovType::Spherical: {
      T var = cov[0] + T(1e-9);
      for (int f = 0; f < nf; ++f) mahal += sq_t<T>(x[f] - mu[f]);
      mahal /= var;
      log_det = nf * std::log(var);
      break;
    }
    case ClusterCovType::Diagonal: {
      for (int f = 0; f < nf; ++f) {
        T var = cov[f] + T(1e-9);
        mahal += sq_t<T>(x[f] - mu[f]) / var;
        log_det += std::log(var);
      }
      break;
    }
    case ClusterCovType::Full: {
      // cov stores precisions_cholesky_: upper triangular L where L @ L^T =
      // precision. y_i = sum_{j<=i} L[j,i] * diff[j] = (L^T @ diff)_i mahal =
      // ||y||^2, log_det_cov = -2*sum(log(diag(L)))
      for (int i = 0; i < nf; ++i) {
        T yi = T(0);
        for (int j = 0; j <= i; ++j) yi += cov[j * nf + i] * (x[j] - mu[j]);
        mahal += yi * yi;
      }
      for (int f = 0; f < nf; ++f)
        log_det -= T(2) * std::log(cov[f * nf + f] + T(1e-12));
      break;
    }
  }
  return T(-0.5) * (mahal + log_det + nf * log2pi);
}

// -----------------------------------------------------------------------
// Typed Impl
// -----------------------------------------------------------------------
template <typename T>
struct ClusteringImplT final : ClusteringNode::Impl {
  ClusteringModelT<T> model;

  int n_features() const noexcept override { return model.n_features; }
  int n_clusters() const noexcept override { return model.n_clusters; }
  omle::rt::DataType dtype() const noexcept override {
    return std::is_same<T, double>::value ? omle::rt::DataType::Float64
                                          : omle::rt::DataType::Float32;
  }

  void compute(const void* features_raw, int n_rows, float* pred_out,
               float* dist_out) const override {
    const T* X = static_cast<const T*>(features_raw);
    const int nc = model.n_clusters;
    const int nf = model.n_features;

    if (model.kind == ClusterKind::Prototype) {
      for (int r = 0; r < n_rows; ++r) {
        const T* x = X + r * nf;
        int best = nearest_prototype_t<T>(x, model.centers.data(), nc, nf,
                                          model.distance);
        pred_out[r] = static_cast<float>(best);
        for (int c = 0; c < nc; ++c) {
          const T* ctr = model.centers.data() + c * nf;
          T d = compute_distance_t<T>(x, ctr, nf, model.distance);
          dist_out[r * nc + c] = static_cast<float>(d);
        }
      }
    } else {
      // Gaussian mixture: predict component with highest posterior
      const int cov_stride = (model.cov_type == ClusterCovType::Spherical) ? 1
                             : (model.cov_type == ClusterCovType::Diagonal)
                                 ? nf
                                 : nf * nf;  // Full

      for (int r = 0; r < n_rows; ++r) {
        const T* x = X + r * nf;
        int best = 0;
        T best_lp = -std::numeric_limits<T>::infinity();
        for (int c = 0; c < nc; ++c) {
          T lw = std::log(model.weights[c] + T(1e-12));
          T lp =
              lw + gmm_log_prob_t<T>(x, model.means.data() + c * nf,
                                     model.covariances.data() + c * cov_stride,
                                     nf, model.cov_type);
          dist_out[r * nc + c] = static_cast<float>(lp);
          if (lp > best_lp) {
            best_lp = lp;
            best = c;
          }
        }
        pred_out[r] = static_cast<float>(best);
      }
      // Convert log-posteriors to probabilities in-place.
      apply_post_transform(dist_out, n_rows, nc, PostTransform::Softmax);
    }
  }
};

// Explicit instantiations — linker finds these; no other TU instantiates them.
template struct ClusteringImplT<float>;
template struct ClusteringImplT<double>;

// -----------------------------------------------------------------------
// Factory functions
// -----------------------------------------------------------------------
std::unique_ptr<ClusteringNode::Impl> make_clustering_impl(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<float> centers, std::vector<float> weights,
    std::vector<float> means, std::vector<float> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features) {
  auto p = std::make_unique<ClusteringImplT<float>>();
  p->model.kind = kind;
  p->model.distance = distance;
  p->model.cov_type = cov_type;
  p->model.centers = std::move(centers);
  p->model.weights = std::move(weights);
  p->model.means = std::move(means);
  p->model.covariances = std::move(covariances);
  p->model.labels = std::move(labels);
  p->model.n_clusters = n_clusters;
  p->model.n_features = n_features;
  return p;
}

std::unique_ptr<ClusteringNode::Impl> make_clustering_impl_f64(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<double> centers, std::vector<double> weights,
    std::vector<double> means, std::vector<double> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features) {
  auto p = std::make_unique<ClusteringImplT<double>>();
  p->model.kind = kind;
  p->model.distance = distance;
  p->model.cov_type = cov_type;
  p->model.centers = std::move(centers);
  p->model.weights = std::move(weights);
  p->model.means = std::move(means);
  p->model.covariances = std::move(covariances);
  p->model.labels = std::move(labels);
  p->model.n_clusters = n_clusters;
  p->model.n_features = n_features;
  return p;
}

// Node-level wrappers — defined here where Impl is complete.
std::unique_ptr<ClusteringNode> make_clustering_node(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<float> centers, std::vector<float> weights,
    std::vector<float> means, std::vector<float> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features) {
  return std::make_unique<ClusteringNode>(make_clustering_impl(
      kind, distance, cov_type, std::move(centers), std::move(weights),
      std::move(means), std::move(covariances), std::move(labels), n_clusters,
      n_features));
}

std::unique_ptr<ClusteringNode> make_clustering_node_f64(
    ClusterKind kind, ClusterDistance distance, ClusterCovType cov_type,
    std::vector<double> centers, std::vector<double> weights,
    std::vector<double> means, std::vector<double> covariances,
    std::vector<std::string> labels, int n_clusters, int n_features) {
  return std::make_unique<ClusteringNode>(make_clustering_impl_f64(
      kind, distance, cov_type, std::move(centers), std::move(weights),
      std::move(means), std::move(covariances), std::move(labels), n_clusters,
      n_features));
}

// -----------------------------------------------------------------------
// ClusteringNode
// -----------------------------------------------------------------------
ClusteringNode::ClusteringNode(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ClusteringNode::~ClusteringNode() = default;

omle::rt::Status ClusteringNode::execute(ValueStore& vs, int n_rows) const {
  const omle::rt::DataType dt = impl_->dtype();
  auto features = gather_slots(vs, in_names, n_rows, dt);

  const int nc = impl_->n_clusters();

  Tensor pred_f32(n_rows, 1);
  Tensor dist_scores =
      omle::rt::Tensor::dense(omle::rt::DataType::Float32, n_rows, nc);

  with_width(dt == omle::rt::DataType::Float64, [&](auto tag) {
    using T = decltype(tag);
    impl_->compute((const void*)data_w<T>(features), n_rows, pred_f32.f32_ptr(),
                   dist_scores.f32_ptr());
  });

  if (!out_names.empty()) {
    Tensor pred = Tensor::dense(omle::rt::DataType::Int32, n_rows, 1);
    int32_t* ip = pred.i32_ptr();
    const float* fp = pred_f32.f32_ptr();
    for (int r = 0; r < n_rows; ++r) ip[r] = static_cast<int32_t>(fp[r]);
    vs.put(out_names[0], std::move(pred));
  }
  if (out_names.size() >= 2) vs.put(out_names[1], std::move(dist_scores));
  return {};
}

}  // namespace omle::rt::impl
