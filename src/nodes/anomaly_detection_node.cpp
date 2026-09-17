#include "anomaly_detection_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

namespace omle::rt::impl {
namespace {

constexpr double kEulerGamma = 0.5772156649015329;

// Expected path length of an unsuccessful search in a binary search tree of
// n points — the normalization every isolation-forest score divides by.
// Matches sklearn's _average_path_length, including its n <= 1 and n == 2
// cases.
double average_path_length(double n) {
  if (n <= 1.0) return 0.0;
  if (n == 2.0) return 1.0;
  return 2.0 * (std::log(n - 1.0) + kEulerGamma) - 2.0 * (n - 1.0) / n;
}

double minkowski_distance(const double* a, const double* b, int d, double p) {
  if (p == 2.0) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) {
      const double diff = a[i] - b[i];
      s += diff * diff;
    }
    return std::sqrt(s);
  }
  if (p == 1.0) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) s += std::fabs(a[i] - b[i]);
    return s;
  }
  if (std::isinf(p)) {
    double m = 0.0;
    for (int i = 0; i < d; ++i) m = std::max(m, std::fabs(a[i] - b[i]));
    return m;
  }
  double s = 0.0;
  for (int i = 0; i < d; ++i) s += std::pow(std::fabs(a[i] - b[i]), p);
  return std::pow(s, 1.0 / p);
}

}  // namespace

struct AnomalyDetectionNode::Impl {
  AnomalyKind kind = AnomalyKind::IsolationForest;
  double offset = 0.0;
  int n_features = 0;
  std::vector<AnomalyOutput> outputs;

  // IsolationForest
  std::vector<IsolationTree> trees;
  int64_t max_samples = 0;

  // EllipticEnvelope
  std::vector<double> location;
  std::vector<double> precision;  // row-major [n_features, n_features]

  // Linear / kernel one-class SVM
  std::vector<double> coefficients;
  std::vector<double> support_vectors;    // [n_sv, n_features]
  std::vector<double> dual_coefficients;  // [n_sv]
  double intercept = 0.0;
  AnomalyKernel kernel = AnomalyKernel::RBF;
  double gamma = 1.0;
  int degree = 3;
  double coef0 = 0.0;
  int n_sv = 0;

  // LocalOutlierFactor
  std::vector<double> reference_samples;  // [n_reference, n_features]
  std::vector<double> reference_lrd;      // precomputed per reference point
  std::vector<double> reference_kdist;    // k-distance per reference point
  int n_reference = 0;
  int n_neighbors = 20;
  double p = 2.0;

  // sklearn's score_samples: higher means more normal.
  double score_one(const double* x) const;

  void precompute_lof();
  // k nearest neighbours of x among the reference samples, excluding `skip`.
  void knn(const double* x, int skip, std::vector<int>& idx,
           std::vector<double>& dist) const;
  double local_reachability_density(const double* x, int skip) const;
};

void AnomalyDetectionNode::Impl::knn(const double* x, int skip,
                                     std::vector<int>& idx,
                                     std::vector<double>& dist) const {
  const int k = std::min(n_neighbors, std::max(0, n_reference - (skip >= 0)));
  idx.clear();
  dist.clear();
  if (k <= 0) return;

  std::vector<std::pair<double, int>> all;
  all.reserve(static_cast<std::size_t>(n_reference));
  for (int i = 0; i < n_reference; ++i) {
    if (i == skip) continue;
    all.emplace_back(
        minkowski_distance(
            x,
            reference_samples.data() + static_cast<std::size_t>(i) * n_features,
            n_features, p),
        i);
  }
  const int kk = std::min<int>(k, static_cast<int>(all.size()));
  std::partial_sort(all.begin(), all.begin() + kk, all.end());
  for (int i = 0; i < kk; ++i) {
    dist.push_back(all[i].first);
    idx.push_back(all[i].second);
  }
}

// lrd(x) = 1 / mean(reachability_distance(x, o) for o in kNN(x)), where
// reachability_distance(x, o) = max(k_distance(o), d(x, o)).
double AnomalyDetectionNode::Impl::local_reachability_density(const double* x,
                                                              int skip) const {
  std::vector<int> idx;
  std::vector<double> dist;
  knn(x, skip, idx, dist);
  if (idx.empty()) return 0.0;
  double sum = 0.0;
  for (std::size_t j = 0; j < idx.size(); ++j)
    sum += std::max(reference_kdist[idx[j]], dist[j]);
  const double mean = sum / static_cast<double>(idx.size());
  // sklearn adds 1e-10 to avoid dividing by zero on duplicate points.
  return 1.0 / (mean + 1e-10);
}

void AnomalyDetectionNode::Impl::precompute_lof() {
  reference_kdist.assign(static_cast<std::size_t>(n_reference), 0.0);
  reference_lrd.assign(static_cast<std::size_t>(n_reference), 0.0);
  if (n_reference == 0) return;

  std::vector<int> idx;
  std::vector<double> dist;
  // Pass 1: k-distance of every reference point (itself excluded).
  for (int i = 0; i < n_reference; ++i) {
    knn(reference_samples.data() + static_cast<std::size_t>(i) * n_features, i,
        idx, dist);
    reference_kdist[i] = dist.empty() ? 0.0 : dist.back();
  }
  // Pass 2: lrd of every reference point, which needs the k-distances above.
  for (int i = 0; i < n_reference; ++i) {
    reference_lrd[i] = local_reachability_density(
        reference_samples.data() + static_cast<std::size_t>(i) * n_features, i);
  }
}

double AnomalyDetectionNode::Impl::score_one(const double* x) const {
  switch (kind) {
    case AnomalyKind::IsolationForest: {
      if (trees.empty()) return 0.0;
      double total = 0.0;
      for (const IsolationTree& t : trees) {
        int node = 0;
        // leaf_value holds depth + c(n_node_samples), folded in by the
        // converter.
        while (node >= 0 && node < static_cast<int>(t.is_leaf.size()) &&
               !t.is_leaf[node]) {
          const int f = t.split_feature[node];
          const double v = (f >= 0 && f < n_features) ? x[f] : 0.0;
          const bool go_left = v <= t.split_threshold[node];
          const int off = t.children_offset[node];
          if (t.children_count[node] < 2 || off < 0 ||
              off + 1 >= static_cast<int>(t.children_index.size()))
            break;
          node = go_left ? t.children_index[off] : t.children_index[off + 1];
        }
        if (node >= 0 && node < static_cast<int>(t.leaf_value.size()))
          total += t.leaf_value[node];
      }
      const double mean_path = total / static_cast<double>(trees.size());
      const double norm = average_path_length(static_cast<double>(max_samples));
      if (norm <= 0.0) return 0.0;
      return -std::pow(2.0, -mean_path / norm);
    }

    case AnomalyKind::EllipticEnvelope: {
      // score_samples = -mahalanobis = -(x-loc)^T P (x-loc)
      double acc = 0.0;
      for (int i = 0; i < n_features; ++i) {
        double row = 0.0;
        for (int j = 0; j < n_features; ++j)
          row += precision[static_cast<std::size_t>(i) * n_features + j] *
                 (x[j] - location[j]);
        acc += (x[i] - location[i]) * row;
      }
      return -acc;
    }

    case AnomalyKind::LinearOneClassSVM: {
      double acc = intercept;
      for (int i = 0; i < n_features; ++i) acc += coefficients[i] * x[i];
      return acc;
    }

    case AnomalyKind::OneClassSVM: {
      double acc = intercept;
      for (int s = 0; s < n_sv; ++s) {
        const double* sv =
            support_vectors.data() + static_cast<std::size_t>(s) * n_features;
        double k = 0.0;
        switch (kernel) {
          case AnomalyKernel::Linear: {
            for (int i = 0; i < n_features; ++i) k += sv[i] * x[i];
            break;
          }
          case AnomalyKernel::RBF: {
            double d2 = 0.0;
            for (int i = 0; i < n_features; ++i) {
              const double diff = x[i] - sv[i];
              d2 += diff * diff;
            }
            k = std::exp(-gamma * d2);
            break;
          }
          case AnomalyKernel::Poly: {
            double dot = 0.0;
            for (int i = 0; i < n_features; ++i) dot += sv[i] * x[i];
            k = std::pow(gamma * dot + coef0, static_cast<double>(degree));
            break;
          }
          case AnomalyKernel::Sigmoid: {
            double dot = 0.0;
            for (int i = 0; i < n_features; ++i) dot += sv[i] * x[i];
            k = std::tanh(gamma * dot + coef0);
            break;
          }
        }
        acc += dual_coefficients[s] * k;
      }
      return acc;
    }

    case AnomalyKind::LocalOutlierFactor: {
      // score_samples = -LOF = -(mean lrd of neighbours / lrd(x))
      std::vector<int> idx;
      std::vector<double> dist;
      knn(x, -1, idx, dist);
      if (idx.empty()) return 0.0;
      const double lrd_x = local_reachability_density(x, -1);
      if (lrd_x <= 0.0) return 0.0;
      double sum = 0.0;
      for (int i : idx) sum += reference_lrd[i];
      const double mean_neighbour_lrd = sum / static_cast<double>(idx.size());
      return -(mean_neighbour_lrd / lrd_x);
    }
  }
  return 0.0;
}

AnomalyDetectionNode::AnomalyDetectionNode(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
AnomalyDetectionNode::~AnomalyDetectionNode() = default;

omle::rt::Status AnomalyDetectionNode::execute(ValueStore& vs,
                                               int n_rows) const {
  Tensor features =
      gather_slots(vs, in_names, n_rows, omle::rt::DataType::Float64);
  const double* xp = features.f64_ptr();
  const int nf = features.n_cols;
  if (nf != impl_->n_features)
    return {omle::rt::ErrorCode::InvalidArgument,
            "omle.ml/AnomalyDetection: input has " + std::to_string(nf) +
                " features, model expects " +
                std::to_string(impl_->n_features)};

  std::vector<double> score(static_cast<std::size_t>(n_rows));
  for (int r = 0; r < n_rows; ++r)
    score[r] = impl_->score_one(xp + static_cast<std::size_t>(r) * nf);

  for (std::size_t o = 0; o < out_names.size(); ++o) {
    const AnomalyOutput role =
        o < impl_->outputs.size() ? impl_->outputs[o] : AnomalyOutput::Score;
    Tensor t = Tensor::dense(omle::rt::DataType::Float64, n_rows, 1);
    double* dp = t.f64_ptr();
    for (int r = 0; r < n_rows; ++r) {
      switch (role) {
        case AnomalyOutput::Score:
          dp[r] = score[r];
          break;
        case AnomalyOutput::DecisionValue:
          dp[r] = score[r] - impl_->offset;
          break;
        case AnomalyOutput::Prediction:
          dp[r] = (score[r] - impl_->offset) >= 0.0 ? 1.0 : -1.0;
          break;
      }
    }
    vs.put(out_names[o], std::move(t));
  }
  return {};
}

// ── Factories ────────────────────────────────────────────────────────────────

static std::unique_ptr<AnomalyDetectionNode> finish(
    std::unique_ptr<AnomalyDetectionNode::Impl> impl) {
  return std::make_unique<AnomalyDetectionNode>(std::move(impl));
}

std::unique_ptr<AnomalyDetectionNode> make_isolation_forest_node(
    std::vector<IsolationTree> trees, int64_t max_samples, double offset,
    std::vector<AnomalyOutput> outputs) {
  auto impl = std::make_unique<AnomalyDetectionNode::Impl>();
  impl->kind = AnomalyKind::IsolationForest;
  impl->trees = std::move(trees);
  impl->max_samples = max_samples;
  impl->offset = offset;
  impl->outputs = std::move(outputs);
  int max_feature = -1;
  for (const auto& t : impl->trees)
    for (int f : t.split_feature) max_feature = std::max(max_feature, f);
  impl->n_features = max_feature + 1;
  return finish(std::move(impl));
}

std::unique_ptr<AnomalyDetectionNode> make_elliptic_envelope_node(
    std::vector<double> location, std::vector<double> precision, int n_features,
    double offset, std::vector<AnomalyOutput> outputs) {
  auto impl = std::make_unique<AnomalyDetectionNode::Impl>();
  impl->kind = AnomalyKind::EllipticEnvelope;
  impl->location = std::move(location);
  impl->precision = std::move(precision);
  impl->n_features = n_features;
  impl->offset = offset;
  impl->outputs = std::move(outputs);
  return finish(std::move(impl));
}

std::unique_ptr<AnomalyDetectionNode> make_linear_one_class_svm_node(
    std::vector<double> coefficients, double intercept, double offset,
    int n_features, std::vector<AnomalyOutput> outputs) {
  auto impl = std::make_unique<AnomalyDetectionNode::Impl>();
  impl->kind = AnomalyKind::LinearOneClassSVM;
  impl->coefficients = std::move(coefficients);
  impl->intercept = intercept;
  impl->n_features = n_features;
  impl->offset = offset;
  impl->outputs = std::move(outputs);
  return finish(std::move(impl));
}

std::unique_ptr<AnomalyDetectionNode> make_one_class_svm_node(
    AnomalyKernel kernel, std::vector<double> support_vectors,
    std::vector<double> dual_coefficients, double intercept, double gamma,
    int degree, double coef0, int n_sv, int n_features, double offset,
    std::vector<AnomalyOutput> outputs) {
  auto impl = std::make_unique<AnomalyDetectionNode::Impl>();
  impl->kind = AnomalyKind::OneClassSVM;
  impl->kernel = kernel;
  impl->support_vectors = std::move(support_vectors);
  impl->dual_coefficients = std::move(dual_coefficients);
  impl->intercept = intercept;
  impl->gamma = gamma;
  impl->degree = degree;
  impl->coef0 = coef0;
  impl->n_sv = n_sv;
  impl->n_features = n_features;
  impl->offset = offset;
  impl->outputs = std::move(outputs);
  return finish(std::move(impl));
}

std::unique_ptr<AnomalyDetectionNode> make_local_outlier_factor_node(
    std::vector<double> reference_samples, int n_reference, int n_features,
    int n_neighbors, const std::string& metric, double p, double offset,
    std::vector<AnomalyOutput> outputs) {
  auto impl = std::make_unique<AnomalyDetectionNode::Impl>();
  impl->kind = AnomalyKind::LocalOutlierFactor;
  impl->reference_samples = std::move(reference_samples);
  impl->n_reference = n_reference;
  impl->n_features = n_features;
  impl->n_neighbors = std::max(1, n_neighbors);
  impl->offset = offset;
  impl->outputs = std::move(outputs);
  // "minkowski" carries its order in p; euclidean and manhattan are the two
  // fixed-order spellings sklearn emits.
  if (metric == "manhattan" || metric == "l1" || metric == "cityblock")
    impl->p = 1.0;
  else if (metric == "euclidean" || metric == "l2")
    impl->p = 2.0;
  else
    impl->p = p;
  impl->precompute_lof();
  return finish(std::move(impl));
}

}  // namespace omle::rt::impl
