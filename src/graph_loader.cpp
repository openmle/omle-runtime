#include "graph_loader.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <thread>

#include "ast_types.h"
#include "expression_eval.h"
#include "graph_executor.h"
#include "model_schema.h"
#include "nodes/anomaly_detection_node.h"
#include "nodes/clustering_node.h"
#include "nodes/composite_node.h"
#include "nodes/linear_node.h"
#include "nodes/naive_bayes_node.h"
#include "nodes/neural_network_node.h"
#include "nodes/ohe_node.h"
#include "nodes/operator_node.h"
#include "nodes/svm_node.h"
#include "nodes/tree_ensemble_node.h"
#include "omle.pb.h"
#include "omle/status.h"
#include "operator_registry.h"
#include "thread_pool.h"

namespace omle::rt::impl {

namespace {

namespace P = omle;

// ---------------------------------------------------------------------------
// Helpers (duplicated from model_loader.cpp — keep them local)
// ---------------------------------------------------------------------------

using ConstantMap = std::unordered_map<std::string, const P::Tensor*>;

// Resolve a TensorValue to a const Tensor*.
// Returns nullptr if tv is unset, a SparseTensor, or the tensor_ref id is not
// in raw.
static const P::Tensor* unwrap_tv(const P::TensorValue& tv,
                                  const ConstantMap& raw) {
  switch (tv.value_case()) {
    case P::TensorValue::kTensor:
      return &tv.tensor();
    case P::TensorValue::kTensorRef: {
      auto it = raw.find(tv.tensor_ref().id());
      return (it != raw.end()) ? it->second : nullptr;
    }
    default:
      return nullptr;
  }
}

static void extract_csr_floats(const P::CSRMatrix& csr,
                               std::vector<float>& dst) {
  dst.clear();
  if (!csr.raw_data().empty()) {
    const auto& raw = csr.raw_data();
    const auto* p = reinterpret_cast<const float*>(raw.data());
    dst.assign(p, p + raw.size() / sizeof(float));
  } else if (csr.has_float32_data()) {
    const auto& vals = csr.float32_data().values();
    dst.assign(vals.begin(), vals.end());
  } else if (csr.has_float64_data()) {
    for (double v : csr.float64_data().values())
      dst.push_back(static_cast<float>(v));
  } else if (csr.has_int32_data()) {
    for (int32_t v : csr.int32_data().values())
      dst.push_back(static_cast<float>(v));
  } else if (csr.has_int64_data()) {
    for (int64_t v : csr.int64_data().values())
      dst.push_back(static_cast<float>(v));
  }
}

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
PredPtr convert_predicate(const P::Predicate& p);
static std::unique_ptr<GraphNode> convert_node_impl(const P::Node& node,
                                                    const ConstantStore& cstore,
                                                    const ConstantMap& raw);

static void extract_floats(const P::Tensor& tensor, std::vector<float>& dst) {
  dst.clear();
  if (!tensor.raw_data().empty()) {
    const auto& raw = tensor.raw_data();
    if (tensor.type().dtype() == P::FLOAT32) {
      const auto* p = reinterpret_cast<const float*>(raw.data());
      dst.assign(p, p + raw.size() / sizeof(float));
    } else if (tensor.type().dtype() == P::FLOAT64) {
      const auto* p = reinterpret_cast<const double*>(raw.data());
      for (size_t i = 0; i < raw.size() / sizeof(double); ++i)
        dst.push_back(static_cast<float>(p[i]));
    } else if (tensor.type().dtype() == P::INT32) {
      const auto* p = reinterpret_cast<const int32_t*>(raw.data());
      for (size_t i = 0; i < raw.size() / sizeof(int32_t); ++i)
        dst.push_back(static_cast<float>(p[i]));
    } else if (tensor.type().dtype() == P::INT64) {
      const auto* p = reinterpret_cast<const int64_t*>(raw.data());
      for (size_t i = 0; i < raw.size() / sizeof(int64_t); ++i)
        dst.push_back(static_cast<float>(p[i]));
    }
  } else if (tensor.has_float32_data()) {
    const auto& vals = tensor.float32_data().values();
    dst.assign(vals.begin(), vals.end());
  } else if (tensor.has_float64_data()) {
    for (double v : tensor.float64_data().values())
      dst.push_back(static_cast<float>(v));
  } else if (tensor.has_int32_data()) {
    for (int32_t v : tensor.int32_data().values())
      dst.push_back(static_cast<float>(v));
  } else if (tensor.has_int64_data()) {
    for (int64_t v : tensor.int64_data().values())
      dst.push_back(static_cast<float>(v));
  }
}

static void extract_doubles(const P::Tensor& tensor, std::vector<double>& dst) {
  dst.clear();
  if (!tensor.raw_data().empty()) {
    const auto& raw = tensor.raw_data();
    if (tensor.type().dtype() == P::FLOAT64) {
      const auto* p = reinterpret_cast<const double*>(raw.data());
      dst.assign(p, p + raw.size() / sizeof(double));
    } else if (tensor.type().dtype() == P::FLOAT32) {
      const auto* p = reinterpret_cast<const float*>(raw.data());
      for (size_t i = 0; i < raw.size() / sizeof(float); ++i)
        dst.push_back(static_cast<double>(p[i]));
    } else if (tensor.type().dtype() == P::INT32) {
      const auto* p = reinterpret_cast<const int32_t*>(raw.data());
      for (size_t i = 0; i < raw.size() / sizeof(int32_t); ++i)
        dst.push_back(static_cast<double>(p[i]));
    } else if (tensor.type().dtype() == P::INT64) {
      const auto* p = reinterpret_cast<const int64_t*>(raw.data());
      for (size_t i = 0; i < raw.size() / sizeof(int64_t); ++i)
        dst.push_back(static_cast<double>(p[i]));
    }
  } else if (tensor.has_float64_data()) {
    const auto& vals = tensor.float64_data().values();
    dst.assign(vals.begin(), vals.end());
  } else if (tensor.has_float32_data()) {
    for (float v : tensor.float32_data().values())
      dst.push_back(static_cast<double>(v));
  } else if (tensor.has_int32_data()) {
    for (int32_t v : tensor.int32_data().values())
      dst.push_back(static_cast<double>(v));
  } else if (tensor.has_int64_data()) {
    for (int64_t v : tensor.int64_data().values())
      dst.push_back(static_cast<double>(v));
  }
}

// Returns true when the tensor's stored dtype is float64.
static bool tensor_is_f64(const P::Tensor& tensor) {
  if (tensor.type().dtype() == P::FLOAT64) return true;
  if (!tensor.raw_data().empty()) return false;  // already handled above
  return tensor.has_float64_data() && !tensor.has_float32_data();
}

static bool tensor_is_f64(const P::TensorValue& tv, const ConstantMap& raw) {
  const P::Tensor* t = unwrap_tv(tv, raw);
  return t && tensor_is_f64(*t);
}

static std::vector<float> resolve_tensor(const P::TensorValue& tv,
                                         const ConstantMap& raw) {
  const P::Tensor* t = unwrap_tv(tv, raw);
  if (!t) throw std::runtime_error("graph_loader: TensorValue not resolvable");
  std::vector<float> v;
  extract_floats(*t, v);
  return v;
}

static PostTransform convert_pt(P::PostTransform pt) {
  switch (pt) {
    case P::IDENTITY:
      return PostTransform::Identity;
    case P::SIGMOID:
      return PostTransform::Sigmoid;
    case P::SOFTMAX:
      return PostTransform::Softmax;
    case P::EXP:
      return PostTransform::Exp;
    case P::LOGIT:
      return PostTransform::Logit;
    case P::PROBIT:
      return PostTransform::Probit;
    case P::CLOGLOG:
      return PostTransform::CLogLog;
    case P::CAUCHIT:
      return PostTransform::Cauchit;
    case P::LOGLOG:
      return PostTransform::LogLog;
    case P::SIGMOID_BINARY:
      return PostTransform::SigmoidBinary;
    default:
      return PostTransform::Identity;
  }
}

static ScalarVal scalar_val(const P::Scalar& s) {
  switch (s.value_case()) {
    case P::Scalar::kIntValue:
      return ScalarVal::from_int(s.int_value());
    case P::Scalar::kFloatValue:
      return ScalarVal::from_float(s.float_value());
    case P::Scalar::kDoubleValue:
      return ScalarVal::from_double(s.double_value());
    case P::Scalar::kStringValue:
      return ScalarVal::from_string(s.string_value());
    case P::Scalar::kBoolValue:
      return ScalarVal::from_bool(s.bool_value());
    default:
      return ScalarVal::from_double(0.0);
  }
}

// ---------------------------------------------------------------------------
// Expression conversion
// ---------------------------------------------------------------------------

ExprPtr convert_expression(const P::Expression& e) {
  switch (e.kind_case()) {
    case P::Expression::kLiteral:
      return make_literal(scalar_val(e.literal()));
    case P::Expression::kRef:
      return make_column(e.ref().value());
    case P::Expression::kApply: {
      std::vector<ExprPtr> args;
      for (const auto& arg : e.apply().arguments())
        args.push_back(convert_expression(arg));
      return make_apply(e.apply().function(), std::move(args));
    }
    default:
      return make_literal(ScalarVal::from_double(0.0));
  }
}

// ---------------------------------------------------------------------------
// Predicate conversion
// ---------------------------------------------------------------------------

static SimpleOp convert_simple_op(P::SimplePredicate::Operator op) {
  switch (op) {
    case P::SimplePredicate::LESS_THAN:
      return SimpleOp::LT;
    case P::SimplePredicate::LESS_OR_EQUAL:
      return SimpleOp::LE;
    case P::SimplePredicate::GREATER_THAN:
      return SimpleOp::GT;
    case P::SimplePredicate::GREATER_OR_EQUAL:
      return SimpleOp::GE;
    case P::SimplePredicate::EQUAL:
      return SimpleOp::EQ;
    case P::SimplePredicate::NOT_EQUAL:
      return SimpleOp::NE;
    case P::SimplePredicate::IS_MISSING:
      return SimpleOp::IsMissing;
    case P::SimplePredicate::IS_NOT_MISSING:
      return SimpleOp::IsNotMissing;
    default:
      return SimpleOp::EQ;
  }
}

PredPtr convert_predicate(const P::Predicate& p) {
  switch (p.kind_case()) {
    case P::Predicate::kTruePredicate:
      return make_true_pred();
    case P::Predicate::kFalsePredicate:
      return make_false_pred();
    case P::Predicate::kSimple: {
      const auto& s = p.simple();
      PredSimple ps;
      ps.column = s.column().value();
      ps.op = convert_simple_op(s.op());
      ps.value = scalar_val(s.value());
      return std::make_shared<Pred>(Pred{std::move(ps)});
    }
    case P::Predicate::kSimpleSet: {
      const auto& ss = p.simple_set();
      PredSet ps;
      ps.column = ss.column().value();
      ps.op = (ss.op() == P::SimpleSetPredicate::IN) ? SetOp::In : SetOp::NotIn;
      for (const auto& v : ss.values()) ps.values.push_back(scalar_val(v));
      return std::make_shared<Pred>(Pred{std::move(ps)});
    }
    case P::Predicate::kCompound: {
      const auto& cp = p.compound();
      PredCompound pc;
      switch (cp.op()) {
        case P::CompoundPredicate::AND:
          pc.op = BoolOp::And;
          break;
        case P::CompoundPredicate::OR:
          pc.op = BoolOp::Or;
          break;
        case P::CompoundPredicate::XOR:
          pc.op = BoolOp::Xor;
          break;
        case P::CompoundPredicate::SURROGATE:
          pc.op = BoolOp::Surrogate;
          break;
        default:
          pc.op = BoolOp::And;
          break;
      }
      for (const auto& child : cp.predicates())
        pc.children.push_back(convert_predicate(child));
      return std::make_shared<Pred>(Pred{std::move(pc)});
    }
    default:
      return make_true_pred();
  }
}

// ---------------------------------------------------------------------------
// Attribute conversion
// ---------------------------------------------------------------------------

static void register_tensor_in_cstore(const P::Tensor& t,
                                      const std::string& key,
                                      ConstantStore& cstore) {
  int64_t rows = 1, cols = 1;
  const auto& shape = t.type().shape();
  if (shape.size() >= 1) cols = shape[shape.size() - 1];
  if (shape.size() >= 2) rows = shape[shape.size() - 2];
  if (shape.size() == 1) {
    rows = 1;
    cols = shape[0];
  }
  std::shared_ptr<Tensor> rt;
  if (t.type().dtype() == P::STRING && t.has_string_data()) {
    std::vector<std::string> sv(t.string_data().values().begin(),
                                t.string_data().values().end());
    rt = std::make_shared<Tensor>(Tensor::strings(
        static_cast<int>(rows), static_cast<int>(cols), std::move(sv)));
  } else {
    std::vector<float> data;
    extract_floats(t, data);
    rt = std::make_shared<Tensor>(Tensor::from_floats(
        static_cast<int>(rows), static_cast<int>(cols), std::move(data)));
  }
  cstore[key] = rt;
}

static AttributeMap convert_attributes(
    const google::protobuf::RepeatedPtrField<P::Attribute>& attrs,
    ConstantStore& cstore, const std::string& node_name) {
  AttributeMap m;
  for (const auto& a : attrs) {
    AttrVal val;
    switch (a.value_case()) {
      case P::Attribute::kI:
        val.kind = AttrVal::Kind::Int;
        val.i = a.i();
        break;
      case P::Attribute::kF32:
        val.kind = AttrVal::Kind::Float;
        val.f = static_cast<double>(a.f32());
        break;
      case P::Attribute::kF64:
        val.kind = AttrVal::Kind::Float;
        val.f = a.f64();  // already double — no truncation
        break;
      case P::Attribute::kS:
        val.kind = AttrVal::Kind::String;
        val.s = a.s();
        break;
      case P::Attribute::kB:
        val.kind = AttrVal::Kind::Bool;
        val.b = a.b();
        break;
      case P::Attribute::kInts:
        val.kind = AttrVal::Kind::Ints;
        for (auto x : a.ints().values()) val.ints.push_back(x);
        break;
      case P::Attribute::kFloat32S:
        val.kind = AttrVal::Kind::Floats;
        for (auto x : a.float32s().values())
          val.floats.push_back(static_cast<double>(x));
        break;
      case P::Attribute::kFloat64S:
        val.kind = AttrVal::Kind::Floats;
        for (double x : a.float64s().values())
          val.floats.push_back(x);  // no truncation
        break;
      case P::Attribute::kStrings:
        val.kind = AttrVal::Kind::Strings;
        for (const auto& x : a.strings().values()) val.strings.push_back(x);
        break;
      case P::Attribute::kBools:
        val.kind = AttrVal::Kind::Ints;
        for (auto x : a.bools().values()) val.ints.push_back(x ? 1 : 0);
        break;
      case P::Attribute::kTensorRef:
        val.kind = AttrVal::Kind::Tensor;
        val.tensor_name = a.tensor_ref().id();
        break;
      case P::Attribute::kTensor: {
        // Inline tensor: register in cstore with a unique key, then reference
        // by name.
        std::string key = node_name + "::" + a.name();
        register_tensor_in_cstore(a.tensor(), key, cstore);
        val.kind = AttrVal::Kind::Tensor;
        val.tensor_name = std::move(key);
        break;
      }
      case P::Attribute::kExpr:
        val.kind = AttrVal::Kind::Expr;
        val.expr = convert_expression(a.expr());
        break;
      case P::Attribute::kPredicate:
        val.kind = AttrVal::Kind::Pred;
        val.pred = convert_predicate(a.predicate());
        break;
      default:
        val.kind = AttrVal::Kind::Int;
        break;
    }
    m[a.name()] = std::move(val);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Node-body converters
// ---------------------------------------------------------------------------

static std::unique_ptr<GraphNode> convert_naive_bayes(const P::Node& node,
                                                      const ConstantMap& raw) {
  const auto& nb = node.naive_bayes();

  bool f64 = tensor_is_f64(nb.class_log_priors(), raw);

  NaiveBayesVariant variant = NaiveBayesVariant::Gaussian;
  std::optional<double> variance_epsilon;
  std::optional<double> binarize_threshold;

  switch (nb.implementation_case()) {
    case P::NaiveBayes::kGaussian:
      variant = NaiveBayesVariant::Gaussian;
      if (nb.gaussian().has_variance_epsilon())
        variance_epsilon =
            scalar_val(nb.gaussian().variance_epsilon()).as_double();
      break;
    case P::NaiveBayes::kMultinomial:
      variant = NaiveBayesVariant::Multinomial;
      break;
    case P::NaiveBayes::kBernoulli:
      variant = NaiveBayesVariant::Bernoulli;
      if (nb.bernoulli().has_binarize_threshold())
        binarize_threshold =
            scalar_val(nb.bernoulli().binarize_threshold()).as_double();
      break;
    case P::NaiveBayes::kCategorical:
      variant = NaiveBayesVariant::Categorical;
      break;
    default:
      break;
  }

  std::unique_ptr<NaiveBayesNode> gn;
  if (f64) {
    std::vector<double> class_log_priors, means, variances, feature_log_prob,
        category_log_prob;
    std::vector<int32_t> category_offset, category_count;
    if (const P::Tensor* t = unwrap_tv(nb.class_log_priors(), raw))
      extract_doubles(*t, class_log_priors);
    const int n_classes = static_cast<int>(class_log_priors.size());
    switch (nb.implementation_case()) {
      case P::NaiveBayes::kGaussian: {
        if (const P::Tensor* t = unwrap_tv(nb.gaussian().means(), raw))
          extract_doubles(*t, means);
        if (const P::Tensor* t = unwrap_tv(nb.gaussian().variances(), raw))
          extract_doubles(*t, variances);
        break;
      }
      case P::NaiveBayes::kMultinomial: {
        if (const P::Tensor* t =
                unwrap_tv(nb.multinomial().feature_log_prob(), raw))
          extract_doubles(*t, feature_log_prob);
        break;
      }
      case P::NaiveBayes::kBernoulli: {
        if (const P::Tensor* t =
                unwrap_tv(nb.bernoulli().feature_log_prob(), raw))
          extract_doubles(*t, feature_log_prob);
        break;
      }
      case P::NaiveBayes::kCategorical: {
        if (const P::Tensor* t =
                unwrap_tv(nb.categorical().category_log_prob(), raw))
          extract_doubles(*t, category_log_prob);
        category_offset.assign(nb.categorical().category_offset().begin(),
                               nb.categorical().category_offset().end());
        category_count.assign(nb.categorical().category_count().begin(),
                              nb.categorical().category_count().end());
        break;
      }
      default:
        break;
    }
    int n_features = 0;
    if (!means.empty() && n_classes > 0)
      n_features = static_cast<int>(means.size()) / n_classes;
    else if (!feature_log_prob.empty() && n_classes > 0)
      n_features = static_cast<int>(feature_log_prob.size()) / n_classes;
    else if (!category_count.empty())
      n_features = static_cast<int>(category_count.size());
    gn = make_naive_bayes_node_f64(
        variant, std::move(class_log_priors), std::move(means),
        std::move(variances), variance_epsilon, std::move(feature_log_prob),
        binarize_threshold, std::move(category_log_prob),
        std::move(category_offset), std::move(category_count), n_classes,
        n_features);
  } else {
    std::vector<float> class_log_priors, means, variances, feature_log_prob,
        category_log_prob;
    std::vector<int32_t> category_offset, category_count;
    class_log_priors = resolve_tensor(nb.class_log_priors(), raw);
    const int n_classes = static_cast<int>(class_log_priors.size());
    switch (nb.implementation_case()) {
      case P::NaiveBayes::kGaussian:
        means = resolve_tensor(nb.gaussian().means(), raw);
        variances = resolve_tensor(nb.gaussian().variances(), raw);
        break;
      case P::NaiveBayes::kMultinomial:
        feature_log_prob =
            resolve_tensor(nb.multinomial().feature_log_prob(), raw);
        break;
      case P::NaiveBayes::kBernoulli:
        feature_log_prob =
            resolve_tensor(nb.bernoulli().feature_log_prob(), raw);
        break;
      case P::NaiveBayes::kCategorical: {
        category_log_prob =
            resolve_tensor(nb.categorical().category_log_prob(), raw);
        category_offset.assign(nb.categorical().category_offset().begin(),
                               nb.categorical().category_offset().end());
        category_count.assign(nb.categorical().category_count().begin(),
                              nb.categorical().category_count().end());
        break;
      }
      default:
        break;
    }
    int n_features = 0;
    if (!means.empty() && n_classes > 0)
      n_features = static_cast<int>(means.size()) / n_classes;
    else if (!feature_log_prob.empty() && n_classes > 0)
      n_features = static_cast<int>(feature_log_prob.size()) / n_classes;
    else if (!category_count.empty())
      n_features = static_cast<int>(category_count.size());
    gn = make_naive_bayes_node(
        variant, std::move(class_log_priors), std::move(means),
        std::move(variances), variance_epsilon, std::move(feature_log_prob),
        binarize_threshold, std::move(category_log_prob),
        std::move(category_offset), std::move(category_count), n_classes,
        n_features);
  }

  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

static std::unique_ptr<GraphNode> convert_clustering(const P::Node& node,
                                                     const ConstantMap& raw) {
  const auto& cl = node.clustering();

  ClusterKind kind = ClusterKind::Prototype;
  ClusterDistance distance = ClusterDistance::Euclidean;
  ClusterCovType cov_type = ClusterCovType::Full;
  int n_clusters = 0;
  int n_features = 0;
  bool f64 = false;
  std::vector<std::string> labels;

  if (cl.has_prototype()) {
    kind = ClusterKind::Prototype;
    if (const P::Tensor* t = unwrap_tv(cl.prototype().centers(), raw)) {
      f64 = tensor_is_f64(*t);
      const auto& shape = t->type().shape();
      if (shape.size() == 2) {
        n_clusters = static_cast<int>(shape[0]);
        n_features = static_cast<int>(shape[1]);
      }
    }
    switch (cl.prototype().distance_measure()) {
      case P::PrototypeClustering::SQUARED_EUCLIDEAN:
        distance = ClusterDistance::SquaredEuclidean;
        break;
      case P::PrototypeClustering::MANHATTAN:
        distance = ClusterDistance::Manhattan;
        break;
      case P::PrototypeClustering::COSINE:
        distance = ClusterDistance::Cosine;
        break;
      default:
        distance = ClusterDistance::Euclidean;
        break;
    }
    for (const auto& lbl : cl.prototype().cluster_labels()) {
      if (lbl.value_case() == P::Scalar::kStringValue)
        labels.push_back(lbl.string_value());
      else
        labels.push_back(std::to_string(labels.size()));
    }
    if (n_clusters == 0) n_clusters = static_cast<int>(labels.size());
  } else if (cl.has_gaussian_mixture()) {
    kind = ClusterKind::GaussianMixture;
    switch (cl.gaussian_mixture().covariance_type()) {
      case P::GaussianMixtureClustering::DIAGONAL:
        cov_type = ClusterCovType::Diagonal;
        break;
      case P::GaussianMixtureClustering::SPHERICAL:
        cov_type = ClusterCovType::Spherical;
        break;
      default:
        cov_type = ClusterCovType::Full;
        break;
    }
    if (const P::Tensor* t = unwrap_tv(cl.gaussian_mixture().means(), raw)) {
      f64 = tensor_is_f64(*t);
      const auto& shape = t->type().shape();
      if (shape.size() == 2) {
        n_clusters = static_cast<int>(shape[0]);
        n_features = static_cast<int>(shape[1]);
      }
    }
    for (const auto& lbl : cl.gaussian_mixture().component_labels()) {
      if (lbl.value_case() == P::Scalar::kStringValue)
        labels.push_back(lbl.string_value());
      else
        labels.push_back(std::to_string(labels.size()));
    }
  }

  std::unique_ptr<ClusteringNode> gn;
  if (f64) {
    std::vector<double> centers, weights, means, covariances;
    if (cl.has_prototype()) {
      if (const P::Tensor* t = unwrap_tv(cl.prototype().centers(), raw)) {
        extract_doubles(*t, centers);
        if (n_clusters > 0 && n_features == 0 && !centers.empty())
          n_features = static_cast<int>(centers.size()) / n_clusters;
      }
    } else if (cl.has_gaussian_mixture()) {
      if (const P::Tensor* t =
              unwrap_tv(cl.gaussian_mixture().weights(), raw)) {
        extract_doubles(*t, weights);
        if (n_clusters == 0) n_clusters = static_cast<int>(weights.size());
      }
      if (const P::Tensor* t = unwrap_tv(cl.gaussian_mixture().means(), raw)) {
        extract_doubles(*t, means);
        if (n_clusters > 0 && n_features == 0 && !means.empty())
          n_features = static_cast<int>(means.size()) / n_clusters;
      }
      if (const P::Tensor* t =
              unwrap_tv(cl.gaussian_mixture().covariances(), raw))
        extract_doubles(*t, covariances);
    }
    gn = make_clustering_node_f64(kind, distance, cov_type, std::move(centers),
                                  std::move(weights), std::move(means),
                                  std::move(covariances), std::move(labels),
                                  n_clusters, n_features);
  } else {
    std::vector<float> centers, weights, means, covariances;
    if (cl.has_prototype()) {
      centers = resolve_tensor(cl.prototype().centers(), raw);
      if (n_clusters > 0 && n_features == 0 && !centers.empty())
        n_features = static_cast<int>(centers.size()) / n_clusters;
    } else if (cl.has_gaussian_mixture()) {
      weights = resolve_tensor(cl.gaussian_mixture().weights(), raw);
      means = resolve_tensor(cl.gaussian_mixture().means(), raw);
      covariances = resolve_tensor(cl.gaussian_mixture().covariances(), raw);
      if (n_clusters == 0) n_clusters = static_cast<int>(weights.size());
      if (n_clusters > 0 && n_features == 0 && !means.empty())
        n_features = static_cast<int>(means.size()) / n_clusters;
    }
    gn = make_clustering_node(kind, distance, cov_type, std::move(centers),
                              std::move(weights), std::move(means),
                              std::move(covariances), std::move(labels),
                              n_clusters, n_features);
  }

  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

// Forward declaration for recursive composite conversion.
static std::unique_ptr<GraphNode> convert_node_impl(const P::Node& node,
                                                    const ConstantStore& cstore,
                                                    const ConstantMap& raw);

static std::unique_ptr<GraphNode> convert_composite(const P::Node& node,
                                                    const ConstantStore& cstore,
                                                    const ConstantMap& raw) {
  const auto& comp = node.composite();
  auto cn = std::make_unique<CompositeNode>();

  for (const auto& alias : comp.input_aliases())
    cn->input_aliases.emplace_back(alias.from_name(), alias.to_name());
  for (const auto& alias : comp.output_aliases())
    cn->output_aliases.emplace_back(alias.from_name(), alias.to_name());

  for (const auto& child : comp.nodes())
    cn->nodes.push_back(convert_node_impl(child, cstore, raw));

  for (const auto& i : node.inputs()) cn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) cn->out_names.push_back(o.name());
  cn->node_name = node.name();
  return cn;
}

static SplitOp convert_split_op_gl(P::Tree::SplitOp op) {
  switch (op) {
    case P::Tree::LESS_THAN:
      return SplitOp::LessThan;
    case P::Tree::LESS_OR_EQUAL:
      return SplitOp::LessOrEqual;
    case P::Tree::GREATER_THAN:
      return SplitOp::GreaterThan;
    case P::Tree::GREATER_OR_EQUAL:
      return SplitOp::GreaterOrEqual;
    case P::Tree::EQUAL:
      return SplitOp::Equal;
    case P::Tree::NOT_EQUAL:
      return SplitOp::NotEqual;
    case P::Tree::IN_SET:
      return SplitOp::InSet;
    case P::Tree::NOT_IN_SET:
      return SplitOp::NotInSet;
    case P::Tree::IS_MISSING:
      return SplitOp::IsMissing;
    default:
      return SplitOp::LessThan;
  }
}

// Build a FlatTree<double> from a proto Tree message.
// Always stores threshold and leaf_value as double (factory narrows to float if
// needed).
static FlatTree<double> convert_flat_tree_input(const P::Tree& t,
                                                const ConstantMap& raw) {
  const int n = t.num_nodes();
  FlatTree<double> ft;
  ft.n_nodes = n;
  ft.leaf_width = std::max(1, t.leaf_width());
  ft.feature.resize(n, -1);
  ft.threshold.resize(n, 0.0);
  ft.left_child.resize(n, -1);
  ft.right_child.resize(n, -1);
  ft.default_child.resize(n, -1);
  ft.split_op.resize(n, SplitOp::LessThan);

  auto is_leaf = [&](int i) {
    return i < t.node_kind_size() && t.node_kind(i) == P::Tree::LEAF;
  };
  std::vector<double> split_thresholds;
  if (const P::Tensor* st = unwrap_tv(t.split_threshold(), raw))
    extract_doubles(*st, split_thresholds);
  for (int i = 0; i < n; ++i) {
    if (is_leaf(i)) continue;
    ft.feature[i] = (i < t.split_feature_size()) ? t.split_feature(i) : -1;
    ft.threshold[i] =
        (i < (int)split_thresholds.size()) ? split_thresholds[i] : 0.0;
    ft.split_op[i] = (i < t.split_op_size())
                         ? convert_split_op_gl(t.split_op(i))
                         : SplitOp::LessThan;
    int off = (i < t.children_offset_size()) ? t.children_offset(i) : -1;
    int cnt = (i < t.children_count_size()) ? t.children_count(i) : 0;
    if (off >= 0 && cnt >= 1 && off < t.children_index_size())
      ft.left_child[i] = t.children_index(off);
    if (off >= 0 && cnt >= 2 && off + 1 < t.children_index_size())
      ft.right_child[i] = t.children_index(off + 1);
    ft.default_child[i] =
        (i < t.default_child_size()) ? t.default_child(i) : ft.right_child[i];
  }
  if (ft.leaf_width == 1) {
    std::vector<double> leaf_values;
    if (const P::Tensor* lv = unwrap_tv(t.leaf_value(), raw))
      extract_doubles(*lv, leaf_values);
    ft.leaf_value.resize(n, 0.0);
    for (int i = 0; i < std::min(n, (int)leaf_values.size()); ++i)
      ft.leaf_value[i] = leaf_values[i];
  } else {
    if (const P::Tensor* lvec = unwrap_tv(t.leaf_vector(), raw))
      extract_doubles(*lvec, ft.leaf_vector);
    ft.leaf_vector_index.resize(n, -1);
    for (int i = 0; i < std::min(n, t.leaf_vector_index_size()); ++i)
      ft.leaf_vector_index[i] = t.leaf_vector_index(i);
  }
  ft.cat_offset.resize(n, 0);
  ft.cat_count.resize(n, 0);
  for (int i = 0; i < std::min(n, t.category_set_offset_size()); ++i)
    ft.cat_offset[i] = t.category_set_offset(i);
  for (int i = 0; i < std::min(n, t.category_set_count_size()); ++i)
    ft.cat_count[i] = t.category_set_count(i);
  ft.category_set.assign(t.category_set().begin(), t.category_set().end());

  for (const auto& cp : t.complex_predicates())
    ft.complex_preds[cp.node_index()] = convert_predicate(cp.predicate());
  ft.has_complex = !ft.complex_preds.empty();

  ft.all_less_than = true;
  ft.all_less_equal = true;
  ft.default_right = true;
  for (int i = 0; i < n; ++i) {
    if (ft.feature[i] < 0) continue;
    if (ft.has_complex && ft.complex_preds.count(i)) {
      ft.all_less_than = false;
      ft.all_less_equal = false;
      continue;
    }
    if (ft.split_op[i] == SplitOp::LessThan)
      ft.all_less_equal = false;
    else if (ft.split_op[i] == SplitOp::LessOrEqual)
      ft.all_less_than = false;
    else {
      ft.all_less_than = false;
      ft.all_less_equal = false;
    }
    if (ft.default_child[i] != ft.right_child[i]) ft.default_right = false;
    if (!ft.all_less_than && !ft.all_less_equal && !ft.default_right) break;
  }
  return ft;
}

// Detect whether any tree in the ensemble stores float64 thresholds/leaves.
static bool tree_ensemble_is_f64(const P::TreeEnsemble& te,
                                 const ConstantMap& raw) {
  for (const auto& t : te.trees()) {
    auto check = [&](const P::TensorValue& tv) {
      const P::Tensor* tp = unwrap_tv(tv, raw);
      return tp && tensor_is_f64(*tp);
    };
    if (check(t.split_threshold())) return true;
    if (check(t.leaf_value())) return true;
    if (check(t.leaf_vector())) return true;
  }
  return false;
}

// Wrap a standalone Tree as a single-tree sum ensemble (no post-transform,
// base=0).
static std::unique_ptr<GraphNode> convert_tree(const P::Node& node,
                                               const ConstantMap& raw) {
  TreeEnsembleModel<double> inp;
  inp.aggregation = Aggregation::Sum;
  inp.post_transform = PostTransform::Identity;
  inp.trees.push_back(convert_flat_tree_input(node.tree(), raw));
  inp.n_trees = 1;
  inp.n_outputs = (inp.trees[0].leaf_width > 1) ? inp.trees[0].leaf_width : 1;
  inp.n_features = 0;

  auto tree_tv_is_f64 = [&](const P::TensorValue& tv) {
    const P::Tensor* tp = unwrap_tv(tv, raw);
    return tp && tensor_is_f64(*tp);
  };
  const bool f64 = tree_tv_is_f64(node.tree().split_threshold()) ||
                   tree_tv_is_f64(node.tree().leaf_value());
  auto impl = f64 ? make_tree_ensemble_impl_f64(inp)
                  : make_tree_ensemble_impl(narrow_tree_ensemble_model(inp));
  auto gn = std::make_unique<TreeEnsembleNode>(std::move(impl));
  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

static std::unique_ptr<GraphNode> convert_tree_ensemble(
    const P::Node& node, const ConstantMap& raw) {
  const auto& te = node.tree_ensemble();
  TreeEnsembleModel<double> inp;
  inp.aggregation = [&] {
    switch (te.aggregation()) {
      case P::TreeEnsemble::SUM:
        return Aggregation::Sum;
      case P::TreeEnsemble::AVERAGE:
        return Aggregation::Average;
      case P::TreeEnsemble::WEIGHTED_SUM:
        return Aggregation::WeightedSum;
      case P::TreeEnsemble::WEIGHTED_AVERAGE:
        return Aggregation::WeightedAvg;
      case P::TreeEnsemble::MAJORITY_VOTE:
        return Aggregation::MajorityVote;
      case P::TreeEnsemble::SOFT_VOTE:
        return Aggregation::SoftVote;
      case P::TreeEnsemble::MIN:
        return Aggregation::Min;
      case P::TreeEnsemble::MAX:
        return Aggregation::Max;
      default:
        return Aggregation::Sum;
    }
  }();
  inp.post_transform = convert_pt(te.post_transform());
  if (te.has_base_score()) {
    inp.base_score = scalar_val(te.base_score()).as_double();
    inp.has_base_score = true;
  }
  if (te.has_tree_weights()) {
    if (const P::Tensor* tw = unwrap_tv(te.tree_weights(), raw))
      extract_doubles(*tw, inp.tree_weights);
  }
  inp.tree_group.assign(te.tree_group().begin(), te.tree_group().end());
  inp.trees.reserve(te.trees_size());
  for (const auto& t : te.trees())
    inp.trees.push_back(convert_flat_tree_input(t, raw));
  inp.n_trees = static_cast<int>(inp.trees.size());
  if (!inp.tree_group.empty())
    inp.n_outputs =
        *std::max_element(inp.tree_group.begin(), inp.tree_group.end()) + 1;
  else if (!inp.trees.empty() && inp.trees[0].leaf_width > 1)
    inp.n_outputs = inp.trees[0].leaf_width;
  else
    inp.n_outputs = 1;
  if (inp.post_transform == PostTransform::SigmoidBinary) inp.n_outputs = 2;
  inp.n_features = 0;

  const bool f64 = tree_ensemble_is_f64(te, raw);
  auto impl = f64 ? make_tree_ensemble_impl_f64(inp)
                  : make_tree_ensemble_impl(narrow_tree_ensemble_model(inp));
  auto gn = std::make_unique<TreeEnsembleNode>(std::move(impl));
  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

static std::unique_ptr<GraphNode> convert_linear(const P::Node& node,
                                                 const ConstantMap& raw) {
  const auto& lin = node.linear();
  int n_features = 0, n_outputs = 1;

  const P::Tensor* coef_ptr = unwrap_tv(lin.coefficients(), raw);
  if (coef_ptr) {
    const auto& shape = coef_ptr->type().shape();
    if (shape.size() == 1) {
      n_outputs = 1;
      n_features = static_cast<int>(shape[0]);
    } else if (shape.size() == 2) {
      n_outputs = static_cast<int>(shape[0]);
      n_features = static_cast<int>(shape[1]);
    }
  }
  PostTransform pt = convert_pt(lin.post_transform());

  // Choose float64 when the coefficient tensor is stored as float64.
  const bool f64 = coef_ptr && tensor_is_f64(*coef_ptr);

  std::unique_ptr<LinearNode> gn;
  if (f64) {
    std::vector<double> coeff, intercept;
    extract_doubles(*coef_ptr, coeff);
    if (lin.has_intercept()) {
      if (const P::Tensor* t = unwrap_tv(lin.intercept(), raw))
        extract_doubles(*t, intercept);
    }
    gn = make_linear_node_f64(std::move(coeff), std::move(intercept), pt,
                              n_features, n_outputs);
  } else {
    std::vector<float> coeff, intercept;
    if (coef_ptr) extract_floats(*coef_ptr, coeff);
    if (lin.has_intercept()) {
      if (const P::Tensor* t = unwrap_tv(lin.intercept(), raw))
        extract_floats(*t, intercept);
    }
    gn = make_linear_node(std::move(coeff), std::move(intercept), pt,
                          n_features, n_outputs);
  }

  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

static std::unique_ptr<GraphNode> convert_svm(const P::Node& node,
                                              const ConstantMap& raw) {
  const auto& ps = node.svm();
  PostTransform pt = convert_pt(ps.post_transform());

  SVMKind kind = SVMKind::Linear;
  KernelType kernel_type = KernelType::Linear;
  double gamma = 1.0;
  int degree = 3;
  double coef0 = 0.0;
  int n_sv = 0;
  int n_features = 0;
  int n_outputs = 1;
  bool f64 = false;

  if (ps.has_linear()) {
    kind = SVMKind::Linear;
    if (const P::Tensor* t = unwrap_tv(ps.linear().coefficients(), raw)) {
      f64 = tensor_is_f64(*t);
      const auto& shape = t->type().shape();
      if (shape.size() == 1) {
        n_outputs = 1;
        n_features = static_cast<int>(shape[0]);
      } else if (shape.size() == 2) {
        n_outputs = static_cast<int>(shape[0]);
        n_features = static_cast<int>(shape[1]);
      }
    }
  } else if (ps.has_kernel()) {
    kind = SVMKind::Kernel;
    const auto& pk = ps.kernel();
    switch (pk.kernel_type()) {
      case P::KernelSVM::POLY:
        kernel_type = KernelType::Poly;
        break;
      case P::KernelSVM::RBF:
        kernel_type = KernelType::RBF;
        break;
      case P::KernelSVM::SIGMOID:
        kernel_type = KernelType::Sigmoid;
        break;
      default:
        kernel_type = KernelType::Linear;
        break;
    }
    if (pk.has_gamma()) gamma = scalar_val(pk.gamma()).as_double();
    if (pk.has_degree()) degree = pk.degree();
    if (pk.has_coef0()) coef0 = scalar_val(pk.coef0()).as_double();
    if (const P::Tensor* t = unwrap_tv(pk.support_vectors(), raw)) {
      f64 = tensor_is_f64(*t);
      const auto& shape = t->type().shape();
      if (shape.size() == 2) {
        n_sv = static_cast<int>(shape[0]);
        n_features = static_cast<int>(shape[1]);
      }
    }
    if (const P::Tensor* t = unwrap_tv(pk.dual_coefficients(), raw)) {
      const auto& shape = t->type().shape();
      if (shape.size() == 1)
        n_outputs = 1;
      else if (shape.size() == 2)
        n_outputs = static_cast<int>(shape[0]);
    }
  }

  // Extract per-class SV counts for multiclass OVO.
  std::vector<int> n_support;
  if (ps.has_kernel()) {
    const auto& pk = ps.kernel();
    for (int i = 0; i < pk.n_support_size(); ++i)
      n_support.push_back(static_cast<int>(pk.n_support(i)));
  }

  std::unique_ptr<SVMNode> gn;
  if (f64) {
    std::vector<double> coefficients, intercept, support_vectors,
        dual_coefficients;
    std::vector<double> prob_a, prob_b;
    if (ps.has_linear()) {
      if (const P::Tensor* t = unwrap_tv(ps.linear().coefficients(), raw))
        extract_doubles(*t, coefficients);
      if (ps.linear().has_intercept()) {
        if (const P::Tensor* t = unwrap_tv(ps.linear().intercept(), raw))
          extract_doubles(*t, intercept);
      }
    } else if (ps.has_kernel()) {
      const auto& pk = ps.kernel();
      if (const P::Tensor* t = unwrap_tv(pk.support_vectors(), raw))
        extract_doubles(*t, support_vectors);
      if (const P::Tensor* t = unwrap_tv(pk.dual_coefficients(), raw))
        extract_doubles(*t, dual_coefficients);
      if (pk.has_intercept()) {
        if (const P::Tensor* t = unwrap_tv(pk.intercept(), raw))
          extract_doubles(*t, intercept);
      }
      if (pk.has_prob_a()) {
        if (const P::Tensor* t = unwrap_tv(pk.prob_a(), raw))
          extract_doubles(*t, prob_a);
      }
      if (pk.has_prob_b()) {
        if (const P::Tensor* t = unwrap_tv(pk.prob_b(), raw))
          extract_doubles(*t, prob_b);
      }
    }
    gn = make_svm_node_f64(kind, kernel_type, std::move(coefficients),
                           std::move(intercept), std::move(support_vectors),
                           std::move(dual_coefficients), gamma, degree, coef0,
                           n_sv, n_features, n_outputs, pt, std::move(prob_a),
                           std::move(prob_b), n_support);
  } else {
    std::vector<float> coefficients, intercept, support_vectors,
        dual_coefficients;
    std::vector<float> prob_a, prob_b;
    if (ps.has_linear()) {
      coefficients = resolve_tensor(ps.linear().coefficients(), raw);
      if (ps.linear().has_intercept()) {
        if (const P::Tensor* t = unwrap_tv(ps.linear().intercept(), raw))
          extract_floats(*t, intercept);
      }
    } else if (ps.has_kernel()) {
      const auto& pk = ps.kernel();
      support_vectors = resolve_tensor(pk.support_vectors(), raw);
      dual_coefficients = resolve_tensor(pk.dual_coefficients(), raw);
      if (pk.has_intercept()) {
        if (const P::Tensor* t = unwrap_tv(pk.intercept(), raw))
          extract_floats(*t, intercept);
      }
      if (pk.has_prob_a()) {
        if (const P::Tensor* t = unwrap_tv(pk.prob_a(), raw))
          extract_floats(*t, prob_a);
      }
      if (pk.has_prob_b()) {
        if (const P::Tensor* t = unwrap_tv(pk.prob_b(), raw))
          extract_floats(*t, prob_b);
      }
    }
    gn = make_svm_node(kind, kernel_type, std::move(coefficients),
                       std::move(intercept), std::move(support_vectors),
                       std::move(dual_coefficients), gamma, degree, coef0, n_sv,
                       n_features, n_outputs, pt, std::move(prob_a),
                       std::move(prob_b));
  }

  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

static std::unique_ptr<GraphNode> convert_neural_network(
    const P::Node& node, const ConstantMap& raw) {
  const auto& pnn = node.neural_network();
  int n_features = 0;
  int n_outputs = 0;

  bool f64 = false;
  if (pnn.layers_size() > 0) {
    f64 = tensor_is_f64(pnn.layers(0).weights(), raw);
  }

  auto act_from_proto = [](int a) -> NNActivation {
    switch (a) {
      case P::NeuralNetwork::LOGISTIC:
        return NNActivation::Logistic;
      case P::NeuralNetwork::TANH:
        return NNActivation::Tanh;
      case P::NeuralNetwork::RELU:
        return NNActivation::Relu;
      case P::NeuralNetwork::SOFTMAX:
        return NNActivation::Softmax;
      default:
        return NNActivation::Identity;
    }
  };

  std::unique_ptr<NeuralNetworkNode> gn;
  if (f64) {
    std::vector<NNLayerDataF64> layers;
    for (const auto& pl : pnn.layers()) {
      NNLayerDataF64 layer;
      layer.activation = act_from_proto(pl.activation());
      if (const P::Tensor* t = unwrap_tv(pl.weights(), raw)) {
        extract_doubles(*t, layer.weights);
        const auto& shape = t->type().shape();
        if (shape.size() == 2) {
          layer.out_features = static_cast<int>(shape[0]);
          layer.in_features = static_cast<int>(shape[1]);
        }
      }
      if (pl.has_bias()) {
        if (const P::Tensor* t = unwrap_tv(pl.bias(), raw))
          extract_doubles(*t, layer.bias);
      }
      layers.push_back(std::move(layer));
    }
    if (!layers.empty()) {
      n_features = layers.front().in_features;
      n_outputs = layers.back().out_features;
    }
    gn = make_neural_network_node_f64(std::move(layers), n_features, n_outputs);
  } else {
    std::vector<NNLayerDataF32> layers;
    for (const auto& pl : pnn.layers()) {
      NNLayerDataF32 layer;
      layer.activation = act_from_proto(pl.activation());
      if (const P::Tensor* t = unwrap_tv(pl.weights(), raw)) {
        extract_floats(*t, layer.weights);
        const auto& shape = t->type().shape();
        if (shape.size() == 2) {
          layer.out_features = static_cast<int>(shape[0]);
          layer.in_features = static_cast<int>(shape[1]);
        }
      }
      if (pl.has_bias()) {
        if (const P::Tensor* t = unwrap_tv(pl.bias(), raw))
          extract_floats(*t, layer.bias);
      }
      layers.push_back(std::move(layer));
    }
    if (!layers.empty()) {
      n_features = layers.front().in_features;
      n_outputs = layers.back().out_features;
    }
    gn = make_neural_network_node(std::move(layers), n_features, n_outputs);
  }

  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

// Read one isolation-forest Tree message into the flattened runtime form.
static IsolationTree convert_isolation_tree(const P::Tree& pt,
                                            const ConstantMap& raw) {
  IsolationTree t;
  const int n = pt.num_nodes();
  t.is_leaf.resize(n, 0);
  t.split_feature.assign(n, -1);
  t.children_offset.assign(n, -1);
  t.children_count.assign(n, 0);
  for (int i = 0; i < n && i < pt.node_kind_size(); ++i)
    t.is_leaf[i] = (pt.node_kind(i) == P::Tree::LEAF) ? 1 : 0;
  for (int i = 0; i < n && i < pt.split_feature_size(); ++i)
    t.split_feature[i] = pt.split_feature(i);
  for (int i = 0; i < n && i < pt.children_offset_size(); ++i)
    t.children_offset[i] = pt.children_offset(i);
  for (int i = 0; i < n && i < pt.children_count_size(); ++i)
    t.children_count[i] = pt.children_count(i);
  t.children_index.assign(pt.children_index().begin(),
                          pt.children_index().end());
  if (const P::Tensor* st = unwrap_tv(pt.split_threshold(), raw))
    extract_doubles(*st, t.split_threshold);
  if (const P::Tensor* lv = unwrap_tv(pt.leaf_value(), raw))
    extract_doubles(*lv, t.leaf_value);
  t.split_threshold.resize(n, 0.0);
  t.leaf_value.resize(n, 0.0);
  return t;
}

// Map declared node outputs to the score / decision_value / prediction roles of
// the omle.ml/AnomalyDetection contract. Role is taken from the output's
// declared role first, then its name; a single unlabelled output means score.
static std::vector<AnomalyOutput> anomaly_output_roles(const P::Node& node) {
  std::vector<AnomalyOutput> roles;
  for (const auto& o : node.outputs()) {
    const std::string& name = o.name();
    if (o.role() == P::OutputRole::PREDICTION ||
        name.find("prediction") != std::string::npos) {
      roles.push_back(AnomalyOutput::Prediction);
    } else if (name.find("decision") != std::string::npos) {
      roles.push_back(AnomalyOutput::DecisionValue);
    } else {
      roles.push_back(AnomalyOutput::Score);
    }
  }
  if (roles.empty()) roles.push_back(AnomalyOutput::Score);
  return roles;
}

static std::unique_ptr<GraphNode> convert_anomaly_detection(
    const P::Node& node, const ConstantMap& raw) {
  const auto& ad = node.anomaly_detection();
  std::vector<AnomalyOutput> roles = anomaly_output_roles(node);
  std::unique_ptr<AnomalyDetectionNode> gn;

  if (ad.has_isolation_forest()) {
    const auto& iso = ad.isolation_forest();
    std::vector<IsolationTree> trees;
    trees.reserve(static_cast<std::size_t>(iso.trees_size()));
    for (const auto& pt : iso.trees())
      trees.push_back(convert_isolation_tree(pt, raw));
    const double offset =
        iso.has_offset()
            ? scalar_val(iso.offset()).as_double()
            : (ad.has_threshold() ? scalar_val(ad.threshold()).as_double()
                                  : 0.0);
    gn = make_isolation_forest_node(std::move(trees), iso.max_samples(), offset,
                                    std::move(roles));
  }

  if (ad.has_elliptic_envelope()) {
    const auto& ee = ad.elliptic_envelope();
    std::vector<double> location, precision;
    if (const P::Tensor* t = unwrap_tv(ee.location(), raw))
      extract_doubles(*t, location);
    if (ee.has_precision()) {
      if (const P::Tensor* t = unwrap_tv(ee.precision(), raw))
        extract_doubles(*t, precision);
    }
    const int nf = static_cast<int>(location.size());
    // precision is optional in the proto; without it there is nothing to invert
    // a covariance with here, so fall back to the identity (plain Euclidean).
    if (static_cast<int>(precision.size()) != nf * nf) {
      precision.assign(static_cast<std::size_t>(nf) * nf, 0.0);
      for (int i = 0; i < nf; ++i)
        precision[static_cast<std::size_t>(i) * nf + i] = 1.0;
    }
    const double offset =
        ee.has_offset()
            ? scalar_val(ee.offset()).as_double()
            : (ad.has_threshold() ? scalar_val(ad.threshold()).as_double()
                                  : 0.0);
    gn = make_elliptic_envelope_node(std::move(location), std::move(precision),
                                     nf, offset, std::move(roles));
  }

  if (ad.has_linear_one_class_svm()) {
    const auto& ls = ad.linear_one_class_svm();
    std::vector<double> coefficients, intercept;
    if (const P::Tensor* t = unwrap_tv(ls.coefficients(), raw))
      extract_doubles(*t, coefficients);
    if (ls.has_intercept()) {
      if (const P::Tensor* t = unwrap_tv(ls.intercept(), raw))
        extract_doubles(*t, intercept);
    }
    const double offset =
        ls.has_offset()
            ? scalar_val(ls.offset()).as_double()
            : (ad.has_threshold() ? scalar_val(ad.threshold()).as_double()
                                  : 0.0);
    const int nf = static_cast<int>(coefficients.size());
    gn = make_linear_one_class_svm_node(std::move(coefficients),
                                        intercept.empty() ? 0.0 : intercept[0],
                                        offset, nf, std::move(roles));
  }

  if (ad.has_one_class_svm()) {
    const auto& oc = ad.one_class_svm();
    const auto& k = oc.kernel_svm();
    AnomalyKernel kernel = AnomalyKernel::RBF;
    switch (k.kernel_type()) {
      case P::KernelSVM::LINEAR:
        kernel = AnomalyKernel::Linear;
        break;
      case P::KernelSVM::POLY:
        kernel = AnomalyKernel::Poly;
        break;
      case P::KernelSVM::SIGMOID:
        kernel = AnomalyKernel::Sigmoid;
        break;
      default:
        kernel = AnomalyKernel::RBF;
        break;
    }
    std::vector<double> sv, dual, intercept;
    int n_sv = 0, nf = 0;
    if (const P::Tensor* t = unwrap_tv(k.support_vectors(), raw)) {
      extract_doubles(*t, sv);
      const auto& shape = t->type().shape();
      if (shape.size() == 2) {
        n_sv = static_cast<int>(shape[0]);
        nf = static_cast<int>(shape[1]);
      }
    }
    if (const P::Tensor* t = unwrap_tv(k.dual_coefficients(), raw))
      extract_doubles(*t, dual);
    if (k.has_intercept()) {
      if (const P::Tensor* t = unwrap_tv(k.intercept(), raw))
        extract_doubles(*t, intercept);
    }
    const double offset =
        oc.has_offset()
            ? scalar_val(oc.offset()).as_double()
            : (ad.has_threshold() ? scalar_val(ad.threshold()).as_double()
                                  : 0.0);
    gn = make_one_class_svm_node(
        kernel, std::move(sv), std::move(dual),
        intercept.empty() ? 0.0 : intercept[0],
        k.has_gamma() ? scalar_val(k.gamma()).as_double() : 1.0,
        k.has_degree() ? k.degree() : 3,
        k.has_coef0() ? scalar_val(k.coef0()).as_double() : 0.0, n_sv, nf,
        offset, std::move(roles));
  }

  if (ad.has_local_outlier_factor()) {
    const auto& lof = ad.local_outlier_factor();
    std::vector<double> ref;
    int n_ref = 0, nf = 0;
    if (const P::Tensor* t = unwrap_tv(lof.reference_samples(), raw)) {
      extract_doubles(*t, ref);
      const auto& shape = t->type().shape();
      if (shape.size() == 2) {
        n_ref = static_cast<int>(shape[0]);
        nf = static_cast<int>(shape[1]);
      }
    }
    double p = 2.0;
    auto it = lof.metric_params().find("p");
    if (it != lof.metric_params().end()) {
      try {
        p = std::stod(it->second);
      } catch (...) {
        p = 2.0;
      }
    }
    const double offset =
        lof.has_offset()
            ? scalar_val(lof.offset()).as_double()
            : (ad.has_threshold() ? scalar_val(ad.threshold()).as_double()
                                  : 0.0);
    gn = make_local_outlier_factor_node(std::move(ref), n_ref, nf,
                                        lof.n_neighbors(), lof.metric(), p,
                                        offset, std::move(roles));
  }

  if (!gn) return nullptr;
  for (const auto& i : node.inputs()) gn->in_names.push_back(i.name().value());
  for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
  gn->node_name = node.name();
  return gn;
}

// Main per-node dispatch
static std::unique_ptr<GraphNode> convert_node_impl(const P::Node& node,
                                                    const ConstantStore& cstore,
                                                    const ConstantMap& raw) {
  switch (node.body_case()) {
    case P::Node::kTree:
      return convert_tree(node, raw);
    case P::Node::kTreeEnsemble:
      return convert_tree_ensemble(node, raw);
    case P::Node::kLinear:
      return convert_linear(node, raw);
    case P::Node::kNaiveBayes:
      return convert_naive_bayes(node, raw);
    case P::Node::kClustering:
      return convert_clustering(node, raw);
    case P::Node::kSvm:
      return convert_svm(node, raw);
    case P::Node::kNeuralNetwork:
      return convert_neural_network(node, raw);
    case P::Node::kAnomalyDetection:
      return convert_anomaly_detection(node, raw);
    case P::Node::kComposite:
      return convert_composite(node, cstore, raw);
    default: {
      // Check if this is a OneHotEncoder op we can accelerate with pre-built
      // hash maps.
      const bool is_ohe =
          (node.domain() == "omle.feature") &&
          (node.op() == "OneHotEncoder" || node.op() == "OneHotEncode");
      if (is_ohe) {
        AttributeMap attrs = convert_attributes(
            node.attributes(), const_cast<ConstantStore&>(cstore), node.name());
        const std::string& cats_name = attr_tensor(attrs, "categories");
        auto cit = (!cats_name.empty()) ? cstore.find(cats_name) : cstore.end();
        if (cit != cstore.end() && cit->second->is_string()) {
          auto ohe = std::make_unique<OHENode>();
          for (const auto& i : node.inputs())
            ohe->in_names.push_back(i.name().value());
          for (const auto& o : node.outputs())
            ohe->out_names.push_back(o.name());
          ohe->node_name = node.name();
          ohe->per_col = ohe->in_names.size() > 1;

          const Tensor* cats_t = cit->second.get();

          const std::string& offs_name = attr_tensor(attrs, "category_offsets");
          const Tensor* offs_t = nullptr;
          if (!offs_name.empty()) {
            auto oit = cstore.find(offs_name);
            if (oit != cstore.end()) offs_t = oit->second.get();
          }

          const int actual_nf =
              ohe->per_col
                  ? static_cast<int>(ohe->in_names.size())
                  : (offs_t ? static_cast<int>(offs_t->numel()) - 1 : 1);

          ohe->cat_maps.resize(actual_nf);
          ohe->feat_widths.resize(actual_nf, 0);
          ohe->total_output_cols = 0;

          for (int f = 0; f < actual_nf; ++f) {
            const int lo =
                (offs_t && offs_t->numel() > static_cast<std::size_t>(f))
                    ? static_cast<int>(offs_t->get(0, f))
                    : 0;
            const int hi =
                (offs_t && offs_t->numel() > static_cast<std::size_t>(f + 1))
                    ? static_cast<int>(offs_t->get(0, f + 1))
                    : (f == 0 ? static_cast<int>(cats_t->numel()) : lo);
            const int w = hi - lo;
            ohe->feat_widths[f] = w;
            ohe->total_output_cols += w;
            ohe->cat_maps[f].reserve(w);
            for (int k = lo; k < hi; ++k)
              ohe->cat_maps[f].emplace(cats_t->str_at(0, k), k - lo);
          }
          return ohe;
        }
      }

      // Generic operator node
      auto gn = std::make_unique<OperatorNode>();
      gn->domain = node.domain();
      gn->op = node.op();
      gn->attrs = convert_attributes(
          node.attributes(), const_cast<ConstantStore&>(cstore), node.name());
      for (const auto& i : node.inputs())
        gn->in_names.push_back(i.name().value());
      for (const auto& o : node.outputs()) gn->out_names.push_back(o.name());
      gn->node_name = node.name();
      return gn;
    }
  }
}

// ---------------------------------------------------------------------------
// Model schema builder
// ---------------------------------------------------------------------------

static std::string expand_range_name(const P::NameRange& r, int i) {
  std::string suffix = std::to_string(i);
  if (r.width() > 0)
    while (static_cast<int>(suffix.size()) < r.width()) suffix = "0" + suffix;
  return r.prefix() + suffix;
}

static CompiledDomain build_domain(const P::ValueDomain& pd) {
  CompiledDomain dom;
  if (pd.has_continuous()) {
    dom.kind = CompiledDomain::Kind::Continuous;
    for (const auto& iv : pd.continuous().intervals()) {
      CompiledInterval ci;
      ci.left = static_cast<float>(scalar_val(iv.left_margin()).as_double());
      ci.right = static_cast<float>(scalar_val(iv.right_margin()).as_double());
      switch (iv.closure()) {
        case P::Interval::OPEN_CLOSED:
          ci.left_open = true;
          ci.right_open = false;
          break;
        case P::Interval::CLOSED_OPEN:
          ci.left_open = false;
          ci.right_open = true;
          break;
        case P::Interval::CLOSED_CLOSED:
          ci.left_open = false;
          ci.right_open = false;
          break;
        default:  // OPEN_OPEN
          ci.left_open = true;
          ci.right_open = true;
          break;
      }
      dom.intervals.push_back(ci);
    }
  } else if (pd.has_discrete()) {
    dom.kind = CompiledDomain::Kind::Discrete;
    for (const auto& dv : pd.discrete().values()) {
      const float fv = static_cast<float>(scalar_val(dv.value()).as_double());
      switch (dv.property()) {
        case P::DomainValue::INVALID:
          dom.invalid_set.insert(fv);
          break;
        case P::DomainValue::MISSING:
          dom.missing_set.insert(fv);
          break;
        default:
          dom.valid_set.insert(fv);
          break;
      }
    }
  }
  return dom;
}

// Compile one proto Feature into one or more SchemaFeature entries.
static void compile_feature(const P::Feature& pf,
                            std::vector<SchemaFeature>& out) {
  auto fill_policies = [&](SchemaFeature& sf) {
    switch (pf.missing_value_policy()) {
      case P::Feature::MISSING_AS_VALUE:
        sf.missing_policy = MissingPolicy::AsValue;
        break;
      case P::Feature::MISSING_AS_INVALID:
        sf.missing_policy = MissingPolicy::AsInvalid;
        break;
      default:
        sf.missing_policy = MissingPolicy::Propagate;
        break;
    }
    sf.missing_replacement = static_cast<float>(
        scalar_val(pf.missing_replacement_value()).as_double());

    switch (pf.invalid_value_policy()) {
      case P::Feature::INVALID_AS_IS:
        sf.invalid_policy = InvalidPolicy::AsIs;
        break;
      case P::Feature::INVALID_AS_MISSING:
        sf.invalid_policy = InvalidPolicy::AsMissing;
        break;
      case P::Feature::INVALID_AS_VALUE:
        sf.invalid_policy = InvalidPolicy::AsValue;
        break;
      default:
        sf.invalid_policy = InvalidPolicy::ReturnInvalid;
        break;
    }
    sf.invalid_replacement = static_cast<float>(
        scalar_val(pf.invalid_replacement_value()).as_double());

    switch (pf.outlier_value_policy()) {
      case P::Feature::OUTLIER_AS_MISSING:
        sf.outlier_policy = OutlierPolicy::AsMissing;
        break;
      case P::Feature::OUTLIER_AS_EXTREME:
        sf.outlier_policy = OutlierPolicy::AsExtreme;
        break;
      default:
        sf.outlier_policy = OutlierPolicy::AsIs;
        break;
    }

    if (pf.has_domain()) sf.domain = build_domain(pf.domain());
  };

  if (pf.naming_case() == P::Feature::kRange) {
    const auto& r = pf.range();
    int col_idx = pf.index();
    for (int i = r.start(); i < r.end(); ++i, ++col_idx) {
      SchemaFeature sf;
      sf.name = expand_range_name(r, i);
      sf.source = pf.source();
      sf.col = col_idx;
      fill_policies(sf);
      out.push_back(std::move(sf));
    }
  } else {
    SchemaFeature sf;
    sf.name = pf.name();
    sf.source = pf.source();
    sf.col = pf.index();
    fill_policies(sf);
    out.push_back(std::move(sf));
  }
}

static std::unique_ptr<ModelSchemaNode> build_schema_node(
    const P::ModelSchema& ms) {
  auto node = std::make_unique<ModelSchemaNode>();
  for (const auto& pf : ms.features()) compile_feature(pf, node->features);

  // Enable batch mode when all features share the same source and are in
  // sequential column order. Then execute() emits one wide tensor instead
  // of n_features individual [n_rows,1] tensors, eliminating O(n_features)
  // allocations and hash-map inserts per inference call.
  if (!node->features.empty()) {
    const std::string& src0 = node->features[0].source;
    bool can_batch = true;
    for (int f = 0; f < static_cast<int>(node->features.size()); ++f) {
      const auto& feat = node->features[f];
      if (feat.source != src0 || feat.col != f) {
        can_batch = false;
        break;
      }
    }
    if (can_batch) {
      node->batch_mode = true;
      node->batch_key = src0 + "__batch__";
    }
  }

  return node;
}

// ---------------------------------------------------------------------------
// Main build entry
// ---------------------------------------------------------------------------

static omle::rt::DataType convert_dtype_gl(P::DataType dt) {
  switch (dt) {
    case P::BOOL:
      return omle::rt::DataType::Bool;
    case P::INT8:
      return omle::rt::DataType::Int8;
    case P::INT16:
      return omle::rt::DataType::Int16;
    case P::INT32:
      return omle::rt::DataType::Int32;
    case P::INT64:
      return omle::rt::DataType::Int64;
    case P::UINT8:
      return omle::rt::DataType::UInt8;
    case P::UINT16:
      return omle::rt::DataType::UInt16;
    case P::UINT32:
      return omle::rt::DataType::UInt32;
    case P::UINT64:
      return omle::rt::DataType::UInt64;
    case P::FLOAT16:
      return omle::rt::DataType::Float16;
    case P::FLOAT32:
      return omle::rt::DataType::Float32;
    case P::FLOAT64:
      return omle::rt::DataType::Float64;
    case P::STRING:
      return omle::rt::DataType::String;
    case P::BYTES:
      return omle::rt::DataType::Bytes;
    case P::DATE:
      return omle::rt::DataType::Date;
    case P::TIME:
      return omle::rt::DataType::Time;
    case P::TIMESTAMP:
      return omle::rt::DataType::Timestamp;
    default:
      return omle::rt::DataType::Unknown;
  }
}

static omle::rt::OutputRole convert_role_gl(P::OutputRole r) {
  switch (r) {
    case P::PREDICTION:
      return omle::rt::OutputRole::Prediction;
    case P::PROBABILITY:
      return omle::rt::OutputRole::Probability;
    case P::SCORE:
      return omle::rt::OutputRole::Score;
    case P::CONFIDENCE:
      return omle::rt::OutputRole::Confidence;
    case P::STANDARD_ERROR:
      return omle::rt::OutputRole::StandardError;
    case P::STANDARD_DEVIATION:
      return omle::rt::OutputRole::StandardDev;
    case P::RESIDUAL:
      return omle::rt::OutputRole::Residual;
    case P::TRANSFORMED_VALUE:
      return omle::rt::OutputRole::TransformedValue;
    case P::ENTITY_ID:
      return omle::rt::OutputRole::EntityId;
    case P::AFFINITY:
      return omle::rt::OutputRole::Affinity;
    case P::CONTRIBUTION:
      return omle::rt::OutputRole::Contribution;
    case P::INTERMEDIATE:
      return omle::rt::OutputRole::Intermediate;
    default:
      return omle::rt::OutputRole::Unspecified;
  }
}

// -----------------------------------------------------------------------
// Tensor entry helpers for verification and warmup
// -----------------------------------------------------------------------

// id → (effective_name, Tensor) built from proto.tensor_entries().
using TensorEntryMap =
    std::unordered_map<std::string, std::pair<std::string, Tensor>>;

static TensorEntryMap build_tensor_entry_map(const P::OMLEModel& proto) {
  TensorEntryMap m;
  for (const auto& entry : proto.tensor_entries()) {
    std::string name;
    Tensor t;
    if (entry.has_dense()) {
      const P::Tensor& dt = entry.dense();
      name = dt.name();
      int rows = 1, cols = 1;
      const auto& shape = dt.type().shape();
      if (shape.size() == 1) {
        cols = static_cast<int>(shape[0]);
      } else if (shape.size() >= 2) {
        rows = static_cast<int>(shape[shape.size() - 2]);
        cols = static_cast<int>(shape[shape.size() - 1]);
      }
      if (dt.type().dtype() == P::STRING && dt.has_string_data()) {
        // 1-D string tensors represent column vectors: shape=[N] → N rows × 1
        // col.
        if (shape.size() == 1) {
          rows = cols;
          cols = 1;
        }
        std::vector<std::string> sv(dt.string_data().values().begin(),
                                    dt.string_data().values().end());
        t = Tensor::strings(rows, cols, std::move(sv));
      } else {
        std::vector<float> data;
        extract_floats(dt, data);
        t = Tensor::from_floats(rows, cols, std::move(data));
      }
    } else if (entry.has_sparse()) {
      const auto& sp = entry.sparse();
      name = sp.name();
      const auto& shape = sp.type().shape();
      int rows = shape.size() >= 1 ? static_cast<int>(shape[0]) : 0;
      int cols = shape.size() >= 2 ? static_cast<int>(shape[1]) : 0;
      float fill =
          sp.has_default_value()
              ? static_cast<float>(scalar_val(sp.default_value()).as_double())
              : 0.f;
      std::vector<float> values;
      std::vector<int32_t> indices;
      std::vector<int32_t> indptr;
      if (sp.has_csr()) {
        extract_csr_floats(sp.csr(), values);
        for (int64_t v : sp.csr().indices())
          indices.push_back(static_cast<int32_t>(v));
        for (int64_t v : sp.csr().indptr())
          indptr.push_back(static_cast<int32_t>(v));
      } else {
        indptr.resize(rows + 1, 0);
      }
      t = Tensor::sparse_csr_f32(rows, cols, std::move(values),
                                 std::move(indices), std::move(indptr), fill);
    } else {
      continue;
    }
    m[entry.id()] = {std::move(name), std::move(t)};
  }
  return m;
}

static std::unordered_map<std::string, Tensor> resolve_refs(
    const TensorEntryMap& entries,
    const google::protobuf::RepeatedPtrField<P::TensorRef>& refs) {
  std::unordered_map<std::string, Tensor> result;
  for (const auto& ref : refs) {
    auto it = entries.find(ref.id());
    if (it == entries.end())
      throw std::runtime_error("omle: TensorRef id '" + ref.id() +
                               "' not found in tensor_entries");
    result[it->second.first] = it->second.second;
  }
  return result;
}

static void run_verification(const P::OMLEModel& proto,
                             const TensorEntryMap& entries, ModelBase& exec) {
  if (!proto.has_verification()) return;
  const auto& verif = proto.verification();
  if (verif.cases_size() == 0) return;

  double atol = 1e-6, rtol = 1e-5;
  if (verif.has_tolerance()) {
    if (verif.tolerance().has_atol())
      atol = scalar_val(verif.tolerance().atol()).as_double();
    if (verif.tolerance().has_rtol())
      rtol = scalar_val(verif.tolerance().rtol()).as_double();
  }

  for (int ci = 0; ci < verif.cases_size(); ++ci) {
    const auto& vc = verif.cases(ci);
    auto inputs = resolve_refs(entries, vc.inputs());
    auto expected = resolve_refs(entries, vc.expected_outputs());
    auto actual = exec.predict_named_serial(inputs);

    for (const auto& [name, exp_t] : expected) {
      // Skip verification of string outputs — numeric comparison is not
      // applicable.
      if (exp_t.is_string()) continue;

      auto ait = actual.find(name);
      if (ait == actual.end())
        throw std::runtime_error("verification case " + std::to_string(ci) +
                                 ": output '" + name + "' was not produced");

      const Tensor act_f = ait->second.to_float32();
      const Tensor exp_f = exp_t.to_float32();
      int n = act_f.n_rows * act_f.n_cols;
      if (n != exp_f.n_rows * exp_f.n_cols)
        throw std::runtime_error(
            "verification case " + std::to_string(ci) + ": output '" + name +
            "' shape mismatch: got " + std::to_string(act_f.n_rows) + "x" +
            std::to_string(act_f.n_cols) + " expected " +
            std::to_string(exp_f.n_rows) + "x" + std::to_string(exp_f.n_cols));

      const float* a = act_f.f32_ptr();
      const float* e = exp_f.f32_ptr();
      for (int i = 0; i < n; ++i) {
        double diff =
            std::fabs(static_cast<double>(a[i]) - static_cast<double>(e[i]));
        double tol = atol + rtol * std::fabs(static_cast<double>(e[i]));
        if (diff > tol)
          throw std::runtime_error(
              "verification case " + std::to_string(ci) + ": output '" + name +
              "' mismatch at element " + std::to_string(i) + ": actual=" +
              std::to_string(a[i]) + " expected=" + std::to_string(e[i]) +
              " (diff=" + std::to_string(diff) + " tol=" + std::to_string(tol) +
              ")");
      }
    }
  }
}

static void run_warmup(const P::OMLEModel& proto, const TensorEntryMap& entries,
                       ModelBase& exec) {
  if (!proto.has_warmup()) return;
  for (const auto& wc : proto.warmup().cases()) {
    auto inputs = resolve_refs(entries, wc.inputs());
    int repeat = std::max(1, wc.repeat());
    for (int i = 0; i < repeat; ++i) exec.predict_named_serial(inputs);
  }
}

LoadedModel build_from_proto(const P::OMLEModel& proto) {
  register_builtin_operators();

  // Build constant store from tensor_entries.
  // raw is keyed by TensorEntry.id (used during loading to resolve TensorRef).
  // cstore is keyed by the tensor's effective name (used at runtime by graph
  // nodes).
  ConstantStore cstore;
  ConstantMap raw;
  for (const auto& c : proto.tensor_entries()) {
    if (c.has_dense()) {
      const P::Tensor& t = c.dense();
      if (c.id().empty()) continue;
      raw[c.id()] = &t;
      int64_t rows = 1, cols = 1;
      const auto& shape = t.type().shape();
      if (shape.size() >= 1) cols = shape[shape.size() - 1];
      if (shape.size() >= 2) rows = shape[shape.size() - 2];
      if (shape.size() == 1) {
        rows = 1;
        cols = shape[0];
      }
      std::shared_ptr<Tensor> rt;
      if (t.type().dtype() == P::STRING && t.has_string_data()) {
        std::vector<std::string> sv(t.string_data().values().begin(),
                                    t.string_data().values().end());
        rt = std::make_shared<Tensor>(Tensor::strings(
            static_cast<int>(rows), static_cast<int>(cols), std::move(sv)));
      } else {
        std::vector<float> data;
        extract_floats(t, data);
        rt = std::make_shared<Tensor>(Tensor::from_floats(
            static_cast<int>(rows), static_cast<int>(cols), std::move(data)));
      }
      const std::string& ckey = t.name().empty() ? c.id() : t.name();
      cstore[ckey] = rt;
    } else if (c.has_sparse()) {
      const P::SparseTensor& sp = c.sparse();
      if (c.id().empty()) continue;
      const auto& shape = sp.type().shape();
      int rows = shape.size() >= 1 ? static_cast<int>(shape[0]) : 0;
      int cols = shape.size() >= 2 ? static_cast<int>(shape[1]) : 0;
      float fill =
          sp.has_default_value()
              ? static_cast<float>(scalar_val(sp.default_value()).as_double())
              : 0.0f;
      std::vector<float> values;
      std::vector<int32_t> indices;
      std::vector<int32_t> indptr;
      if (sp.has_csr()) {
        const auto& csr = sp.csr();
        extract_csr_floats(csr, values);
        indices.reserve(csr.indices_size());
        for (int64_t v : csr.indices())
          indices.push_back(static_cast<int32_t>(v));
        indptr.reserve(csr.indptr_size());
        for (int64_t v : csr.indptr())
          indptr.push_back(static_cast<int32_t>(v));
      } else {
        indptr.resize(rows + 1, 0);
      }
      auto rt = std::make_shared<Tensor>(Tensor::sparse_csr_f32(
          rows, cols, std::move(values), std::move(indices), std::move(indptr),
          static_cast<float>(fill)));
      const std::string& ckey = sp.name().empty() ? c.id() : sp.name();
      cstore[ckey] = rt;
    }
  }

  auto exec = std::make_unique<GraphExecutor>();
  exec->constants = std::move(cstore);

  // Populate input names from declared inputs.
  for (const auto& inp : proto.inputs())
    exec->input_names.push_back(inp.name());

  // Compile user-defined functions declared in the model.
  for (const auto& df : proto.functions()) {
    UserFunction uf;
    for (const auto& param : df.parameters()) uf.params.push_back(param.name());
    uf.body = convert_expression(df.body());
    exec->user_functions[df.name()] = std::move(uf);
  }

  // Schema preprocessing node runs first if the model declares features.
  ModelSchemaNode* schema_ptr = nullptr;
  if (proto.model_schema().features_size() > 0) {
    auto schema = build_schema_node(proto.model_schema());
    schema_ptr = schema.get();
    exec->nodes.push_back(std::move(schema));
  }

  // Build feature-name→index map for batch-mode patching.
  std::unordered_map<std::string, int> schema_feat_idx;
  if (schema_ptr && schema_ptr->batch_mode) {
    for (int f = 0; f < static_cast<int>(schema_ptr->features.size()); ++f)
      schema_feat_idx[schema_ptr->features[f].name] = f;
  }

  // Convert all graph nodes. When the schema is in batch mode and a node's
  // in_names are exactly the schema feature names in order, replace them
  // with the single batch key to skip O(n_features) hash-map lookups in
  // gather_slots.
  for (const auto& node : proto.nodes()) {
    auto gn = convert_node_impl(node, exec->constants, raw);
    if (schema_ptr && schema_ptr->batch_mode && !gn->in_names.empty() &&
        gn->in_names.size() == schema_ptr->features.size()) {
      bool all_in_order = true;
      for (int i = 0; i < static_cast<int>(gn->in_names.size()); ++i) {
        auto it = schema_feat_idx.find(gn->in_names[i]);
        if (it == schema_feat_idx.end() || it->second != i) {
          all_in_order = false;
          break;
        }
      }
      if (all_in_order) gn->in_names = {schema_ptr->batch_key};
    }
    exec->nodes.push_back(std::move(gn));
  }

  // Flatten CompositeNodes: inline inner nodes into the parent graph, applying
  // alias substitutions so inner tensor names resolve in the parent ValueStore.
  // This eliminates the per-call child-scope creation, map copies, and
  // ref-count bumps that CompositeNode::execute() otherwise incurs on every
  // inference call.
  {
    std::vector<std::unique_ptr<GraphNode>> flat;
    flat.reserve(exec->nodes.size());
    for (auto& node : exec->nodes) {
      auto* cn = dynamic_cast<CompositeNode*>(node.get());
      if (!cn) {
        flat.push_back(std::move(node));
        continue;
      }

      // Build rename map: internal_name → parent_name
      std::unordered_map<std::string, std::string> rename;
      // input_aliases: (from=external, to=internal) — internal becomes external
      for (const auto& [frm, to] : cn->input_aliases)
        if (frm != to) rename[to] = frm;
      // output_aliases: (from=internal, to=external) — internal becomes
      // external
      for (const auto& [frm, to] : cn->output_aliases)
        if (frm != to) rename[frm] = to;

      for (auto& child : cn->nodes) {
        for (auto& nm : child->in_names)
          if (auto it = rename.find(nm); it != rename.end()) nm = it->second;
        for (auto& nm : child->out_names)
          if (auto it = rename.find(nm); it != rename.end()) nm = it->second;
        flat.push_back(std::move(child));
      }
    }
    exec->nodes = std::move(flat);
  }

  // Resolve TakeSlots "names" attribute (list of feature names) → "indices"
  // (column ints). This allows the graph to carry readable column names while
  // the operator still uses integer indices at runtime.
  if (schema_ptr && !schema_ptr->features.empty()) {
    std::unordered_map<std::string, int> feat_col_idx;
    for (const auto& feat : schema_ptr->features)
      feat_col_idx[feat.name] = feat.col;

    for (auto& gn : exec->nodes) {
      auto* op = dynamic_cast<OperatorNode*>(gn.get());
      if (!op || op->op != "TakeSlots") continue;
      auto it = op->attrs.find("names");
      if (it == op->attrs.end() || it->second.kind != AttrVal::Kind::Strings)
        continue;

      AttrVal idx_val;
      idx_val.kind = AttrVal::Kind::Ints;
      for (const auto& nm : it->second.strings) {
        auto col_it = feat_col_idx.find(nm);
        if (col_it == feat_col_idx.end())
          throw std::runtime_error("TakeSlots: unknown feature name '" + nm +
                                   "'");
        idx_val.ints.push_back(col_it->second);
      }
      op->attrs["indices"] = std::move(idx_val);
      op->attrs.erase("names");
    }
  }

  // Populate output specs.
  for (const auto& out : proto.outputs()) {
    OutputSpec spec;
    spec.name = out.name();
    // cols = product of trailing dims (skip leading N dim)
    spec.cols = 1;
    for (int d = 1; d < out.type().shape_size(); ++d)
      spec.cols *= static_cast<int>(out.type().shape(d));
    if (spec.cols == 0) spec.cols = 1;
    exec->output_specs.push_back(spec);
    exec->total_output_cols += spec.cols;
  }
  if (exec->total_output_cols == 0) exec->total_output_cols = 1;

  // Build LoadedModel
  LoadedModel result;
  result.executor = std::move(exec);

  for (const auto& inp : proto.inputs()) {
    omle::rt::InputSpec info;
    info.name = inp.name();
    info.dtype = convert_dtype_gl(inp.type().dtype());
    for (int64_t d : inp.type().shape()) info.shape.push_back(d);
    result.inputs.push_back(std::move(info));
  }
  for (const auto& out : proto.outputs()) {
    omle::rt::OutputSpec info;
    info.name = out.name();
    info.dtype = convert_dtype_gl(out.type().dtype());
    for (int64_t d : out.type().shape()) info.shape.push_back(d);
    info.role = convert_role_gl(out.role());
    result.outputs.push_back(std::move(info));
  }

  return result;
}

}  // anonymous namespace

LoadedModel build_graph_executor(const void* data, std::size_t size,
                                 const LoadOptions& opts) {
  P::OMLEModel proto;
  if (!proto.ParseFromArray(data, static_cast<int>(size)))
    throw std::runtime_error("omle: failed to parse protobuf model");

  LoadedModel result = build_from_proto(proto);

  // Resolve n_threads: 0 = auto, 1 = single-threaded (no pool created).
  int nt = opts.n_threads;
  if (nt == 0) nt = static_cast<int>(std::thread::hardware_concurrency());
  nt = std::max(1, nt);

  if (nt > 1) {
    auto* exec = static_cast<GraphExecutor*>(result.executor.get());
    exec->thread_pool = std::make_unique<ThreadPool>(nt);
    exec->min_parallel_rows = std::max(1, opts.min_parallel_rows);
    // Propagate the pool to nodes that support intra-call tree parallelism.
    for (auto& node : exec->nodes)
      node->set_thread_pool(exec->thread_pool.get());
  }

  TensorEntryMap tensor_map = build_tensor_entry_map(proto);
  if (opts.run_verification)
    run_verification(proto, tensor_map, *result.executor);
  if (opts.run_warmup) run_warmup(proto, tensor_map, *result.executor);

  return result;
}

}  // namespace omle::rt::impl
