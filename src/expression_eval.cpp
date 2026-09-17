#include "expression_eval.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "math_utils.h"
#include "status_macros.h"

namespace omle::rt::impl {

namespace {

using EvalFn = std::vector<float> (*)(const std::vector<std::vector<float>>&,
                                      int);

#define UNARY(name, expr)           \
  {name, [](const auto& a, int n) { \
     std::vector<float> r(n);       \
     for (int i = 0; i < n; ++i) {  \
       float x = a[0][i];           \
       r[i] = (expr);               \
     }                              \
     return r;                      \
   }}

#define BINARY(name, expr)             \
  {name, [](const auto& a, int n) {    \
     std::vector<float> r(n);          \
     for (int i = 0; i < n; ++i) {     \
       float x = a[0][i], y = a[1][i]; \
       r[i] = (expr);                  \
     }                                 \
     return r;                         \
   }}

static const std::unordered_map<std::string, EvalFn> fn_table = {
    BINARY("add", x + y),
    BINARY("subtract", x - y),
    BINARY("multiply", x* y),
    BINARY("divide",
           y != 0.0f ? x / y : std::numeric_limits<float>::quiet_NaN()),
    UNARY("negate", -x),
    UNARY("abs", std::abs(x)),
    BINARY("mod", std::fmod(x, y)),
    UNARY("exp", std::exp(x)),
    UNARY("exp2", std::exp2(x)),
    UNARY("log",
          x > 0.0f ? std::log(x) : std::numeric_limits<float>::quiet_NaN()),
    UNARY("log2",
          x > 0.0f ? std::log2(x) : std::numeric_limits<float>::quiet_NaN()),
    UNARY("log10",
          x > 0.0f ? std::log10(x) : std::numeric_limits<float>::quiet_NaN()),
    UNARY("log1p", std::log1p(x)),
    UNARY("sqrt",
          x >= 0.0f ? std::sqrt(x) : std::numeric_limits<float>::quiet_NaN()),
    UNARY("cbrt", std::cbrt(x)),
    BINARY("pow", std::pow(x, y)),
    UNARY("square", x* x),
    UNARY("sign", x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f)),
    UNARY("sin", std::sin(x)),
    UNARY("cos", std::cos(x)),
    UNARY("tan", std::tan(x)),
    UNARY("asin", std::asin(x)),
    UNARY("acos", std::acos(x)),
    UNARY("atan", std::atan(x)),
    BINARY("atan2", std::atan2(x, y)),
    UNARY("floor", std::floor(x)),
    UNARY("ceil", std::ceil(x)),
    UNARY("round", std::round(x)),
    UNARY("trunc", std::trunc(x)),
    BINARY("min", std::min(x, y)),
    BINARY("max", std::max(x, y)),
    BINARY("equal", x == y ? 1.0f : 0.0f),
    BINARY("not_equal", x != y ? 1.0f : 0.0f),
    BINARY("less", x < y ? 1.0f : 0.0f),
    BINARY("less_or_equal", x <= y ? 1.0f : 0.0f),
    BINARY("greater", x > y ? 1.0f : 0.0f),
    BINARY("greater_or_equal", x >= y ? 1.0f : 0.0f),
    BINARY("logical_and", (x != 0.0f && y != 0.0f) ? 1.0f : 0.0f),
    BINARY("logical_or", (x != 0.0f || y != 0.0f) ? 1.0f : 0.0f),
    UNARY("logical_not", x == 0.0f ? 1.0f : 0.0f),
    UNARY("sigmoid", 1.0f / (1.0f + std::exp(-x))),
    UNARY("tanh", std::tanh(x)),
    UNARY("relu", x > 0.0f ? x : 0.0f),
    UNARY("softplus", std::log1p(std::exp(x))),
    UNARY("is_missing", is_nan_safe(x) ? 1.0f : 0.0f),
    UNARY("is_not_missing", is_nan_safe(x) ? 0.0f : 1.0f),
    // Spark SQLTransformer aliases
    BINARY("sub", x - y),
    BINARY("mul", x* y),
    BINARY("div", y != 0.0f ? x / y : std::numeric_limits<float>::quiet_NaN()),
    UNARY("neg", -x),
    BINARY("greater_than", x > y ? 1.0f : 0.0f),
    BINARY("less_than", x < y ? 1.0f : 0.0f),
    BINARY("and", (x != 0.0f && y != 0.0f) ? 1.0f : 0.0f),
    BINARY("or", (x != 0.0f || y != 0.0f) ? 1.0f : 0.0f),
    UNARY("not", x == 0.0f ? 1.0f : 0.0f),
};

omle::rt::StatusOr<std::vector<float>> eval_apply(const ExprApply& apply,
                                                  const ValueStore& vs,
                                                  int n_rows) {
  std::vector<std::vector<float>> args;
  args.reserve(apply.args.size());
  for (const auto& a : apply.args) {
    ASSIGN_OR_RETURN(auto col, eval_expr(*a, vs, n_rows));
    args.push_back(std::move(col));
  }

  // Strip namespace prefix (e.g. "omle.functions.log" → "log") first so
  // all special-case checks below work with either qualified or bare names.
  static const std::string ns_prefix = "omle.functions.";
  const auto& fn = apply.function;
  const std::string fn_base = (fn.size() > ns_prefix.size() &&
                               fn.substr(0, ns_prefix.size()) == ns_prefix)
                                  ? fn.substr(ns_prefix.size())
                                  : fn;

  if (fn_base == "if" && args.size() == 3) {
    std::vector<float> r(n_rows);
    for (int i = 0; i < n_rows; ++i)
      r[i] = (args[0][i] != 0.0f) ? args[1][i] : args[2][i];
    return r;
  }
  if (fn_base == "clamp" && args.size() == 3) {
    std::vector<float> r(n_rows);
    for (int i = 0; i < n_rows; ++i)
      r[i] = std::max(args[1][i], std::min(args[0][i], args[2][i]));
    return r;
  }
  if (fn_base == "coalesce") {
    std::vector<float> r(n_rows, std::numeric_limits<float>::quiet_NaN());
    for (int k = 0; k < (int)args.size(); ++k)
      for (int i = 0; i < n_rows; ++i)
        if (is_nan_safe(r[i]) && !is_nan_safe(args[k][i])) r[i] = args[k][i];
    return r;
  }
  if ((fn_base == "fill_missing" || fn_base == "replace_missing") &&
      args.size() == 2) {
    std::vector<float> r(n_rows);
    for (int i = 0; i < n_rows; ++i)
      r[i] = is_nan_safe(args[0][i]) ? args[1][i] : args[0][i];
    return r;
  }
  if (fn_base == "leaky_relu" && args.size() >= 1) {
    float alpha = args.size() >= 2 ? args[1][0] : 0.01f;
    std::vector<float> r(n_rows);
    for (int i = 0; i < n_rows; ++i)
      r[i] = args[0][i] > 0.0f ? args[0][i] : alpha * args[0][i];
    return r;
  }
  // String functions: return NaN (string ops not supported in float expr
  // pipeline).
  static const std::unordered_set<std::string> string_fns = {
      "concat",    "lower",         "upper",          "trim",     "ltrim",
      "rtrim",     "substring",     "substr",         "left",     "right",
      "length",    "char_length",   "regexp_replace", "replace",  "split",
      "lpad",      "rpad",          "repeat",         "reverse",  "initcap",
      "translate", "ascii",         "base64",         "unbase64", "decode",
      "encode",    "format_string", "printf",         "nvl",      "nvl2",
  };
  if (string_fns.count(fn_base))
    return std::vector<float>(n_rows, std::numeric_limits<float>::quiet_NaN());

  auto it = fn_table.find(fn_base);
  if (it != fn_table.end()) return it->second(args, n_rows);

  const UserFunctionMap* user_fns = UserFunctionContext::current();
  if (user_fns) {
    auto uit = user_fns->find(fn);
    if (uit != user_fns->end()) {
      const UserFunction& uf = uit->second;
      if (args.size() != uf.params.size())
        return omle::rt::Status{omle::rt::ErrorCode::InvalidArgument,
                                "function '" + fn + "' expects " +
                                    std::to_string(uf.params.size()) +
                                    " argument(s), got " +
                                    std::to_string(args.size())};
      ValueStore fn_vs = vs.child_scope(&vs);
      for (std::size_t p = 0; p < uf.params.size(); ++p) {
        Tensor col(n_rows, 1);
        col.set_floats(std::move(args[p]));
        fn_vs.put(uf.params[p], std::move(col));
      }
      return eval_expr(*uf.body, fn_vs, n_rows);
    }
  }

  return omle::rt::Status{omle::rt::ErrorCode::UnknownOperator,
                          "unknown expression function '" + fn + "'"};
}

}  // anonymous namespace

// -----------------------------------------------------------------------
// Thread-local user function context
// -----------------------------------------------------------------------

namespace {
thread_local const UserFunctionMap* t_user_fns = nullptr;
}

UserFunctionContext::UserFunctionContext(const UserFunctionMap& map) noexcept
    : prev_(t_user_fns) {
  t_user_fns = &map;
}

UserFunctionContext::~UserFunctionContext() noexcept { t_user_fns = prev_; }

const UserFunctionMap* UserFunctionContext::current() noexcept {
  return t_user_fns;
}

// -----------------------------------------------------------------------
// Public implementation
// -----------------------------------------------------------------------

omle::rt::StatusOr<std::vector<float>> eval_expr(const Expr& expr,
                                                 const ValueStore& vs,
                                                 int n_rows) {
  return std::visit(
      [&](const auto& node) -> omle::rt::StatusOr<std::vector<float>> {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, ExprLiteral>) {
          return std::vector<float>(n_rows, node.value.as_float());
        } else if constexpr (std::is_same_v<T, ExprColumn>) {
          const auto& t = vs.get(node.name);
          if (t.is_string())
            return std::vector<float>(n_rows,
                                      std::numeric_limits<float>::quiet_NaN());
          std::vector<float> out(n_rows);
          for (int i = 0; i < n_rows; ++i)
            out[i] = static_cast<float>(t.get(i, 0));
          return out;
        } else {
          return eval_apply(node, vs, n_rows);
        }
      },
      expr.node);
}

omle::rt::StatusOr<std::vector<uint8_t>> eval_pred(const Pred& pred,
                                                   const ValueStore& vs,
                                                   int n_rows) {
  return std::visit(
      [&](const auto& node) -> omle::rt::StatusOr<std::vector<uint8_t>> {
        using T = std::decay_t<decltype(node)>;

        if constexpr (std::is_same_v<T, PredTrue>)
          return std::vector<uint8_t>(n_rows, 1);
        if constexpr (std::is_same_v<T, PredFalse>)
          return std::vector<uint8_t>(n_rows, 0);

        if constexpr (std::is_same_v<T, PredSimple>) {
          const auto& col = vs.get(node.column);
          const float thr = node.value.as_float();
          std::vector<uint8_t> r(n_rows);
          for (int i = 0; i < n_rows; ++i) {
            const float v = col.at(i, 0);
            if (node.op == SimpleOp::IsMissing) {
              r[i] = is_nan_safe(v) ? 1 : 0;
              continue;
            }
            if (node.op == SimpleOp::IsNotMissing) {
              r[i] = is_nan_safe(v) ? 0 : 1;
              continue;
            }
            if (is_nan_safe(v)) {
              r[i] = 0;
              continue;
            }
            switch (node.op) {
              case SimpleOp::LT:
                r[i] = v < thr ? 1 : 0;
                break;
              case SimpleOp::LE:
                r[i] = v <= thr ? 1 : 0;
                break;
              case SimpleOp::GT:
                r[i] = v > thr ? 1 : 0;
                break;
              case SimpleOp::GE:
                r[i] = v >= thr ? 1 : 0;
                break;
              case SimpleOp::EQ:
                r[i] = v == thr ? 1 : 0;
                break;
              case SimpleOp::NE:
                r[i] = v != thr ? 1 : 0;
                break;
              default:
                r[i] = 0;
            }
          }
          return r;
        }

        if constexpr (std::is_same_v<T, PredSet>) {
          const auto& col = vs.get(node.column);
          std::unordered_set<float> vset;
          for (const auto& sv : node.values) vset.insert(sv.as_float());
          std::vector<uint8_t> r(n_rows);
          for (int i = 0; i < n_rows; ++i) {
            bool found = vset.count(col.at(i, 0)) > 0;
            r[i] = ((node.op == SetOp::In) ? found : !found) ? 1 : 0;
          }
          return r;
        }

        if constexpr (std::is_same_v<T, PredCompound>) {
          if (node.children.empty()) return std::vector<uint8_t>(n_rows, 0);

          ASSIGN_OR_RETURN(std::vector<uint8_t> r,
                           eval_pred(*node.children[0], vs, n_rows));

          for (std::size_t k = 1; k < node.children.size(); ++k) {
            ASSIGN_OR_RETURN(std::vector<uint8_t> rk,
                             eval_pred(*node.children[k], vs, n_rows));
            switch (node.op) {
              case BoolOp::And:
                for (int i = 0; i < n_rows; ++i) r[i] = r[i] & rk[i];
                break;
              case BoolOp::Or:
                for (int i = 0; i < n_rows; ++i) r[i] = r[i] | rk[i];
                break;
              case BoolOp::Xor:
                for (int i = 0; i < n_rows; ++i) r[i] = r[i] ^ rk[i];
                break;
              case BoolOp::Surrogate:
                for (int i = 0; i < n_rows; ++i)
                  if (!r[i]) r[i] = rk[i];
                break;
            }
          }
          return r;
        }

        return std::vector<uint8_t>(n_rows, 0);
      },
      pred.node);
}

}  // namespace omle::rt::impl
