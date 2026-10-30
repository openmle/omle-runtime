#ifndef OMLE_AST_TYPES_H_
#define OMLE_AST_TYPES_H_

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

// Internal AST types for Expression and Predicate.
// Built once at model-load time from proto messages; used at runtime
// with no proto dependency.

namespace omle::rt::impl {

// -----------------------------------------------------------------------
// Scalar literal (mirrors proto Scalar oneof)
// -----------------------------------------------------------------------
struct ScalarVal {
  enum class Kind : uint8_t {
    Bool,
    Int,
    Float,
    Double,
    String
  } kind = Kind::Double;
  bool b = false;
  int64_t i = 0;
  float f = 0.0f;
  double d = 0.0;
  std::string s;

  static ScalarVal from_bool(bool v) {
    ScalarVal r;
    r.kind = Kind::Bool;
    r.b = v;
    return r;
  }
  static ScalarVal from_int(int64_t v) {
    ScalarVal r;
    r.kind = Kind::Int;
    r.i = v;
    return r;
  }
  static ScalarVal from_float(float v) {
    ScalarVal r;
    r.kind = Kind::Float;
    r.f = v;
    return r;
  }
  static ScalarVal from_double(double v) {
    ScalarVal r;
    r.kind = Kind::Double;
    r.d = v;
    return r;
  }
  static ScalarVal from_string(const std::string& v) {
    ScalarVal r;
    r.kind = Kind::String;
    r.s = v;
    return r;
  }

  double as_double() const {
    switch (kind) {
      case Kind::Bool:
        return b ? 1.0 : 0.0;
      case Kind::Int:
        return static_cast<double>(i);
      case Kind::Float:
        return static_cast<double>(f);
      case Kind::Double:
        return d;
      default:
        return std::numeric_limits<double>::quiet_NaN();
    }
  }
  float as_float() const {
    if (kind == Kind::Float) return f;
    return static_cast<float>(as_double());
  }
  bool as_bool() const { return as_double() != 0.0; }
};

// -----------------------------------------------------------------------
// Expression AST
// -----------------------------------------------------------------------
struct Expr;
using ExprPtr = std::shared_ptr<Expr>;

struct ExprLiteral {
  ScalarVal value;
};
struct ExprColumn {
  std::string name;
};
struct ExprApply {
  std::string function;
  std::vector<ExprPtr> args;
};

struct Expr {
  std::variant<ExprLiteral, ExprColumn, ExprApply> node;
};

inline ExprPtr make_literal(ScalarVal v) {
  return std::make_shared<Expr>(Expr{ExprLiteral{v}});
}
inline ExprPtr make_column(const std::string& name) {
  return std::make_shared<Expr>(Expr{ExprColumn{name}});
}
inline ExprPtr make_apply(const std::string& fn, std::vector<ExprPtr> args) {
  return std::make_shared<Expr>(Expr{ExprApply{fn, std::move(args)}});
}

// -----------------------------------------------------------------------
// Predicate AST
// -----------------------------------------------------------------------
enum class SimpleOp : uint8_t {
  LT,
  LE,
  GT,
  GE,
  EQ,
  NE,
  IsMissing,
  IsNotMissing
};
enum class SetOp : uint8_t { In, NotIn };
enum class BoolOp : uint8_t { And, Or, Xor, Surrogate };

struct Pred;
using PredPtr = std::shared_ptr<Pred>;

struct PredTrue {};
struct PredFalse {};
struct PredSimple {
  std::string column;
  SimpleOp op;
  ScalarVal value;
};
struct PredSet {
  std::string column;
  SetOp op;
  std::vector<ScalarVal> values;
};
struct PredCompound {
  BoolOp op;
  std::vector<PredPtr> children;
};

struct Pred {
  std::variant<PredTrue, PredFalse, PredSimple, PredSet, PredCompound> node;
};

inline PredPtr make_true_pred() {
  return std::make_shared<Pred>(Pred{PredTrue{}});
}
inline PredPtr make_false_pred() {
  return std::make_shared<Pred>(Pred{PredFalse{}});
}

// -----------------------------------------------------------------------
// Attribute value (mirrors proto Attribute oneof)
// -----------------------------------------------------------------------
struct AttrVal {
  enum class Kind : uint8_t {
    Int,
    Float,
    String,
    Bool,
    Ints,
    Floats,
    Strings,
    Tensor,  // resolved: name of a constant already in ConstantStore
    Expr,
    Pred
  } kind;

  int64_t i = 0;
  double f = 0.0;
  std::string s;
  bool b = false;
  std::vector<int64_t> ints;
  std::vector<double> floats;
  std::vector<std::string> strings;
  std::string tensor_name;  // key into ConstantStore
  ExprPtr expr;
  PredPtr pred;
};

using AttributeMap = std::unordered_map<std::string, AttrVal>;

// Helpers for attribute lookup with defaults
inline int64_t attr_int(const AttributeMap& attrs, const std::string& key,
                        int64_t def = 0) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  return it->second.kind == AttrVal::Kind::Int ? it->second.i : def;
}
inline double attr_float(const AttributeMap& attrs, const std::string& key,
                         double def = 0.0) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  return it->second.kind == AttrVal::Kind::Float ? it->second.f : def;
}
inline std::string attr_str(const AttributeMap& attrs, const std::string& key,
                            const std::string& def = "") {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  return it->second.kind == AttrVal::Kind::String ? it->second.s : def;
}
inline bool attr_bool(const AttributeMap& attrs, const std::string& key,
                      bool def = false) {
  auto it = attrs.find(key);
  if (it == attrs.end()) return def;
  return it->second.kind == AttrVal::Kind::Bool ? it->second.b : def;
}
inline const std::string& attr_tensor(const AttributeMap& attrs,
                                      const std::string& key) {
  static const std::string empty;
  auto it = attrs.find(key);
  if (it == attrs.end()) return empty;
  return it->second.tensor_name;
}

}  // namespace omle::rt::impl

#endif  // OMLE_AST_TYPES_H_
