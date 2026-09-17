#ifndef OMLE_STATUS_MACROS_H_
#define OMLE_STATUS_MACROS_H_

#include <utility>

#include "omle/port.h"

// Internal-only helpers — not part of the public API.
//
//   RETURN_IF_ERROR(expr)
//     Evaluate expr (Status or StatusOr) once. Return its Status on error.
//
//   ASSIGN_OR_RETURN(decl, expr)
//     Evaluate expr (StatusOr<T>) once. Return its Status on error.
//     On success, bind the value to decl (e.g. `auto x`).

#define STATUS_MACROS_CONCAT_INNER(x, y) x##y
#define STATUS_MACROS_CONCAT(x, y) STATUS_MACROS_CONCAT_INNER(x, y)

#define RETURN_IF_ERROR(expr)                                     \
  do {                                                            \
    auto STATUS_MACROS_CONCAT(_st_, __LINE__) = (expr);           \
    if (!STATUS_MACROS_CONCAT(_st_, __LINE__).ok()) OMLE_UNLIKELY \
    return STATUS_MACROS_CONCAT(_st_, __LINE__).status();         \
  } while (false)

#define ASSIGN_OR_RETURN_IMPL(tmp, decl, expr) \
  auto tmp = (expr);                           \
  if (!tmp.ok()) OMLE_UNLIKELY                 \
  return tmp.status();                         \
  decl = std::move(*tmp)

#define ASSIGN_OR_RETURN(decl, expr) \
  ASSIGN_OR_RETURN_IMPL(STATUS_MACROS_CONCAT(_sor_, __LINE__), decl, expr)

#endif  // OMLE_STATUS_MACROS_H_
