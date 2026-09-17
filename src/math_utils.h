#ifndef OMLE_MATH_UTILS_H_
#define OMLE_MATH_UTILS_H_
#include <cstdint>
#include <cstring>

namespace omle::rt::impl {

// NaN detection that is correct even under -ffast-math / -ffinite-math-only.
// Uses bit manipulation so the compiler cannot optimize it away.
inline bool is_nan_safe(float x) noexcept {
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return (u & 0x7FFFFFFFu) > 0x7F800000u;
}

}  // namespace omle::rt::impl

#endif  // OMLE_MATH_UTILS_H_
