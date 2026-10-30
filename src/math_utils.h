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

// Double overload. Without it a double argument narrows to float to reach the
// function above, which is the one implicit conversion this test exists to
// avoid -- and a double NaN payload has no business round-tripping through
// float to be recognised.
inline bool is_nan_safe(double x) noexcept {
  uint64_t u;
  std::memcpy(&u, &x, sizeof(u));
  return (u & 0x7FFFFFFFFFFFFFFFull) > 0x7FF0000000000000ull;
}

}  // namespace omle::rt::impl

#endif  // OMLE_MATH_UTILS_H_
