#include "simd_traits.h"

namespace omle::rt::impl {

void upcast_f32_to_f64(const float* src, double* dst, int n) noexcept {
#ifdef OMLE_AVX2
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    // _mm256_cvtps_pd converts 4 floats (128-bit) to 4 doubles (256-bit).
    _mm256_storeu_pd(dst + i, _mm256_cvtps_pd(_mm_loadu_ps(src + i)));
  }
  for (; i < n; ++i) dst[i] = static_cast<double>(src[i]);
#elif defined(OMLE_NEON)
  int i = 0;
  for (; i + 2 <= n; i += 2) {
    // vcvt_f64_f32 converts 2 floats (64-bit) to 2 doubles (128-bit).
    vst1q_f64(dst + i, vcvt_f64_f32(vld1_f32(src + i)));
  }
  for (; i < n; ++i) dst[i] = static_cast<double>(src[i]);
#else
  for (int i = 0; i < n; ++i) dst[i] = static_cast<double>(src[i]);
#endif
}

void downcast_f64_to_f32(const double* src, float* dst, int n) noexcept {
#ifdef OMLE_AVX2
  int i = 0;
  for (; i + 4 <= n; i += 4) {
    // _mm256_cvtpd_ps converts 4 doubles (256-bit) to 4 floats (128-bit).
    _mm_storeu_ps(dst + i, _mm256_cvtpd_ps(_mm256_loadu_pd(src + i)));
  }
  for (; i < n; ++i) dst[i] = static_cast<float>(src[i]);
#elif defined(OMLE_NEON)
  int i = 0;
  for (; i + 2 <= n; i += 2) {
    // vcvt_f32_f64 converts 2 doubles (128-bit) to 2 floats (64-bit).
    vst1_f32(dst + i, vcvt_f32_f64(vld1q_f64(src + i)));
  }
  for (; i < n; ++i) dst[i] = static_cast<float>(src[i]);
#else
  for (int i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]);
#endif
}

}  // namespace omle::rt::impl
