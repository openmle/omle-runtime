#ifndef OMLE_SIMD_TRAITS_H_
#define OMLE_SIMD_TRAITS_H_

// Internal SIMD trait types — NOT included from any public header.
// Provides typed wrappers for AVX2 and NEON operations parameterised on T.

// The specialisations below are written for GCC/Clang and are exercised on
// Linux x86-64 (AVX2) and Apple arm64 (NEON). MSVC has never compiled them:
// /arch:AVX2 defines __AVX2__, which pulled this block in for the first time
// when the Windows job was added, and it fails to parse there. Until someone
// can verify the intrinsics on MSVC, that compiler takes the scalar path.
//
// This is safe because every SimdTraits consumer has a scalar fallback --
// checked by building the whole library with both macros forced off, which
// compiles clean and passes the full ctest suite. /arch:AVX2 is still passed
// on MSVC, so its auto-vectoriser can still widen the scalar loops.
//
// clang-cl (_MSC_VER and __clang__) is a real Clang front end, so it keeps the
// AVX2 path.
#if defined(_MSC_VER) && !defined(__clang__)
// No hand-written SIMD on MSVC; SimdTraits exposes only scalar_t.
#elif defined(__AVX2__)
#include <immintrin.h>
#define OMLE_AVX2 1
#elif defined(__ARM_NEON__)
#include <arm_neon.h>
#define OMLE_NEON 1
#endif

#include <cstdint>
#include <type_traits>

namespace omle::rt::impl {

// Primary template — intentionally left undefined.
// Only float and double are valid instantiations.
template <typename T>
struct SimdTraits;

// ============================================================
// float32 specialisation
// ============================================================
template <>
struct SimdTraits<float> {
  using scalar_t = float;

#ifdef OMLE_AVX2
  using vec_t = __m256;
  static constexpr int AVX_WIDTH = 8;

  static vec_t zero() noexcept { return _mm256_setzero_ps(); }
  static vec_t set1(float v) noexcept { return _mm256_set1_ps(v); }
  static vec_t loadu(const float* p) noexcept { return _mm256_loadu_ps(p); }
  static void storeu(float* p, vec_t v) noexcept { _mm256_storeu_ps(p, v); }
  static vec_t add(vec_t a, vec_t b) noexcept { return _mm256_add_ps(a, b); }
  static vec_t fmadd(vec_t a, vec_t b, vec_t c) noexcept {
    return _mm256_fmadd_ps(a, b, c);
  }
  static vec_t cmplt(vec_t a, vec_t b) noexcept {
    return _mm256_cmp_ps(a, b, _CMP_LT_OS);
  }
  static vec_t blend(vec_t f, vec_t t, vec_t mask) noexcept {
    return _mm256_blendv_ps(f, t, mask);
  }

  // Horizontal sum of 8 floats.
  static float hsum(vec_t v) noexcept {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
  }

  // Gather 8 floats at arbitrary int32 indices (scale = sizeof(float) = 4).
  static vec_t gather(const float* base, __m256i idx) noexcept {
    return _mm256_i32gather_ps(base, idx, 4);
  }
#endif  // OMLE_AVX2

#ifdef OMLE_NEON
  using nvec_t = float32x4_t;
  static constexpr int NEON_WIDTH = 4;

  static nvec_t nzero() noexcept { return vdupq_n_f32(0.f); }
  static nvec_t nset1(float v) noexcept { return vdupq_n_f32(v); }
  static nvec_t nloadu(const float* p) noexcept { return vld1q_f32(p); }
  static void nstoreu(float* p, nvec_t v) noexcept { vst1q_f32(p, v); }
  static nvec_t nadd(nvec_t a, nvec_t b) noexcept { return vaddq_f32(a, b); }
  static nvec_t nfma(nvec_t acc, nvec_t a, nvec_t b) noexcept {
    return vfmaq_f32(acc, a, b);
  }
  static uint32x4_t ncmplt(nvec_t a, nvec_t b) noexcept {
    return vcltq_f32(a, b);
  }
  static nvec_t nselect(uint32x4_t m, nvec_t t, nvec_t f) noexcept {
    return vbslq_f32(m, t, f);
  }
  static float nhsum(nvec_t v) noexcept { return vaddvq_f32(v); }
#endif  // OMLE_NEON
};

// ============================================================
// float64 specialisation
// ============================================================
template <>
struct SimdTraits<double> {
  using scalar_t = double;

#ifdef OMLE_AVX2
  using vec_t = __m256d;
  static constexpr int AVX_WIDTH = 4;  // 4 doubles per AVX2 register

  static vec_t zero() noexcept { return _mm256_setzero_pd(); }
  static vec_t set1(double v) noexcept { return _mm256_set1_pd(v); }
  static vec_t loadu(const double* p) noexcept { return _mm256_loadu_pd(p); }
  static void storeu(double* p, vec_t v) noexcept { _mm256_storeu_pd(p, v); }
  static vec_t add(vec_t a, vec_t b) noexcept { return _mm256_add_pd(a, b); }
  static vec_t fmadd(vec_t a, vec_t b, vec_t c) noexcept {
    return _mm256_fmadd_pd(a, b, c);
  }
  static vec_t cmplt(vec_t a, vec_t b) noexcept {
    return _mm256_cmp_pd(a, b, _CMP_LT_OS);
  }
  static vec_t blend(vec_t f, vec_t t, vec_t mask) noexcept {
    return _mm256_blendv_pd(f, t, mask);
  }

  static double hsum(vec_t v) noexcept {
    __m128d lo = _mm256_castpd256_pd128(v);
    __m128d hi = _mm256_extractf128_pd(v, 1);
    __m128d sum = _mm_add_pd(lo, hi);
    sum = _mm_hadd_pd(sum, sum);
    double r;
    _mm_store_sd(&r, sum);
    return r;
  }

  // AVX2 gather: 32-bit indices, scale=8 (sizeof(double)).
  // Takes __m128i (4 × int32) rather than __m256i.
  static vec_t gather(const double* base, __m128i idx) noexcept {
    return _mm256_i32gather_pd(base, idx, 8);
  }
#endif  // OMLE_AVX2

#ifdef OMLE_NEON
  using nvec_t = float64x2_t;           // ARMv8 only
  static constexpr int NEON_WIDTH = 2;  // 2 doubles per NEON register

  static nvec_t nzero() noexcept { return vdupq_n_f64(0.0); }
  static nvec_t nset1(double v) noexcept { return vdupq_n_f64(v); }
  static nvec_t nloadu(const double* p) noexcept { return vld1q_f64(p); }
  static void nstoreu(double* p, nvec_t v) noexcept { vst1q_f64(p, v); }
  static nvec_t nadd(nvec_t a, nvec_t b) noexcept { return vaddq_f64(a, b); }
  static nvec_t nfma(nvec_t acc, nvec_t a, nvec_t b) noexcept {
    return vfmaq_f64(acc, a, b);
  }
  static uint64x2_t ncmplt(nvec_t a, nvec_t b) noexcept {
    return vcltq_f64(a, b);
  }
  static nvec_t nselect(uint64x2_t m, nvec_t t, nvec_t f) noexcept {
    return vbslq_f64(m, t, f);
  }
  static double nhsum(nvec_t v) noexcept {
    return vgetq_lane_f64(v, 0) + vgetq_lane_f64(v, 1);
  }
#endif  // OMLE_NEON
};

// ============================================================
// SIMD conversion helpers: upcast float[]→double[], downcast double[]→float[]
// Implementations are in simd_utils.cpp.
// ============================================================
void upcast_f32_to_f64(const float* src, double* dst, int n) noexcept;
void downcast_f64_to_f32(const double* src, float* dst, int n) noexcept;

}  // namespace omle::rt::impl

#endif  // OMLE_SIMD_TRAITS_H_
