#ifndef OMLE_PORT_H_
#define OMLE_PORT_H_

// Compiler portability shims. Kept dependency-free so every other header can
// include it.

// OMLE_LANG_VERSION — the C++ standard actually in effect.
//
// MSVC reports it in _MSVC_LANG and leaves __cplusplus at 199711L unless
// /Zc:__cplusplus is passed, so __cplusplus alone is not reliable there.
#if defined(_MSVC_LANG)
#define OMLE_LANG_VERSION _MSVC_LANG
#else
#define OMLE_LANG_VERSION __cplusplus
#endif

// OMLE_UNLIKELY — the C++20 [[unlikely]] attribute wherever the compiler
// actually accepts it, and nothing otherwise.
//
// This library is built as C++17, but the test is compiler support rather than
// language version on purpose: GCC and Clang both report
// __has_cpp_attribute(unlikely) == 201803 in C++17 mode and honour the hint, so
// keying off __cplusplus would throw away branch hints on Linux and macOS.
//
// MSVC is the one that genuinely rejects it below C++20, warning C5051
// ("attribute [[unlikely]] requires at least '/std:c++20'; ignored") at every
// use, which buries real diagnostics. It is excluded explicitly so the outcome
// does not depend on whether its __has_cpp_attribute agrees with its own
// warning. clang-cl is a Clang front end and keeps the attribute.
//
// Placed after the condition, matching the attribute it stands in for:
//   if (rare) OMLE_UNLIKELY { ... }
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(unlikely) >= 201803L && \
    !(defined(_MSC_VER) && !defined(__clang__) && OMLE_LANG_VERSION < 202002L)
#define OMLE_UNLIKELY [[unlikely]]
#endif
#endif

#ifndef OMLE_UNLIKELY
#define OMLE_UNLIKELY
#endif

// OMLE_RESTRICT — the restrict qualifier, under each compiler's spelling.
// GCC and Clang take __restrict__; MSVC takes __restrict and rejects the GNU
// form with a bare syntax error, which then derails the rest of the parse.
#if defined(_MSC_VER) && !defined(__clang__)
#define OMLE_RESTRICT __restrict
#else
#define OMLE_RESTRICT __restrict__
#endif

// OMLE_EXPECT_FALSE / OMLE_EXPECT_TRUE — branch-probability hints.
//
// __builtin_expect is a GCC/Clang builtin with no MSVC counterpart, so on MSVC
// these degrade to the bare condition. That loses only the hint: MSVC gets the
// same information from [[likely]]/[[unlikely]] under C++20, or from PGO.
#if defined(__GNUC__) || defined(__clang__)
#define OMLE_EXPECT_FALSE(x) __builtin_expect(!!(x), 0)
#define OMLE_EXPECT_TRUE(x) __builtin_expect(!!(x), 1)
#else
#define OMLE_EXPECT_FALSE(x) (x)
#define OMLE_EXPECT_TRUE(x) (x)
#endif

#endif  // OMLE_PORT_H_
