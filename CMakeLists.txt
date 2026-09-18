cmake_minimum_required(VERSION 3.16)
project(omle_runtime VERSION 0.1.0 LANGUAGES CXX)

# Pinned source for OMLE_BUNDLE_DEPS builds. Protobuf selects a matching Abseil
# itself, so only this one version is pinned here.
set(OMLE_PROTOBUF_VERSION "33.6")
set(OMLE_PROTOBUF_SHA256
    "16498d7dc7967e9b100632138babd4b86b61592beeccdd556f67539d9c231355")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Detect compiler and apply aggressive opts
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O3 -DNDEBUG -ffast-math -fno-finite-math-only -funroll-loops)
    add_compile_options(-fomit-frame-pointer)

    # -march=native used to live above. It bakes the *build* machine's ISA into
    # the binary, which is wrong for anything distributed: python/pyproject.toml
    # builds the wheel from this same CMakeLists with build-type Release, so a
    # wheel built on an AVX-512 runner would SIGILL on an older CPU.
    #
    # AVX2 + FMA is the explicit floor instead. It keeps every hand-written
    # kernel in simd_traits.h active (those are gated on __AVX2__, which nothing
    # else defines once -march=native is gone) while staying runnable on any
    # Haswell-or-newer x86, i.e. 2013 onwards.
    #
    # arm64 needs nothing: NEON is mandatory in ARMv8, so __ARM_NEON__ is
    # already defined and the NEON path stays active either way. The guard is a
    # target-architecture test rather than a flag probe because Clang on arm64
    # *accepts* -mavx2 and then warns "argument unused" on every file.
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
      #if !defined(__x86_64__) && !defined(_M_X64) && !defined(__i386__)
      #error not x86
      #endif
      int main() { return 0; }" OMLE_TARGET_IS_X86)
    if(OMLE_TARGET_IS_X86)
      add_compile_options(-mavx2 -mfma)
    endif()
  elseif(MSVC)
    add_compile_options(/O2 /DNDEBUG /arch:AVX2 /fp:fast)
  endif()
endif()

# Any translation unit that pulls in <windows.h> gets min/max as macros unless
# NOMINMAX is set, which breaks every std::min / std::max it can see. Defining
# it for the whole target means a new Windows-only include cannot reintroduce
# the problem.
if(MSVC)
  add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ---------------------------------------------------------------------------
# Protobuf and Abseil
# ---------------------------------------------------------------------------
# .omle files are protobuf-encoded, so the generated omle.pb.cc is compiled into
# this library and protobuf's runtime comes with it. Protobuf >= 22 depends on
# Abseil, so a system protobuf drags ~57 further shared objects behind it. That
# is fine for a local build and fatal for a redistributable artifact: the paths
# are recorded absolute (/opt/homebrew/opt/protobuf/lib/...), so the library only
# loads on a machine with the same packages in the same place.
#
# OMLE_BUNDLE_DEPS builds protobuf and Abseil from pinned source as static
# archives, which the linker folds into libomleruntime. The result depends on
# nothing but libc/libc++, which is what makes a Python wheel or a JNA fat jar
# work on a machine that has never heard of protobuf. It is off by default
# because it costs several minutes of build time; release builds turn it on.
option(OMLE_BUNDLE_DEPS
  "Fetch protobuf and Abseil and link them statically into libomleruntime" OFF)

if(OMLE_BUNDLE_DEPS)
  include(FetchContent)

  # Static archives get linked into a SHARED library, so every object in them
  # has to be position-independent or the link fails outright on Linux.
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)

  # BUILD_SHARED_LIBS is deliberately left alone (it defaults to OFF) — that
  # global is what makes protobuf and Abseil build as static archives, and
  # omle_runtime is declared SHARED explicitly so it is unaffected.
  set(protobuf_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_EXAMPLES     OFF CACHE BOOL "" FORCE)
  set(protobuf_BUILD_SHARED_LIBS  OFF CACHE BOOL "" FORCE)
  set(protobuf_INSTALL            OFF CACHE BOOL "" FORCE)
  set(ABSL_ENABLE_INSTALL         OFF CACHE BOOL "" FORCE)
  set(ABSL_PROPAGATE_CXX_STD      ON  CACHE BOOL "" FORCE)
  set(BUILD_TESTING               OFF CACHE BOOL "" FORCE)

  # Without this protobuf runs find_package(absl CONFIG) first and silently
  # picks up a system Abseil — which is exactly the shared dependency being
  # eliminated here. Forcing the fetch also means protobuf chooses the Abseil
  # version it was tested against, so that pin does not have to live here.
  set(protobuf_FORCE_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)

  if(MSVC)
    # protobuf declares
    #   cmake_dependent_option(protobuf_MSVC_STATIC_RUNTIME ... ON
    #                          "NOT protobuf_BUILD_SHARED_LIBS" OFF)
    # so asking for a static protobuf silently also asks for the static CRT
    # (/MT), and protobuf hands the same choice to Abseil through
    # ABSL_MSVC_STATIC_RUNTIME. This project's own objects use the dynamic CRT
    # (/MD), and the link then fails with a wall of LNK2038 "RuntimeLibrary
    # mismatch" errors.
    #
    # /MD is the side that has to win. A Python extension must use the same CRT
    # as the interpreter that loads it, or memory allocated by one is freed by
    # the other's heap. Static protobuf and a dynamic CRT is a supported
    # combination; it is only the default that disagrees.
    set(protobuf_MSVC_STATIC_RUNTIME OFF CACHE BOOL "" FORCE)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
  endif()

  FetchContent_Declare(protobuf
    URL      "https://github.com/protocolbuffers/protobuf/releases/download/v${OMLE_PROTOBUF_VERSION}/protobuf-${OMLE_PROTOBUF_VERSION}.tar.gz"
    URL_HASH SHA256=${OMLE_PROTOBUF_SHA256}
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(protobuf)

  # find_package(Protobuf) normally supplies these; with a vendored build they
  # come from the targets the sub-project just defined.
  set(Protobuf_PROTOC_EXECUTABLE "$<TARGET_FILE:protoc>")
  set(Protobuf_INCLUDE_DIRS      "${protobuf_SOURCE_DIR}/src")
  set(Protobuf_VERSION           "${OMLE_PROTOBUF_VERSION} (vendored, static)")
else()
  list(APPEND CMAKE_PREFIX_PATH "/opt/anaconda3" "/usr/local" "/opt/homebrew")
  find_package(Protobuf REQUIRED)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
message(STATUS "Protobuf version   : ${Protobuf_VERSION}")
message(STATUS "Protobuf include   : ${Protobuf_INCLUDE_DIRS}")
message(STATUS "Protobuf protoc    : ${Protobuf_PROTOC_EXECUTABLE}")

# -----------------------------------------------------------------------
# Proto code generation
# -----------------------------------------------------------------------
# The schema lives in the sibling `omle` repository, so the default assumes the
# two are checked out next to each other. An isolated build — a wheel built in a
# manylinux container, say — has no sibling to point at, so the location is a
# cache variable: stage omle.proto anywhere and set OMLE_PROTO_DIR to it.
set(OMLE_PROTO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../omle/protobuf" CACHE PATH
    "Directory containing omle.proto")
set(PROTO_SRCDIR  "${OMLE_PROTO_DIR}")
set(PROTO_FILE    "${PROTO_SRCDIR}/omle.proto")
set(PROTO_GENDIR  "${CMAKE_CURRENT_BINARY_DIR}/proto_gen")
file(MAKE_DIRECTORY "${PROTO_GENDIR}")

add_custom_command(
  OUTPUT
    "${PROTO_GENDIR}/omle.pb.cc"
    "${PROTO_GENDIR}/omle.pb.h"
  COMMAND
    "${Protobuf_PROTOC_EXECUTABLE}"
    "--cpp_out=${PROTO_GENDIR}"
    "--proto_path=${PROTO_SRCDIR}"
    "${PROTO_FILE}"
  DEPENDS "${PROTO_FILE}"
  COMMENT "Generating protobuf C++ from omle.proto"
  VERBATIM
)

# -----------------------------------------------------------------------
# Runtime library
# -----------------------------------------------------------------------
find_package(Threads REQUIRED)

# One shared library carrying everything: the C++ API and the flat C ABI
# (src/c_api.cpp below) live in the same artifact, so consumers pick whichever
# surface suits them and there is a single file to ship.
set(OMLE_RUNTIME_SOURCES
  src/model_loader.cpp
  src/graph_loader.cpp
  src/graph_executor.cpp
  src/thread_pool.cpp
  src/model_schema.cpp
  src/expression_eval.cpp
  src/operator_registry.cpp
  src/runtime.cpp
  src/c_api.cpp
  src/simd_utils.cpp
  src/nodes/tree_ensemble_node.cpp
  src/nodes/linear_node.cpp
  src/nodes/naive_bayes_node.cpp
  src/nodes/clustering_node.cpp
  src/nodes/anomaly_detection_node.cpp
  src/nodes/composite_node.cpp
  src/nodes/ohe_node.cpp
  src/nodes/operator_node.cpp
  src/nodes/svm_node.cpp
  src/nodes/neural_network_node.cpp
  "${PROTO_GENDIR}/omle.pb.cc"
)

add_library(omle_runtime SHARED ${OMLE_RUNTIME_SOURCES})

# A static twin, Windows only.
#
# The generated protobuf classes cannot cross a DLL boundary here.
# WINDOWS_EXPORT_ALL_SYMBOLS exports data symbols, but a consumer has to declare
# them __declspec(dllimport) to read them correctly, and protoc emits no such
# annotation by default. omle.pb.cc lives in this library, so a test that builds
# a fixture with P::OMLEModel gets the import thunk's address for
# _OMLEModel_default_instance_ and faults the moment it constructs one --
# SEH 0xc0000005 in every test that touches a proto, which is exactly what CI
# showed.
#
# Linking those tests against a static build keeps the generated code in the
# same binary that uses it. They must not also link the DLL: two copies of
# omle.pb.cc would both register omle.proto in protobuf's single descriptor
# pool and abort on the duplicate. Everything else keeps linking the DLL, so
# the shipped artifact is still what the suite exercises.
# The static build is also what the CLI tools link, on every platform, so that
# omle-predict and omle-benchmark are standalone executables rather than files
# that only run next to a copy of the shared library. That costs one extra
# compile of the runtime sources per build.
add_library(omle_runtime_static STATIC ${OMLE_RUNTIME_SOURCES})
set(OMLE_RUNTIME_TARGETS omle_runtime omle_runtime_static)

foreach(_omle_tgt IN LISTS OMLE_RUNTIME_TARGETS)
target_include_directories(${_omle_tgt}
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nodes"
    "${PROTO_GENDIR}"
    "${Protobuf_INCLUDE_DIRS}"
)
# protobuf::libprotobuf is protobuf's own imported target, so it carries the
# include directories and whatever dependencies protobuf's config declares.
target_link_libraries(${_omle_tgt} PUBLIC protobuf::libprotobuf Threads::Threads)

# Abseil. No source in this project includes absl, but protobuf >= 22 generated
# code calls into it (absl::hash_internal::CombineContiguousImpl and friends),
# and not every protobuf packaging propagates those symbols, so they are linked
# explicitly where available.
#
# Skipped entirely for a bundled build: the vendored protobuf carries its own
# static Abseil through protobuf::libprotobuf, and find_package(absl) here would
# locate the *system* shared Abseil and link that alongside it — putting back
# the shared dependency the bundled build exists to remove, and risking two
# copies of Abseil in one library.
if(NOT OMLE_BUNDLE_DEPS)
  # Each target is probed before use. Naming them unconditionally is what broke
  # CI: Ubuntu's libabsl-dev sets absl_FOUND but does not export the internal
  # targets such as absl::log_internal_message, and linking a non-existent
  # target fails the configure step outright. Where a distribution ships absl as
  # a shared library, the omitted pieces resolve at load time anyway.
  find_package(absl QUIET)
  if(absl_FOUND)
    foreach(_absl_target
        absl::base absl::hash absl::strings absl::status absl::log
        absl::log_internal_message absl::log_internal_check_op)
      if(TARGET ${_absl_target})
        target_link_libraries(${_omle_tgt} PUBLIC ${_absl_target})
      endif()
    endforeach()
  endif()
endif()
# MSVC does not understand the GCC/Clang warning flags.
target_compile_options(${_omle_tgt} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wno-unused-parameter>)
endforeach()
# The CMake target stays `omle_runtime` (other projects link it by that name);
# OUTPUT_NAME controls the artifact, which matches the Python package spelling:
#   libomleruntime.so   libomleruntime.dylib   omleruntime.dll
# On Windows the DLL is accompanied by the import library omleruntime.lib.
#
# WINDOWS_EXPORT_ALL_SYMBOLS is what makes the C++ surface visible on MSVC;
# c_api.h carries no dllexport annotations either, so it covers both. The DLL
# export table caps at 64k entries and this library exports ~2.3k, so there is
# ample headroom.
#
# Everything lands in python/omleruntime/ because that is where the Python
# package expects it and where the JVM bindings point JNA. ARCHIVE matters on
# Windows: it is where the DLL's import library goes.
#
# The trailing `$<0:>` is load-bearing. Multi-config generators (Visual Studio,
# Xcode) append a per-configuration subdirectory to a plain *_OUTPUT_DIRECTORY,
# so the DLL would land in python/omleruntime/Release/ and neither the Python
# package nor JNA would find it. An output directory containing a generator
# expression suppresses that append; `$<0:>` expands to nothing, so the path is
# unchanged everywhere else.
set_target_properties(omle_runtime PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "omleruntime"
  WINDOWS_EXPORT_ALL_SYMBOLS ON
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/omleruntime/$<0:>"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/omleruntime/$<0:>"
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/omleruntime/$<0:>")

# -----------------------------------------------------------------------
# Executables
# -----------------------------------------------------------------------
# As with omle_runtime, the CMake target keeps the underscored spelling and
# OUTPUT_NAME controls the artifact. The dash matches the other top-level
# commands in the project (omle, omle-convert); the library keeps the
# unseparated libomleruntime, which is the convention for shared objects.
add_executable(omle_predict examples/predict.cpp)
# Static, so the tool runs from anywhere without the shared library beside it.
target_link_libraries(omle_predict PRIVATE omle_runtime_static)
set_target_properties(omle_predict PROPERTIES OUTPUT_NAME "omle-predict")

add_executable(omle_benchmark examples/benchmark.cpp)
# Static, so the tool runs from anywhere without the shared library beside it.
target_link_libraries(omle_benchmark PRIVATE omle_runtime_static)
set_target_properties(omle_benchmark PROPERTIES OUTPUT_NAME "omle-benchmark")
if(WIN32)
  # GetProcessMemoryInfo, used for the RSS figures in the benchmark output.
  target_link_libraries(omle_benchmark PRIVATE psapi)
endif()

add_executable(bench_precision examples/bench_precision.cpp)
target_link_libraries(bench_precision PRIVATE omle_runtime)
target_include_directories(bench_precision PRIVATE src)

add_executable(bench_tensor_convert examples/bench_tensor_convert.cpp)
target_link_libraries(bench_tensor_convert PRIVATE omle_runtime)

add_executable(bench_c_api examples/bench_c_api.cpp)
target_link_libraries(bench_c_api PRIVATE omle_runtime)

# Same rpath gap as the test binaries: omleruntime.dll is built into
# python/omleruntime/, so on Windows these executables cannot start unless the
# DLL sits beside them. omle-predict and omle-benchmark are absent from this
# list because they link the static runtime and carry no DLL dependency.
if(WIN32)
  foreach(_exe bench_precision bench_tensor_convert bench_c_api)
    add_custom_command(TARGET ${_exe} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:omle_runtime>" "$<TARGET_FILE_DIR:${_exe}>"
      COMMENT "Staging omleruntime.dll next to ${_exe}")
  endforeach()
endif()

# -----------------------------------------------------------------------
# Python bindings (opt-in)
# -----------------------------------------------------------------------
# The shared library above already carries the C ABI, so the JVM bindings need
# nothing extra. This option only adds the pybind11 fast-path extension.
option(BUILD_PYTHON "Build the Python extension module" OFF)
if(BUILD_PYTHON)
  add_subdirectory(python)
endif()

# -----------------------------------------------------------------------
# Tests (opt-in)
# -----------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  list(APPEND CMAKE_PREFIX_PATH "/opt/anaconda3")
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
