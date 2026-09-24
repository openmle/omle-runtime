cmake_minimum_required(VERSION 3.16)

# ---------------------------------------------------------------------------
# omle_ext — pybind11 extension (fast-path predict, calls the C API).
#
# There is no separate C-ABI library: omle_runtime is a single shared library
# exporting both the C++ API and the flat C ABI, and it is built into this
# directory, so the extension just links it. This file is only added when
# BUILD_PYTHON is on, so nothing here is guarded on it.
# ---------------------------------------------------------------------------
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Try to locate via whichever python3 is on PATH
    find_program(_python3 python3)
    if(_python3)
        execute_process(
            COMMAND "${_python3}" -c
                "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE _pb11_cmake_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(_pb11_cmake_dir)
            find_package(pybind11 REQUIRED PATHS "${_pb11_cmake_dir}" NO_DEFAULT_PATH)
        endif()
    endif()
endif()

if(pybind11_FOUND)
    pybind11_add_module(omle_ext omle_ext.cpp)
    target_include_directories(omle_ext PRIVATE
        "${CMAKE_SOURCE_DIR}/include")
    target_link_libraries(omle_ext PRIVATE omle_runtime)
    # A module library uses LIBRARY_OUTPUT_DIRECTORY on every platform,
    # Windows included -- RUNTIME_OUTPUT_DIRECTORY only governs executables and
    # DLLs. The trailing `$<0:>` stops multi-config generators (Visual Studio,
    # Xcode) from appending a per-configuration subdirectory: without it the
    # .pyd is written to omle_runtime/Release/ and `from omle_runtime import
    # omle_ext` fails even though the build succeeded.
    set_target_properties(omle_ext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/omle_runtime/$<0:>"
    )
    add_dependencies(omle_ext omle_runtime)

    # ---------------------------------------------------------------------
    # Bundle the CLI tools with the wheel.
    # ---------------------------------------------------------------------
    # omle-predict and omle-benchmark are documented as the project's CLI
    # tools, but nothing shipped them: there are no install rules, and the
    # wheel is assembled from this directory, so a user had no way to get them.
    #
    # omle-predict only. It links the static runtime (see the top-level
    # CMakeLists), so it is a standalone executable: no rpath, no shared library
    # to find, nothing to stage beside it on Windows, and it keeps working if
    # copied out of the installation entirely.
    #
    # Being standalone is what lets it install straight into the wheel's scripts
    # directory, which pip puts on PATH. A tool linked against the shared
    # library could not: from bin/ the hop back to the library inside
    # site-packages embeds the Python version, and has no Windows equivalent.
    #
    # omle-benchmark is deliberately NOT shipped. It generates random inputs and
    # reports latency, throughput and RSS — a tool for tuning the runtime, not
    # for using it — and at ~1.5 MB (3.6 MB with vendored deps) it would roughly
    # double the CLI payload of every wheel for an audience of runtime
    # developers, who build from source anyway. It is still built by the normal
    # C++ build and sits in the build directory.
    install(TARGETS omle_predict
            RUNTIME DESTINATION ${SKBUILD_SCRIPTS_DIR})
else()
    message(STATUS "pybind11 not found — omle_ext skipped (ctypes fallback active)")
endif()
