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
    set_target_properties(omle_ext PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/omleruntime"
    )
    add_dependencies(omle_ext omle_runtime)
else()
    message(STATUS "pybind11 not found — omle_ext skipped (ctypes fallback active)")
endif()
