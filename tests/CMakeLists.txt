find_package(GTest REQUIRED)

# Helper to register a test executable.
# Each test file becomes its own CTest entry.
function(add_omle_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE omle_runtime GTest::GTest GTest::Main)
  target_include_directories(${name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../src"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/nodes"
  )
  target_compile_options(${name} PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall;-Wextra;-Wno-unused-parameter>)
  add_test(NAME ${name} COMMAND ${name})

  # Windows has no rpath: a DLL is located through the executable's own
  # directory or PATH, and omleruntime.dll is built into python/omleruntime/
  # (where the Python package and JNA expect it), not next to these binaries.
  # Without this copy the test executables cannot start at all, and a process
  # that fails to start on a headless runner hangs instead of reporting -- the
  # symptom is ctest printing "Start 1: test_linear" and then nothing.
  if(WIN32)
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:omle_runtime>" "$<TARGET_FILE_DIR:${name}>"
      COMMENT "Staging omleruntime.dll next to ${name}")
  endif()

  # A hung test should fail the job in minutes, not occupy the runner until the
  # workflow-level limit. The whole suite runs in about three seconds.
  set_tests_properties(${name} PROPERTIES TIMEOUT 300)
endfunction()

add_omle_test(test_linear          test_linear.cpp)
add_omle_test(test_tree_ensemble   test_tree_ensemble.cpp)
add_omle_test(test_naive_bayes     test_naive_bayes.cpp)
add_omle_test(test_svm             test_svm.cpp)
add_omle_test(test_neural_network  test_neural_network.cpp)
add_omle_test(test_composite       test_composite.cpp)
add_omle_test(test_clustering      test_clustering.cpp)
add_omle_test(test_anomaly_detection test_anomaly_detection.cpp)
add_omle_test(test_expression_eval test_expression_eval.cpp)
add_omle_test(test_operators          test_operators.cpp)
add_omle_test(test_operators_extended test_operators_extended.cpp)
add_omle_test(test_knn                test_knn.cpp)
add_omle_test(test_model_schema    test_model_schema.cpp)

# These two build their fixture models with the generated protobuf classes, so
# on Windows they link the static twin rather than the DLL -- see the comment in
# the top-level CMakeLists for why the generated code cannot cross that
# boundary. They must not link both: two copies of omle.pb.cc would each
# register omle.proto in protobuf's single descriptor pool and abort.
function(omle_test_uses_proto name)
  target_include_directories(${name} PRIVATE
    "${PROTO_GENDIR}" "${Protobuf_INCLUDE_DIRS}")
  if(WIN32)
    # Replace, not append: PRIVATE linkage was set by add_omle_test.
    set_property(TARGET ${name} PROPERTY LINK_LIBRARIES "")
    target_link_libraries(${name} PRIVATE
      omle_runtime_static GTest::GTest GTest::Main)
  endif()
endfunction()

# test_runtime exercises the public Model / Session / Tensor API.
# It needs the proto-generated header to build fixture models.
add_omle_test(test_runtime         test_runtime.cpp)
omle_test_uses_proto(test_runtime)

# test_api covers extended Tensor C++ API and the full C API
# (omle_tensor_*, omle_model_*, omle_session_*).
add_omle_test(test_api             test_api.cpp)
omle_test_uses_proto(test_api)
