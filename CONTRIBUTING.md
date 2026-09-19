# Contributing to omle-runtime

Thanks for your interest. This document covers how to work on this repository —
setup, the layout, and the checks a change needs to pass.

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).

## What lives here

The C++ inference engine that executes OMLE models, plus its Python (pybind11)
and Java bindings and a stable C ABI.

The format this runtime consumes is defined in
[omle](https://github.com/openmle/omle) — the proto schema, the operator
registry, and the semantics of each operator. A change to what an operator
*means* belongs there; this repository implements it.

| Repository | Owns |
|---|---|
| [omle](https://github.com/openmle/omle) | Format, registries, validation |
| [omle-convert](https://github.com/openmle/omle-convert) | Converters that produce `.omle` files |
| [omle-server](https://github.com/openmle/omle-server) | Inference server built on this runtime |
| [omle.js](https://github.com/openmle/omle.js) | TypeScript engine — the other implementation of these semantics |

## Getting started

Requires a C++17 compiler, CMake, protobuf, and GoogleTest.

```bash
# macOS
brew install cmake protobuf googletest

mkdir -p build && cd build
cmake ..
cmake --build . -j8
ctest --output-on-failure
```

10 test targets, all of which should pass.

The protobuf C++ sources are generated at build time from
`../omle/protobuf/omle.proto` — the sibling `omle` checkout must be
present. If you upgrade protobuf, delete `build/` and reconfigure: previously
generated `.pb.h` files are version-locked to the runtime they were built
against, and the mismatch surfaces as
`"Protobuf C++ gencode is built with an incompatible version"`.

### Python bindings

Off by default. They need pybind11:

```bash
pip install pybind11
cmake -DBUILD_PYTHON=ON -Dpybind11_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())") ..
cmake --build . --target omle_ext -j8
```

Without this the CMake output says `pybind11 not found — omle_ext skipped`
and `import omle_runtime` fails. Worth knowing because omle-convert's
runtime-verification tests silently *skip* rather than fail when the extension
is missing — a passing test run with several hundred skips usually means this.

## Layout

```
include/omle/    public headers — the API other projects compile against
  runtime.h         Model, Session, LoadOptions, InputSpec, OutputSpec
  tensor.h          Tensor, DataType
  status.h          Status, ErrorCode
  c_api.h           stable C ABI
src/                engine internals
  graph_executor.*  DAG execution
  operator_registry.*  the omle.core / feature / text operator implementations
  nodes/            structured model bodies (tree ensemble, linear, SVM, …)
  expression_eval.* the elementwise expression DSL
python/             pybind11 module + the omle_runtime package
java/               JNI bindings (Maven)
spark/              Spark integration (Scala)
tests/              GoogleTest suites, one per area
```

### Namespaces

Three, and they are easy to confuse:

| Namespace | Contents |
|---|---|
| `omle::rt` | Public API — `Model`, `Session`, `Tensor`, `DataType`, `LoadOptions` |
| `omle::rt::impl` | Engine internals |
| `omle` | **Generated protobuf types** (the proto declares `package omle`) |

There is no `omle::v1`. Code referring to it predates the current schema.

## Code style

C++ follows the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)
strictly — 2-space indent, 80 columns, no deviations. `.clang-format` is just
`BasedOnStyle: Google`.

```bash
find src include tests examples -name '*.cpp' -o -name '*.h' | xargs clang-format -i
```

Headers use `#define` include guards named `<PROJECT>_<PATH>_<FILE>_H_` with the
source root stripped, matching the style guide — `include/omle/tensor.h` is
`OMLE_TENSOR_H_`, `src/nodes/linear_node.h` is
`OMLE_NODES_LINEAR_NODE_H_`. Not `#pragma once`.

## Adding an operator

1. Implement it in `src/operator_registry.cpp` and register it in the table at
   the bottom of that file, under the right namespace name.
2. Match the contract in the `omle` registry exactly — input kinds, attribute
   names, output shape and dtype rules. Attributes that the registry marks
   optional must be handled when absent; `StandardScaler` skipping centering
   when `mean` is missing is the pattern.
3. Add a GoogleTest case in `tests/`.
4. The same operator in [omle.js](https://github.com/openmle/omle.js)
   should agree numerically. Note in your PR if it needs a matching change.

## Pre-commit hooks (optional)

```bash
pip install pre-commit
pre-commit install
```

Runs `clang-format` plus the hygiene hooks.

## Tests

Add a GoogleTest case with any behaviour change. For a bug fix, a test that
fails before the fix is the most useful thing you can include.

Please run `ctest` before pushing — and if you touched anything under
`include/`, also rebuild `omle_ext` and the dependent projects, since the
public headers are compiled against by omle-server and the Python bindings.

## Reporting bugs

Include the model file (or the script that produced it), the platform and
compiler, and the observed versus expected output. For a numeric discrepancy,
the same model scored by omle-convert's source framework is the most useful
comparison.

## License

Contributions are accepted under the [Apache License 2.0](LICENSE), in
accordance with section 5 of that license. There is no separate CLA.
