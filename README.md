# OMLE Runtime

High-performance C++ inference engine for classical machine learning models serialized in OMLE format. Supports batch prediction with multi-threaded execution, bindings for Python/NumPy, Java, Scala/Spark and PySpark, and a stable C ABI for FFI integration.

The runtime implements the OMLE standard in full: every model type, and every operator in the `omle.core`, `omle.feature`, `omle.ml` and `omle.text` domains.

## Building

**Requirements:** CMake ≥ 3.16, C++17 compiler, Protobuf (3.x).

Release builds target AVX2 + FMA on x86, so binaries require a Haswell-era
or newer CPU (2013+); arm64 uses NEON, which every ARMv8 chip has. The
baseline is explicit rather than `-march=native` so a binary built on one
machine runs on another — the Python wheel is built from this same
configuration.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

**CMake options:**

| Option | Default | Description |
|---|---|---|
| `BUILD_TESTS` | `ON` | Build the gtest test suite |
| `BUILD_PYTHON` | `OFF` | Build the Python/pybind11 extension |

The build produces a shared library and two CLI tools, `omle-predict` and
`omle-benchmark`. Artifact names follow each platform's convention:

| Linux | macOS | Windows |
|---|---|---|
| `libomleruntime.so` | `libomleruntime.dylib` | `omleruntime.dll` |

There is one library, and it exports both surfaces: the C++ API
(`omle::rt::Model`, `omle::rt::Tensor`) and the flat C ABI from `omle/c_api.h`.
C++ callers such as omle-server link it directly; the Java, Scala and PySpark
bindings load the same file through JNA and use only the C entry points.

Because the C++ surface crosses the shared-library boundary, a C++ consumer
should be built with the same compiler and standard library as the runtime.
Consumers that stick to the C ABI have no such constraint.

### Python extension

```bash
cmake -S . -B build -DBUILD_PYTHON=ON
cmake --build build
pip install -e python/
```

### JVM bindings (Java and Spark)

The Java, Scala and PySpark bindings all call the C ABI through JNA, so they need
the shared library `libomleruntime`, which the default build already produces —
no extra option and no Python toolchain required:

```bash
# 1. native shared library (libomleruntime)
cmake -S . -B build
cmake --build build

# 2. Java bindings  →  java/target/omle-runtime-0.1.0.jar
cd java && mvn package && cd ..

# 3. Spark transformer  →  spark/target/scala-2.13/omle-spark_2.13-0.1.0.jar
#    (reads the jar produced in step 2, so run it after)
cd spark && sbt package && cd ..
```

Point JNA at the directory holding `libomleruntime` with either `-Djna.library.path=<dir>`
or the binding-specific `-Dio.github.openmle.libpath=<dir>`.

This applies to a build from source, where the library sits in the build tree.
A **released** jar needs none of it: it carries `libomleruntime` for every
supported platform at the layout JNA searches, so `Native.load` finds and
extracts the right one on its own.

| | linux | macOS | Windows |
|---|---|---|---|
| x86-64 | yes | yes | yes |
| arm64 | yes | yes | — |

Anything outside that table still needs a locally built library and one of the
two properties above.

## C++ API

### Loading a model

```cpp
#include "omle/runtime.h"

// From file
auto result = omle::Model::load("model.omle");
if (!result.ok()) {
    std::cerr << result.message() << "\n";
    return 1;
}
auto model = std::move(*result);

// From memory buffer
auto result = omle::Model::load(data_ptr, data_size);
```

`Model::load` accepts an optional `LoadOptions`:

```cpp
omle::LoadOptions opts;
opts.n_threads         = 4;    // 0 = auto (hardware_concurrency), 1 = single-threaded
opts.min_parallel_rows = 64;   // minimum batch size to engage the thread pool
opts.run_verification  = true; // run built-in correctness cases embedded in the model
opts.run_warmup        = true; // execute warmup passes to pre-populate caches

auto model = omle::Model::load("model.omle", opts).value();
```

### Batch prediction

```cpp
// Build input tensors (row-major float32)
omle::Tensor input = omle::Tensor::from_floats(n_samples, n_features, feature_data);

// Named inputs → named outputs
auto outputs = model->predict({{"features", input}}).value();
const omle::Tensor& scores = outputs.at("score");

// Access results row by row
for (int i = 0; i < scores.n_rows; ++i) {
    const float* row = scores.row(i);  // pointer to n_cols floats
}
```

### Session API (per-thread, reusable)

`Session` is cheaper than calling `predict()` in a tight loop because it retains its execution buffers between runs.

```cpp
auto session = model->create_session();

session->bind_input("features", omle::Tensor::from_floats(n, d, data));
omle::Status st = session->run();
if (!st.ok()) { /* handle error */ }

const auto& results = session->results();         // all outputs
auto score = session->get_output("score").value(); // single named output
```

### Tensor construction

```cpp
// Dense float32 — row-major layout
omle::Tensor t = omle::Tensor::from_floats(n_rows, n_cols, flat_vector);
omle::Tensor t = omle::Tensor::f32(n_rows, n_cols, fill_value);

// Sparse CSR float32
omle::Tensor t = omle::Tensor::sparse_csr_f32(
    n_rows, n_cols, values, col_indices, row_indptr, default_fill);

// Zero-copy view of an existing buffer
omle::Tensor t = omle::Tensor::from_buffer(
    omle::DataType::Float32, n_rows, n_cols, ptr, byte_count);
```

### Error handling

```cpp
// StatusOr<T> — carries either T or an ErrorCode + message
auto model_or = omle::Model::load("model.omle");
if (!model_or.ok()) {
    omle::ErrorCode code = model_or.code();   // e.g. ErrorCode::FileNotFound
    std::string msg         = model_or.message();
}
auto model = std::move(model_or).value();        // throws std::runtime_error if !ok()

// Status — success or error, no value
omle::Status st = session->run();
if (!st.ok()) { /* st.code(), st.message() */ }
```

### Model introspection

```cpp
for (const auto& inp : model->inputs())
    printf("input: %s\n", inp.name.c_str());

for (const auto& out : model->outputs())
    printf("output: %s  role: %d\n", out.name.c_str(), (int)out.role);
```

## C API

A stable C ABI is provided for use from any language with a C FFI.

```c
#include "omle/c_api.h"

// Load
omle_model_t* model = NULL;
omle_load_options_t opts = {0};
opts.n_threads = 1;
omle_status_t st = omle_model_load_file("model.omle", &opts, &model);
if (st != OMLE_OK) {
    fprintf(stderr, "%s\n", omle_last_error());
    return 1;
}

// Build input tensor
omle_tensor_t* input = NULL;
omle_tensor_create_f32(n_samples, n_features, data, &input);

// Predict
const char* input_name = "features";
omle_tensor_t* input_tensors[] = {input};
const char** out_names = NULL;
omle_tensor_t** out_tensors = NULL;
size_t n_outputs = 0;
st = omle_model_predict(model, 1, &input_name, input_tensors,
                           0, NULL, &n_outputs, &out_names, &out_tensors);

// Cleanup
omle_model_destroy(model);
omle_free_tensor(input);
```

Error codes: `OMLE_OK`, `OMLE_ERR_FILE_NOT_FOUND`, `OMLE_ERR_PARSE`, `OMLE_ERR_INVALID_GRAPH`, `OMLE_ERR_VERIFICATION_FAILED`, `OMLE_ERR_MISSING_INPUT`, `OMLE_ERR_OUTPUT_NOT_PRODUCED`, `OMLE_ERR_OUTPUT_NOT_FOUND`, `OMLE_ERR_UNKNOWN_OPERATOR`, `OMLE_ERR_INVALID_ARGUMENT`.

## Python API

```python
import omleruntime as rt
import numpy as np

model = rt.load("model.omle", n_threads=4)

X = np.random.rand(1000, 20).astype(np.float32)
scores = model.predict(X)   # returns float32 ndarray [n_samples, n_outputs]

# Session interface
session = model.create_session()
session.bind_input("features", X)
session.run()
result = session.get_output("score")
```

Input arrays are automatically coerced to float32 C-contiguous layout. The GIL is released during inference.

## Java API

Maven coordinates `io.github.openmle:omle-runtime:0.1.0`, built with `mvn package` in `java/`.
The `io.github.openmle` namespace is the GitHub-backed one on Maven Central: it is
verified by owning the [openmle](https://github.com/openmle) GitHub organization, so
publishing needs no custom domain.
The binding is a JNA wrapper over the C ABI, so the native `libomleruntime` has to
be loadable at run time. The released jar bundles it for the platforms listed
above and JNA extracts it automatically, so a consumer of that jar configures
nothing. Building from source, point JNA at the build output with
`jna.library.path` or `-Dio.github.openmle.libpath=<dir>`.

`Model`, `Session` and `Tensor` all implement `AutoCloseable`; use
try-with-resources so the native handles are released deterministically.

```java
import io.github.openmle.runtime.Model;
import io.github.openmle.runtime.Session;

// Load from a file (or Model.loadBytes(byte[]) for an in-memory model).
// The second argument is the thread count; the single-argument form uses 1.
try (Model model = Model.loadFile("model.omle", 4)) {

    System.out.println(model.numInputs() + " inputs, " + model.numOutputs() + " outputs");

    // Stateless batch prediction: one row per sample.
    float[][] X = {
        {0.1f, 0.9f},
        {0.7f, 0.2f},
    };
    float[] scores = model.predict(X);       // flat, n_samples * n_outputs

    // Or pass an already-flat array to avoid the row-array allocation.
    float[] flat = {0.1f, 0.9f, 0.7f, 0.2f};
    float[] same = model.predict(flat, 2, 2);

    // Reusable per-thread session — create one per thread, not one per call.
    try (Session session = model.createSession()) {
        float[] s = session.predict(X);
    }
}
```

`Model.loadFile` and `Model.loadBytes` throw `OMLEException` on a missing file,
an unparseable model, or an invalid graph.

For schema-driven input, `predictColumns(String[] names, int[] types, Object[] cols, int nRows)`
feeds named columns directly, which suits models converted from a pipeline whose
inputs are scalar columns rather than one assembled vector.

## Spark (Scala / JVM)

`io.github.openmle:omle-spark:0.1.0`, built with `sbt package` in `spark/`. `OMLEModel` is a
plain Spark ML `Transformer`, so it drops into a `Pipeline` like any other stage.
Both the `omle-spark` JAR and the `omle-runtime` JAR must be on the driver and
executor class-paths. With a released `omle-runtime` jar that is the whole setup:
the native library travels inside it, so every executor JVM extracts its own copy
and nothing has to be staged on the nodes.

Only when the `omle-runtime` jar was built locally — and so carries no bundled
library — does JNA need pointing at one that exists on **every node**:

```
--conf spark.driver.extraJavaOptions=-Djna.library.path=/opt/omle/lib
--conf spark.executor.extraJavaOptions=-Djna.library.path=/opt/omle/lib
```

```scala
import io.github.openmle.spark.OMLEModel
import org.apache.spark.ml.feature.VectorAssembler

val df = Seq((0.1f, 0.9f), (0.7f, 0.2f)).toDF("f0", "f1")

val features = new VectorAssembler()
  .setInputCols(Array("f0", "f1"))
  .setOutputCol("features")
  .transform(df)

// loadFile opens the model once on the driver, so a bad path fails here
// rather than inside transform on every executor.
val model = OMLEModel.loadFile("/path/to/model.omle")
  .setFeaturesCol("features")
  .setPredictionCol("prediction")
  .setProbabilityCol("probability")

model.transform(features).select("prediction", "probability").show()
```

`prediction` is always added, as a `Double`. `probability` is a `Vector` and is only
added when the model produces more than one output column per row — so a regression
model yields `prediction` alone, and selecting `probability` on one would fail.

Scoring runs inside the executors, one native session per partition.

## PySpark

`pip install omle-spark`. A thin wrapper that delegates `transform` to the same
JVM `io.github.openmle.spark.OMLEModel`, so it needs the same two JARs on the
class-path — and, like the Scala API, no `jna.library.path` when the
`omle-runtime` jar is a released one.

Input columns are resolved from the model's declared input specs. A model with a
single rank-2 input (`[-1, n_features]`) reads from `featuresCol`, the usual Spark
ML convention for a pre-assembled vector:

```python
from omle.spark import OMLEModel
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["f0", "f1"], outputCol="features")

# loadFile opens the model once on the driver, so a bad path fails here
# rather than inside transform on every executor.
model = OMLEModel.loadFile("/path/to/model.omle")
predictions = model.transform(assembler.transform(df))
predictions.select("prediction", "probability").show()
```

As on the JVM side, `probability` is only present for models with more than one
output column per row; a regression model produces just `prediction`.

A model with multiple or scalar inputs reads each input by its spec name straight
from the DataFrame, so no `VectorAssembler` is needed:

```python
# Converted from Pipeline([VectorAssembler(["a", "b"]), RandomForestClassifier()]);
# the model's input specs are the scalar columns "a" and "b".
model = OMLEModel.loadFile("/path/to/pipeline.omle")
predictions = model.transform(raw_df)      # raw_df has columns "a" and "b"
```

The API mirrors the Scala side: the `OMLEModel.loadFile` factory, and the
setters `setModelPath`, `setFeaturesCol`, `setPredictionCol` and
`setProbabilityCol`, each returning `self` for chaining. `OMLEModel(modelPath=...)`
still works and defers loading, which is what you want if the file only becomes
readable on the executors.

## CLI tools

Both tools are statically linked, so they run from anywhere with nothing beside
them. `pip install omle-runtime` puts them on `PATH`; a source build leaves them
in the build directory, where they are invoked as `./omle-predict`.

### `omle-predict`

```
omle-predict <model.omle> [features.csv] [predictions.csv]
```

Input CSV: one sample per line, comma-separated floats, no header. Without it,
the metadata is printed and nothing is scored; without `predictions.csv`, the
input is scored but not written. Predictions go to the file, never to stdout.

```console
$ omle-predict model.omle features.csv predictions.csv
Model loaded: 2 features, 1 outputs

Inputs (1):
  X                     dtype=11        shape=[-1,2]

Outputs (1):
  score                 dtype=0         role=0     shape=[]

Scored 4 records (2 features in, 1 column out) in 0.02 ms
Wrote 4 rows to predictions.csv
```

### `omle-benchmark`

```
omle-benchmark <model.omle> [n_samples] [n_threads] [n_warmup] [n_reps]
```

Generates random inputs and reports mean, min and max latency in milliseconds,
throughput in samples/sec, and resident memory. Only the model is required; the
remaining arguments are positional.

```bash
omle-benchmark model.omle 10000 4 5 20
```

## Running tests

```bash
cmake -S . -B build -DBUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

The binding suites are separate, and both need `libomleruntime` built first (see
[JVM bindings](#jvm-bindings-java-and-spark)):

```bash
cd java  && mvn test     # JUnit 5: loading, schema, tensors, sessions
cd spark && sbt test     # ScalaTest: Transformer against a local SparkSession
```

## Architecture notes

**Graph execution** — models are compiled into a static DAG of `GraphNode` objects sharing a `ValueStore` (name → tensor map). Each node reads its declared inputs and writes its outputs; no dynamic dispatch after load. Model types compose freely: any combination can be nested into a single graph via the `Composite` node type, which runs its sub-DAG in an isolated scope.

**Thread model** — a fixed-size `ThreadPool` is created once at load time. `Model::predict` uses `parallel_for` to divide the batch across threads. `Session::run` always executes serially on the calling thread, making it suitable for per-request use in an already-concurrent server.

**Sparse tensors** — CSR (Compressed Sparse Row) format is supported throughout. Most compute nodes convert to dense float32 on first use; the schema preprocessing node handles missing-value policies before any model node runs.

**Anomaly-detection outputs** — these models expose a `score` / `decision_value` / `prediction` contract: `score` is the detector's normalized score where higher means more normal, `decision_value` is `score - offset`, and `prediction` is `+1` for an inlier and `-1` for an outlier. A node gets whichever of the three it declares, so a score-only model stays a single output.

**Verification** — model files may embed reference input/output pairs. When `LoadOptions::run_verification = true` (the default), these are run immediately after loading and the load fails if results fall outside the declared tolerance.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and the checks a
change needs to pass, and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for community
expectations.

## License

[Apache License 2.0](LICENSE)
