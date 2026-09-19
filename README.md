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

# 2. Java bindings  →  java/target/omle-runtime-<version>.jar
cd java && mvn package && cd ..

# 3. Spark transformer  →  spark/target/scala-2.13/omle-spark_2.13-<version>.jar
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
| x86-64 | yes | — | yes |
| arm64 | yes | yes | — |

Anything outside that table still needs a locally built library and one of the
two properties above.

## Performance

Speedup over the framework's own predictor, so `>1` means faster than native.
`small` models, `min` latency, Apple M-series, **every engine pinned to one
thread**. Cells read *ORT / OMLE*; bold marks the faster of the two. Full method,
model configs and the medium/large sizes are in
[`benchmark/README.md`](benchmark/README.md).

| Model | batch 1 | batch 10 | batch 100 | batch 1 000 | batch 10 000 |
|-------|----------:|----------:|----------:|----------:|----------:|
| xgb/regression | 13.99x / **16.64x** | 5.41x / **7.74x** | 0.87x / **2.43x** | 0.31x / **1.39x** | 0.26x / **1.21x** |
| xgb/binary | 10.45x / **15.31x** | 3.89x / **7.87x** | 0.82x / **2.31x** | 0.48x / **1.30x** | 0.44x / **1.11x** |
| xgb/multiclass | 6.82x / **13.75x** | 1.22x / **3.99x** | 0.29x / **1.64x** | 0.23x / **1.08x** | 0.22x / **1.03x** |
| xgb/mnist | 7.43x / **9.63x** | 1.62x / **2.28x** | 0.37x / **0.93x** | 0.28x / **0.75x** | 0.27x / **0.68x** |
| lgbm/regression | 4.97x / **5.72x** | 3.33x / **3.97x** | 2.16x / **3.36x** | 0.86x / **3.36x** | 0.75x / **3.31x** |
| lgbm/binary | 3.60x / **5.54x** | 1.26x / **3.85x** | 1.26x / **3.09x** | 1.46x / **2.65x** | 1.46x / **2.84x** |
| lgbm/multiclass | 2.12x / **3.62x** | 1.02x / **3.37x** | 0.71x / **3.43x** | 0.67x / **3.37x** | 0.67x / **3.33x** |
| lgbm/mnist | 2.64x / **3.16x** | 1.37x / **1.89x** | 0.79x / **1.71x** | 0.65x / **1.53x** | 0.70x / **1.43x** |
| adult pipeline | 147.92x / **181.00x** | **96.89x** / 86.77x | **20.47x** / 15.88x | **2.78x** / 1.83x | **1.39x** / 0.75x |

**Small batches are where the margin is largest.** At batch=1 OMLE is 9.6–16.6x
faster than XGBoost and 3.2–5.7x faster than LightGBM, because neither
framework's entry point is built for one row — XGBoost constructs a `DMatrix` per
call. The adult pipeline is the extreme at 181x, since the native path re-runs
the whole sklearn `ColumnTransformer` every call while OMLE compiles it into the
graph.

**The margin narrows with batch size but does not invert**, except on
`xgb/mnist`. ORT is the contrast: at batch 10 000 it drops below native on seven
of the nine models, to 0.22x on `xgb/multiclass`, so it is a latency engine here,
while OMLE holds 1.03–3.33x everywhere but `xgb/mnist`. Note both
libraries predict on *every core* unless told otherwise — leave OMLE at its
one-thread default against a stock XGBoost and you are measuring core count, not
kernels. Given equal threads OMLE scales slightly better than either framework
(4.7–5.5x on 11 cores against 4.2–5.3x), so at batch 10 000 the lead *widens*
with threads rather than holding: `xgb/regression` goes 1.37x → 1.76x and
`lgbm/binary` 3.23x → 3.72x.

`xgb/mnist` is the one model below parity at scale. Wide feature rows cost OMLE
more than they cost XGBoost: from 8 features to 784, XGBoost's time per node
visit rises about 20% while OMLE's roughly doubles. The cause is not the
cache-block size — sweeping it from 4 to 256 rows moves MNIST by under 2% — nor
the row-major layout, since transposing each block to column-major measured
consistently slower. It is the one model that also scales worse than native
(`lgbm/mnist`, 3.96x against 5.14x on 11 cores), which points at memory
bandwidth rather than cache locality.

**Tuning for throughput.** Two load-time settings, and the product is what
counts. They are `LoadOptions` fields in C++ (*C++ API* below), the same two
fields in the C API, and keyword arguments in Python; the JVM bindings take the
thread count only and leave the row floor at 64:

```cpp
omle::LoadOptions opts;
opts.n_threads         = 8;   // 0 = auto (hardware_concurrency); default 1
opts.min_parallel_rows = 64;  // per-thread row floor; default 64
```

A batch is split only once it holds `n_threads * min_parallel_rows` rows, so at
`n_threads = 8` anything under 512 rows still runs serially — raising the thread
count alone can leave mid-sized batches untouched. Serving batches in the
hundreds, lower `min_parallel_rows` alongside it; the benchmark README measures
2.4-4.3x left unclaimed at batches of 100-500 on the default.

`Session::run` is always serial on the calling thread whatever these are set to,
which is what you want when the server is already concurrent. `Model::predict` is
the one that uses the pool.

See [`benchmark/README.md`](benchmark/README.md) for the full five-batch matrix,
ONNX Runtime comparison, memory figures and how to reproduce all of it.

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
opts.min_parallel_rows = 64;   // rows per thread needed before the pool is used;
                               // a batch engages it at n_threads * this (see Performance)
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
import omle_runtime as rt
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

Maven coordinates `io.github.openmle:omle-runtime`, built with `mvn package` in `java/`.
The version comes from the git tag — `mvn -Drevision=<version>` in CI, and an
untagged local build produces `0.1.0-SNAPSHOT`.
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

`io.github.openmle:omle-spark`, built with `sbt package` in `spark/`, versioned
from the git tag by sbt-dynver. `OMLEModel` is a
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
from omle_spark import OMLEModel
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
them. A source build leaves both in the build directory, where they are invoked
as `./omle-predict`.

Only `omle-predict` ships in the wheel, where `pip install omle-runtime` puts it
on `PATH`. `omle-benchmark` is a tool for tuning the runtime rather than using
it, so it is not worth ~1.5 MB in every wheel — build from source to get it.

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
