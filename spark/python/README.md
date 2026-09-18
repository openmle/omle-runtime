# omle-spark

Score OMLE models on Spark DataFrames.

`OMLEModel` is a Spark ML `Transformer` that delegates to the JVM-side
`io.github.openmle.spark.OMLEModel`, so scoring runs natively on each executor
rather than through a Python UDF.

```python
from omle.spark import OMLEModel

model = OMLEModel(modelPath="/path/to/model.omle")
predictions = model.transform(df)
```

Input columns are resolved from the model's own input specs. A model with a
single rank-2 input reads from `featuresCol`, following the usual Spark ML
convention for a pre-assembled vector:

```python
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=["f0", "f1"], outputCol="features")
predictions = model.transform(assembler.transform(df))
```

A model with multiple or scalar inputs reads each one by name straight from the
DataFrame, so no `VectorAssembler` is needed.

Output depends on the model's output count: a single output produces
`predictionCol` (`DoubleType`); multiple outputs produce `probabilityCol`
(`VectorType`) plus `predictionCol` holding the argmax.

## Requirements

Two JARs on the driver and executor class-paths: `omle-spark` and
`omle-runtime`. The released `omle-runtime` jar carries the native library for
linux, macOS and Windows on x86-64, plus linux and macOS on arm64, and JNA
extracts the right one per JVM — so there is nothing to install on the nodes and
no `jna.library.path` to set. A locally built jar has no bundled library, and
then JNA does need pointing at one on every node.

Spark 3.5 (Scala 2.12) and Spark 4.x (Scala 2.13) are both supported, each with
its own build of the `omle-spark` JAR.

## Related packages

- [`omle`](https://pypi.org/project/omle/) — the model IR and converters
- [`omle-runtime`](https://pypi.org/project/omle-runtime/) — the local runtime

## License

Apache-2.0
