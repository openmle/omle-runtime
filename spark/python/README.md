# omle-spark

Score OMLE models on Spark DataFrames.

`OMLEModel` is a Spark ML `Transformer` that delegates to the JVM-side
`io.github.openmle.spark.OMLEModel`, so scoring runs natively on each executor
rather than through a Python UDF.

```python
from omle_spark import OMLEModel

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

## Setting up the session

The JARs ship inside this package, so there is nothing to build and no native
library to install on the nodes. They do have to be on the class-path before the
JVM starts, so name them when the session is created:

```python
import omle_spark
from pyspark.sql import SparkSession

spark = (SparkSession.builder
         .config("spark.jars", omle_spark.jars_classpath())
         .getOrCreate())
```

`spark.jars` also ships them to the executors, and JNA extracts the right
`libomleruntime` for each node's platform out of the bundled `omle-runtime` JAR
— covering linux, macOS and Windows on x86-64 plus linux and macOS on arm64 — so
`jna.library.path` is not needed anywhere.

Configuring this after `getOrCreate()` has no effect: a JVM already running
cannot be given new JARs, and `transform` then fails with `'JavaPackage' object
is not callable`, which names no cause.

Spark 3.5 (Scala 2.12) and Spark 4.x (Scala 2.13) are both supported. The two
are binary-incompatible, so the package carries a build for each and
`jars_classpath()` picks the one matching the installed PySpark — read from
PySpark's own `spark-core` JAR name, not guessed from its version.

`omle_spark.jars()` returns the same paths as a list, for anything that wants
them individually.

## Related packages

- [`omle`](https://pypi.org/project/omle/) — the model IR and converters
- [`omle-runtime`](https://pypi.org/project/omle-runtime/) — the local runtime

## License

Apache-2.0
