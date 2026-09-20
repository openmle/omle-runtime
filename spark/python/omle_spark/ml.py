"""PySpark wrapper for OMLEModel.

Mirrors the Spark ML Python API pattern: a thin Python class that delegates
``transform`` to the JVM-side ``io.github.openmle.spark.OMLEModel`` Scala class.

The JARs this needs ship inside the package, so nothing has to be built and no
native library has to be installed. They must be on the class-path before the
JVM starts, which means naming them when the session is built::

    import omle_spark
    spark = (SparkSession.builder
             .config("spark.jars", omle_spark.jars_classpath())
             .getOrCreate())

``spark.jars`` also ships them to the executors, and JNA extracts the right
``libomleruntime`` for each node's platform out of the omle-runtime JAR, so
``-Djna.library.path`` is not needed. Setting this after ``getOrCreate()`` has
no effect and ``transform`` then fails with ``'JavaPackage' object is not
callable``.

Input columns are resolved from the model's declared input specs:

* **Single rank-2 input** (shape ``[-1, n_features]``): reads from *featuresCol*
  (standard Spark ML convention for pre-assembled feature vectors).
* **Multiple inputs or scalar inputs**: reads each input by its spec name
  directly from the DataFrame — no ``VectorAssembler`` required.

Examples
--------
Pre-assembled vector (single featuresCol)::

    from pyspark.ml.feature import VectorAssembler
    assembler = VectorAssembler(inputCols=["f0", "f1"], outputCol="features")
    model = OMLEModel.loadFile("/path/to/model.omle")
    predictions = model.transform(assembler.transform(df))

Scalar columns from a raw DataFrame (multi-input pipeline model)::

    # model was converted from Pipeline([VectorAssembler([\"a\",\"b\"]), RF])
    # its input specs are \"a\" and \"b\" (scalar)
    model = OMLEModel.loadFile("/path/to/pipeline.omle")
    predictions = model.transform(raw_df)   # raw_df has columns \"a\", \"b\"
"""

from __future__ import annotations

from pyspark import keyword_only
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.wrapper import JavaTransformer


class OMLEModel(JavaTransformer):
    """Spark ML Transformer that scores an OMLE model.

    Loads an OMLE protobuf model from *modelPath* and applies it to a
    DataFrame.  The input features are read from a single assembled
    :class:`pyspark.ml.linalg.Vector` column (*featuresCol*).  Use
    :class:`~pyspark.ml.feature.VectorAssembler` to build one from individual
    numeric columns.

    Output columns depend on the model's output count:

    * ``numOutputs == 1`` → *predictionCol* (``DoubleType``)
    * ``numOutputs  > 1`` → *probabilityCol* (``VectorType``) +
      *predictionCol* (``DoubleType``, argmax index)

    Parameters
    ----------
    modelPath : str
        Local (or HDFS-cached) path to the ``.omle`` model file.
    featuresCol : str, default ``"features"``
        Column containing the assembled feature ``Vector``.
    predictionCol : str, default ``"prediction"``
        Name of the output prediction column.
    probabilityCol : str, default ``"probability"``
        Name of the output probability column (multi-output models only).
    """

    modelPath = Param(
        Params._dummy(),
        "modelPath",
        "Local (or HDFS-cached) path to the OMLE serialised model (.omle file)",
        typeConverter=TypeConverters.toString,
    )
    featuresCol = Param(
        Params._dummy(),
        "featuresCol",
        "Column containing the assembled feature Vector",
        typeConverter=TypeConverters.toString,
    )
    predictionCol = Param(
        Params._dummy(),
        "predictionCol",
        "Output column name for the model prediction (Double)",
        typeConverter=TypeConverters.toString,
    )
    probabilityCol = Param(
        Params._dummy(),
        "probabilityCol",
        "Output column name for class probabilities (Vector); added only for multi-output models",
        typeConverter=TypeConverters.toString,
    )

    @keyword_only
    def __init__(
        self,
        *,
        modelPath: str | None = None,
        featuresCol: str = "features",
        predictionCol: str = "prediction",
        probabilityCol: str = "probability",
    ) -> None:
        super().__init__()
        self._java_obj = self._new_java_obj(
            "io.github.openmle.spark.OMLEModel", self.uid
        )
        self._setDefault(
            featuresCol="features",
            predictionCol="prediction",
            probabilityCol="probability",
        )
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(
        self,
        *,
        modelPath: str | None = None,
        featuresCol: str = "features",
        predictionCol: str = "prediction",
        probabilityCol: str = "probability",
    ) -> "OMLEModel":
        """Set parameters; returns ``self``."""
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    def loadFile(cls, path: str) -> "OMLEModel":
        """Build a transformer for the ``.omle`` model at *path*.

        The file is opened and closed on the driver first, so a missing path or
        an unreadable model raises here rather than inside ``transform`` on
        every executor. Only the path is kept: the native model handle is not
        serialisable, so each partition reloads it at run time.

        Validation is delegated to the Scala companion's ``loadFile``, so both
        bindings accept and reject exactly the same files.

        Named ``loadFile`` rather than ``load_file`` to match the rest of this
        class and the Spark ML Python API, which are camelCase throughout, and
        ``load`` is left free for ``MLReadable``.

        Parameters
        ----------
        path : str
            Local (or HDFS-cached) path to the ``.omle`` model file.

        Returns
        -------
        OMLEModel
            A transformer with *modelPath* set to *path*.

        Raises
        ------
        RuntimeError
            If no active :class:`~pyspark.SparkContext` is available, since the
            JVM is needed to read the model.
        Py4JJavaError
            If the model cannot be loaded.

        Examples
        --------
        ::

            model = OMLEModel.loadFile("/path/to/model.omle")
            predictions = model.transform(df)
        """
        from pyspark import SparkContext

        sc = SparkContext._active_spark_context
        if sc is None:
            raise RuntimeError(
                "OMLEModel.loadFile needs an active SparkContext to read the "
                "model on the driver; create a SparkSession first, or use "
                "OMLEModel(modelPath=...) to defer loading."
            )
        # Discarded: this call exists for its exceptions. Constructing through
        # the Param below is what makes copy() and JVM sync work.
        sc._jvm.io.github.openmle.spark.OMLEModel.loadFile(path)
        return cls(modelPath=path)

    # ── Setters ──────────────────────────────────────────────────────────────

    def setModelPath(self, value: str) -> "OMLEModel":
        return self._set(modelPath=value)

    def setFeaturesCol(self, value: str) -> "OMLEModel":
        return self._set(featuresCol=value)

    def setPredictionCol(self, value: str) -> "OMLEModel":
        return self._set(predictionCol=value)

    def setProbabilityCol(self, value: str) -> "OMLEModel":
        return self._set(probabilityCol=value)

    # ── Getters ──────────────────────────────────────────────────────────────

    def getModelPath(self) -> str:
        return self.getOrDefault(self.modelPath)

    def getFeaturesCol(self) -> str:
        return self.getOrDefault(self.featuresCol)

    def getPredictionCol(self) -> str:
        return self.getOrDefault(self.predictionCol)

    def getProbabilityCol(self) -> str:
        return self.getOrDefault(self.probabilityCol)
