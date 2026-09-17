package io.github.openmle.spark

import io.github.openmle.runtime.{DataType, Model}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.{SQLDataTypes, Vector, Vectors}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/**
 * OMLEModel — wraps an OMLE protobuf model as a Spark ML [[Transformer]].
 *
 * Loads a model from a local `.omle` file and applies it to a Spark DataFrame.
 *
 * Input columns are determined by the model's declared input specs:
 *  - Single rank-2 input (shape `[-1, n]`): reads from *featuresCol*
 *    (standard Spark ML convention for pre-assembled feature vectors).
 *  - Multiple inputs, or a single rank-1 (scalar) input: reads each input by
 *    its spec name directly from the DataFrame.  Individual columns may be
 *    scalar numerics (`DoubleType`, `LongType`, etc.) or assembled
 *    `Vector` columns.
 *
 * Output columns depend on the model's per-row output count:
 *  - `outputCols == 1` → *predictionCol* (`DoubleType`)
 *  - `outputCols  > 1` → *probabilityCol* (`VectorType`) +
 *                        *predictionCol* (`DoubleType`, argmax index)
 *
 * Example — pre-assembled vector (single featuresCol):
 * {{{
 *   val model = OMLEModel.loadFile("/path/to/model.omle")
 *   val predictions = model.transform(assembler.transform(df))
 * }}}
 *
 * Example — scalar columns directly from a raw DataFrame:
 * {{{
 *   // model was converted from Pipeline([VectorAssembler, RF])
 *   // its inputs are "size", "weight", "is_premium" (scalar)
 *   val model = OMLEModel.loadFile("/path/to/pipeline.omle")
 *   val predictions = model.transform(rawDf)
 * }}}
 */
class OMLEModel(override val uid: String) extends Transformer {

  def this() = this(Identifiable.randomUID("omleModel"))

  // ── Params ──────────────────────────────────────────────────────────────────

  val modelPath: Param[String] =
    new Param(this, "modelPath",
      "Local (or HDFS-cached) path to the OMLE serialised model (.omle file)")

  // Needed because a single rank-2 input carries no usable column name: the
  // converter emits the placeholder "X" for it, which is not what the caller's
  // VectorAssembler wrote the vector to. The model cannot know that name, so it
  // has to come from here -- the same reason Spark's own PredictionModel gives
  // every *fitted* model a setFeaturesCol.
  //
  // Models with multiple or scalar inputs do not use this param at all;
  // inputLayout looks those up by their declared spec names, which are real
  // DataFrame column names by construction.
  val featuresCol: Param[String] =
    new Param(this, "featuresCol",
      "Column containing the assembled feature Vector — used only when the model " +
      "has a single rank-2 input (standard Spark ML convention).")

  val predictionCol: Param[String] =
    new Param(this, "predictionCol",
      "Output column name for the model prediction (Double). " +
      "For multi-output models this is the argmax class index.")

  val probabilityCol: Param[String] =
    new Param(this, "probabilityCol",
      "Output column name for class probabilities (Vector). " +
      "Only added when the model has more than one output column per row.")

  setDefault(
    featuresCol    -> "features",
    predictionCol  -> "prediction",
    probabilityCol -> "probability",
  )

  def setModelPath(value: String):      this.type = set(modelPath,     value)
  def setFeaturesCol(value: String):    this.type = set(featuresCol,   value)
  def setPredictionCol(value: String):  this.type = set(predictionCol, value)
  def setProbabilityCol(value: String): this.type = set(probabilityCol, value)

  def getModelPath:      String = $(modelPath)
  def getFeaturesCol:    String = $(featuresCol)
  def getPredictionCol:  String = $(predictionCol)
  def getProbabilityCol: String = $(probabilityCol)

  // ── Output column count ──────────────────────────────────────────────────────

  /**
   * Number of output values the model produces per input row.
   * Determined by a single dummy prediction on the driver.
   */
  private[spark] lazy val outputColsPerRow: Int = {
    val m = Model.loadFile($(modelPath), 1)
    try {
      val n        = m.numInputs()
      val hasStr   = (0 until n).exists(i => m.inputSpec(i).dtype() == DataType.STRING)
      if (hasStr) {
        val colNames = Array.tabulate(n)(i => m.inputSpec(i).name())
        val colTypes = Array.tabulate(n)(i =>
          if (m.inputSpec(i).dtype() == DataType.STRING) 1 else 0)
        val cols: Array[Object] = Array.tabulate(n)(i =>
          if (colTypes(i) == 1) Array("").asInstanceOf[Object]
          else new Array[Float](1).asInstanceOf[Object])
        m.predictColumns(colNames, colTypes, cols, 1).length
      } else {
        val nFeatures = totalInputFeatures(m)
        m.predict(new Array[Float](nFeatures), 1, nFeatures).length
      }
    } finally m.close()
  }

  // ── Schema ───────────────────────────────────────────────────────────────────

  override def transformSchema(schema: StructType): StructType = {
    val nOut = outputColsPerRow
    var s    = schema
    if (nOut > 1)
      s = s.add(StructField($(probabilityCol), SQLDataTypes.VectorType, nullable = false))
    s.add(StructField($(predictionCol), DoubleType, nullable = false))
  }

  // ── Transformer ──────────────────────────────────────────────────────────────

  override def transform(dataset: Dataset[_]): DataFrame = {
    val path         = $(modelPath)
    val nOut         = outputColsPerRow
    val schema       = dataset.schema
    val outputSchema = transformSchema(schema)

    // Compute input layout on the driver (one short model load, closed immediately).
    val (colIndices, colWidths, colIsVector, colIsString) = inputLayout(path, schema)
    val nFeaturesTotal = colWidths.sum
    val hasStringInput = colIsString.contains(true)

    val rdd = dataset.toDF().rdd.mapPartitions { iter =>
      val model = Model.loadFile(path, 1)
      try {
        val rows = iter.toArray
        if (rows.isEmpty) {
          Iterator.empty
        } else {
          val nRows = rows.length

          val output =
            if (hasStringInput) {
              // Per-column path: pass each input as its own typed array.
              val colNames = Array.tabulate(colIndices.length)(i =>
                model.inputSpec(i).name())
              val colTypes = colIsString.map(s => if (s) 1 else 0)
              val cols: Array[Object] = Array.tabulate(colIndices.length) { i =>
                if (colIsString(i)) {
                  Array.tabulate(nRows)(r => rows(r).getString(colIndices(i))): Object
                } else {
                  Array.tabulate(nRows)(r => toFloat(rows(r), colIndices(i))): Object
                }
              }
              model.predictColumns(colNames, colTypes, cols, nRows)
            } else {
              // Flat float path (original).
              val flat = new Array[Float](nRows * nFeaturesTotal)
              for (r <- rows.indices) {
                var col = 0
                for (i <- colIndices.indices) {
                  if (colIsVector(i)) {
                    rows(r).getAs[Vector](colIndices(i)).foreachActive { (c, v) =>
                      flat(r * nFeaturesTotal + col + c) = v.toFloat
                    }
                    col += colWidths(i)
                  } else {
                    flat(r * nFeaturesTotal + col) = toFloat(rows(r), colIndices(i))
                    col += 1
                  }
                }
              }
              model.predict(flat, nRows, nFeaturesTotal)
            }

          rows.zipWithIndex.map { case (row, r) =>
            if (nOut == 1) {
              Row.fromSeq(row.toSeq :+ output(r).toDouble)
            } else {
              val probs = Vectors.dense(
                Array.tabulate(nOut)(j => output(r * nOut + j).toDouble))
              val pred  = (0 until nOut).maxBy(j => output(r * nOut + j)).toDouble
              Row.fromSeq(row.toSeq :+ probs :+ pred)
            }
          }.iterator
        }
      } finally {
        model.close()
      }
    }

    dataset.sparkSession.createDataFrame(rdd, outputSchema)
  }

  override def copy(extra: ParamMap): OMLEModel = defaultCopy(extra)

  // ── Private helpers ──────────────────────────────────────────────────────────

  /** Total number of scalar features across all model inputs. */
  private def totalInputFeatures(m: Model): Int =
    (0 until m.numInputs()).map { i =>
      val s = m.inputSpec(i).shape()
      if (s.length == 1) 1 else s(1).toInt
    }.sum

  /**
   * Returns `(colIndices, colWidths, colIsVector, colIsString)` describing how to read
   * input values from a DataFrame row.
   *
   * For a single rank-2 input uses `featuresCol`; for all other cases
   * uses the model's declared input names.
   */
  private def inputLayout(path: String, schema: StructType)
      : (Array[Int], Array[Int], Array[Boolean], Array[Boolean]) = {
    val m = Model.loadFile(path, 1)
    try {
      val n = m.numInputs()
      if (n == 1 && m.inputSpec(0).shape().length >= 2) {
        // Single pre-assembled vector: use featuresCol (legacy / standard Spark ML path).
        val idx   = schema.fieldIndex($(featuresCol))
        val width = m.inputSpec(0).shape()(1).toInt
        (Array(idx), Array(width), Array(true), Array(false))
      } else {
        // Multiple inputs or scalar inputs: look up each by its declared name.
        val indices  = Array.tabulate(n)(i => schema.fieldIndex(m.inputSpec(i).name()))
        val widths   = Array.tabulate(n) { i =>
          val s = m.inputSpec(i).shape()
          if (s.length == 1) 1 else s(1).toInt
        }
        val isVec    = Array.tabulate(n)(i => m.inputSpec(i).shape().length >= 2)
        val isString = Array.tabulate(n)(i => m.inputSpec(i).dtype() == DataType.STRING)
        (indices, widths, isVec, isString)
      }
    } finally m.close()
  }

  /** Extract a numeric column value as Float, handling all Spark numeric types. */
  private def toFloat(row: Row, idx: Int): Float =
    row.get(idx) match {
      case v: Double            => v.toFloat
      case v: Float             => v
      case v: Int               => v.toFloat
      case v: Long              => v.toFloat
      case v: java.lang.Double  => v.toFloat
      case v: java.lang.Float   => v
      case v: java.lang.Integer => v.toFloat
      case v: java.lang.Long    => v.toFloat
      case v: Number            => v.floatValue()
      case null                 => 0f
    }
}

/**
 * Companion object — factory for building an [[OMLEModel]] from a `.omle` file.
 */
object OMLEModel {

  /**
   * Build a transformer for the OMLE model at `path`.
   *
   * The file is opened and closed immediately so that a missing path or an
   * unreadable model fails here, on the driver, rather than inside `transform`
   * on every executor. Only the path is retained: the native model handle is
   * not serialisable, so each partition reloads it from this path at run time.
   *
   * Named `loadFile` to match the Java binding's `Model.loadFile`, and to leave
   * `load` free for `MLReadable`, whose `load(path: String)` means something
   * different -- reading a Spark-ML-serialised stage directory.
   *
   * {{{
   *   val model = OMLEModel.loadFile("/path/to/model.omle")
   *   val predictions = model.transform(df)
   * }}}
   *
   * @throws java.lang.RuntimeException if the model cannot be loaded.
   */
  def loadFile(path: String): OMLEModel = {
    val m = Model.loadFile(path, 1)
    m.close()
    new OMLEModel().setModelPath(path)
  }
}
