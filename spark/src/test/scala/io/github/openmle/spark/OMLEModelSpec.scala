package io.github.openmle.spark

import io.github.openmle.runtime.Model
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{SQLDataTypes, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import java.io.{File, FileOutputStream}
import java.nio.file.Files

/**
 * Comprehensive tests for [[OMLEModel]].
 *
 * Two model fixtures:
 *   test_model_2f.omle    — 2-feature regression stump (1 output per row)
 *                         feat0 < 0.5  →  +2.0,  feat0 >= 0.5  →  -2.0
 *
 *   test_model_3class.omle — 2-feature 3-class classifier (3 outputs per row, SOFTMAX)
 *                         feat0 < 0.5  →  softmax([+2, 0, -1]) ≈ [0.844, 0.114, 0.042], pred=0
 *                         feat0 >= 0.5 →  softmax([-1, 0, +2]) ≈ [0.042, 0.114, 0.844], pred=2
 *
 * Each "native match" test loads the same model directly via [[Model]] and
 * compares its output with the Spark transformer output to verify they are
 * identical (within float32 precision).
 */
class OMLEModelSpec extends AnyFunSuite with BeforeAndAfterAll {

  private var spark:        SparkSession = _
  private var regrFile:     File         = _
  private var classFile:    File         = _

  // ── Softmax analytic values (feat0 < 0.5) ─────────────────────────────────
  // softmax([2.0, 0.0, -1.0])
  private val ExpProb0Left  = Array(0.8437947, 0.1141952, 0.0420101)
  // softmax([-1.0, 0.0, 2.0])
  private val ExpProb0Right = Array(0.0420101, 0.1141952, 0.8437947)

  override def beforeAll(): Unit = {
    spark = SparkSession.builder()
      .master("local[4]")
      .appName("OMLEModelSpec")
      .config("spark.sql.shuffle.partitions", "4")
      .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    regrFile  = extractResource("/test_model_2f.omle")
    classFile = extractResource("/test_model_3class.omle")
  }

  override def afterAll(): Unit = {
    Seq(regrFile, classFile).filter(_ != null).foreach(_.delete())
    if (spark != null) spark.stop()
  }

  // ── Helpers ──────────────────────────────────────────────────────────────────

  private def extractResource(resource: String): File = {
    val stream = getClass.getResourceAsStream(resource)
    assume(stream != null, s"$resource not found in test resources")
    val f = Files.createTempFile("omle_test_", ".omle").toFile
    val os = new FileOutputStream(f)
    try { stream.transferTo(os) } finally { os.close(); stream.close() }
    f
  }

  private def regrModel():  OMLEModel = new OMLEModel().setModelPath(regrFile.getAbsolutePath)
  private def classModel(): OMLEModel = new OMLEModel().setModelPath(classFile.getAbsolutePath)

  /** DataFrame with two float columns + assembled `features` Vector. */
  private def featuresDF(rows: Seq[(Float, Float)], partitions: Int = 2) = {
    val ss = spark; import ss.implicits._
    val df = rows.toDF("f0", "f1").repartition(partitions)
    new VectorAssembler()
      .setInputCols(Array("f0", "f1"))
      .setOutputCol("features")
      .transform(df)
  }

  private def assertClose(expected: Double, actual: Double, tol: Double = 1e-4): Unit =
    assert(math.abs(actual - expected) <= tol,
      f"expected $expected%.6f but got $actual%.6f (delta ${math.abs(actual-expected)}%.2e)")

  // ══════════════════════════════════════════════════════════════════════════════
  // Regression model (1 output per row)
  // ══════════════════════════════════════════════════════════════════════════════

  // ── Schema ───────────────────────────────────────────────────────────────────

  test("regr: outputColsPerRow is 1") {
    assert(regrModel().outputColsPerRow == 1)
  }

  test("regr: transform adds predictionCol") {
    val result = regrModel().transform(featuresDF(Seq((0.1f, 0f))))
    assert(result.schema.fieldNames.contains("prediction"))
  }

  test("regr: predictionCol is DoubleType") {
    val result = regrModel().transform(featuresDF(Seq((0.1f, 0f))))
    assert(result.schema("prediction").dataType == DoubleType)
  }

  test("regr: no probabilityCol for 1-output model") {
    val result = regrModel().transform(featuresDF(Seq((0.1f, 0f))))
    assert(!result.schema.fieldNames.contains("probability"))
  }

  test("regr: original columns are preserved") {
    val result = regrModel().transform(featuresDF(Seq((0.1f, 0f))))
    val names  = result.schema.fieldNames.toSet
    assert(names.contains("f0"))
    assert(names.contains("f1"))
    assert(names.contains("features"))
  }

  test("regr: prediction column is appended last") {
    val result = regrModel().transform(featuresDF(Seq((0.1f, 0f))))
    val names  = result.schema.fieldNames
    assert(names.last == "prediction")
  }

  test("regr: transformSchema is consistent with transform") {
    val input   = featuresDF(Seq((0.1f, 0f)))
    val model   = regrModel()
    val schema1 = model.transformSchema(input.schema)
    val schema2 = model.transform(input).schema
    assert(schema1 == schema2)
  }

  // ── Correctness ──────────────────────────────────────────────────────────────

  test("regr: left branch feat0 < 0.5 predicts +2.0") {
    val pred = regrModel().transform(featuresDF(Seq((0.1f, 99f))))
      .select("prediction").head().getDouble(0)
    assertClose(2.0, pred)
  }

  test("regr: right branch feat0 >= 0.5 predicts -2.0") {
    val pred = regrModel().transform(featuresDF(Seq((0.9f, 99f))))
      .select("prediction").head().getDouble(0)
    assertClose(-2.0, pred)
  }

  test("regr: boundary feat0 == 0.5 predicts -2.0") {
    val pred = regrModel().transform(featuresDF(Seq((0.5f, 0f))))
      .select("prediction").head().getDouble(0)
    assertClose(-2.0, pred)
  }

  test("regr: feat1 is ignored") {
    val model = regrModel()
    for (f1 <- Seq(-1000f, 0f, 1000f)) {
      val pred = model.transform(featuresDF(Seq((0.1f, f1))))
        .select("prediction").head().getDouble(0)
      assertClose(2.0, pred)
    }
  }

  test("regr: batch of 100 rows all correct") {
    val rows   = (0 until 100).map(i => (if (i % 2 == 0) 0.1f else 0.9f, 0f))
    val preds  = regrModel().transform(featuresDF(rows))
      .orderBy("f0")   // stable sort: 0.1-group first
      .select("prediction").collect().map(_.getDouble(0))
    // After orderBy, first 50 should be 2.0, last 50 should be -2.0
    preds.take(50).foreach(p => assertClose(2.0,  p))
    preds.drop(50).foreach(p => assertClose(-2.0, p))
  }

  // ── Native API match ─────────────────────────────────────────────────────────

  test("regr: predictions match Java Model.predict bit-for-bit") {
    val testRows = Seq(
      (0.0f, 0f), (0.1f, 5f), (0.4f, -3f), (0.5f, 0f),
      (0.6f, 1f), (0.9f, 99f), (1.0f, 0f),
    )
    val javaModel = Model.loadFile(regrFile.getAbsolutePath)
    val expected  = try {
      javaModel.predict(testRows.map { case (a, b) => Array(a, b) }.toArray)
    } finally javaModel.close()

    val preds = regrModel()
      .transform(featuresDF(testRows, partitions = 3))
      .select("f0", "f1", "prediction")
      .collect()
      .sortBy(r => (r.getFloat(0), r.getFloat(1)))
      .map(_.getDouble(2))

    val expSorted = testRows.sortBy { case (a, b) => (a, b) }.zipWithIndex.map { case ((a, b), i) =>
      expected(testRows.indexOf((a, b))).toDouble
    }
    preds.zip(expSorted).foreach { case (actual, exp) =>
      assertClose(exp, actual, tol = 1e-5)
    }
  }

  // ── Vector types ─────────────────────────────────────────────────────────────

  test("regr: sparse Vector input produces same result as dense") {
    val ss = spark; import ss.implicits._
    // Dense vector [0.1, 0.0] vs sparse [0.1, 0.0]
    val dense  = Vectors.dense(0.1, 0.0)
    val sparse = Vectors.sparse(2, Array(0), Array(0.1))
    val df = Seq((dense, 0.1f), (sparse, 0.1f)).toDF("features", "f0_orig")

    val result = regrModel().transform(df)
    val preds  = result.select("prediction").collect().map(_.getDouble(0))
    preds.foreach(p => assertClose(2.0, p))
  }

  // ── Partition behavior ────────────────────────────────────────────────────────

  test("regr: single partition gives correct results") {
    val rows  = (0 until 20).map(i => (if (i % 2 == 0) 0.1f else 0.9f, 0f))
    val preds = regrModel().transform(featuresDF(rows, partitions = 1))
      .select("f0", "prediction").collect()
    preds.foreach { r =>
      val expected = if (r.getFloat(0) < 0.5f) 2.0 else -2.0
      assertClose(expected, r.getDouble(1))
    }
  }

  test("regr: eight partitions give correct results") {
    val rows  = (0 until 80).map(i => (if (i % 2 == 0) 0.1f else 0.9f, 0f))
    val preds = regrModel().transform(featuresDF(rows, partitions = 8))
      .select("f0", "prediction").collect()
    preds.foreach { r =>
      val expected = if (r.getFloat(0) < 0.5f) 2.0 else -2.0
      assertClose(expected, r.getDouble(1))
    }
  }

  // ── Params ───────────────────────────────────────────────────────────────────

  test("regr: getModelPath returns set value") {
    val m = regrModel()
    assert(m.getModelPath == regrFile.getAbsolutePath)
  }

  test("regr: getFeaturesCol default is 'features'") {
    assert(regrModel().getFeaturesCol == "features")
  }

  test("regr: setFeaturesCol changes input column") {
    val ss = spark; import ss.implicits._
    val df = new VectorAssembler()
      .setInputCols(Array("f0", "f1"))
      .setOutputCol("myFeats")
      .transform(Seq((0.1f, 0f)).toDF("f0", "f1"))
    val result = regrModel().setFeaturesCol("myFeats").transform(df)
    assert(result.schema.fieldNames.contains("prediction"))
    assertClose(2.0, result.select("prediction").head().getDouble(0))
  }

  test("regr: setPredictionCol renames output") {
    val result = regrModel().setPredictionCol("score")
      .transform(featuresDF(Seq((0.1f, 0f))))
    assert( result.schema.fieldNames.contains("score"))
    assert(!result.schema.fieldNames.contains("prediction"))
  }

  test("regr: setProbabilityCol has no effect for 1-output model") {
    val result = regrModel().setProbabilityCol("myProb")
      .transform(featuresDF(Seq((0.1f, 0f))))
    assert(!result.schema.fieldNames.contains("myProb"))
    assert(!result.schema.fieldNames.contains("probability"))
  }

  test("regr: copy() preserves all params") {
    val original = regrModel()
      .setFeaturesCol("feats")
      .setPredictionCol("score")
      .setProbabilityCol("prob")
    val copied = original.copy(new ParamMap())
    assert(copied.getModelPath     == original.getModelPath)
    assert(copied.getFeaturesCol   == original.getFeaturesCol)
    assert(copied.getPredictionCol == original.getPredictionCol)
    assert(copied.getProbabilityCol == original.getProbabilityCol)
  }

  // ── Edge cases ───────────────────────────────────────────────────────────────

  test("regr: empty DataFrame produces zero rows") {
    val result = regrModel().transform(featuresDF(Seq.empty))
    assert(result.count() == 0)
  }

  test("regr: single row DataFrame works") {
    val result = regrModel().transform(featuresDF(Seq((0.1f, 0f))))
    assert(result.count() == 1)
    assertClose(2.0, result.select("prediction").head().getDouble(0))
  }

  // ══════════════════════════════════════════════════════════════════════════════
  // 3-class classifier (3 output columns per row, SOFTMAX)
  // ══════════════════════════════════════════════════════════════════════════════

  // ── Schema ───────────────────────────────────────────────────────────────────

  test("cls: outputColsPerRow is 3") {
    assume(classFile != null)
    assert(classModel().outputColsPerRow == 3)
  }

  test("cls: transform adds both probabilityCol and predictionCol") {
    assume(classFile != null)
    val result = classModel().transform(featuresDF(Seq((0.1f, 0f))))
    assert(result.schema.fieldNames.contains("probability"))
    assert(result.schema.fieldNames.contains("prediction"))
  }

  test("cls: probabilityCol is VectorType") {
    assume(classFile != null)
    val schema = classModel().transformSchema(featuresDF(Seq((0.1f, 0f))).schema)
    assert(schema("probability").dataType == SQLDataTypes.VectorType)
  }

  test("cls: predictionCol is DoubleType") {
    assume(classFile != null)
    val schema = classModel().transformSchema(featuresDF(Seq((0.1f, 0f))).schema)
    assert(schema("prediction").dataType == DoubleType)
  }

  test("cls: probabilityCol comes before predictionCol") {
    assume(classFile != null)
    val names = classModel().transformSchema(featuresDF(Seq((0.1f, 0f))).schema).fieldNames
    val probIdx = names.indexOf("probability")
    val predIdx = names.indexOf("prediction")
    assert(probIdx < predIdx)
  }

  test("cls: original columns are preserved") {
    assume(classFile != null)
    val result = classModel().transform(featuresDF(Seq((0.1f, 0f))))
    val names  = result.schema.fieldNames.toSet
    assert(names.contains("f0") && names.contains("f1") && names.contains("features"))
  }

  // ── Correctness ──────────────────────────────────────────────────────────────

  test("cls: feat0 < 0.5 predicts class 0") {
    assume(classFile != null)
    val pred = classModel().transform(featuresDF(Seq((0.1f, 0f))))
      .select("prediction").head().getDouble(0)
    assertClose(0.0, pred)
  }

  test("cls: feat0 >= 0.5 predicts class 2") {
    assume(classFile != null)
    val pred = classModel().transform(featuresDF(Seq((0.9f, 0f))))
      .select("prediction").head().getDouble(0)
    assertClose(2.0, pred)
  }

  test("cls: probabilities sum to 1.0") {
    assume(classFile != null)
    val rows = Seq((0.1f, 0f), (0.9f, 0f), (0.5f, 1f))
    classModel().transform(featuresDF(rows))
      .select("probability").collect().foreach { row =>
        val v = row.getAs[Vector](0)
        assertClose(1.0, v.toArray.sum, tol = 1e-5)
      }
  }

  test("cls: probability vector has 3 elements") {
    assume(classFile != null)
    val v = classModel().transform(featuresDF(Seq((0.1f, 0f))))
      .select("probability").head().getAs[Vector](0)
    assert(v.size == 3)
  }

  test("cls: predictionCol equals argmax of probabilityCol") {
    assume(classFile != null)
    val rows = Seq((0.1f, 0f), (0.9f, 0f), (0.5f, 1f), (0.4f, -5f), (0.6f, 3f))
    classModel().transform(featuresDF(rows))
      .select("probability", "prediction").collect().foreach { row =>
        val probs  = row.getAs[Vector](0).toArray
        val pred   = row.getDouble(1)
        val argmax = probs.indices.maxBy(probs).toDouble
        assertClose(argmax, pred, tol = 0.5)  // integer comparison
      }
  }

  test("cls: analytic probabilities for feat0 < 0.5") {
    assume(classFile != null)
    val probs = classModel().transform(featuresDF(Seq((0.1f, 0f))))
      .select("probability").head().getAs[Vector](0).toArray
    ExpProb0Left.zipWithIndex.foreach { case (exp, i) =>
      assertClose(exp, probs(i), tol = 1e-4)
    }
  }

  test("cls: analytic probabilities for feat0 >= 0.5") {
    assume(classFile != null)
    val probs = classModel().transform(featuresDF(Seq((0.9f, 0f))))
      .select("probability").head().getAs[Vector](0).toArray
    ExpProb0Right.zipWithIndex.foreach { case (exp, i) =>
      assertClose(exp, probs(i), tol = 1e-4)
    }
  }

  // ── Native API match (3-class) ────────────────────────────────────────────────

  test("cls: probability columns match Java Model.predict output exactly") {
    assume(classFile != null)
    val testRows = Seq(
      (0.0f, 0f), (0.1f, 0f), (0.4f, 1f), (0.5f, 0f), (0.9f, 0f), (1.0f, 2f),
    )
    val javaModel = Model.loadFile(classFile.getAbsolutePath)
    // Java predict returns flat [row0_class0, row0_class1, row0_class2, row1_class0, ...]
    val javaOut = try {
      javaModel.predict(testRows.map { case (a, b) => Array(a, b) }.toArray)
    } finally javaModel.close()

    // Collect Spark results, sort by (f0, f1) to match testRows order
    val sparkRows = classModel()
      .transform(featuresDF(testRows, partitions = 3))
      .select("f0", "f1", "probability").collect()
      .sortBy(r => (r.getFloat(0), r.getFloat(1)))

    sparkRows.zipWithIndex.foreach { case (row, r) =>
      val sparkProbs = row.getAs[Vector](2).toArray
      (0 until 3).foreach { j =>
        val javaVal = javaOut(r * 3 + j).toDouble
        assertClose(javaVal, sparkProbs(j), tol = 1e-5)
      }
    }
  }

  test("cls: prediction matches argmax of Java output") {
    assume(classFile != null)
    val testRows = Seq((0.1f, 0f), (0.5f, 0f), (0.9f, 0f))
    val javaModel = Model.loadFile(classFile.getAbsolutePath)
    val javaOut = try {
      javaModel.predict(testRows.map { case (a, b) => Array(a, b) }.toArray)
    } finally javaModel.close()

    val sparkRows = classModel()
      .transform(featuresDF(testRows, partitions = 1))
      .select("f0", "f1", "prediction").collect()
      .sortBy(r => (r.getFloat(0), r.getFloat(1)))

    sparkRows.zipWithIndex.foreach { case (row, r) =>
      val javaArgmax = (0 until 3).maxBy(j => javaOut(r * 3 + j)).toDouble
      assertClose(javaArgmax, row.getDouble(2), tol = 0.5)
    }
  }

  // ── Params (classifier) ───────────────────────────────────────────────────────

  test("cls: setProbabilityCol renames probability output") {
    assume(classFile != null)
    val result = classModel().setProbabilityCol("myProbs")
      .transform(featuresDF(Seq((0.1f, 0f))))
    assert( result.schema.fieldNames.contains("myProbs"))
    assert(!result.schema.fieldNames.contains("probability"))
  }

  test("cls: copy() preserves params") {
    assume(classFile != null)
    val original = classModel().setPredictionCol("cls").setProbabilityCol("probs")
    val copied   = original.copy(new ParamMap())
    assert(copied.getPredictionCol  == "cls")
    assert(copied.getProbabilityCol == "probs")
  }

  test("cls: batch across multiple partitions is correct") {
    assume(classFile != null)
    val rows  = (0 until 60).map(i => (if (i % 2 == 0) 0.1f else 0.9f, 0f))
    classModel().transform(featuresDF(rows, partitions = 6))
      .select("f0", "prediction").collect().foreach { r =>
        val expected = if (r.getFloat(0) < 0.5f) 0.0 else 2.0
        assertClose(expected, r.getDouble(1), tol = 0.5)
      }
  }

  // ── Companion factory ────────────────────────────────────────────────────────

  test("loadFile: builds a usable transformer") {
    val result = OMLEModel.loadFile(regrFile.getAbsolutePath)
      .transform(featuresDF(Seq((0.1f, 0f))))
    assertClose(2.0, result.select("prediction").head().getDouble(0))
  }

  test("loadFile: sets modelPath, so copy() and the setters still work") {
    val m = OMLEModel.loadFile(regrFile.getAbsolutePath)
    assert(m.getModelPath == regrFile.getAbsolutePath)
    // The factory must go through the Param, or defaultCopy would drop the path.
    assert(m.copy(new ParamMap()).getModelPath == regrFile.getAbsolutePath)
  }

  test("loadFile: equivalent to setModelPath") {
    val viaFactory = OMLEModel.loadFile(regrFile.getAbsolutePath)
    val viaSetter  = new OMLEModel().setModelPath(regrFile.getAbsolutePath)
    assert(viaFactory.getModelPath == viaSetter.getModelPath)
    assert(viaFactory.outputColsPerRow == viaSetter.outputColsPerRow)
  }

  test("loadFile: a missing file fails immediately, not at transform") {
    val missing = new File(System.getProperty("java.io.tmpdir"), "no_such_model.omle")
    assert(!missing.exists())
    intercept[Throwable] { OMLEModel.loadFile(missing.getAbsolutePath) }
    // The lazy path would have accepted the bad path and only failed later.
    val lazyModel = new OMLEModel().setModelPath(missing.getAbsolutePath)
    assert(lazyModel.getModelPath == missing.getAbsolutePath)
  }

  test("loadFile: multi-output model still reports both columns") {
    val result = OMLEModel.loadFile(classFile.getAbsolutePath)
      .transform(featuresDF(Seq((0.1f, 0f))))
    assert(result.schema.fieldNames.contains("probability"))
    assert(result.schema.fieldNames.contains("prediction"))
  }
}

