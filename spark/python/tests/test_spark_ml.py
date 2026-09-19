"""Integration tests for omle_spark.OMLEModel (Python/PySpark API).

Mirrors OMLEModelSpec.scala — covers params, schema, correctness,
native-match, edge cases, and multi-partition behaviour.

All tests are skipped when the omle-spark JAR is not present on disk.
"""

import math

import numpy as np
import pytest

from .conftest import skip_no_jar

pytestmark = skip_no_jar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_regr_df(spark, rows):
    """Create a features DataFrame from (f0, f1) tuples."""
    from pyspark.ml.feature import VectorAssembler
    df = spark.createDataFrame(rows, ["f0", "f1"])
    return VectorAssembler(inputCols=["f0", "f1"], outputCol="features").transform(df)


def _make_class_df(spark, rows):
    """Same as _make_regr_df — the fixture model also uses 2 features."""
    return _make_regr_df(spark, rows)


def _collect_sorted(df, *cols):
    """Collect rows sorted by the first column, return plain Python values."""
    sorted_rows = df.orderBy("f0").select(*cols).collect()
    return sorted_rows


# ---------------------------------------------------------------------------
# 1-output (regression) tests
# ---------------------------------------------------------------------------

class TestRegressionParams:
    def test_default_params(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel(modelPath=regr_model_path)
        assert m.getModelPath()    == regr_model_path
        assert m.getFeaturesCol()  == "features"
        assert m.getPredictionCol() == "prediction"
        assert m.getProbabilityCol() == "probability"

    def test_set_model_path(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel().setModelPath(regr_model_path)
        assert m.getModelPath() == regr_model_path

    def test_set_features_col(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel(modelPath=regr_model_path).setFeaturesCol("feats")
        assert m.getFeaturesCol() == "feats"

    def test_set_prediction_col(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel(modelPath=regr_model_path).setPredictionCol("score")
        assert m.getPredictionCol() == "score"

    def test_set_probability_col(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel(modelPath=regr_model_path).setProbabilityCol("probs")
        assert m.getProbabilityCol() == "probs"

    def test_constructor_kwargs(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel(
            modelPath=regr_model_path,
            featuresCol="myFeats",
            predictionCol="myPred",
            probabilityCol="myProb",
        )
        assert m.getFeaturesCol()   == "myFeats"
        assert m.getPredictionCol() == "myPred"
        assert m.getProbabilityCol() == "myProb"

    def test_set_params(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel()
        m.setParams(modelPath=regr_model_path, predictionCol="p")
        assert m.getModelPath()     == regr_model_path
        assert m.getPredictionCol() == "p"


class TestRegressionSchema:
    def test_prediction_col_added(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        from pyspark.sql.types import DoubleType
        df  = _make_regr_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        assert "prediction" in out.columns
        assert isinstance(out.schema["prediction"].dataType, DoubleType)

    def test_no_probability_col_for_regression(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        assert "probability" not in out.columns

    def test_original_cols_preserved(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.1, 99.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        assert "f0" in out.columns
        assert "f1" in out.columns
        assert "features" in out.columns

    def test_custom_prediction_col_name(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=regr_model_path, predictionCol="score").transform(df)
        assert "score" in out.columns
        assert "prediction" not in out.columns


class TestRegressionCorrectness:
    """2-feature model: feat0 < 0.5 → +2.0, feat0 >= 0.5 → −2.0."""

    def test_left_branch(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        pred = out.select("prediction").collect()[0][0]
        assert abs(pred - 2.0) < 1e-5

    def test_right_branch(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.9, 0.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        pred = out.select("prediction").collect()[0][0]
        assert abs(pred - (-2.0)) < 1e-5

    def test_boundary_goes_right(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.5, 0.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        pred = out.select("prediction").collect()[0][0]
        assert abs(pred - (-2.0)) < 1e-5

    def test_batch(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        rows = [(0.1, 0.0), (0.9, 0.0), (0.3, 0.0), (0.7, 0.0)]
        df   = _make_regr_df(spark, rows)
        out  = OMLEModel(modelPath=regr_model_path).transform(df)
        preds = {r.f0: r.prediction for r in out.select("f0", "prediction").collect()}
        assert abs(preds[0.1] - 2.0)  < 1e-5
        assert abs(preds[0.9] + 2.0)  < 1e-5
        assert abs(preds[0.3] - 2.0)  < 1e-5
        assert abs(preds[0.7] + 2.0)  < 1e-5

    def test_single_row(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.2, 5.0)])
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        assert out.count() == 1

    def test_empty_dataframe(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        df  = _make_regr_df(spark, [(0.1, 0.0)]).filter("f0 < 0")
        out = OMLEModel(modelPath=regr_model_path).transform(df)
        assert out.count() == 0

    def test_multipartition(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        rows = [(float(i) / 10, 0.0) for i in range(20)]
        df   = _make_regr_df(spark, rows).repartition(8)
        out  = OMLEModel(modelPath=regr_model_path).transform(df)
        assert out.count() == 20
        # All preds should be ±2
        preds = [r[0] for r in out.select("prediction").collect()]
        assert all(abs(abs(p) - 2.0) < 1e-5 for p in preds)


class TestRegressionNativeMatch:
    """Spark transformer predictions must match omle_runtime.predict()."""

    def test_predictions_match_native(self, spark, regr_model_path, native_regr_model):
        from omle_spark import OMLEModel
        test_rows = [(0.1, 0.0), (0.5, 0.0), (0.9, 0.0), (0.0, 0.0), (1.0, 0.0)]
        df = _make_regr_df(spark, test_rows)
        out = OMLEModel(modelPath=regr_model_path).transform(df)

        spark_preds = {(r.f0, r.f1): r.prediction
                       for r in out.select("f0", "f1", "prediction").collect()}

        X = np.array(test_rows, dtype=np.float32)
        native_preds = native_regr_model.predict(X)

        for (f0, f1), sp in spark_preds.items():
            idx = next(i for i, (a, b) in enumerate(test_rows) if a == f0 and b == f1)
            assert abs(sp - float(native_preds[idx])) < 1e-4, (
                f"row ({f0},{f1}): spark={sp}, native={native_preds[idx]}"
            )


# ---------------------------------------------------------------------------
# 3-class classifier tests
# ---------------------------------------------------------------------------

_SOFTMAX_2_0   = math.exp(2) / (math.exp(2) + 1 + math.exp(-1))   # ≈ 0.8438
_SOFTMAX_0_0   = 1.0          / (math.exp(2) + 1 + math.exp(-1))   # ≈ 0.1142
_SOFTMAX_N1_0  = math.exp(-1) / (math.exp(2) + 1 + math.exp(-1))   # ≈ 0.0420


class TestClassifierParams:
    def test_set_probability_col(self, spark, class_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel(modelPath=class_model_path, probabilityCol="probs")
        assert m.getProbabilityCol() == "probs"
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = m.transform(df)
        assert "probs" in out.columns

    def test_set_prediction_col(self, spark, class_model_path):
        from omle_spark import OMLEModel
        m   = OMLEModel(modelPath=class_model_path, predictionCol="cls")
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = m.transform(df)
        assert "cls" in out.columns
        assert "prediction" not in out.columns


class TestClassifierSchema:
    def test_probability_col_added(self, spark, class_model_path):
        from omle_spark import OMLEModel
        from pyspark.ml.linalg import VectorUDT
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=class_model_path).transform(df)
        assert "probability" in out.columns
        assert isinstance(out.schema["probability"].dataType, VectorUDT)

    def test_prediction_col_added(self, spark, class_model_path):
        from omle_spark import OMLEModel
        from pyspark.sql.types import DoubleType
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=class_model_path).transform(df)
        assert "prediction" in out.columns
        assert isinstance(out.schema["prediction"].dataType, DoubleType)

    def test_probability_col_before_prediction_col(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df   = _make_class_df(spark, [(0.1, 0.0)])
        out  = OMLEModel(modelPath=class_model_path).transform(df)
        cols = out.columns
        assert cols.index("probability") < cols.index("prediction")

    def test_original_cols_preserved(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=class_model_path).transform(df)
        for col in ["f0", "f1", "features"]:
            assert col in out.columns


class TestClassifierCorrectness:
    """3-class model: feat0 < 0.5 → pred=0, feat0 >= 0.5 → pred=2."""

    def test_prediction_left(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=class_model_path).transform(df)
        assert out.select("prediction").collect()[0][0] == 0.0

    def test_prediction_right(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df  = _make_class_df(spark, [(0.9, 0.0)])
        out = OMLEModel(modelPath=class_model_path).transform(df)
        assert out.select("prediction").collect()[0][0] == 2.0

    def test_proba_sums_to_one(self, spark, class_model_path):
        from omle_spark import OMLEModel
        rows = [(0.1, 0.0), (0.9, 0.0)]
        df   = _make_class_df(spark, rows)
        out  = OMLEModel(modelPath=class_model_path).transform(df)
        for r in out.select("probability").collect():
            s = sum(r[0].toArray())
            assert abs(s - 1.0) < 1e-5, f"probs sum to {s}"

    def test_proba_vector_size(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df  = _make_class_df(spark, [(0.1, 0.0)])
        out = OMLEModel(modelPath=class_model_path).transform(df)
        vec = out.select("probability").collect()[0][0]
        assert len(vec) == 3

    def test_prediction_equals_argmax_of_proba(self, spark, class_model_path):
        from omle_spark import OMLEModel
        rows = [(0.1, 0.0), (0.9, 0.0), (0.5, 0.0)]
        df   = _make_class_df(spark, rows)
        out  = OMLEModel(modelPath=class_model_path).transform(df)
        for r in out.select("probability", "prediction").collect():
            probs   = r[0].toArray()
            argmax  = float(np.argmax(probs))
            assert r[1] == argmax, f"pred={r[1]} but argmax={argmax}, probs={probs}"

    def test_analytic_proba_left(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df   = _make_class_df(spark, [(0.1, 0.0)])
        out  = OMLEModel(modelPath=class_model_path).transform(df)
        prob = out.select("probability").collect()[0][0].toArray()
        assert abs(prob[0] - _SOFTMAX_2_0)  < 1e-4
        assert abs(prob[1] - _SOFTMAX_0_0)  < 1e-4
        assert abs(prob[2] - _SOFTMAX_N1_0) < 1e-4

    def test_analytic_proba_right(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df   = _make_class_df(spark, [(0.9, 0.0)])
        out  = OMLEModel(modelPath=class_model_path).transform(df)
        prob = out.select("probability").collect()[0][0].toArray()
        assert abs(prob[0] - _SOFTMAX_N1_0) < 1e-4
        assert abs(prob[1] - _SOFTMAX_0_0)  < 1e-4
        assert abs(prob[2] - _SOFTMAX_2_0)  < 1e-4

    def test_batch_multipartition(self, spark, class_model_path):
        from omle_spark import OMLEModel
        rows = [(float(i) / 20, 0.0) for i in range(20)]
        df   = _make_class_df(spark, rows).repartition(4)
        out  = OMLEModel(modelPath=class_model_path).transform(df)
        assert out.count() == 20

    def test_empty_dataframe(self, spark, class_model_path):
        from omle_spark import OMLEModel
        df  = _make_class_df(spark, [(0.1, 0.0)]).filter("f0 < 0")
        out = OMLEModel(modelPath=class_model_path).transform(df)
        assert out.count() == 0


class TestClassifierNativeMatch:
    """Spark transformer must match omle_runtime.predict_proba()."""

    def test_probability_matches_native(self, spark, class_model_path, native_class_model):
        from omle_spark import OMLEModel
        test_rows = [(0.1, 0.0), (0.9, 0.0), (0.5, 0.0), (0.0, 0.0), (1.0, 0.0)]
        df  = _make_class_df(spark, test_rows)
        out = OMLEModel(modelPath=class_model_path).transform(df)

        spark_data = {(r.f0, r.f1): r.probability.toArray()
                      for r in out.select("f0", "f1", "probability").collect()}

        X = np.array(test_rows, dtype=np.float32)
        native_proba = native_class_model.predict_proba(X)

        for i, (f0, f1) in enumerate(test_rows):
            sp = spark_data[(f0, f1)]
            np.testing.assert_allclose(sp, native_proba[i], atol=1e-4,
                err_msg=f"row ({f0},{f1}): spark={sp}, native={native_proba[i]}")

    def test_prediction_matches_native_argmax(self, spark, class_model_path, native_class_model):
        from omle_spark import OMLEModel
        test_rows = [(0.1, 0.0), (0.9, 0.0), (0.5, 0.0)]
        df  = _make_class_df(spark, test_rows)
        out = OMLEModel(modelPath=class_model_path).transform(df)

        spark_preds = {(r.f0, r.f1): r.prediction
                       for r in out.select("f0", "f1", "prediction").collect()}

        X = np.array(test_rows, dtype=np.float32)
        native_proba = native_class_model.predict_proba(X)

        for i, (f0, f1) in enumerate(test_rows):
            sp      = spark_preds[(f0, f1)]
            argmax  = float(np.argmax(native_proba[i]))
            assert sp == argmax, (
                f"row ({f0},{f1}): spark pred={sp}, native argmax={argmax}"
            )


# ---------------------------------------------------------------------------
# Companion factory — mirrors the loadFile tests in OMLEModelSpec.scala
# ---------------------------------------------------------------------------

class TestLoadFileFactory:
    def test_builds_a_usable_transformer(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        out = OMLEModel.loadFile(regr_model_path).transform(
            _make_regr_df(spark, [(0.1, 0.0)]))
        assert math.isclose(out.select("prediction").head()[0], 2.0, abs_tol=1e-4)

    def test_sets_model_path_param(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        m = OMLEModel.loadFile(regr_model_path)
        assert m.getModelPath() == regr_model_path
        # Must go through the Param, or copy() would drop the path.
        assert m.copy().getModelPath() == regr_model_path

    def test_equivalent_to_set_model_path(self, spark, regr_model_path):
        from omle_spark import OMLEModel
        via_factory = OMLEModel.loadFile(regr_model_path)
        via_setter  = OMLEModel().setModelPath(regr_model_path)
        assert via_factory.getModelPath() == via_setter.getModelPath()
        assert via_factory.getFeaturesCol() == via_setter.getFeaturesCol()

    def test_missing_file_fails_immediately(self, spark, tmp_path):
        from omle_spark import OMLEModel
        missing = str(tmp_path / "no_such_model.omle")
        with pytest.raises(Exception):
            OMLEModel.loadFile(missing)
        # The deferred path accepts the same bad path without complaint.
        assert OMLEModel(modelPath=missing).getModelPath() == missing

    def test_multi_output_model_keeps_both_columns(self, spark, class_model_path):
        from omle_spark import OMLEModel
        out = OMLEModel.loadFile(class_model_path).transform(
            _make_class_df(spark, [(0.1, 0.0)]))
        assert "probability" in out.schema.fieldNames()
        assert "prediction" in out.schema.fieldNames()
