"""
Tests for the omle_runtime Python API (pybind11 extension module).

Run with:
    cd python && python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import omle_runtime as omr
from omle_runtime import (
    DataType,
    InputSpec,
    Model,
    OutputRole,
    OutputSpec,
    Session,
    Tensor,
    load_bytes,
)


# ===========================================================================
# Tensor
# ===========================================================================


class TestTensor:
    def test_2d_shape(self):
        t = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert t.shape == (2, 2)
        assert t.n_rows == 2
        assert t.n_cols == 2

    def test_dtype_always_float32(self):
        t = Tensor(np.array([[1.0]], dtype=np.float64))
        assert t.numpy().dtype == np.float32

    def test_1d_input(self):
        t = Tensor([1.0, 2.0, 3.0])
        assert t.shape == (3,)
        assert t.n_rows == 3
        assert t.n_cols == 1

    def test_name_attribute(self):
        t = Tensor([[1.0]], name="score")
        assert t.name == "score"

    def test_default_dtype_field(self):
        t = Tensor([[1.0]])
        assert t.dtype == DataType.Float32

    def test_pandas_input(self):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = Tensor(df)
        assert t.shape == (2, 2)

    def test_3d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            Tensor(np.zeros((2, 2, 2)))

    def test_numpy_roundtrip(self):
        arr = np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
        t = Tensor(arr)
        np.testing.assert_array_equal(t.numpy(), arr)

    def test_repr_contains_name_and_shape(self):
        t = Tensor([[1.0, 2.0]], name="feat")
        r = repr(t)
        assert "feat" in r
        assert "shape" in r


# ===========================================================================
# Model loading
# ===========================================================================


class TestLoad:
    def test_load_bytes_returns_model(self, model_bytes_2f):
        m = load_bytes(model_bytes_2f)
        assert isinstance(m, Model)

    def test_load_bytes_helper(self, model_bytes_2f):
        m = omr.load_bytes(model_bytes_2f)
        assert isinstance(m, Model)

    def test_load_file(self, model_bytes_2f, tmp_path):
        pb_file = tmp_path / "model.omle"
        pb_file.write_bytes(model_bytes_2f)
        m = omr.load(str(pb_file))
        assert isinstance(m, Model)

    def test_load_bytes_invalid_raises(self):
        with pytest.raises(RuntimeError):
            load_bytes(b"not protobuf")

    def test_load_file_missing_raises(self):
        with pytest.raises(RuntimeError):
            omr.load("/nonexistent/does_not_exist.omle")

    def test_multiple_loads_are_independent(self, model_bytes_2f):
        m1 = load_bytes(model_bytes_2f)
        m2 = load_bytes(model_bytes_2f)
        assert m1 is not m2
        assert m1.n_features_in_ == m2.n_features_in_


# ===========================================================================
# Model schema
# ===========================================================================


class TestModelSchema:
    def test_n_features_in_(self, model):
        assert model.n_features_in_ == 2

    def test_feature_names_in_(self, model):
        assert list(model.feature_names_in_) == ["X"]

    def test_num_inputs(self, model):
        assert model.num_inputs == 1

    def test_num_outputs(self, model):
        assert model.num_outputs == 1

    def test_inputs_types(self, model):
        assert all(isinstance(s, InputSpec) for s in model.inputs)

    def test_inputs_names(self, model):
        assert [s.name for s in model.inputs] == ["X"]

    def test_outputs_types(self, model):
        assert all(isinstance(s, OutputSpec) for s in model.outputs)

    def test_outputs_names(self, model):
        assert model.outputs[0].name == "score"

    def test_repr(self, model):
        r = repr(model)
        assert "2" in r
        assert "1" in r


# ===========================================================================
# Numerical correctness
# ===========================================================================


class TestNumerical:
    def test_stump_left_branch(self, model):
        # feat0 < 0.5 → +2.0
        X = np.array([[0.0, 99.0]], dtype=np.float32)
        np.testing.assert_allclose(model.predict(X), [2.0], rtol=1e-5)

    def test_stump_right_branch(self, model):
        # feat0 >= 0.5 → −2.0
        X = np.array([[1.0, 99.0]], dtype=np.float32)
        np.testing.assert_allclose(model.predict(X), [-2.0], rtol=1e-5)

    def test_boundary_is_strict_less_than(self, model):
        # LESS_THAN is strict; 0.5 goes to the right branch → −2.0
        X = np.array([[0.5, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(model.predict(X), [-2.0], rtol=1e-5)

    def test_feat1_is_ignored(self, model):
        # Only feat0 matters; varying feat1 should not change the result
        for f1 in [-100.0, 0.0, 100.0]:
            X = np.array([[0.1, f1]], dtype=np.float32)
            np.testing.assert_allclose(model.predict(X), [2.0], rtol=1e-5)

    def test_batch_correctness(self, model):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 2)).astype(np.float32)
        out = model.predict(X)
        expected = np.where(X[:, 0] < 0.5, 2.0, -2.0)
        np.testing.assert_allclose(out, expected, rtol=1e-5)


# ===========================================================================
# predict / predict_proba shapes
# ===========================================================================


class TestShapes:
    def test_predict_single_output_is_1d(self, model):
        X = np.ones((5, 2), dtype=np.float32)
        out = model.predict(X)
        assert out.ndim == 1
        assert out.shape == (5,)

    def test_predict_proba_always_2d(self, model):
        X = np.ones((5, 2), dtype=np.float32)
        out = model.predict_proba(X)
        assert out.ndim == 2
        assert out.shape == (5, 1)

    def test_predict_single_row(self, model):
        X = np.array([[0.1, 0.0]], dtype=np.float32)
        out = model.predict(X)
        assert out.shape == (1,)
        np.testing.assert_allclose(out, [2.0], rtol=1e-5)

    def test_predict_proba_single_row(self, model):
        X = np.array([[0.1, 0.0]], dtype=np.float32)
        out = model.predict_proba(X)
        assert out.shape == (1, 1)
        np.testing.assert_allclose(out[0, 0], 2.0, rtol=1e-5)


# ===========================================================================
# Input coercion
# ===========================================================================


class TestCoercion:
    def test_float64_array(self, model):
        X = np.array([[0.1, 0.0], [0.9, 0.0]], dtype=np.float64)
        out = model.predict(X)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_int_array(self, model):
        X = np.array([[0, 0], [1, 0]], dtype=np.int32)
        out = model.predict(X)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_list_of_lists(self, model):
        out = model.predict([[0.1, 0.0], [0.9, 0.0]])
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_1d_treated_as_single_sample(self, model):
        X = np.array([0.1, 0.0], dtype=np.float32)
        out = model.predict(X)
        assert out.shape == (1,)
        np.testing.assert_allclose(out, [2.0], rtol=1e-5)

    def test_pandas_dataframe(self, model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.0, 0.0]})
        out = model.predict(df)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_scipy_sparse_csr(self, model):
        sp = pytest.importorskip("scipy.sparse")
        X = sp.csr_matrix(np.array([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32))
        out = model.predict(X)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_tensor_input(self, model):
        t = Tensor([[0.1, 0.0], [0.9, 0.0]])
        out = model.predict(t)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_wrong_feature_count_raises(self, model):
        X = np.ones((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="feature"):
            model.predict(X)

    def test_3d_array_raises(self, model):
        X = np.ones((2, 2, 2), dtype=np.float32)
        with pytest.raises(ValueError):
            model.predict(X)


# ===========================================================================
# sklearn estimator protocol
# ===========================================================================


class TestSklearnProtocol:
    def test_get_params_returns_dict(self, model):
        assert isinstance(model.get_params(), dict)

    def test_get_params_deep(self, model):
        assert isinstance(model.get_params(deep=True), dict)

    def test_set_params_returns_self(self, model):
        assert model.set_params() is model

    def test_fit_returns_self(self, model):
        X = np.ones((3, 2), dtype=np.float32)
        assert model.fit(X) is model

    def test_fit_with_y_ignored(self, model):
        X = np.ones((3, 2), dtype=np.float32)
        y = np.array([0, 1, 0])
        assert model.fit(X, y) is model

    def test_fit_validates_wrong_features(self, model):
        X = np.ones((3, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="feature"):
            model.fit(X)

    def test_pipeline(self, model):
        sklearn = pytest.importorskip("sklearn")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        X = np.array([[0.1, 0.0], [0.9, 0.0], [-0.5, 1.0]], dtype=np.float64)
        pipe.fit(X)
        out = pipe.predict(X)
        assert out.shape == (3,)
        assert out.dtype == np.float32


# ===========================================================================
# Session
# ===========================================================================


class TestSession:
    def test_create_returns_session(self, model):
        assert isinstance(model.create_session(), Session)

    def test_repr(self, model):
        s = model.create_session()
        assert "Session" in repr(s)

    def test_num_outputs(self, model):
        s = model.create_session()
        assert s.num_outputs == 1

    def test_predict_correctness(self, model):
        s = model.create_session()
        X = np.array([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32)
        out = s.predict(X)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_predict_1d_output_for_single_output(self, model):
        s = model.create_session()
        X = np.ones((4, 2), dtype=np.float32)
        out = s.predict(X)
        assert out.ndim == 1
        assert out.shape == (4,)

    def test_predict_proba_always_2d(self, model):
        s = model.create_session()
        X = np.ones((4, 2), dtype=np.float32)
        out = s.predict_proba(X)
        assert out.ndim == 2
        assert out.shape == (4, 1)

    def test_run_returns_tensor(self, model):
        s = model.create_session()
        X = np.array([[0.1, 0.0]], dtype=np.float32)
        t = s.run(X)
        assert isinstance(t, Tensor)

    def test_run_tensor_name(self, model):
        s = model.create_session()
        t = s.run(np.array([[0.1, 0.0]], dtype=np.float32))
        assert t.name == "score"

    def test_run_tensor_value(self, model):
        s = model.create_session()
        t = s.run(np.array([[0.1, 0.0]], dtype=np.float32))
        np.testing.assert_allclose(t.numpy().ravel(), [2.0], rtol=1e-5)

    def test_run_squeeze_false_is_2d(self, model):
        s = model.create_session()
        t = s.run(np.array([[0.1, 0.0]], dtype=np.float32), squeeze=False)
        assert t.numpy().ndim == 2

    def test_reuse_across_calls(self, model):
        s = model.create_session()
        cases = [(0.1, 2.0), (0.9, -2.0), (0.2, 2.0), (0.8, -2.0)]
        for feat0, expected in cases:
            X = np.array([[feat0, 0.0]], dtype=np.float32)
            out = s.predict(X)
            np.testing.assert_allclose(out, [expected], rtol=1e-5)

    def test_session_matches_model_predict(self, model):
        s = model.create_session()
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, 2)).astype(np.float32)
        np.testing.assert_allclose(s.predict(X), model.predict(X), rtol=1e-5)

    def test_session_coercion_list(self, model):
        s = model.create_session()
        out = s.predict([[0.1, 0.0], [0.9, 0.0]])
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_session_coercion_pandas(self, model):
        pd = pytest.importorskip("pandas")
        s = model.create_session()
        df = pd.DataFrame({"f0": [0.1, 0.9], "f1": [0.0, 0.0]})
        out = s.predict(df)
        np.testing.assert_allclose(out, [2.0, -2.0], rtol=1e-5)

    def test_session_wrong_features_raises(self, model):
        s = model.create_session()
        X = np.ones((2, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="feature"):
            s.predict(X)

    def test_independent_sessions_same_result(self, model):
        s1 = model.create_session()
        s2 = model.create_session()
        X = np.array([[0.1, 0.0], [0.9, 0.0]], dtype=np.float32)
        np.testing.assert_allclose(s1.predict(X), s2.predict(X), rtol=1e-5)


# ===========================================================================
# Mixed-type DataFrame (string + numeric columns)
# ===========================================================================


class TestStringDataFrame:
    """String-only DataFrame routed through a LabelEncoder model."""

    def test_predict_string_df(self, label_encoder_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"category": ["A", "C", "B"]})
        out = label_encoder_model.predict(df)
        # A→0, C→2, B→1
        np.testing.assert_allclose(out, [0.0, 2.0, 1.0], rtol=1e-5)

    def test_predict_output_shape(self, label_encoder_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"category": ["A", "B", "C", "A"]})
        out = label_encoder_model.predict(df)
        assert out.shape == (4,)

    def test_session_predict_string_df(self, label_encoder_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"category": ["B", "A", "C"]})
        s = label_encoder_model.create_session()
        out = s.predict(df)
        # B→1, A→0, C→2
        np.testing.assert_allclose(out, [1.0, 0.0, 2.0], rtol=1e-5)

    def test_session_run_returns_tensor(self, label_encoder_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"category": ["A"]})
        s = label_encoder_model.create_session()
        t = s.run(df)
        assert isinstance(t, Tensor)
        np.testing.assert_allclose(t.numpy().ravel(), [0.0], rtol=1e-5)

    def test_unknown_label_gives_nan(self, label_encoder_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"category": ["Z"]})
        out = label_encoder_model.predict(df)
        assert np.isnan(out[0])


class TestMixedDataFrame:
    """DataFrame with both numeric and string columns dispatched per-column."""

    def test_predict_mixed_df(self, mixed_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [25.0, 40.0, 55.0], "category": ["low", "high", "med"]})
        out = mixed_model.predict(df)
        # low→0, high→2, med→1
        np.testing.assert_allclose(out, [0.0, 2.0, 1.0], rtol=1e-5)

    def test_predict_output_shape(self, mixed_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [1.0, 2.0], "category": ["low", "med"]})
        out = mixed_model.predict(df)
        assert out.shape == (2,)

    def test_session_predict_mixed_df(self, mixed_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [10.0, 20.0], "category": ["high", "low"]})
        s = mixed_model.create_session()
        out = s.predict(df)
        # high→2, low→0
        np.testing.assert_allclose(out, [2.0, 0.0], rtol=1e-5)

    def test_session_run_mixed_df(self, mixed_model):
        pd = pytest.importorskip("pandas")
        df = pd.DataFrame({"age": [5.0], "category": ["med"]})
        s = mixed_model.create_session()
        t = s.run(df)
        assert isinstance(t, Tensor)
        np.testing.assert_allclose(t.numpy().ravel(), [1.0], rtol=1e-5)

    def test_reuse_session_across_mixed_calls(self, mixed_model):
        pd = pytest.importorskip("pandas")
        s = mixed_model.create_session()
        cases = [("low", 0.0), ("med", 1.0), ("high", 2.0)]
        for cat, expected in cases:
            df = pd.DataFrame({"age": [0.0], "category": [cat]})
            out = s.predict(df)
            np.testing.assert_allclose(out, [expected], rtol=1e-5)
