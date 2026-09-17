"""Runtime prediction tests for XGBoost, LightGBM, and CatBoost tree ensembles.

Covers: float32/float64 inputs, mixed-dtype DataFrame pipelines, and categorical
string features.  Each model is converted to OMLE, loaded with omleruntime,
and verified to produce predictions identical to the native framework.
"""

import numpy as np
import pytest

# pandas is only needed by the DataFrame pipeline cases, and the conversion
# dependencies below are optional too, so it is skipped rather than imported at
# module scope -- a hard import breaks collection for the whole file.
pd = pytest.importorskip("pandas", reason="pandas not installed")

omle          = pytest.importorskip("omle",           reason="omle not installed")
omr              = pytest.importorskip("omleruntime",    reason="omleruntime not installed")
omle_convert  = pytest.importorskip("omle_convert",  reason="omle_convert not installed")

_RTOL = 1e-4
_ATOL = 1e-4


# ── Helper ────────────────────────────────────────────────────────────────────

def _load(ir_model) -> omr.Model:
    return omr.load_bytes(omle.to_proto_bytes(ir_model))


# ── Shared data fixtures ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def regression_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    y = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.standard_normal(200).astype(np.float32)
    return X, y


@pytest.fixture(scope="session")
def binary_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture(scope="session")
def multiclass_data():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 6)).astype(np.float32)
    y = (np.abs(X[:, 0]) * 3).astype(int).clip(0, 2)
    return X, y


@pytest.fixture(scope="session")
def regression_data_f64():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6))
    y = X[:, 0] * 2.0 + X[:, 1] - X[:, 2] + rng.standard_normal(200)
    return X, y


@pytest.fixture(scope="session")
def binary_data_f64():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((200, 6))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture(scope="session")
def multiclass_data_f64():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((300, 6))
    y = (np.abs(X[:, 0]) * 3).astype(int).clip(0, 2)
    return X, y


@pytest.fixture(scope="session")
def mixed_df():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({
        "age":       rng.integers(18, 80, size=n),
        "education": rng.integers(6, 22, size=n).astype(np.int32),
        "income":    rng.uniform(20_000, 150_000, size=n),
        "stock":     rng.uniform(0.0, 10_000.0, size=n).astype(np.float32),
        "active":    rng.integers(0, 2, size=n).astype(bool),
        "gender":    rng.choice(["M", "F"], size=n),
        "segment":   rng.choice(["low", "mid", "high", "premium"], size=n),
    })
    y_regression = (df["income"].to_numpy() / 1000 + rng.standard_normal(n)).astype(np.float32)
    y_binary     = (df["age"].to_numpy() > 40).astype(int)
    y_multiclass = np.where(df["segment"] == "low", 0,
                   np.where(df["segment"] == "mid", 1,
                   np.where(df["segment"] == "high", 2, 3)))
    return df, y_regression, y_binary, y_multiclass


@pytest.fixture(scope="session")
def cat_mixed_df(mixed_df):
    df, y_reg, y_bin, y_mc = mixed_df
    df_cat = df.copy()
    df_cat["gender"]  = df_cat["gender"].astype("category")
    df_cat["segment"] = df_cat["segment"].astype("category")
    return df_cat, y_reg, y_bin, y_mc


# ── XGBoost fixtures ──────────────────────────────────────────────────────────

xgb = pytest.importorskip("xgboost", reason="xgboost not installed")

_N = 10


@pytest.fixture(scope="session")
def xgb_regressor(regression_data):
    X, y = regression_data
    m = xgb.XGBRegressor(n_estimators=_N, max_depth=3, random_state=0, verbosity=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def xgb_binary(binary_data):
    X, y = binary_data
    m = xgb.XGBClassifier(n_estimators=_N, max_depth=3, random_state=0,
                           verbosity=0, eval_metric="logloss")
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def xgb_multiclass(multiclass_data):
    X, y = multiclass_data
    m = xgb.XGBClassifier(n_estimators=_N, max_depth=3, num_class=3,
                           random_state=0, verbosity=0, eval_metric="mlogloss")
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def xgb_regressor_f64(regression_data_f64):
    X, y = regression_data_f64
    m = xgb.XGBRegressor(n_estimators=_N, max_depth=3, random_state=0, verbosity=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def xgb_binary_f64(binary_data_f64):
    X, y = binary_data_f64
    m = xgb.XGBClassifier(n_estimators=_N, max_depth=3, random_state=0,
                           verbosity=0, eval_metric="logloss")
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def xgb_multiclass_f64(multiclass_data_f64):
    X, y = multiclass_data_f64
    m = xgb.XGBClassifier(n_estimators=_N, max_depth=3, num_class=3,
                           random_state=0, verbosity=0, eval_metric="mlogloss")
    m.fit(X, y)
    return m


def _make_xgb_mixed_pipeline(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    import inspect
    ct_kwargs = {"remainder": "passthrough"}
    if "force_int_remainder_cols" in inspect.signature(ColumnTransformer).parameters:
        ct_kwargs["force_int_remainder_cols"] = False
    ct = ColumnTransformer([("ord", OrdinalEncoder(), ["gender", "segment"])], **ct_kwargs)
    return Pipeline([("preprocessor", ct), ("model", estimator)])


@pytest.fixture(scope="session")
def xgb_mixed_regression(mixed_df):
    df, y_reg, _, _ = mixed_df
    pipeline = _make_xgb_mixed_pipeline(
        xgb.XGBRegressor(n_estimators=_N, max_depth=3, random_state=0, verbosity=0)
    )
    pipeline.fit(df, y_reg)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="session")
def xgb_mixed_binary(mixed_df):
    df, _, y_bin, _ = mixed_df
    pipeline = _make_xgb_mixed_pipeline(
        xgb.XGBClassifier(n_estimators=_N, max_depth=3, random_state=0,
                           verbosity=0, eval_metric="logloss")
    )
    pipeline.fit(df, y_bin)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="session")
def xgb_mixed_multiclass(mixed_df):
    df, _, _, y_mc = mixed_df
    pipeline = _make_xgb_mixed_pipeline(
        xgb.XGBClassifier(n_estimators=_N, max_depth=3, num_class=4,
                           random_state=0, verbosity=0, eval_metric="mlogloss")
    )
    pipeline.fit(df, y_mc)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="session")
def xgb_cat_gender_target(cat_mixed_df):
    df, _, _, _ = cat_mixed_df
    y = (df["gender"].astype(str) == "M").astype(int)
    model = xgb.XGBClassifier(n_estimators=_N, max_depth=4, random_state=0,
                               verbosity=0, eval_metric="logloss",
                               enable_categorical=True)
    model.fit(df, y)
    return model, df, y


# ── XGBoost runtime tests ─────────────────────────────────────────────────────

class TestXGBRegressionRuntime:
    def test_predict_matches_native(self, xgb_regressor, regression_data):
        X, _ = regression_data
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(xgb_regressor))
        np.testing.assert_allclose(rt.predict(X), xgb_regressor.predict(X),
                                   rtol=_RTOL, atol=_ATOL)


class TestXGBBinaryRuntime:
    def test_predict_matches_native(self, xgb_binary, binary_data):
        X, _ = binary_data
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(xgb_binary))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            xgb_binary.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), xgb_binary.predict(X))


class TestXGBMulticlassRuntime:
    def test_predict_matches_native(self, xgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(xgb_multiclass))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            xgb_multiclass.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), xgb_multiclass.predict(X))


class TestXGBFloat64Runtime:
    def test_regression_predict_matches_native(self, xgb_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(xgb_regressor_f64))
        np.testing.assert_allclose(rt.predict(X), xgb_regressor_f64.predict(X),
                                   rtol=_RTOL, atol=_ATOL)

    def test_binary_predict_matches_native(self, xgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(xgb_binary_f64))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            xgb_binary_f64.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), xgb_binary_f64.predict(X))

    def test_multiclass_predict_matches_native(self, xgb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(xgb_multiclass_f64))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            xgb_multiclass_f64.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), xgb_multiclass_f64.predict(X))


class TestXGBMixedRegressionRuntime:
    def test_predict_matches_native(self, xgb_mixed_regression, mixed_df):
        pipeline, cols = xgb_mixed_regression
        df, _, _, _ = mixed_df
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(pipeline, X=df[cols]))
        np.testing.assert_allclose(rt.predict(df[cols]), pipeline.predict(df[cols]),
                                   rtol=_RTOL, atol=_ATOL)


class TestXGBMixedBinaryRuntime:
    def test_predict_matches_native(self, xgb_mixed_binary, mixed_df):
        pipeline, cols = xgb_mixed_binary
        df, _, _, _ = mixed_df
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(pipeline, X=df[cols]))
        np.testing.assert_allclose(
            rt.predict_proba(df[cols]),
            pipeline.predict_proba(df[cols]),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(df[cols]), pipeline.predict(df[cols]))


class TestXGBMixedMulticlassRuntime:
    def test_predict_matches_native(self, xgb_mixed_multiclass, mixed_df):
        pipeline, cols = xgb_mixed_multiclass
        df, _, _, _ = mixed_df
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(pipeline, X=df[cols]))
        np.testing.assert_allclose(
            rt.predict_proba(df[cols]),
            pipeline.predict_proba(df[cols]),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(df[cols]), pipeline.predict(df[cols]))


class TestXGBCatStringRuntime:
    def test_predict_matches_native(self, xgb_cat_gender_target):
        model, df, _ = xgb_cat_gender_target
        from omle_convert.xgboost import from_xgboost
        rt = _load(from_xgboost(model, X=df))
        np.testing.assert_allclose(
            rt.predict_proba(df),
            model.predict_proba(df),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(df), model.predict(df))


# ── LightGBM fixtures ─────────────────────────────────────────────────────────

lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")


@pytest.fixture(scope="session")
def lgb_regressor(regression_data):
    X, y = regression_data
    m = lgb.LGBMRegressor(n_estimators=_N, max_depth=3, random_state=0, verbose=-1)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def lgb_binary(binary_data):
    X, y = binary_data
    m = lgb.LGBMClassifier(n_estimators=_N, max_depth=3, random_state=0, verbose=-1)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def lgb_multiclass(multiclass_data):
    X, y = multiclass_data
    m = lgb.LGBMClassifier(n_estimators=_N, max_depth=3, num_class=3,
                            random_state=0, verbose=-1)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def lgb_regressor_f64(regression_data_f64):
    X, y = regression_data_f64
    m = lgb.LGBMRegressor(n_estimators=_N, max_depth=3, random_state=0, verbose=-1)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def lgb_binary_f64(binary_data_f64):
    X, y = binary_data_f64
    m = lgb.LGBMClassifier(n_estimators=_N, max_depth=3, random_state=0, verbose=-1)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def lgb_multiclass_f64(multiclass_data_f64):
    X, y = multiclass_data_f64
    m = lgb.LGBMClassifier(n_estimators=_N, max_depth=3, num_class=3,
                            random_state=0, verbose=-1)
    m.fit(X, y)
    return m


def _make_lgb_mixed_pipeline(estimator):
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OrdinalEncoder
    import inspect
    ct_kwargs = {"remainder": "passthrough"}
    if "force_int_remainder_cols" in inspect.signature(ColumnTransformer).parameters:
        ct_kwargs["force_int_remainder_cols"] = False
    ct = ColumnTransformer([("ord", OrdinalEncoder(), ["gender", "segment"])], **ct_kwargs)
    return Pipeline([("preprocessor", ct), ("model", estimator)])


@pytest.fixture(scope="session")
def lgb_mixed_regression(mixed_df):
    df, y_reg, _, _ = mixed_df
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMRegressor(n_estimators=_N, max_depth=3, random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_reg)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="session")
def lgb_mixed_binary(mixed_df):
    df, _, y_bin, _ = mixed_df
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMClassifier(n_estimators=_N, max_depth=3, random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_bin)
    return pipeline, df.columns.tolist()


@pytest.fixture(scope="session")
def lgb_mixed_multiclass(mixed_df):
    df, _, _, y_mc = mixed_df
    pipeline = _make_lgb_mixed_pipeline(
        lgb.LGBMClassifier(n_estimators=_N, max_depth=3, num_class=4,
                            random_state=0, verbose=-1)
    )
    pipeline.fit(df, y_mc)
    return pipeline, df.columns.tolist()


# ── LightGBM runtime tests ────────────────────────────────────────────────────

class TestLGBRegressionRuntime:
    def test_predict_matches_native(self, lgb_regressor, regression_data):
        X, _ = regression_data
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(lgb_regressor))
        np.testing.assert_allclose(rt.predict(X), lgb_regressor.predict(X),
                                   rtol=_RTOL, atol=_ATOL)


class TestLGBBinaryRuntime:
    def test_predict_matches_native(self, lgb_binary, binary_data):
        X, _ = binary_data
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(lgb_binary))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            lgb_binary.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), lgb_binary.predict(X))


class TestLGBMulticlassRuntime:
    def test_predict_matches_native(self, lgb_multiclass, multiclass_data):
        X, _ = multiclass_data
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(lgb_multiclass))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            lgb_multiclass.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), lgb_multiclass.predict(X))


class TestLGBFloat64Runtime:
    def test_regression_predict_matches_native(self, lgb_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(lgb_regressor_f64))
        np.testing.assert_allclose(rt.predict(X), lgb_regressor_f64.predict(X),
                                   rtol=_RTOL, atol=_ATOL)

    def test_binary_predict_matches_native(self, lgb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(lgb_binary_f64))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            lgb_binary_f64.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), lgb_binary_f64.predict(X))

    def test_multiclass_predict_matches_native(self, lgb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(lgb_multiclass_f64))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            lgb_multiclass_f64.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), lgb_multiclass_f64.predict(X))


class TestLGBMixedRegressionRuntime:
    def test_predict_matches_native(self, lgb_mixed_regression, mixed_df):
        pipeline, cols = lgb_mixed_regression
        df, _, _, _ = mixed_df
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(pipeline, X=df[cols]))
        np.testing.assert_allclose(rt.predict(df[cols]), pipeline.predict(df[cols]),
                                   rtol=_RTOL, atol=_ATOL)


class TestLGBMixedBinaryRuntime:
    def test_predict_matches_native(self, lgb_mixed_binary, mixed_df):
        pipeline, cols = lgb_mixed_binary
        df, _, _, _ = mixed_df
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(pipeline, X=df[cols]))
        np.testing.assert_allclose(
            rt.predict_proba(df[cols]),
            pipeline.predict_proba(df[cols]),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(df[cols]), pipeline.predict(df[cols]))


class TestLGBMixedMulticlassRuntime:
    def test_predict_matches_native(self, lgb_mixed_multiclass, mixed_df):
        pipeline, cols = lgb_mixed_multiclass
        df, _, _, _ = mixed_df
        from omle_convert.lightgbm import from_lightgbm
        rt = _load(from_lightgbm(pipeline, X=df[cols]))
        np.testing.assert_allclose(
            rt.predict_proba(df[cols]),
            pipeline.predict_proba(df[cols]),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(df[cols]), pipeline.predict(df[cols]))


# ── CatBoost fixtures ─────────────────────────────────────────────────────────

cb = pytest.importorskip("catboost", reason="catboost not installed")

_DEPTH = 4


@pytest.fixture(scope="session")
def cb_regressor(regression_data):
    X, y = regression_data
    m = cb.CatBoostRegressor(n_estimators=_N, depth=_DEPTH, random_seed=0, verbose=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def cb_binary(binary_data):
    X, y = binary_data
    m = cb.CatBoostClassifier(n_estimators=_N, depth=_DEPTH, random_seed=0, verbose=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def cb_multiclass(multiclass_data):
    X, y = multiclass_data
    m = cb.CatBoostClassifier(n_estimators=_N, depth=_DEPTH, random_seed=0, verbose=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def cb_regressor_f64(regression_data_f64):
    X, y = regression_data_f64
    m = cb.CatBoostRegressor(n_estimators=_N, depth=_DEPTH, random_seed=0, verbose=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def cb_binary_f64(binary_data_f64):
    X, y = binary_data_f64
    m = cb.CatBoostClassifier(n_estimators=_N, depth=_DEPTH, random_seed=0, verbose=0)
    m.fit(X, y)
    return m


@pytest.fixture(scope="session")
def cb_multiclass_f64(multiclass_data_f64):
    X, y = multiclass_data_f64
    m = cb.CatBoostClassifier(n_estimators=_N, depth=_DEPTH, random_seed=0, verbose=0)
    m.fit(X, y)
    return m


# ── CatBoost runtime tests ────────────────────────────────────────────────────

class TestCBRegressionRuntime:
    def test_predict_matches_native(self, cb_regressor, regression_data):
        X, _ = regression_data
        from omle_convert.catboost import from_catboost
        rt = _load(from_catboost(cb_regressor))
        np.testing.assert_allclose(rt.predict(X), cb_regressor.predict(X),
                                   rtol=_RTOL, atol=_ATOL)


class TestCBBinaryRuntime:
    def test_predict_matches_native(self, cb_binary, binary_data):
        X, _ = binary_data
        from omle_convert.catboost import from_catboost
        rt = _load(from_catboost(cb_binary))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            cb_binary.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), cb_binary.predict(X))


class TestCBMulticlassRuntime:
    def test_predict_matches_native(self, cb_multiclass, multiclass_data):
        X, _ = multiclass_data
        from omle_convert.catboost import from_catboost
        rt = _load(from_catboost(cb_multiclass))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            cb_multiclass.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), cb_multiclass.predict(X).ravel())


class TestCBFloat64Runtime:
    def test_regression_predict_matches_native(self, cb_regressor_f64, regression_data_f64):
        X, _ = regression_data_f64
        from omle_convert.catboost import from_catboost
        rt = _load(from_catboost(cb_regressor_f64))
        np.testing.assert_allclose(rt.predict(X), cb_regressor_f64.predict(X),
                                   rtol=_RTOL, atol=_ATOL)

    def test_binary_predict_matches_native(self, cb_binary_f64, binary_data_f64):
        X, _ = binary_data_f64
        from omle_convert.catboost import from_catboost
        rt = _load(from_catboost(cb_binary_f64))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            cb_binary_f64.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), cb_binary_f64.predict(X))

    def test_multiclass_predict_matches_native(self, cb_multiclass_f64, multiclass_data_f64):
        X, _ = multiclass_data_f64
        from omle_convert.catboost import from_catboost
        rt = _load(from_catboost(cb_multiclass_f64))
        np.testing.assert_allclose(
            rt.predict_proba(X),
            cb_multiclass_f64.predict_proba(X),
            rtol=_RTOL, atol=_ATOL,
        )
        assert np.array_equal(rt.predict(X), cb_multiclass_f64.predict(X).ravel())
