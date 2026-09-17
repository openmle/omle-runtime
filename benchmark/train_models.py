"""Train XGBoost/LightGBM models and convert to OMLE protobuf + ONNX.

Datasets
--------
Regression  : California Housing  (sklearn)  — 8 features,  20 640 samples
Binary      : Breast Cancer Wisc. (sklearn)  — 30 features,    569 samples
Multiclass  : Digits (MNIST-tiny) (sklearn)  — 64 features,  1 797 samples, 10 classes
MNIST       : MNIST-784            (OpenML)  — 784 features, 70 000 samples, 10 classes
Adult       : Adult Census Income  (OpenML)  — 14 features (6 numeric + 8 categorical)
              sklearn Pipeline: ColumnTransformer(StandardScaler, OneHotEncoder)
                                → RandomForestClassifier

Model configurations
--------------------
  small  : n_estimators=100, max_depth=4  (mnist: 50 trees, depth 3)
  medium : n_estimators=500, max_depth=6  (mnist: 100 trees, depth 4)
  large  : n_estimators=1000, max_depth=8 (mnist: 200 trees, depth 5)

Outputs written to ./models/
  {task}_{size}_xgb.json   — XGBoost native model (JSON)
  {task}_{size}_xgb.onnx   — XGBoost ONNX model
  {task}_{size}_lgb.txt    — LightGBM native model
  {task}_{size}_lgb.onnx   — LightGBM ONNX model
  {task}_{size}_X_test.npy — float32 test features  (n_test × n_features)
  {task}_{size}_y_test.npy — float32 ground truth
  {task}_{size}.omle        — OMLE model (XGBoost source)
  {task}_{size}_lgb.omle   — OMLE model (LightGBM source)
  adult_{size}_pipe.pkl    — sklearn Pipeline (joblib)
  adult_{size}_pipe.onnx   — ONNX model (skl2onnx)
  adult_{size}_pipe.omle   — OMLE model (sklearn converter)
  adult_{size}_X_test.pkl  — test DataFrame with mixed dtypes (pickle)
  adult_{size}_y_test.npy  — test labels

Usage
-----
  python train_models.py [--tasks regression binary multiclass mnist adult]
                         [--sizes small medium large]
                         [--force]            # re-train even if outputs exist

Note: MNIST downloads ~170 MB from OpenML on first run (cached by sklearn).
      Training takes ~20 s (small) / ~40 s (medium) / ~100 s (large).
      Adult requires skl2onnx for ONNX export: pip install skl2onnx
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# ── Path bootstrap ────────────────────────────────────────────────────────────
_BENCH = Path(__file__).resolve().parent
_ROOT  = _BENCH.parent
_SIBLING = _ROOT.parent

sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_SIBLING / "omle" / "src"))
sys.path.insert(0, str(_SIBLING / "omle-convert" / "src"))

# ── Imports ───────────────────────────────────────────────────────────────────
import numpy as np
import xgboost as xgb
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_breast_cancer,
    load_digits,
)
from sklearn.model_selection import train_test_split

import omle
from omle_convert.xgboost import from_xgboost

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR = _BENCH / "models"
SEED = 42
TEST_FRAC = 0.2

# XGBoost: (n_estimators, max_depth, learning_rate)
SIZE_CONFIGS: dict[str, tuple[int, int, float]] = {
    "small":  (100,   4, 0.1),
    "medium": (500,   6, 0.1),
    "large":  (1000,  8, 0.05),
}

# Multiclass (digits) and MNIST use shallower trees; each round trains n_classes trees
MULTICLASS_DEPTHS = {"small": 3, "medium": 4, "large": 5}

# MNIST overrides both n_estimators and depth
MNIST_CONFIGS: dict[str, tuple[int, int, float]] = {
    "small":  (50,  3, 0.1),
    "medium": (100, 4, 0.1),
    "large":  (200, 5, 0.05),
}

# LightGBM: (n_estimators, max_depth, num_leaves, learning_rate)
# num_leaves = 2^max_depth to match XGBoost tree complexity
LGB_SIZE_CONFIGS: dict[str, tuple[int, int, int, float]] = {
    "small":  (100,   4, 16,  0.1),
    "medium": (500,   6, 64,  0.1),
    "large":  (1000,  8, 256, 0.05),
}
LGB_MULTICLASS_DEPTHS = {"small": 3, "medium": 4, "large": 5}
LGB_MULTICLASS_LEAVES = {"small": 8, "medium": 16, "large": 32}
LGB_MNIST_CONFIGS: dict[str, tuple[int, int, int, float]] = {
    "small":  (50,  3, 8,  0.1),
    "medium": (100, 4, 16, 0.1),
    "large":  (200, 5, 32, 0.05),
}


def _stamp(msg: str) -> None:
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}")


# ── ONNX export helpers ───────────────────────────────────────────────────────

def _export_xgb_onnx(xgb_model, n_feat: int, path: Path) -> None:
    try:
        from onnxmltools.convert import convert_xgboost
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print("    [onnx] onnxmltools not found — skipping ONNX export")
        return
    onx = convert_xgboost(
        xgb_model.get_booster(),
        initial_types=[("float_input", FloatTensorType([None, n_feat]))],
    )
    with open(str(path), "wb") as f:
        f.write(onx.SerializeToString())
    _stamp(f"saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


def _export_lgb_onnx(lgb_model, n_feat: int, path: Path) -> None:
    try:
        from onnxmltools.convert import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
    except ImportError:
        print("    [onnx] onnxmltools not found — skipping ONNX export")
        return
    onx = convert_lightgbm(
        lgb_model.booster_,
        initial_types=[("float_input", FloatTensorType([None, n_feat]))],
        target_opset=12,
    )
    with open(str(path), "wb") as f:
        f.write(onx.SerializeToString())
    _stamp(f"saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


# ── Dataset loaders ───────────────────────────────────────────────────────────

def load_regression():
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    feat_names = list(data.feature_names)
    return X, y, feat_names, "MedHouseVal"


def load_binary():
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    feat_names = [n.replace(" ", "_").replace("(", "").replace(")", "") for n in data.feature_names]
    return X, y, feat_names, "malignant"


def load_multiclass():
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int32)
    feat_names = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, y, feat_names, "digit"


def load_mnist():
    _stamp("fetching MNIST-784 from OpenML (downloads ~170 MB on first run, then cached)")
    data = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = data.data.astype(np.float32)
    y = data.target.astype(np.int32)
    feat_names = [f"pixel_{i}" for i in range(X.shape[1])]
    return X, y, feat_names, "digit"


LOADERS = {
    "regression":  load_regression,
    "binary":      load_binary,
    "multiclass":  load_multiclass,
    "mnist":       load_mnist,
}

# Tasks that are multiclass classification (share training/accuracy logic)
MULTICLASS_TASKS = {"multiclass", "mnist"}


# ── Adult Census (mixed-type pipeline) ────────────────────────────────────────

ADULT_PIPE_SIZES: dict[str, tuple[int, int]] = {
    "small":  (100, 4),
    "medium": (500, 6),
    "large":  (1000, 8),
}


def load_adult_df():
    """Return (df, y, num_cols, cat_cols) for the Adult Census Income dataset."""
    _stamp("fetching Adult Census Income from OpenML (cached after first run)")
    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame.copy()
    df = df.dropna().reset_index(drop=True)
    target = df["class"].astype(str).str.strip().str.rstrip(".")
    y = (target == ">50K").astype(np.int32).values
    df = df.drop(columns=["class"])
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].astype(np.float32)
    for c in cat_cols:
        df[c] = df[c].astype(str)
    return df, y, num_cols, cat_cols


def build_adult_pipeline(n_est: int, depth: int, num_cols: list, cat_cols: list):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), cat_cols),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, random_state=SEED, n_jobs=1,
        )),
    ])


def _export_adult_onnx(pipe, num_cols: list, cat_cols: list, path: Path) -> None:
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType, StringTensorType
    except ImportError:
        print("    [onnx] skl2onnx not found — skipping ONNX export (pip install skl2onnx)")
        return
    initial_types = (
        [(c, FloatTensorType([None, 1])) for c in num_cols] +
        [(c, StringTensorType([None, 1])) for c in cat_cols]
    )
    try:
        onx = convert_sklearn(pipe, initial_types=initial_types,
                              options={"zipmap": False}, target_opset=15)
        with open(str(path), "wb") as f:
            f.write(onx.SerializeToString())
        _stamp(f"saved {path.name}  ({path.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"    [onnx] skl2onnx conversion failed: {e}")


def _encode_adult_for_omle(
    X_df,
    all_col_order: list,
    cat_cols: list,
    ohe_categories: list,
) -> np.ndarray:
    """Return a flat float32 array ordered by feature_names_in_ for OMLE inference.

    OMLE's TakeSlots uses column indices based on feature_names_in_, so we must
    build the flat array in that same order.  Categorical columns are replaced by
    their ordinal index into the OHE's sorted category list so that OMLE's OHE
    node (which accepts numeric ordinal-index input) produces identical output.
    """
    n = len(X_df)
    cat_to_ord = {
        col: {str(c): float(i) for i, c in enumerate(cats)}
        for col, cats in zip(cat_cols, ohe_categories)
    }
    out = np.zeros((n, len(all_col_order)), dtype=np.float32)
    for j, col in enumerate(all_col_order):
        if col in cat_to_ord:
            vals = X_df[col].astype(str).values
            for r, v in enumerate(vals):
                out[r, j] = cat_to_ord[col].get(v, -1.0)
        else:
            out[:, j] = X_df[col].values.astype(np.float32)
    return np.ascontiguousarray(out)


def convert_and_save_adult(
    size: str,
    pipe,
    num_cols: list,
    cat_cols: list,
    X_test_df,
    y_test: np.ndarray,
    force: bool,
) -> bool:
    import joblib
    from omle_convert.sklearn import from_sklearn

    prefix    = MODELS_DIR / f"adult_{size}"
    pb_path   = Path(str(prefix) + "_pipe.omle")
    pkl_path  = Path(str(prefix) + "_pipe.pkl")
    onnx_path = Path(str(prefix) + "_pipe.onnx")
    X_path    = Path(str(prefix) + "_X_test.pkl")
    y_path    = Path(str(prefix) + "_y_test.npy")

    if pb_path.exists() and not force:
        print(f"  skip   {pb_path.name} (exists; use --force to overwrite)")
        return False

    joblib.dump(pipe, str(pkl_path))
    X_test_df.to_pickle(str(X_path))
    np.save(str(y_path), y_test)

    _stamp(f"converting adult/{size} → {onnx_path.name}")
    _export_adult_onnx(pipe, num_cols, cat_cols, onnx_path)

    _stamp(f"converting adult/{size} → {pb_path.name}")
    try:
        # Per-column mode: pass X_test_df so from_sklearn generates named per-column
        # InputSpecs with correct dtypes (float32 for numeric, string for categorical).
        # This lets session.predict_proba() accept a mixed-type DataFrame directly.
        omle_model = from_sklearn(
            pipe, X=X_test_df,
            n_verify=min(5, len(X_test_df)), n_warmup=10,
            verify_atol=0.05, verify_rtol=0.05,
        )
        omle.save(omle_model, str(pb_path))
        _stamp(f"saved {pb_path.name}  ({pb_path.stat().st_size / 1024:.0f} KB)")
    except Exception as e:
        print(f"    [omle] from_sklearn failed: {e}")
        return False
    return True


def run_adult(sizes: list[str], force: bool) -> None:
    """Train and convert the Adult Census pipeline benchmark."""
    MODELS_DIR.mkdir(exist_ok=True)
    print(f"\n{'='*60}")
    print("Task: ADULT (mixed-type sklearn pipeline)")
    print(f"{'='*60}")

    df, y, num_cols, cat_cols = load_adult_df()
    print(f"  dataset: {len(df):,} samples × {df.shape[1]} features")
    print(f"  numeric ({len(num_cols)}): {num_cols}")
    print(f"  categorical ({len(cat_cols)}): {cat_cols}")

    df_tr, df_te, y_tr, y_te = train_test_split(
        df, y, test_size=TEST_FRAC, random_state=SEED, stratify=y,
    )
    print(f"  split: {len(df_tr):,} train / {len(df_te):,} test")

    for size in sizes:
        n_est, depth = ADULT_PIPE_SIZES[size]
        print(f"\n  [pipeline] {size}: RandomForest {n_est} trees × depth {depth}")

        pb_path = Path(str(MODELS_DIR / f"adult_{size}") + "_pipe.omle")
        if pb_path.exists() and not force:
            print(f"  skip   {pb_path.name} (exists; use --force to overwrite)")
            continue

        _stamp("training pipeline")
        t0 = time.perf_counter()
        pipe = build_adult_pipeline(n_est, depth, num_cols, cat_cols)
        pipe.fit(df_tr, y_tr)
        _stamp(f"done in {time.perf_counter() - t0:.1f}s")

        from sklearn.metrics import accuracy_score
        print(f"  Accuracy on test: {accuracy_score(y_te, pipe.predict(df_te)):.4f}")

        convert_and_save_adult(size, pipe, num_cols, cat_cols, df_te, y_te, force)


# ── Training ──────────────────────────────────────────────────────────────────

def train_regression(X_tr, y_tr, n_est, depth, lr):
    model = xgb.XGBRegressor(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        tree_method="hist",
        device="cpu",
        verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model


def train_binary(X_tr, y_tr, n_est, depth, lr):
    model = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        objective="binary:logistic",
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        tree_method="hist",
        device="cpu",
        verbosity=0,
        eval_metric="logloss",
    )
    model.fit(X_tr, y_tr)
    return model


def train_multiclass(X_tr, y_tr, n_classes, n_est, depth, lr):
    model = xgb.XGBClassifier(
        n_estimators=n_est,
        max_depth=depth,
        learning_rate=lr,
        objective="multi:softprob",
        num_class=n_classes,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        tree_method="hist",
        device="cpu",
        verbosity=0,
        eval_metric="mlogloss",
    )
    model.fit(X_tr, y_tr)
    return model


# ── LightGBM training ─────────────────────────────────────────────────────────

def train_lgb_regression(X_tr, y_tr, n_est, depth, n_leaves, lr):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(
        n_estimators=n_est, max_depth=depth, num_leaves=n_leaves,
        learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def train_lgb_binary(X_tr, y_tr, n_est, depth, n_leaves, lr):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=n_est, max_depth=depth, num_leaves=n_leaves,
        learning_rate=lr, objective="binary",
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


def train_lgb_multiclass(X_tr, y_tr, n_classes, n_est, depth, n_leaves, lr):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=n_est, max_depth=depth, num_leaves=n_leaves,
        learning_rate=lr, objective="multiclass", num_class=n_classes,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    return model


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_and_save(
    task: str,
    size: str,
    xgb_model,
    feat_names: list[str],
    X_test: np.ndarray,
    y_test: np.ndarray,
    force: bool,
) -> bool:
    prefix = MODELS_DIR / f"{task}_{size}"
    pb_path    = prefix.with_suffix(".omle")
    json_path  = Path(str(prefix) + "_xgb.json")
    onnx_path  = Path(str(prefix) + "_xgb.onnx")
    X_path     = Path(str(prefix) + "_X_test.npy")
    y_path     = Path(str(prefix) + "_y_test.npy")

    if pb_path.exists() and not force:
        print(f"  skip   {pb_path.name} (exists; use --force to overwrite)")
        return False

    import joblib

    # Save XGBoost model (native JSON for booster, joblib for sklearn wrapper)
    xgb_model.get_booster().save_model(str(json_path))
    joblib.dump(xgb_model, str(prefix) + "_xgb.pkl")

    # Save test split
    np.save(str(X_path), X_test)
    np.save(str(y_path), y_test)

    # Convert to ONNX
    _stamp(f"converting {task}/{size} → {onnx_path.name}")
    _export_xgb_onnx(xgb_model, len(feat_names), onnx_path)

    # Convert to OMLE
    _stamp(f"converting {task}/{size} → {pb_path.name}")
    omle_model = from_xgboost(
        xgb_model,
        feature_names=feat_names,
        X=X_test,
        n_verify=min(5, len(X_test)),
        n_warmup=10,
        verify_atol=1e-3,
        verify_rtol=1e-3,
    )
    omle.save(omle_model, str(pb_path))
    _stamp(f"saved {pb_path.name}  ({pb_path.stat().st_size / 1024:.0f} KB)")
    return True


def convert_and_save_lgb(
    task: str,
    size: str,
    lgb_model,
    feat_names: list[str],
    X_test: np.ndarray,
    force: bool,
) -> bool:
    from omle_convert.lightgbm import from_lightgbm
    import joblib

    prefix    = MODELS_DIR / f"{task}_{size}"
    pb_path   = Path(str(prefix) + "_lgb.omle")
    txt_path  = Path(str(prefix) + "_lgb.txt")
    onnx_path = Path(str(prefix) + "_lgb.onnx")

    if pb_path.exists() and not force:
        print(f"  skip   {pb_path.name} (exists; use --force to overwrite)")
        return False

    lgb_model.booster_.save_model(str(txt_path))
    joblib.dump(lgb_model, str(prefix) + "_lgb.pkl")

    # Convert to ONNX
    _stamp(f"converting {task}/{size} lgbm → {onnx_path.name}")
    _export_lgb_onnx(lgb_model, len(feat_names), onnx_path)

    _stamp(f"converting {task}/{size} lgbm → {pb_path.name}")
    omle_model = from_lightgbm(
        lgb_model,
        feature_names=feat_names,
        X=X_test,
        n_verify=min(5, len(X_test)),
        n_warmup=10,
        verify_atol=1e-3,
        verify_rtol=1e-3,
    )
    omle.save(omle_model, str(pb_path))
    _stamp(f"saved {pb_path.name}  ({pb_path.stat().st_size / 1024:.0f} KB)")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def _run_xgboost(task, size, is_mc, n_classes, feat_names, X_tr, X_te, y_tr, y_te, force):
    if task == "mnist":
        n_est, depth, lr = MNIST_CONFIGS[size]
    else:
        n_est, depth, lr = SIZE_CONFIGS[size]
        if is_mc:
            depth = MULTICLASS_DEPTHS[size]

    n_trees_total = n_est * (n_classes if is_mc else 1)
    print(f"\n  [xgboost] {size}: {n_est} trees × depth {depth}"
          + (f" × {n_classes} classes = {n_trees_total:,} total" if is_mc else ""))

    pb_path = MODELS_DIR / f"{task}_{size}.omle"
    if pb_path.exists() and not force:
        print(f"  skip   {pb_path.name} (exists; use --force to overwrite)")
        return

    _stamp("training xgboost")
    t0 = time.perf_counter()
    if task == "regression":
        model = train_regression(X_tr, y_tr, n_est, depth, lr)
    elif task == "binary":
        model = train_binary(X_tr, y_tr, n_est, depth, lr)
    else:
        model = train_multiclass(X_tr, y_tr, n_classes, n_est, depth, lr)
    _stamp(f"done in {time.perf_counter() - t0:.1f}s")

    if task == "regression":
        from sklearn.metrics import mean_absolute_error
        print(f"  MAE on test: {mean_absolute_error(y_te, model.predict(X_te)):.4f}")
    else:
        from sklearn.metrics import accuracy_score
        pred = model.predict(X_te)
        print(f"  Accuracy on test: {accuracy_score(y_te.astype(int), pred.astype(int)):.4f}")

    convert_and_save(task, size, model, feat_names, X_te, y_te, force)


def _run_lgbm(task, size, is_mc, n_classes, feat_names, X_tr, X_te, y_tr, y_te, force):
    if task == "mnist":
        n_est, depth, n_leaves, lr = LGB_MNIST_CONFIGS[size]
    else:
        n_est, depth, n_leaves, lr = LGB_SIZE_CONFIGS[size]
        if is_mc:
            depth    = LGB_MULTICLASS_DEPTHS[size]
            n_leaves = LGB_MULTICLASS_LEAVES[size]

    n_trees_total = n_est * (n_classes if is_mc else 1)
    print(f"\n  [lgbm]    {size}: {n_est} trees × depth {depth} × {n_leaves} leaves"
          + (f" × {n_classes} classes = {n_trees_total:,} total" if is_mc else ""))

    pb_path = Path(str(MODELS_DIR / f"{task}_{size}") + "_lgb.omle")
    if pb_path.exists() and not force:
        print(f"  skip   {pb_path.name} (exists; use --force to overwrite)")
        return

    _stamp("training lgbm")
    t0 = time.perf_counter()
    if task == "regression":
        model = train_lgb_regression(X_tr, y_tr, n_est, depth, n_leaves, lr)
    elif task == "binary":
        model = train_lgb_binary(X_tr, y_tr, n_est, depth, n_leaves, lr)
    else:
        model = train_lgb_multiclass(X_tr, y_tr, n_classes, n_est, depth, n_leaves, lr)
    _stamp(f"done in {time.perf_counter() - t0:.1f}s")

    if task == "regression":
        from sklearn.metrics import mean_absolute_error
        print(f"  MAE on test: {mean_absolute_error(y_te, model.predict(X_te)):.4f}")
    else:
        from sklearn.metrics import accuracy_score
        pred = model.predict(X_te)
        print(f"  Accuracy on test: {accuracy_score(y_te.astype(int), pred.astype(int)):.4f}")

    # Ensure test arrays exist (written by XGBoost step; write here if missing)
    X_path = Path(str(MODELS_DIR / f"{task}_{size}") + "_X_test.npy")
    y_path = Path(str(MODELS_DIR / f"{task}_{size}") + "_y_test.npy")
    if not X_path.exists():
        np.save(str(X_path), X_te)
        np.save(str(y_path), y_te)

    convert_and_save_lgb(task, size, model, feat_names, X_te, force)


def run(tasks: list[str], sizes: list[str], force: bool,
        frameworks: list[str] | None = None) -> None:
    if frameworks is None:
        frameworks = ["xgboost", "lgbm"]
    MODELS_DIR.mkdir(exist_ok=True)

    # Adult pipeline task is handled separately (sklearn Pipeline, not bare XGB/LGB)
    if "adult" in tasks:
        run_adult(sizes, force)

    tree_tasks = [t for t in tasks if t != "adult"]
    for task in tree_tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task.upper()}")
        print(f"{'='*60}")

        _stamp("loading dataset")
        X, y, feat_names, target_name = LOADERS[task]()
        is_mc = task in MULTICLASS_TASKS
        n_classes = int(np.max(y) + 1) if is_mc else 1
        print(f"  dataset: {X.shape[0]:,} samples × {X.shape[1]} features"
              + (f", {n_classes} classes" if is_mc else ""))

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_FRAC, random_state=SEED,
            stratify=(y if task != "regression" else None),
        )
        print(f"  split: {len(X_tr):,} train / {len(X_te):,} test")

        for size in sizes:
            if "xgboost" in frameworks:
                _run_xgboost(task, size, is_mc, n_classes, feat_names,
                             X_tr, X_te, y_tr, y_te, force)
            if "lgbm" in frameworks:
                _run_lgbm(task, size, is_mc, n_classes, feat_names,
                          X_tr, X_te, y_tr, y_te, force)

    print(f"\nAll done. Models written to: {MODELS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks",  nargs="+",
                        choices=["regression", "binary", "multiclass", "mnist", "adult"],
                        default=["regression", "binary", "multiclass", "mnist", "adult"])
    parser.add_argument("--sizes",  nargs="+",
                        choices=["small", "medium", "large"],
                        default=["small", "medium", "large"])
    parser.add_argument("--frameworks", nargs="+",
                        choices=["xgboost", "lgbm"],
                        default=["xgboost", "lgbm"],
                        help="frameworks to train (default: both)")
    parser.add_argument("--force",  action="store_true",
                        help="re-train and overwrite existing model files")
    args = parser.parse_args()
    run(args.tasks, args.sizes, args.force, args.frameworks)


if __name__ == "__main__":
    main()
