"""Compare XGBoost / LightGBM native inference vs OMLE runtime.

For each (framework, task, model-size) combination this script:
  1. Loads the native model and the corresponding OMLE protobuf model.
  2. Verifies predictions are numerically identical (tolerance 1e-4).
  3. Benchmarks latency at multiple batch sizes using min-over-N-reps timing.

Batch sizes tested: 1, 10, 100, 1000, 10000

Native paths benchmarked (per framework)
-----------------------------------------
  xgb-sklearn  : XGBRegressor/Classifier.predict(X)   — sklearn interface
  xgb-booster  : booster.inplace_predict(X)           — lowest-overhead native path
  lgb-sklearn  : LGBMRegressor/Classifier.predict(X)  — sklearn interface
  lgb-booster  : booster.predict(X)                   — lowest-overhead native path

Pipeline task (adult) — mixed numeric + categorical DataFrame
--------------------------------------------------------------
  pipe-sklearn : Pipeline.predict_proba(df)  — full sklearn pipeline
  ort-session  : ORT inference with per-column dict input (skl2onnx model)
  omle-session : session.predict_proba(df)   — accepts mixed-type DataFrame directly

OMLE paths benchmarked (same for both frameworks)
------------------------------------------------------
  omle-model   : model.predict(X)   — stateless, new session per call
  omle-session : session.predict(X) — persistent session, no allocation per call

Usage
-----
  python run_benchmark.py [--frameworks xgboost lgbm]
                          [--tasks regression binary multiclass mnist adult]
                          [--sizes small medium large]
                          [--batch-sizes 1 10 100 1000 10000]
                          [--reps 200]
                          [--warmup 20]
                          [--threads 1]

Run train_models.py first to generate model files.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Callable

# LightGBM 4.x stores auto-generated feature names; suppress the sklearn warning
# that fires when numpy arrays are passed to predict on such models.
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import xgboost as xgb

try:
    import onnxruntime as _ort
    _ORT_AVAILABLE = True
except ImportError:
    _ort = None
    _ORT_AVAILABLE = False
    print("  [warning] onnxruntime not found — ORT path skipped (pip install onnxruntime)")

# ── Memory helper ─────────────────────────────────────────────────────────────

try:
    import psutil as _psutil
    _PROC = _psutil.Process(os.getpid())
    def rss_mb() -> float:
        return _PROC.memory_info().rss / 1048576.0
    _MEM_AVAILABLE = True
except ImportError:
    def rss_mb() -> float:
        return 0.0
    _MEM_AVAILABLE = False
    print("  [warning] psutil not found — memory metrics disabled (pip install psutil)")

# ── Path bootstrap ────────────────────────────────────────────────────────────
_BENCH   = Path(__file__).resolve().parent
_RESULTS = _BENCH / "results"
_ROOT    = _BENCH.parent
_SIBLING = _ROOT.parent

sys.path.insert(0, str(_ROOT / "python"))
sys.path.insert(0, str(_SIBLING / "omle" / "src"))
sys.path.insert(0, str(_SIBLING / "omle-convert" / "src"))

import omleruntime as omr

MODELS_DIR = _BENCH / "models"

# ── Timing helpers ────────────────────────────────────────────────────────────

def bench(fn: Callable, warmup: int, reps: int) -> tuple[float, float]:
    """Return (min_us, median_us) over `reps` timed calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[0], times[len(times) // 2]


def throughput(n: int, us: float) -> float:
    if us <= 0:
        return float("inf")
    return n / (us * 1e-6)


# ── Data helpers ──────────────────────────────────────────────────────────────

def make_batch(X_test: np.ndarray, batch_size: int) -> np.ndarray:
    """Return a contiguous float32 array of exactly `batch_size` rows."""
    n = len(X_test)
    if batch_size <= n:
        return np.ascontiguousarray(X_test[:batch_size], dtype=np.float32)
    reps = (batch_size + n - 1) // n
    return np.ascontiguousarray(
        np.tile(X_test, (reps, 1))[:batch_size], dtype=np.float32
    )


# ── Correctness check ─────────────────────────────────────────────────────────

def check_correctness(
    task: str,
    native_sk_model,
    omle_model: omr.Model,
    X_test: np.ndarray,
) -> tuple[bool, float, float]:
    """Compare native sklearn model vs OMLE predictions on the full test set.

    Returns (passed, max_abs_err, max_rel_err).
    """
    X = np.ascontiguousarray(X_test, dtype=np.float32)

    if task == "regression":
        ref_pred  = native_sk_model.predict(X).astype(np.float32).ravel()
        omle_pred = omle_model.predict(X).astype(np.float32).ravel()
    elif task == "binary":
        ref_pred  = native_sk_model.predict_proba(X)[:, 1].astype(np.float32).ravel()
        omle_raw  = omle_model.predict_proba(X).astype(np.float32)
        omle_pred = (omle_raw[:, 1] if omle_raw.shape[1] == 2 else omle_raw.ravel())
    else:  # multiclass / mnist
        ref_pred  = native_sk_model.predict_proba(X).astype(np.float32).ravel()
        omle_pred = omle_model.predict_proba(X).astype(np.float32).ravel()

    abs_err = np.abs(ref_pred - omle_pred)
    max_abs = float(abs_err.max())
    denom   = np.abs(ref_pred)
    rel_err = abs_err / np.where(denom > 1e-6, denom, 1e-6)
    max_rel = float(rel_err.max())

    return max_abs <= 1e-4, max_abs, max_rel


# ── Benchmark runners ─────────────────────────────────────────────────────────

def bench_task(
    task: str,
    size: str,
    batch_sizes: list[int],
    reps: int,
    warmup: int,
    n_threads: int,
    framework: str = "xgboost",
) -> dict:
    import joblib

    prefix = MODELS_DIR / f"{task}_{size}"
    X_path = Path(str(prefix) + "_X_test.npy")

    rss_baseline = rss_mb()

    if framework == "xgboost":
        pb_path    = prefix.with_suffix(".omle")
        pkl_path   = Path(str(prefix) + "_xgb.pkl")
        json_path  = Path(str(prefix) + "_xgb.json")
        onnx_path  = Path(str(prefix) + "_xgb.onnx")
        sk_label   = "xgb-sklearn"
        bp_label   = "xgb-booster"
        native_file = json_path

        if not pb_path.exists():
            print(f"  MISSING: {pb_path} — run train_models.py first")
            return {}

        native_sk = joblib.load(str(pkl_path))
        _booster  = xgb.Booster()
        _booster.load_model(str(json_path))

        def _bp_fn(X):
            return _booster.inplace_predict(X)

    else:  # lgbm
        import lightgbm as lgb
        pb_path    = Path(str(prefix) + "_lgb.omle")
        pkl_path   = Path(str(prefix) + "_lgb.pkl")
        txt_path   = Path(str(prefix) + "_lgb.txt")
        onnx_path  = Path(str(prefix) + "_lgb.onnx")
        sk_label   = "lgb-sklearn"
        bp_label   = "lgb-booster"
        native_file = txt_path

        if not pb_path.exists():
            print(f"  MISSING: {pb_path} — run train_models.py --frameworks lgbm first")
            return {}

        native_sk = joblib.load(str(pkl_path))
        _booster  = lgb.Booster(model_file=str(txt_path))

        def _bp_fn(X):
            return _booster.predict(X)

    rss_after_native = rss_mb()

    # ── ORT session ───────────────────────────────────────────────────────────
    ort_sess = None
    ort_input_name = "float_input"
    if _ORT_AVAILABLE and onnx_path.exists():
        _ort_opts = _ort.SessionOptions()
        _ort_opts.intra_op_num_threads = 1
        _ort_opts.inter_op_num_threads = 1
        ort_sess = _ort.InferenceSession(
            str(onnx_path),
            sess_options=_ort_opts,
            providers=["CPUExecutionProvider"],
        )
        ort_input_name = ort_sess.get_inputs()[0].name

    rss_after_ort = rss_mb()

    omle_model   = omr.Model.load(str(pb_path), n_threads=n_threads)
    omle_session = omle_model.create_session()

    rss_after_omle = rss_mb()

    # ── File sizes ────────────────────────────────────────────────────────────
    file_sizes = {
        "native_kb": native_file.stat().st_size / 1024 if native_file.exists() else 0,
        "onnx_kb":   onnx_path.stat().st_size   / 1024 if onnx_path.exists()  else 0,
        "omle_kb":   pb_path.stat().st_size      / 1024 if pb_path.exists()    else 0,
    }

    X_test = np.load(str(X_path))
    n_feat = X_test.shape[1]
    n_test = X_test.shape[0]

    print(f"\n  Correctness check ({n_test} test rows):")
    ok, max_abs, max_rel = check_correctness(task, native_sk, omle_model, X_test)
    status = "PASS" if ok else "FAIL"
    print(f"    max_abs_err = {max_abs:.2e}  max_rel_err = {max_rel:.2e}  [{status}]")
    if not ok:
        print(f"    WARNING: max_abs_err {max_abs:.2e} > 1e-4 tolerance")

    rows: list[dict] = []
    for bs in batch_sizes:
        X_b = make_batch(X_test, bs)

        def fn_sk():
            return native_sk.predict(X_b) if task == "regression" else native_sk.predict_proba(X_b)

        def fn_bp():
            return _bp_fn(X_b)

        def fn_ort():
            return ort_sess.run(None, {ort_input_name: X_b})

        def fn_st():
            return omle_model.predict(X_b)

        def fn_se():
            return omle_session.predict(X_b)

        sk_mn, sk_me = bench(fn_sk, warmup, reps)
        bp_mn, bp_me = bench(fn_bp, warmup, reps)
        ort_mn, ort_me = bench(fn_ort, warmup, reps) if ort_sess else (0.0, 0.0)
        st_mn, st_me = bench(fn_st, warmup, reps)
        se_mn, se_me = bench(fn_se, warmup, reps)

        rss_before_inf = rss_mb()
        for _ in range(3):
            fn_se()
        rss_after_inf = rss_mb()

        rows.append({
            "batch_size":     bs,
            "nat_sk_min_us":  sk_mn,  "nat_sk_med_us":  sk_me,
            "nat_bp_min_us":  bp_mn,  "nat_bp_med_us":  bp_me,
            "ort_min_us":     ort_mn, "ort_med_us":     ort_me,
            "omle_st_min_us": st_mn,  "omle_st_med_us": st_me,
            "omle_se_min_us": se_mn,  "omle_se_med_us": se_me,
            "rss_before_inf_mb": rss_before_inf,
            "rss_after_inf_mb":  rss_after_inf,
        })

    rss_after_bench = rss_mb()

    return {
        "correctness": (ok, max_abs, max_rel),
        "rows": rows,
        "n_feat": n_feat,
        "sk_label": sk_label,
        "bp_label": bp_label,
        "file_sizes": file_sizes,
        "mem": {
            "baseline_mb":     rss_baseline,
            "native_delta_mb": rss_after_native - rss_baseline,
            "ort_delta_mb":    rss_after_ort    - rss_after_native,
            "omle_delta_mb":   rss_after_omle   - rss_after_ort,
            "after_bench_mb":  rss_after_bench,
        },
    }


# ── Report ────────────────────────────────────────────────────────────────────

def fmt_us(us: float) -> str:
    if us >= 1e6:
        return f"{us/1e6:8.2f}s "
    if us >= 1000:
        return f"{us/1000:8.2f}ms"
    return f"{us:8.1f}µs"


def fmt_tput(n: int, us: float) -> str:
    t = throughput(n, us)
    if t >= 1e6:
        return f"{t/1e6:6.2f}M/s"
    if t >= 1e3:
        return f"{t/1e3:6.1f}k/s"
    return f"{t:6.0f}/s "


_XGB_TASK_CONFIGS = {
    "regression":  {"small": (100, 4),  "medium": (500, 6),  "large": (1000, 8)},
    "binary":      {"small": (100, 4),  "medium": (500, 6),  "large": (1000, 8)},
    "multiclass":  {"small": (100, 3),  "medium": (500, 4),  "large": (1000, 5)},
    "mnist":       {"small": (50,  3),  "medium": (100, 4),  "large": (200,  5)},
}

_ADULT_PIPE_CONFIGS = {
    "small":  (100, 4),
    "medium": (500, 6),
    "large":  (1000, 8),
}

_LGB_TASK_CONFIGS = {
    "regression":  {"small": (100, 4, 16),   "medium": (500, 6, 64),   "large": (1000, 8, 256)},
    "binary":      {"small": (100, 4, 16),   "medium": (500, 6, 64),   "large": (1000, 8, 256)},
    "multiclass":  {"small": (100, 3, 8),    "medium": (500, 4, 16),   "large": (1000, 5, 32)},
    "mnist":       {"small": (50,  3, 8),    "medium": (100, 4, 16),   "large": (200,  5, 32)},
}


def _make_df_batch(X_df, batch_size: int):
    """Return a DataFrame of exactly batch_size rows (tile/truncate as needed)."""
    n = len(X_df)
    if batch_size <= n:
        return X_df.iloc[:batch_size].copy()
    reps = (batch_size + n - 1) // n
    import pandas as pd
    return pd.concat([X_df] * reps, ignore_index=True).iloc[:batch_size]


def _make_ort_input_adult(df_b, num_cols: list, cat_cols: list, ort_input_names: set) -> dict:
    """Build a per-column ORT input dict for the mixed-type adult pipeline.

    skl2onnx sanitizes column names (hyphens → underscores), so we remap our
    hyphenated names to match the ONNX model's input names.
    """
    def _sanitize(name: str) -> str:
        return name.replace("-", "_")

    ort_in = {}
    for c in num_cols:
        key = _sanitize(c) if _sanitize(c) in ort_input_names else c
        ort_in[key] = df_b[c].values.astype(np.float32).reshape(-1, 1)
    for c in cat_cols:
        key = _sanitize(c) if _sanitize(c) in ort_input_names else c
        ort_in[key] = df_b[c].astype(str).values.reshape(-1, 1)
    return ort_in


def _encode_adult_for_omle(
    df_b,
    all_col_order: list,
    cat_cols: list,
    ohe_categories: list,
) -> np.ndarray:
    """Convert mixed-type DataFrame to flat float32 array in feature_names_in_ order.

    OMLE's TakeSlots uses column indices based on feature_names_in_, so we must
    build the flat array in that same order.  Categorical columns are replaced by
    their ordinal index into the OHE's sorted category list so that OMLE's OHE
    node (which accepts numeric ordinal-index input) produces identical output.
    """
    n = len(df_b)
    cat_to_ord = {
        col: {str(c): float(i) for i, c in enumerate(cats)}
        for col, cats in zip(cat_cols, ohe_categories)
    }
    out = np.zeros((n, len(all_col_order)), dtype=np.float32)
    for j, col in enumerate(all_col_order):
        if col in cat_to_ord:
            vals = df_b[col].astype(str).values
            for r, v in enumerate(vals):
                out[r, j] = cat_to_ord[col].get(v, -1.0)
        else:
            out[:, j] = df_b[col].values.astype(np.float32)
    return np.ascontiguousarray(out)


def bench_task_adult(
    size: str,
    batch_sizes: list[int],
    reps: int,
    warmup: int,
    n_threads: int,
) -> dict:
    """Benchmark the Adult Census mixed-type sklearn pipeline.

    Three paths are compared:
      pipe-sklearn : full Pipeline.predict_proba(DataFrame) — includes preprocessing
      ort-session  : ORT inference with per-column string/float dict (skl2onnx model)
      omle-session : OMLE inference with mixed-type DataFrame directly (per-column named
                     inputs; string columns handled natively by StringHStack + OHE nodes)
    """
    import joblib
    import pandas as pd

    prefix    = MODELS_DIR / f"adult_{size}"
    pb_path   = Path(str(prefix) + "_pipe.omle")
    pkl_path  = Path(str(prefix) + "_pipe.pkl")
    onnx_path = Path(str(prefix) + "_pipe.onnx")
    X_path    = Path(str(prefix) + "_X_test.pkl")
    y_path    = Path(str(prefix) + "_y_test.npy")

    if not pb_path.exists():
        print(f"  MISSING: {pb_path} — run train_models.py --tasks adult first")
        return {}

    rss_baseline = rss_mb()

    native_pipe = joblib.load(str(pkl_path))
    ct = native_pipe.named_steps["prep"]
    num_cols       = list(ct.transformers_[0][2])
    cat_cols       = list(ct.transformers_[1][2])
    ohe_categories = ct.named_transformers_["cat"].categories_
    all_col_order  = list(ct.feature_names_in_)   # feature_names_in_ order for OMLE flat X

    rss_after_native = rss_mb()

    ort_sess = None
    ort_input_names: set = set()
    if _ORT_AVAILABLE and onnx_path.exists():
        _ort_opts = _ort.SessionOptions()
        _ort_opts.intra_op_num_threads = 1
        _ort_opts.inter_op_num_threads = 1
        ort_sess = _ort.InferenceSession(
            str(onnx_path),
            sess_options=_ort_opts,
            providers=["CPUExecutionProvider"],
        )
        ort_input_names = {inp.name for inp in ort_sess.get_inputs()}

    rss_after_ort = rss_mb()

    omle_model   = omr.Model.load(str(pb_path), n_threads=n_threads)
    omle_session = omle_model.create_session()

    rss_after_omle = rss_mb()

    file_sizes = {
        "native_kb": pkl_path.stat().st_size  / 1024 if pkl_path.exists()  else 0,
        "onnx_kb":   onnx_path.stat().st_size / 1024 if onnx_path.exists() else 0,
        "omle_kb":   pb_path.stat().st_size   / 1024 if pb_path.exists()   else 0,
    }

    X_test = pd.read_pickle(str(X_path))
    y_test = np.load(str(y_path))
    n_test = len(X_test)

    # Correctness check: compare sklearn vs OMLE — DataFrame passed directly
    print(f"\n  Correctness check ({n_test} test rows):")
    ref_prob = native_pipe.predict_proba(X_test)[:, 1].astype(np.float32)
    # Use omle_model (not session) — model.predict_proba filters to y_prob only: shape (n, 1)
    omle_raw = omle_model.predict_proba(X_test)
    omle_prob = np.asarray(omle_raw, dtype=np.float32)[:, 0]
    abs_err = np.abs(ref_prob - omle_prob)
    max_abs = float(abs_err.max())
    denom   = np.abs(ref_prob)
    max_rel = float((abs_err / np.where(denom > 1e-6, denom, 1e-6)).max())
    # Tolerance 5e-2: float32 StandardScaler vs float64 sklearn can shift borderline splits
    ok = max_abs <= 5e-2
    print(f"    max_abs_err = {max_abs:.2e}  max_rel_err = {max_rel:.2e}  [{'PASS' if ok else 'FAIL'}]")
    if not ok:
        print(f"    WARNING: max_abs_err {max_abs:.2e} > 5e-2 tolerance")

    rows: list[dict] = []
    for bs in batch_sizes:
        df_b   = _make_df_batch(X_test, bs)
        ort_in = _make_ort_input_adult(df_b, num_cols, cat_cols, ort_input_names) if ort_sess else None

        def fn_sk():
            return native_pipe.predict_proba(df_b)

        def fn_ort():
            return ort_sess.run(None, ort_in)

        def fn_st():
            return omle_model.predict_proba(df_b)

        def fn_se():
            return omle_session.predict_proba(df_b)

        sk_mn,  sk_me  = bench(fn_sk, warmup, reps)
        ort_mn, ort_me = bench(fn_ort, warmup, reps) if ort_sess else (0.0, 0.0)
        st_mn,  st_me  = bench(fn_st, warmup, reps)
        se_mn,  se_me  = bench(fn_se, warmup, reps)

        rss_before_inf = rss_mb()
        for _ in range(3):
            fn_se()
        rss_after_inf = rss_mb()

        rows.append({
            "batch_size":        bs,
            "nat_sk_min_us":     sk_mn,  "nat_sk_med_us":  sk_me,
            "nat_bp_min_us":     sk_mn,  "nat_bp_med_us":  sk_me,  # no booster path; duplicate
            "ort_min_us":        ort_mn, "ort_med_us":     ort_me,
            "omle_st_min_us":    st_mn,  "omle_st_med_us": st_me,
            "omle_se_min_us":    se_mn,  "omle_se_med_us": se_me,
            "rss_before_inf_mb": rss_before_inf,
            "rss_after_inf_mb":  rss_after_inf,
        })

    rss_after_bench = rss_mb()
    return {
        "correctness": (ok, max_abs, max_rel),
        "rows":        rows,
        "n_feat":      len(num_cols) + len(cat_cols),
        "sk_label":    "pipe-sklearn",
        "bp_label":    "(same)",
        "file_sizes":  file_sizes,
        "mem": {
            "baseline_mb":     rss_baseline,
            "native_delta_mb": rss_after_native - rss_baseline,
            "ort_delta_mb":    rss_after_ort    - rss_after_native,
            "omle_delta_mb":   rss_after_omle   - rss_after_ort,
            "after_bench_mb":  rss_after_bench,
        },
    }


def print_table(task: str, size: str, result: dict, framework: str) -> None:
    if not result:
        return
    rows = result["rows"]
    ok, max_abs, _ = result["correctness"]
    sk_label   = result["sk_label"]
    bp_label   = result["bp_label"]
    mem        = result.get("mem", {})
    fsz        = result.get("file_sizes", {})

    if framework == "pipeline":
        n_est, depth = _ADULT_PIPE_CONFIGS[size]
        model_desc = f"RandomForest {n_est} trees × depth {depth}, ColumnTransformer(StandardScaler + OHE)"
    elif framework == "xgboost":
        n_est, depth = _XGB_TASK_CONFIGS.get(task, _XGB_TASK_CONFIGS["regression"])[size]
        model_desc = f"{n_est} trees, depth {depth}"
    else:
        n_est, depth, n_leaves = _LGB_TASK_CONFIGS.get(task, _LGB_TASK_CONFIGS["regression"])[size]
        model_desc = f"{n_est} trees, depth {depth}, {n_leaves} leaves"

    W = 134
    print(f"\n{'─'*W}")
    print(f"  [{framework}]  Task: {task.upper():<12}  Model: {size} ({model_desc})")
    print(f"  Correctness: max_abs_err={max_abs:.2e}  [{'PASS' if ok else 'FAIL'}]")
    print(f"{'─'*W}")

    # ── File sizes ────────────────────────────────────────────────────────────
    if fsz:
        if framework == "pipeline":
            native_label = "pipe .pkl"
        elif framework == "xgboost":
            native_label = "xgb .json"
        else:
            native_label = "lgb .txt"
        print(f"  File sizes:")
        print(f"    {native_label:12s}: {fsz['native_kb']:8.1f} KB")
        print(f"    {'ONNX':12s}: {fsz['onnx_kb']:8.1f} KB"
              + (f"  ({fsz['onnx_kb']/fsz['native_kb']:.2f}× native)" if fsz['native_kb'] else ""))
        print(f"    {'OMLE .omle':12s}: {fsz['omle_kb']:8.1f} KB"
              + (f"  ({fsz['omle_kb']/fsz['native_kb']:.2f}× native)" if fsz['native_kb'] else ""))
        print()

    # ── Memory (RSS) ──────────────────────────────────────────────────────────
    if mem and _MEM_AVAILABLE:
        rss0 = mem['baseline_mb']
        rss1 = rss0 + mem['native_delta_mb']
        rss2 = rss1 + mem['ort_delta_mb']
        rss3 = rss2 + mem['omle_delta_mb']
        print(f"  Memory (RSS load deltas):")
        print(f"    {'baseline':22s}: {rss0:7.1f} MB")
        print(f"    {'+ native model':22s}: {mem['native_delta_mb']:+7.1f} MB  →  {rss1:.1f} MB total")
        print(f"    {'+ ORT session':22s}: {mem['ort_delta_mb']:+7.1f} MB  →  {rss2:.1f} MB total")
        print(f"    {'+ OMLE session':22s}: {mem['omle_delta_mb']:+7.1f} MB  →  {rss3:.1f} MB total")
        print(f"    {'after benchmark':22s}: {mem['after_bench_mb']:7.1f} MB")
        if rows:
            r_last = rows[-1]
            inf_delta = r_last["rss_after_inf_mb"] - r_last["rss_before_inf_mb"]
            print(f"    {'omle inference delta':22s}: {inf_delta:+7.1f} MB  (batch={r_last['batch_size']:,})")
        print()

    # ── Latency table ─────────────────────────────────────────────────────────
    print(f"  {'Batch':>7}  │ {sk_label:^23} │ {bp_label:^23} │ {'ort-session':^23} │ {'omle-model':^23} │ {'omle-session':^23} │ {'vs best-native':>14}")
    print(f"  {'size':>7}  │ {'min':>10}  {'tput':>10} │ {'min':>10}  {'tput':>10} │ {'min':>10}  {'tput':>10} │ {'min':>10}  {'tput':>10} │ {'min':>10}  {'tput':>10} │ {'ORT':>6}  {'OMLE':>6}")
    print(f"  {'─'*7}──┼{'─'*23}─┼{'─'*23}─┼{'─'*23}─┼{'─'*23}─┼{'─'*23}─┼{'─'*14}")

    for r in rows:
        bs = r["batch_size"]
        best_native  = min(r["nat_sk_min_us"], r["nat_bp_min_us"])
        ort_speedup  = best_native / r["ort_min_us"]  if r["ort_min_us"]     > 0 else float("nan")
        omle_speedup = best_native / r["omle_se_min_us"] if r["omle_se_min_us"] > 0 else float("nan")

        ort_s  = f"{ort_speedup:+.2f}x"  if r["ort_min_us"]     > 0 else "  n/a  "
        omle_s = f"{omle_speedup:+.2f}x" if r["omle_se_min_us"] > 0 else "  n/a  "

        print(
            f"  {bs:>7,}  │"
            f" {fmt_us(r['nat_sk_min_us'])}  {fmt_tput(bs, r['nat_sk_min_us'])} │"
            f" {fmt_us(r['nat_bp_min_us'])}  {fmt_tput(bs, r['nat_bp_min_us'])} │"
            f" {fmt_us(r['ort_min_us'])}  {fmt_tput(bs, r['ort_min_us'])} │"
            f" {fmt_us(r['omle_st_min_us'])}  {fmt_tput(bs, r['omle_st_min_us'])} │"
            f" {fmt_us(r['omle_se_min_us'])}  {fmt_tput(bs, r['omle_se_min_us'])} │"
            f"  {ort_s}  {omle_s}"
        )


def save_csv(path: Path, all_results: dict) -> None:
    """Write a flat CSV with all results for further analysis."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "framework", "task", "size", "batch_size",
            "nat_sk_min_us", "nat_sk_med_us",
            "nat_bp_min_us", "nat_bp_med_us",
            "ort_min_us", "ort_med_us",
            "omle_model_min_us", "omle_model_med_us",
            "omle_session_min_us", "omle_session_med_us",
            "max_abs_err", "correctness_pass",
            "native_file_kb", "onnx_file_kb", "omle_file_kb",
            "rss_baseline_mb", "native_load_delta_mb",
            "ort_load_delta_mb", "omle_load_delta_mb",
            "rss_after_bench_mb", "omle_inf_delta_mb",
        ])
        for (framework, task, size), res in all_results.items():
            if not res:
                continue
            ok, max_abs, _ = res["correctness"]
            mem = res.get("mem", {})
            fsz = res.get("file_sizes", {})
            for r in res["rows"]:
                inf_delta = r["rss_after_inf_mb"] - r["rss_before_inf_mb"]
                w.writerow([
                    framework, task, size, r["batch_size"],
                    f"{r['nat_sk_min_us']:.3f}",  f"{r['nat_sk_med_us']:.3f}",
                    f"{r['nat_bp_min_us']:.3f}",  f"{r['nat_bp_med_us']:.3f}",
                    f"{r['ort_min_us']:.3f}",     f"{r['ort_med_us']:.3f}",
                    f"{r['omle_st_min_us']:.3f}",  f"{r['omle_st_med_us']:.3f}",
                    f"{r['omle_se_min_us']:.3f}",  f"{r['omle_se_med_us']:.3f}",
                    f"{max_abs:.6e}", "1" if ok else "0",
                    f"{fsz.get('native_kb', 0):.1f}",
                    f"{fsz.get('onnx_kb', 0):.1f}",
                    f"{fsz.get('omle_kb', 0):.1f}",
                    f"{mem.get('baseline_mb', 0):.1f}",
                    f"{mem.get('native_delta_mb', 0):.1f}",
                    f"{mem.get('ort_delta_mb', 0):.1f}",
                    f"{mem.get('omle_delta_mb', 0):.1f}",
                    f"{mem.get('after_bench_mb', 0):.1f}",
                    f"{inf_delta:.1f}",
                ])
    print(f"\n  CSV saved to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--frameworks",  nargs="+",
                        choices=["xgboost", "lgbm"],
                        default=["xgboost", "lgbm"])
    parser.add_argument("--tasks",       nargs="+",
                        choices=["regression", "binary", "multiclass", "mnist", "adult"],
                        default=["regression", "binary", "multiclass", "mnist", "adult"])
    parser.add_argument("--sizes",       nargs="+",
                        choices=["small", "medium", "large"],
                        default=["small", "medium", "large"])
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 10, 100, 1000, 10000])
    parser.add_argument("--reps",    type=int, default=200,
                        help="timing repetitions per measurement (default: 200)")
    parser.add_argument("--warmup",  type=int, default=20,
                        help="warmup passes before timing (default: 20)")
    parser.add_argument("--threads", type=int, default=1,
                        help="worker threads for OMLE runtime (default: 1)")
    parser.add_argument("--csv",     type=Path, default=None,
                        help="write results CSV to this path")
    args = parser.parse_args()

    print("OMLE vs XGBoost/LightGBM latency benchmark")
    print(f"frameworks={args.frameworks}  reps={args.reps}  warmup={args.warmup}  threads={args.threads}")
    print(f"batch sizes: {args.batch_sizes}")

    all_results = {}

    # Adult pipeline task: one entry per size, framework="pipeline"
    if "adult" in args.tasks:
        for size in args.sizes:
            label = f"pipeline/adult/{size}"
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            result = bench_task_adult(
                size,
                batch_sizes=args.batch_sizes,
                reps=args.reps,
                warmup=args.warmup,
                n_threads=args.threads,
            )
            all_results[("pipeline", "adult", size)] = result
            print_table("adult", size, result, "pipeline")

    # XGBoost/LightGBM tree tasks
    tree_tasks = [t for t in args.tasks if t != "adult"]
    for framework in args.frameworks:
        for task in tree_tasks:
            for size in args.sizes:
                label = f"{framework}/{task}/{size}"
                print(f"\n{'='*60}")
                print(f"  {label}")
                print(f"{'='*60}")
                result = bench_task(
                    task, size,
                    batch_sizes=args.batch_sizes,
                    reps=args.reps,
                    warmup=args.warmup,
                    n_threads=args.threads,
                    framework=framework,
                )
                all_results[(framework, task, size)] = result
                print_table(task, size, result, framework)

    if args.csv:
        save_csv(args.csv, all_results)
    else:
        _RESULTS.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        csv_path = _RESULTS / f"results_{ts}.csv"
        save_csv(csv_path, all_results)

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    main()
