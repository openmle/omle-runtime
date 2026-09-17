# OMLE vs XGBoost Latency Benchmark

End-to-end benchmark: train XGBoost, convert to OMLE protobuf, compare predictions and latency.

## Quick start

```bash
# 1. Install Python deps (if not already present)
python3 -m pip install xgboost lightgbm scikit-learn joblib

# 2. Train and convert all models (both XGBoost and LightGBM)  ~2 minutes
python3 train_models.py

# Or train only one framework:
python3 train_models.py --frameworks xgboost
python3 train_models.py --frameworks lgbm

# 3. Run the full benchmark (both frameworks)  ~10 minutes
python3 run_benchmark.py

# Or a quick subset:
python3 run_benchmark.py --frameworks xgboost lgbm \
                         --tasks regression binary --sizes small medium \
                         --batch-sizes 1 10 100 1000 --reps 100
```

## Models

| Task | Dataset | Features | Samples | Classes |
|------|---------|----------|---------|---------|
| Regression | California Housing (sklearn) | 8 | 20 640 | — |
| Binary | Breast Cancer Wisconsin (sklearn) | 30 | 569 | 2 |
| Multiclass | Digits / MNIST-tiny (sklearn) | 64 | 1 797 | 10 |
| MNIST | MNIST-784 (OpenML) | **784** | **70 000** | 10 |

Three sizes per task (80/20 train/test split, seed=42):

| Size | Trees | Max depth | MNIST trees | Notes |
|------|-------|-----------|-------------|-------|
| small | 100 | 4 (mc: 3) | 50, depth 3 | fast dev loop |
| medium | 500 | 6 (mc: 4) | 100, depth 4 | production-like |
| large | 1 000 | 8 (mc: 5) | 200, depth 5 | upper bound |

MNIST uses fewer estimators because the 56k-row training set is 40× larger than Digits,
making each boosting round proportionally more expensive.

## Correctness

All predictions verified against the native framework before benchmarking.
Tolerance: absolute error ≤ 1e-4.

### XGBoost

| Task | Max absolute error | Note |
|------|--------------------|------|
| Regression | 0.00e+00 | Bit-exact |
| Binary | 0.00e+00 | Bit-exact |
| Multiclass (Digits) | ~2.38e-07 | Float32 softmax rounding |
| MNIST | ~3.58e-07 | Float32 softmax rounding |

### LightGBM

| Task | Max absolute error | Note |
|------|--------------------|------|
| Regression | 0.00e+00 | Bit-exact |
| Binary | 0.00e+00 | Bit-exact |
| Multiclass (Digits) | 0.00e+00 | Bit-exact — float64 end-to-end |
| MNIST | 0.00e+00 | Bit-exact — float64 end-to-end |

LightGBM models are stored and executed in float64 throughout (thresholds, leaf values, post-transform), so predictions are bit-identical to the native booster even for softmax multiclass.

## Benchmark paths

| Label | What it times |
|-------|--------------|
| `xgb-sklearn` | `XGBRegressor/Classifier.predict(X_np)` — standard sklearn interface |
| `xgb-booster` | `booster.inplace_predict(X_np)` — lowest-level XGBoost native path |
| `lgb-sklearn` | `LGBMRegressor/Classifier.predict(X_np)` — standard sklearn interface |
| `lgb-booster` | `booster.predict(X_np)` — lowest-level LightGBM native path |
| `omle-model` | `model.predict(X)` — stateless, allocates a session per call |
| `omle-session` | `session.predict(X)` — persistent session, reused across calls |

## Key findings (Apple M-series, single thread)

### vs XGBoost

**Single-output (regression/binary), small model — latency-sensitive:**

- Batch=1: OMLE session **6–8x faster** than XGBoost (no DMatrix overhead)
- Batch=10: **5–7x faster**
- Batch=100: **2–3x faster**
- Batch≥1000: XGBoost gains ground (SIMD-heavy AVX prediction kernels)

**Multiclass/MNIST vs XGBoost:**

| Batch | Multiclass/small | MNIST/small |
|-------|-----------------|-------------|
| 1 | **+3.9x** | **+1.2x** |
| 10 | **+2.5x** | ~1.0x (tied) |
| 100 | ~1.0x (tied) | −0.8x (XGB faster) |

### vs LightGBM

LightGBM's native booster is significantly lighter than XGBoost's at small batches
(no DMatrix — direct numpy input). This shifts the crossover point earlier.

**Single-output (regression/binary), small model:**

| Batch | Regression/small | Binary/small |
|-------|-----------------|--------------|
| 1 | **+3.9x** | **+2.7x** |
| 10 | **+2.6x** | **+1.8x** |
| 100 | ~1.0x (tied) | ~1.0x (tied) |
| 1000 | −0.3x (LGB faster) | −0.6x (LGB faster) |

**Multiclass/MNIST vs LightGBM:**
OMLE is slower on multiclass and MNIST for all batch sizes. LightGBM's prediction
kernel processes all class outputs for a sample in a single vectorized pass, while
OMLE traverses each tree independently with a scalar scatter-add to the output.

| Batch | Multiclass/small | MNIST/small |
|-------|-----------------|-------------|
| 1 | −0.84x (LGB faster) | −0.44x (LGB faster) |
| 10 | −0.69x | −0.37x |
| 100 | −0.35x | −0.25x |

**Why LightGBM booster beats XGBoost booster at small batches:**
LightGBM's `booster.predict()` accepts raw numpy arrays directly — regression/small
batch=1: 35 µs vs XGBoost's 82 µs. XGBoost internally converts the input via a
DMatrix path even for `inplace_predict`, adding ~40–50 µs of fixed overhead.

**Why OMLE wins on single-output tasks:**
The OMLE session at 9 µs (regression/small batch=1) beats both frameworks by
avoiding Python-level validation entirely. The C API path receives a raw float32
pointer with no conversion overhead.

**Why LightGBM wins on multiclass:**
LightGBM's prediction kernel is structured around computing all class outputs for
each tree in one pass (interleaved leaf-value access). OMLE's current blocked
multiclass path calls `traverse_one_lt` per tree per class per sample using a
scalar scatter-add — this cannot exploit LightGBM's class-interleaved layout.

## Model configs

LightGBM is configured to match XGBoost tree complexity via `num_leaves ≈ 2^max_depth`:

| Size | XGBoost | LightGBM |
|------|---------|----------|
| small | 100 trees, depth 4 | 100 trees, depth 4, 16 leaves |
| medium | 500 trees, depth 6 | 500 trees, depth 6, 64 leaves |
| large | 1 000 trees, depth 8 | 1 000 trees, depth 8, 256 leaves |

Both use `subsample=0.8`, `colsample_bytree=0.8`, `random_state=42`, same 80/20 train/test split.

## Files

```
benchmark/
├── train_models.py      # train both frameworks, convert, save to models/
├── run_benchmark.py     # load, verify correctness, benchmark
├── models/              # auto-created (gitignored)
│   ├── {task}_{size}.omle            # XGBoost → OMLE protobuf
│   ├── {task}_{size}_xgb.json      # XGBoost booster (JSON)
│   ├── {task}_{size}_xgb.pkl       # XGBoost sklearn wrapper (joblib)
│   ├── {task}_{size}_lgb.omle        # LightGBM → OMLE protobuf
│   ├── {task}_{size}_lgb.txt       # LightGBM native text model
│   ├── {task}_{size}_lgb.pkl       # LightGBM sklearn wrapper (joblib)
│   ├── {task}_{size}_X_test.npy    # float32 test features (shared)
│   └── {task}_{size}_y_test.npy    # ground truth (shared)
└── results_*.csv        # timing output (auto-saved, includes framework column)
```

## CLI reference

### train_models.py

```
python3 train_models.py [--frameworks xgboost lgbm]
                        [--tasks regression binary multiclass mnist]
                        [--sizes small medium large]
                        [--force]
```

Note: `--tasks mnist` downloads ~170 MB from OpenML on first run (cached afterward).

### run_benchmark.py

```
python3 run_benchmark.py [--frameworks xgboost lgbm]
                         [--tasks regression binary multiclass mnist]
                         [--sizes small medium large]
                         [--batch-sizes 1 10 100 1000 10000]
                         [--reps 200]
                         [--warmup 20]
                         [--threads 1]
                         [--csv PATH]
```
