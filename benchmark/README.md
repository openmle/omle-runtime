# OMLE Latency Benchmark

End-to-end benchmark: train a model, convert it to OMLE protobuf and to ONNX,
then compare predictions and latency across four execution engines — the native
framework (XGBoost or LightGBM), ONNX Runtime, and OMLE.

## Quick start

```bash
# 1. Install Python deps. Use the requirements file rather than a hand-written
#    list: a missing package does not fail the run, it silently drops a column
#    from the comparison (no onnxruntime means no ORT numbers).
python3 -m pip install -r requirements.txt

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

OMLE and ORT predictions are both verified against the native framework before
benchmarking, on the full test set. Tolerance: absolute error ≤ 1e-4 (5e-2 for
the adult pipeline, explained below).

Both engines are checked against the native framework on every task and size,
because a converter that produces a fast but inaccurate model should not be able
to hide behind a good latency number.

ORT agrees with the native framework everywhere except LightGBM regression,
where the `.onnx` export loses accuracy as the forest grows:

| Model | OMLE | ORT |
|-------|-----:|----:|
| lgbm/regression/small (100 trees) | 0.00e+00 | 3.34e-06 |
| lgbm/regression/medium (500 trees) | **0.00e+00** | **7.12e-02** |
| lgbm/regression/large (1000 trees) | **0.00e+00** | **9.32e-02** |

The error grows with tree count, which is the signature of accumulated
precision loss rather than a single bad conversion: `onnxmltools` writes
LightGBM's float64 thresholds and leaf values as float32, and summing a thousand
of them drifts. OMLE keeps the same model in float64 end to end and stays exact.
On California Housing, where targets run roughly 0.5–5, an absolute error of
0.09 is a real accuracy loss, not rounding.

Everywhere else ORT lands between 1e-7 and 2e-5, and on the adult pipeline it
matches OMLE's error to three significant figures — both declare float32 inputs,
so both round at the boundary and route the same borderline samples the same way.

### XGBoost

| Task | Max absolute error | Note |
|------|--------------------|------|
| Regression | 0.00e+00 | Bit-exact, all three sizes |
| Binary | 5.96e-08 – 1.19e-07 | Float32 sigmoid rounding |
| Multiclass (Digits) | 2.38e-07 | Float32 softmax rounding |
| MNIST | 3.58e-07 | Float32 softmax rounding |

### Adult pipeline

| Task | Max absolute error | Note |
|------|--------------------|------|
| adult/small | 9.73e-03 | Float32 model input shifts borderline splits |
| adult/medium | 5.97e-03 | |
| adult/large | 1.19e-02 | |

Tolerance here is 5e-2 rather than 1e-4 because the error is not rounding but
routing. Around 85% of the 9045 test rows match sklearn *exactly*; the rest
differ because a sample sitting on a split threshold is sent down the other
branch. The model declares its six numeric columns as `FLOAT32`, so the feature
values are rounded at the boundary before any scaling happens, while sklearn
carries them in float64 end to end. That is enough to change which side of a
threshold a borderline value falls on.

The tree itself is not the source: the converter stores this model's thresholds
in float64, and the runtime selects its float64 path whenever thresholds or leaf
values are float64. The same mechanism is why LightGBM is bit-exact above — it
is float64 from input to output — while XGBoost is float32, matching the
precision XGBoost itself predicts in.

The worst case scales with the forest, consistent with more trees giving more
chances for one to be borderline: the largest disagreement corresponds to about
1 tree of 100 for `small`, 3 of 500 for `medium`, and 12 of 1000 for `large`.
A single tree changing its leaf moves an averaged probability by a fraction of
1/n_trees, which is why the absolute error stays near 1e-2 rather than growing.

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
| `ort-session` | `InferenceSession.run(...)` — ONNX Runtime on the `.onnx` export of the same model |
| `omle-model` | `model.predict(X)` — stateless, allocates a session per call |
| `omle-session` | `session.predict(X)` — persistent session, reused across calls |

## Benchmark matrix results

> **These numbers are not thread-matched and are being re-measured.** They were
> taken when `--threads` governed only OMLE: OMLE and ORT ran on one thread while
> XGBoost and LightGBM predicted on all 11 cores, because neither library
> defaults to single-threaded inference. Everything below batch≈100 is unaffected
> (native threading does not engage on tiny batches), but the large-batch columns
> compare 1 core against 11 and understate OMLE by roughly 4-6x. `--threads` now
> applies to every engine; see *Threading* below.

Speedup over the faster of the two native paths, so `>1` means faster than the
framework's own predictor. Apple M-series, `min` latency, `small` models. Bold
marks the faster of ORT and OMLE.

**XGBoost** — ORT / OMLE

| Batch | regression | binary | multiclass | mnist |
|------:|-----------:|-------:|-----------:|------:|
| 1 | 38.84x / **42.20x** | **30.60x** / 21.73x | 15.48x / **29.19x** | 18.04x / **23.65x** |
| 10 | 11.78x / **14.13x** | **10.07x** / 9.37x | 1.90x / **6.11x** | 3.17x / **4.39x** |
| 100 | 1.36x / **2.99x** | 1.33x / **2.31x** | 0.26x / **1.18x** | 0.35x / **0.85x** |
| 1 000 | 0.15x / **0.51x** | 0.25x / **0.38x** | 0.05x / **0.16x** | 0.07x / **0.17x** |
| 10 000 | 0.06x / **0.22x** | 0.10x / **0.15x** | 0.04x / **0.12x** | 0.05x / **0.12x** |

**LightGBM** — ORT / OMLE

| Batch | regression | binary | multiclass | mnist |
|------:|-----------:|-------:|-----------:|------:|
| 1 | 12.49x / **13.35x** | **9.25x** / 7.52x | 3.08x / **5.23x** | 4.65x / **5.69x** |
| 10 | **5.39x** / 5.18x | 2.09x / **3.26x** | 0.72x / **1.68x** | 1.03x / **1.24x** |
| 100 | 1.04x / **1.50x** | 0.60x / **0.94x** | 0.22x / **1.05x** | 0.23x / **0.52x** |
| 1 000 | 0.24x / **0.77x** | 0.39x / **0.46x** | 0.14x / **0.70x** | 0.14x / **0.29x** |
| 10 000 | 0.17x / **0.58x** | 0.31x / **0.40x** | 0.13x / **0.62x** | 0.12x / **0.23x** |

`lgbm/regression` at medium and large is the one place where the two engines are
not computing the same answer: those ONNX exports carry an absolute error around
7e-02 to 9e-02 (see *Correctness*), so their timings are not comparable.

**Adult pipeline** — ORT / OMLE. The widest margins in the suite, because the
native path re-runs the whole sklearn `ColumnTransformer` on every call.

| Batch | ORT / OMLE |
|------:|-----------:|
| 1 | 167.50x / **186.22x** |
| 10 | **97.21x** / 90.60x |
| 100 | **20.62x** / 15.30x |
| 1 000 | **2.87x** / 1.88x |
| 10 000 | **1.42x** / 0.79x |

## ONNX Runtime

ORT executes the `.onnx` export of the same trained model, so the comparison is
engine-versus-engine on identical maths. Its numbers are in *Benchmark matrix
results* above; what follows is what they mean.

ORT wins heavily at batch=1, where the native frameworks are dominated by
per-call setup, and loses from roughly batch=100 onward, ending up **3–25x
slower than native** at batch 10 000. It is a latency engine here, not a
throughput one. The effect is larger against XGBoost than LightGBM because
XGBoost's native path builds a `DMatrix` per call, which LightGBM's does not.

Against OMLE the two are close at batch=1 — ORT is ahead on binary for both
frameworks — and OMLE pulls away as batches grow, by 3–5x at batch 10 000 on
multiclass. The adult pipeline is the exception: ORT stays ahead from batch 10
upward, the only place in the suite where it leads at scale.

ORT also costs more to hold resident. Session RSS deltas on the `large` models,
where the numbers are above allocator noise:

| Model | ORT session | OMLE session |
|-------|------------:|-------------:|
| xgboost/regression | +134.3 MB | **+34.4 MB** |
| xgboost/mnist | +41.0 MB | **+13.5 MB** |
| lgbm/regression | +64.0 MB | **+24.4 MB** |
| lgbm/mnist | +39.8 MB | **+15.2 MB** |
| pipeline/adult | +60.3 MB | **+32.8 MB** |

On the `small` models both engines sit within ±1 MB, so those rows say nothing.

On disk, `.onnx` is larger than `.omle` for all 26 models — 0.23–0.84x of the
native file against 0.22–0.55x, which is 1.05–1.77x the size of the equivalent
`.omle`.

## Threading

Neither XGBoost nor LightGBM defaults to single-threaded prediction, and neither
does sklearn's `RandomForestClassifier` — leave `n_jobs`/`num_threads` unset and
all three use every core. OMLE defaults to one. That made the original
"native wins at scale" result mostly an artefact of core count, so `--threads`
now sets the budget for every engine:

```bash
python run_benchmark.py --threads 1    # one core each: compares the kernels
python run_benchmark.py --threads 8    # eight cores each: compares deployments
```

Separating the two effects on `lgbm/regression medium`, batch 10 000, 11 cores
available:

| | native | OMLE | OMLE advantage |
|---|-------:|-----:|---------------:|
| 1 thread each | 161.09 ms | 34.83 ms | **4.62x** |
| all cores each | 28.63 ms | 6.31 ms | **4.54x** |
| 1 thread OMLE vs all-core native | 28.63 ms | 34.83 ms | 0.82x |

The last row is what the matrix above reports. Thread scaling is close to
identical on both sides — 5.63x native against 5.52x OMLE on 5 performance plus
6 efficiency cores — so a matched comparison holds its ratio at either thread
count. Per thread, OMLE is at parity with XGBoost (0.91–1.03x on
`regression`/`binary` medium) and 4.6x faster than LightGBM, whose native
predictor is the slowest single-threaded kernel in the suite.

### min_parallel_rows

OMLE splits a batch across threads only when it has at least
`n_threads * min_parallel_rows` rows, with `min_parallel_rows` defaulting to 64.
Because the threshold scales with the thread count, asking for more threads
raises the batch size at which any of them are used — at 11 threads nothing below
704 rows is parallelised at all:

| Batch | 1 thread | 11 threads, default 64 | 11 threads, tuned |
|------:|---------:|-----------------------:|------------------:|
| 100 | 0.348 ms | 0.348 ms | **0.142 ms** (mpr=8) |
| 200 | 0.695 ms | 0.695 ms | **0.200 ms** (mpr=16) |
| 500 | 1.735 ms | 1.738 ms | **0.403 ms** (mpr=32) |
| 1 000 | 3.468 ms | **0.736 ms** | 0.731 ms (mpr=8) |

So batches of 100–500 leave a 2.4–4.3x speedup unclaimed on the default. No
setting down to `mpr=8` was slower than the default at any batch size measured,
which suggests 64 rows per thread is a more conservative floor than the dispatch
cost warrants. `--min-parallel-rows` exposes it for measurement; the library
default in `runtime.h` is unchanged.

## Key findings (Apple M-series)

Two things drive the shape of *Benchmark matrix results*.

**Per-call overhead dominates small batches.** At batch=1 OMLE is 21–42x faster
than XGBoost and 5–13x faster than LightGBM. The gap is wider against XGBoost
because its native path constructs a `DMatrix` per call; LightGBM's booster takes
numpy directly, so it starts from a lighter baseline and the margin is smaller.

**The apparent crossover at large batches is mostly core count.** From
batch≈1000 the matrix shows native ahead by 1.6–8x, but that is 11 native cores
against one OMLE thread. Thread-matched, OMLE holds its lead on LightGBM and
reaches parity with XGBoost — see *Threading*. A genuine per-thread gap remains
only against XGBoost, and it is under 10%.

The practical read: OMLE's advantage is largest for latency-sensitive serving of
one row or a few, where it is a large constant factor faster. For bulk scoring it
is competitive once given the same cores, but a native library already tuned for
your batch size is not obviously worth replacing. The adult pipeline is the
outlier, at **186.22x** at batch=1 and still **1.88x** at batch 1000, because
the native path re-runs the entire sklearn `ColumnTransformer` on every call
while OMLE compiles it into the graph.

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
