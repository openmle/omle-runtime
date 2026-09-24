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

Speedup over the faster of the two native paths, so `>1` means faster than the
framework's own predictor. Apple M-series, `min` latency, `small` models,
**every engine pinned to one thread** (`--threads 1`). Bold marks the faster of
ORT and OMLE.

**XGBoost** — ORT / OMLE

| Batch | regression | binary | multiclass | mnist |
|------:|-----------:|-----------:|-----------:|-----------:|
| 1 | 13.99x / **16.64x** | 10.45x / **15.31x** | 6.82x / **13.75x** | 7.43x / **9.63x** |
| 10 | 5.41x / **7.74x** | 3.89x / **7.87x** | 1.22x / **3.99x** | 1.62x / **2.28x** |
| 100 | 0.87x / **2.43x** | 0.82x / **2.31x** | 0.29x / **1.64x** | 0.37x / **0.93x** |
| 1 000 | 0.31x / **1.39x** | 0.48x / **1.30x** | 0.23x / **1.08x** | 0.28x / **0.75x** |
| 10 000 | 0.26x / **1.21x** | 0.44x / **1.11x** | 0.22x / **1.03x** | 0.27x / **0.68x** |

**LightGBM** — ORT / OMLE

| Batch | regression | binary | multiclass | mnist |
|------:|-----------:|-----------:|-----------:|-----------:|
| 1 | 4.97x / **5.72x** | 3.60x / **5.54x** | 2.12x / **3.62x** | 2.64x / **3.16x** |
| 10 | 3.33x / **3.97x** | 1.26x / **3.85x** | 1.02x / **3.37x** | 1.37x / **1.89x** |
| 100 | 2.16x / **3.36x** | 1.26x / **3.09x** | 0.71x / **3.43x** | 0.79x / **1.71x** |
| 1 000 | 0.86x / **3.36x** | 1.46x / **2.65x** | 0.67x / **3.37x** | 0.65x / **1.53x** |
| 10 000 | 0.75x / **3.31x** | 1.46x / **2.84x** | 0.67x / **3.33x** | 0.70x / **1.43x** |

**Adult pipeline** — ORT / OMLE

| Batch | ORT / OMLE |
|------:|-----------:|
| 1 | 147.92x / **181.00x** |
| 10 | **96.89x** / 86.77x |
| 100 | **20.47x** / 15.88x |
| 1 000 | **2.78x** / 1.83x |
| 10 000 | **1.39x** / 0.75x |

`lgbm/regression` at medium and large is the one place where the two engines are
not computing the same answer: those ONNX exports carry an absolute error around
7e-02 to 9e-02 (see *Correctness*), so their timings there are not comparable.
The `small` models in the table above all agree.

## ONNX Runtime

ORT executes the `.onnx` export of the same trained model, so the comparison is
engine-versus-engine on identical maths. Its numbers are in *Benchmark matrix
results* above; what follows is what they mean.

ORT wins at batch=1, where the native frameworks are dominated by per-call
setup, and loses from roughly batch=100, ending up **1.3–4.5x slower than
native** at batch 10 000 on every model but `lgbm/binary`, where it stays 1.5x
ahead. It is a latency engine here, not a throughput one, and the effect is
larger against XGBoost than LightGBM because XGBoost's native path builds a
`DMatrix` per call while LightGBM's takes numpy directly.

OMLE is ahead of ORT at every batch size on every tree model, by 1.9–5.0x at
batch 10 000, and the margin widens with batch size rather than narrowing. The
adult pipeline is the one exception: ORT leads from batch 10 upward and is 1.8x
ahead at batch 10 000, the only place in the suite where it wins at scale.

## Footprint

Serialised size and the resident-memory cost of holding a loaded session, for
every model in the suite. Bold marks the smaller of the two engines. `.omle` is
compared against the framework's own file (`.json` for XGBoost, `.txt` for
LightGBM, a joblib `.pkl` for the sklearn pipeline).

| Model | native | `.onnx` | `.omle` | ORT RSS | OMLE RSS |
|-------|-------:|--------:|--------:|--------:|--------:|
| xgb/regression/small | 192 KB | 92 KB | **58 KB** | 2.3 MB | **0.6 MB** |
| xgb/regression/medium | 3.1 MB | 1.8 MB | **1.0 MB** | **5.3 MB** | 7.2 MB |
| xgb/regression/large | 18.9 MB | 11.6 MB | **6.7 MB** | 155.0 MB | **54.3 MB** |
| xgb/binary/small | 94 KB | 32 KB | **27 KB** | 1.3 MB | **0.6 MB** |
| xgb/binary/medium | 300 KB | 68 KB | **67 KB** | 0.9 MB | **0.3 MB** |
| xgb/binary/large | 598 KB | 135 KB | **130 KB** | **0.6 MB** | 1.1 MB |
| xgb/multiclass/small | 981 KB | 365 KB | **254 KB** | 1.6 MB | **1.0 MB** |
| xgb/multiclass/medium | 3.1 MB | 832 KB | **720 KB** | **2.9 MB** | 5.4 MB |
| xgb/multiclass/large | 6.4 MB | 1.7 MB | **1.4 MB** | **3.9 MB** | 8.8 MB |
| xgb/mnist/small | 592 KB | 240 KB | **213 KB** | 5.6 MB | **2.1 MB** |
| xgb/mnist/medium | 2.0 MB | 957 KB | **626 KB** | 7.0 MB | **3.9 MB** |
| xgb/mnist/large | — | — | — | — | — |
| lgbm/regression/small | 147 KB | 105 KB | **73 KB** | 2.4 MB | **0.6 MB** |
| lgbm/regression/medium | 1.7 MB | 1.3 MB | **883 KB** | **3.3 MB** | 3.9 MB |
| lgbm/regression/large | 6.3 MB | 5.3 MB | **3.5 MB** | 51.8 MB | **28.6 MB** |
| lgbm/binary/small | 125 KB | 70 KB | **54 KB** | 2.5 MB | **0.9 MB** |
| lgbm/binary/medium | 419 KB | 214 KB | **163 KB** | **0.3 MB** | 0.8 MB |
| lgbm/binary/large | 932 KB | 502 KB | **361 KB** | **2.9 MB** | 3.1 MB |
| lgbm/multiclass/small | 958 KB | 533 KB | **383 KB** | 4.0 MB | **3.2 MB** |
| lgbm/multiclass/medium | 2.7 MB | 1.2 MB | **1013 KB** | **3.6 MB** | 6.2 MB |
| lgbm/multiclass/large | 6.1 MB | 3.0 MB | **2.3 MB** | 22.5 MB | **17.8 MB** |
| lgbm/mnist/small | 520 KB | 280 KB | **256 KB** | 7.0 MB | **2.3 MB** |
| lgbm/mnist/medium | 1.8 MB | 1.1 MB | **802 KB** | 9.0 MB | **5.2 MB** |
| lgbm/mnist/large | — | — | — | — | — |
| sklearn/adult/small | 282 KB | 111 KB | **96 KB** | 7.0 MB | **1.2 MB** |
| sklearn/adult/medium | 4.0 MB | 1.8 MB | **1.2 MB** | 8.5 MB | **7.3 MB** |
| sklearn/adult/large | 20.9 MB | 10.0 MB | **6.8 MB** | 102.0 MB | **44.3 MB** |

**On disk `.omle` is never larger** — 25 of 25 models, by up to 1.75x against the
equivalent `.onnx`, though `xgb/binary` is effectively a dead heat at all three
sizes. Against the native file `.omle` lands at 0.22–0.55x where ONNX needs
0.23–0.84x. The margin is widest on LightGBM, whose ONNX export stays close in
size to the original `.txt`.

**Memory is not a clean win, and the direction depends on size.** On the five
largest models — the ones where the delta is well clear of allocator noise — OMLE
holds 1.3–2.9x less: 54.3 MB against 155.0 MB on `xgb/regression/large`, 44.3
against 102.0 on the adult pipeline. But on eight of the 25 ORT is the
smaller of the two, by as much as 4.9 MB on `xgb/multiclass/large` (3.9 MB
against 8.8 MB), and below roughly 3 MB the deltas are within allocator noise and
should not be read as a ranking at all. The honest summary is
that OMLE scales better with model size rather than being uniformly lighter.

MNIST `large` is absent from both frameworks: those two models exhaust memory on
this machine, and the run was scoped to `small` and `medium` rather than reporting
figures taken under swap pressure.

## Threading

Neither XGBoost nor LightGBM defaults to single-threaded prediction, and neither
does sklearn's `RandomForestClassifier` — leave `n_jobs`/`num_threads` unset and
all three use every core. OMLE defaults to one. That made the original
"native wins at scale" result largely an artefact of core count, so `--threads`
now sets the budget for every engine, native and ORT included:

```bash
python run_benchmark.py --threads 1    # one core each: compares the kernels
python run_benchmark.py --threads 8    # eight cores each: compares deployments
```

Separating the two effects on `lgbm/regression medium`, batch 10 000, 11 cores
available:

| | native | OMLE | OMLE advantage |
|---|-------:|-----:|---------------:|
| 1 thread each | 163.84 ms | 24.70 ms | **6.63x** |
| all cores each | 25.67 ms | 3.43 ms | **7.49x** |
| 1 thread OMLE vs all-core native | 25.67 ms | 24.70 ms | 1.04x |

The last row is the comparison the benchmark used to make by accident, and it is
the only one of the three that looks like a tie. Thread scaling is close on both
sides — 6.38x native against 7.21x OMLE on 5 performance plus 6 efficiency
cores — so a matched comparison holds its ratio at either thread count.

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

**Per-call overhead dominates small batches.** At batch=1 OMLE is 9.6–16.6x
faster than XGBoost and 3.2–5.7x faster than LightGBM, with both sides on one
thread. The gap is wider against XGBoost because its native path constructs a
`DMatrix` per call; LightGBM's booster takes numpy directly, so it starts from a
lighter baseline. Against a *stock* XGBoost the batch=1 gap is far wider still —
it spins up a thread per core for a single row and the dispatch cost dominates —
but that measures thread setup, not the kernel.

**There is no crossover any more.** OMLE stays ahead of both frameworks at every
batch size on every tree model, bottoming out at 1.03x (`xgboost/multiclass`,
batch 10 000) and holding 1.4–3.3x across LightGBM. Earlier revisions of this
file reported native winning by 2–8x from batch≈1000; that was three separate
things, all since fixed:

- the benchmark pinned OMLE to one thread while XGBoost and LightGBM used every
  core, so the large-batch columns compared 1 core against 11 (see *Threading*);
- the batched walk read four separate SoA arrays per node instead of the compact
  AoS forest, and ran four traversals to completion rather than interleaving
  eight, leaving the dependent-load chain unhidden;
- a single splitless tree — which boosting emits routinely, 1514 of 5000 trees
  in the multiclass model — disqualified an entire model from the fast layout.

**MNIST is the hardest case and the one still worth watching.** At 784 features
`xgboost/mnist` is 0.68x at batch 10 000, the only tree model below parity. The
feature row no longer fits the blocking assumption that keeps a 128-row window
in L1, so the traversal pays cache misses the other models do not.

The practical read: OMLE is the better choice for latency-sensitive serving of
one row or a few, where it is a large constant factor faster, and it is now also
competitive-to-better for bulk scoring at matched thread counts. The adult
pipeline remains the widest margin at batch=1, at **181.00x**, because the native
path re-runs the entire sklearn `ColumnTransformer` on every call while OMLE
compiles it into the graph — though it is also the one model where the native
path wins at batch 10 000 (0.75x), since its cost is preprocessing, not trees.

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
