"""
omle_runtime — fast inference for classical ML models.

pybind11 bindings over the omle C API.  Requires the compiled
omle_ext extension module (built with cmake -DBUILD_PYTHON=ON).

The :class:`Model` class follows the scikit-learn estimator interface:
``predict``, ``predict_proba``, ``fit`` (no-op), ``get_params``,
``set_params``, ``n_features_in_``, and ``feature_names_in_`` are all
present, so ``Model`` can be used as the final step of a
``sklearn.pipeline.Pipeline``.

Quick start::

    import omle_runtime as omr
    import numpy as np

    model   = omr.load("model.omle", n_threads=4)   # thread-safe
    session = model.create_session()               # one per thread

    X = np.random.randn(1000, 10).astype(np.float32)

    # sklearn-style (Model or Session):
    scores = model.predict(X)          # (n_samples,)  — single-output
    proba  = model.predict_proba(X)    # (n_samples, n_outputs)

    # or via Session (avoids per-call session creation):
    scores = session.predict(X)

    # Pipeline example:
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
    pipe.predict(X)
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Union

import numpy as np


# ---------------------------------------------------------------------------
# Windows DLL search path
# ---------------------------------------------------------------------------
# omle_ext.pyd links omleruntime.dll, which in turn links libprotobuf and
# Abseil. Since Python 3.8 an extension module's dependent DLLs are resolved
# against the system directories, the directory holding the .pyd, and whatever
# os.add_dll_directory() has registered — PATH is deliberately not consulted.
# omleruntime.dll sits next to the .pyd so it resolves on its own, but protobuf
# and Abseil usually live in a conda or vcpkg prefix that is only on PATH, and
# without this the import fails with a bare "DLL load failed".
#
# The cookies returned by add_dll_directory() unregister the directory when
# closed, so they are parked in a module-level list to keep them alive.
_dll_directories = []

if sys.platform == "win32":
    _candidates = [os.path.dirname(os.path.abspath(__file__))]
    # An explicit override comes first; it is the only knob available when the
    # dependencies live somewhere PATH does not mention.
    _candidates += os.environ.get("OMLE_RUNTIME_DLL_PATH", "").split(os.pathsep)
    _candidates += os.environ.get("PATH", "").split(os.pathsep)

    _seen = set()
    for _d in _candidates:
        if not _d:
            continue
        try:
            _key = os.path.normcase(os.path.abspath(_d))
            if _key in _seen or not os.path.isdir(_key):
                continue
            _seen.add(_key)
            _dll_directories.append(os.add_dll_directory(_key))
        except OSError:
            # An unreadable or malformed PATH entry is not worth failing over.
            continue

# pybind11 extension — required
try:
    from omle_runtime import omle_ext as _ext
except ImportError as _e:
    raise ImportError(
        "omle_runtime: could not import the pybind11 extension 'omle_ext'.\n"
        f"Underlying error: {_e}\n"
        "If the module is missing, build it with:\n"
        "    cmake -DBUILD_PYTHON=ON .. && cmake --build . --target omle_ext\n"
        "If it was found but failed to load, a dependent shared library "
        "(omle_runtime, protobuf, Abseil) is not on the loader's search path; "
        "on Windows point OMLE_RUNTIME_DLL_PATH at the directory holding them."
    ) from _e


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DataType(IntEnum):
    Unknown   = 0
    Bool      = 1
    Int8      = 2
    Int16     = 3
    Int32     = 4
    Int64     = 5
    UInt8     = 6
    UInt16    = 7
    UInt32    = 8
    UInt64    = 9
    Float16   = 10
    Float32   = 11
    Float64   = 12
    String    = 13
    Bytes     = 14
    Date      = 15
    Time      = 16
    Timestamp = 17


class OutputRole(IntEnum):
    Unspecified      = 0
    Prediction       = 1
    Probability      = 2
    Score            = 3
    Confidence       = 4
    StandardError    = 5
    StandardDev      = 6
    Residual         = 7
    TransformedValue = 8
    EntityId         = 9
    Affinity         = 10
    Contribution     = 11
    Intermediate     = 12


# ---------------------------------------------------------------------------
# Spec dataclasses  (returned by Model.inputs / Model.outputs)
# ---------------------------------------------------------------------------

@dataclass
class InputSpec:
    name:  str
    dtype: DataType
    shape: List[int]

    def __repr__(self) -> str:
        return f"InputSpec(name='{self.name}', shape={self.shape})"


@dataclass
class OutputSpec:
    name:  str
    dtype: DataType
    shape: List[int]
    role:  OutputRole = OutputRole.Unspecified

    def __repr__(self) -> str:
        return f"OutputSpec(name='{self.name}', shape={self.shape})"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _read_spec(model_ptr: int, idx: int, is_output: bool):
    fn = _ext.model_output_spec if is_output else _ext.model_input_spec
    result = fn(model_ptr, idx)
    if is_output:
        name, dtype_int, shape, role_int = result
        dtype = DataType(dtype_int)
        try:
            role = OutputRole(role_int)
        except ValueError:
            role = OutputRole.Unspecified
        return OutputSpec(name=name, dtype=dtype, shape=list(shape), role=role)
    name, dtype_int, shape = result
    dtype = DataType(dtype_int)
    return InputSpec(name=name, dtype=dtype, shape=list(shape))


def _dict_to_array(d: dict, output_specs: List[OutputSpec]) -> np.ndarray:
    cols = [d[s.name] for s in output_specs if s.name in d]
    if not cols:
        cols = list(d.values())
    return np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]


def _df_has_strings(df) -> bool:
    """Return True if any column of df has a string/object dtype."""
    try:
        import pandas as pd
        return any(
            df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
            for c in df.columns
        )
    except Exception:
        return False


def _prepare_col_data(df, col_str_mask=None):
    """Return (col_names, col_data) for _ext.model_predict_columns / session_bind_columns.

    Each entry in col_data is either a float32 ndarray (for numeric columns)
    or a numpy object ndarray (for string/object columns).  The C++ extension
    handles encoding via PyUnicode_AsUTF8 — no Python .encode() overhead.

    col_str_mask: optional pre-computed list[bool] per column; skips dtype checks.
    """
    import pandas as pd
    col_names = [str(c) for c in df.columns]
    col_data: list = []
    for i, col in enumerate(df.columns):
        s = df[col]
        is_str = col_str_mask[i] if col_str_mask is not None else (
            s.dtype == object or pd.api.types.is_string_dtype(s))
        if is_str:
            # to_numpy(dtype=object), not .values: pandas 3.0 stores strings in
            # an Arrow-backed StringDtype whose .values is an ArrowStringArray,
            # not the numpy object array the C++ side reads PyObject* out of.
            arr = s.to_numpy(dtype=object)
            # Missing values arrive as a float nan, which the C++ reader hands
            # to PyUnicode_AsUTF8 and fails on with a bare TypeError. Converters
            # stringify categories with str(), so a NaN category is stored in
            # the model as "nan" -- spell it the same way here so the lookup
            # matches instead of crashing.
            if s.isna().any():
                arr = np.where(pd.isna(arr), "nan", arr)
            col_data.append(arr)
        else:
            # Keep float64 columns at full width. Narrowing here and scaling
            # afterwards is not the same as scaling in float64 and narrowing
            # once, and the ulp of difference is enough to move a value across
            # a tree split threshold. Anything else still goes over as float32.
            vals = s.to_numpy()
            # Integers go over as float64 too. float32 carries only 24 bits of
            # mantissa, so an int64 column silently loses values above 2**24 --
            # 16777217 arrives as 16777216. float64 is exact to 2**53, which
            # covers ids, counts and epoch-second timestamps. Beyond that a
            # true integer column type would be needed.
            dt = (np.float64
                  if vals.dtype == np.float64 or vals.dtype.kind in "iu"
                  else np.float32)
            col_data.append(np.ascontiguousarray(vals, dtype=dt))
    return col_names, col_data


def _session_run_array(session_ptr: int, arr: np.ndarray,
                       output_specs: List[OutputSpec], input_name: str) -> np.ndarray:
    """Session run with a single dense numpy input array.

    Binds float64 input as float64. Forcing float32 here would make a Session
    score differently from Model.predict on the same array, since that path
    now carries the caller's width through.
    """
    if arr.dtype == np.float64:
        t_ptr = _ext.tensor_create_f64(arr.shape[0], arr.shape[1], arr)
    else:
        t_ptr = _ext.tensor_create_f32(arr.shape[0], arr.shape[1], arr)
    try:
        _ext.session_clear_inputs(session_ptr)
        _ext.session_bind_input(session_ptr, input_name, t_ptr)
        _ext.session_run(session_ptr)
    finally:
        _ext.free_tensor(t_ptr)
    cols = [_ext.tensor_read(_ext.session_get_output(session_ptr, s.name))
            for s in output_specs]
    return np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------

def _coerce(
    X,
    expected_features: int = 0,
) -> np.ndarray:
    """Return X as a C-contiguous 2-D float array, validating feature count.

    Accepts numpy arrays, pandas DataFrames (numeric), scipy sparse matrices,
    Python list / nested list, or :class:`Tensor`.
    A 1-D input is treated as a single sample ``(1, n_features)``.

    float64 input stays float64; everything else becomes float32. The dense
    entry path used to narrow unconditionally, which meant an all-numeric
    DataFrame lost precision before the first operator ran -- the columns
    path (taken only when some column is a string) already kept full width,
    so the same model scored differently depending on whether it happened to
    have a string feature.
    """
    # Fast path: already the right format — avoid all numpy overhead
    if (isinstance(X, np.ndarray) and X.dtype in (np.float32, np.float64)
            and X.ndim == 2 and X.flags['C_CONTIGUOUS']):
        if expected_features > 0 and X.shape[1] != expected_features:
            raise ValueError(
                f"X has {X.shape[1]} feature(s) but model expects {expected_features}."
            )
        return X

    # scipy sparse → dense
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except ImportError:
        pass

    if isinstance(X, Tensor):
        arr = X.numpy()
    else:
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                X = X.values
        except ImportError:
            pass
        arr = np.asarray(X)
        # Integers go over as float64 for the same reason _prepare_col_data
        # sends them that way: float32 carries 24 bits of mantissa, so an id
        # or count above 2**24 would arrive as a different number.
        want = (np.float64
                if arr.dtype == np.float64 or arr.dtype.kind in "iu"
                else np.float32)
        arr = np.asarray(arr, dtype=want)

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim != 2:
        raise ValueError(f"Input must be 1-D or 2-D, got {arr.ndim}-D")

    if expected_features > 0 and arr.shape[1] != expected_features:
        raise ValueError(
            f"X has {arr.shape[1]} feature(s) but model expects {expected_features}."
        )

    dt = np.float64 if arr.dtype == np.float64 else np.float32
    return np.ascontiguousarray(arr, dtype=dt)


def _maybe_squeeze(arr: np.ndarray, squeeze: bool, n_outputs: int) -> np.ndarray:
    if squeeze and n_outputs == 1 and arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


# ---------------------------------------------------------------------------
# Tensor  (pure Python — no C handle)
# ---------------------------------------------------------------------------

class Tensor:
    """Named, typed data container backed by a NumPy float32 array.

    Parameters
    ----------
    data : array-like
        Converted to ``float32`` C-contiguous automatically.
        A 1-D array is treated as a single sample ``(1, n_features)``.
    name : str, optional
        Logical name (e.g. the output column name).
    dtype : DataType, optional
        Declared element type (informational; storage is always float32).
    """

    def __init__(
        self,
        data: Union[np.ndarray, "pd.DataFrame", list],
        name: str = "",
        dtype: DataType = DataType.Float32,
    ) -> None:
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                data = data.values
        except ImportError:
            pass
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim > 2:
            raise ValueError(f"Tensor data must be 1-D or 2-D, got {arr.ndim}-D")
        self._arr  = np.ascontiguousarray(arr)
        self.name  = name
        self.dtype = dtype

    @property
    def shape(self) -> tuple:
        return self._arr.shape

    @property
    def n_rows(self) -> int:
        return self._arr.shape[0]

    @property
    def n_cols(self) -> int:
        return self._arr.shape[1] if self._arr.ndim == 2 else 1

    def numpy(self) -> np.ndarray:
        """Return the underlying NumPy array (zero-copy)."""
        return self._arr

    def __repr__(self) -> str:
        tag = f"'{self.name}'" if self.name else "unnamed"
        return f"<omle_runtime.Tensor {tag} shape={self.shape}>"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _restore_model(data: bytes, n_threads: int, min_parallel_rows: int) -> "Model":
    return Model.load_bytes(data, n_threads=n_threads,
                            min_parallel_rows=min_parallel_rows)


class Model:
    """Immutable model handle — thread-safe.

    Load via :func:`load` / :func:`load_bytes`, or the class methods below.
    Multiple threads may call :meth:`predict` or :meth:`create_session` concurrently.
    """

    def __init__(self, ptr: int, inputs: List[InputSpec], outputs: List[OutputSpec]) -> None:
        self._ptr     = ptr
        self._inputs  = inputs
        self._outputs = outputs
        self._model_bytes = None
        self._n_threads = 1
        self._min_parallel_rows = 64

    @classmethod
    def _from_ptr(cls, ptr: int) -> "Model":
        n_in  = _ext.model_num_inputs(ptr)
        n_out = _ext.model_num_outputs(ptr)
        # The C library may over-report num_inputs for matrix inputs (it counts
        # columns rather than named input tensors). Stop reading at the first error.
        ins: List[InputSpec] = []
        for i in range(n_in):
            try:
                ins.append(_read_spec(ptr, i, False))
            except RuntimeError:
                break
        outs: List[OutputSpec] = []
        for i in range(n_out):
            try:
                outs.append(_read_spec(ptr, i, True))
            except RuntimeError:
                break
        return cls(ptr, ins, outs)

    @classmethod
    def load(cls, path: str, *, n_threads: int = 1, min_parallel_rows: int = 64) -> "Model":
        """Load from a protobuf binary file.

        Reads the file in Python rather than handing the path to the
        extension, so the Model keeps the bytes it was built from and can be
        pickled.  A consequence worth knowing: an unreadable path raises the
        matching OSError subclass — FileNotFoundError, PermissionError,
        IsADirectoryError — while a readable file whose contents are not a
        valid model still raises RuntimeError from load_bytes.
        """
        with open(path, "rb") as file:
            return cls.load_bytes(file.read(), n_threads=n_threads,
                                  min_parallel_rows=min_parallel_rows)

    @classmethod
    def load_bytes(cls, data: bytes, *, n_threads: int = 1, min_parallel_rows: int = 64) -> "Model":
        """Load from a bytes object (e.g. from a database or object store)."""
        ptr = _ext.model_load_memory(data, n_threads, min_parallel_rows)
        model = cls._from_ptr(ptr)
        model._model_bytes = data
        model._n_threads = n_threads
        model._min_parallel_rows = min_parallel_rows

        # Advisories the loader collected rather than printed — a schema version
        # this build does not recognise on a model with no verification cases to
        # settle it, for instance. Raised through the warnings module so they can
        # be filtered, captured in tests, or escalated with -W error, none of
        # which is possible for a library that writes to stderr.
        for message in _ext.model_warnings(ptr):
            warnings.warn(f"omle-runtime: {message}", RuntimeWarning,
                          stacklevel=3)
        return model

    def __reduce__(self):
        """Rebuild the native handle from the embedded model on unpickle."""
        return (_restore_model, (self._model_bytes, self._n_threads,
                                 self._min_parallel_rows))

    def __del__(self) -> None:
        if getattr(self, "_ptr", None):
            _ext.free_model(self._ptr)
            self._ptr = None

    # ------------------------------------------------------------------
    # sklearn-compatible attributes
    # ------------------------------------------------------------------

    @property
    def n_features_in_(self) -> int:
        """Number of input features seen at fit time (sklearn convention)."""
        if len(self._inputs) == 1 and len(self._inputs[0].shape) == 2:
            return self._inputs[0].shape[1]
        return len(self._inputs)

    @property
    def feature_names_in_(self) -> np.ndarray:
        """Input feature names as a numpy string array (sklearn convention)."""
        return np.array([s.name for s in self._inputs])

    # ------------------------------------------------------------------
    # Omle-style schema accessors
    # ------------------------------------------------------------------

    @property
    def num_inputs(self) -> int:
        return len(self._inputs)

    @property
    def num_outputs(self) -> int:
        return len(self._outputs)

    @property
    def inputs(self) -> List[InputSpec]:
        return self._inputs

    @property
    def outputs(self) -> List[OutputSpec]:
        return self._outputs

    # ------------------------------------------------------------------
    # Session factory
    # ------------------------------------------------------------------

    def create_session(self) -> "Session":
        """Return a new per-thread :class:`Session` backed by this model."""
        return Session(_ext.session_create(self._ptr), self)

    # ------------------------------------------------------------------
    # sklearn estimator protocol
    # ------------------------------------------------------------------

    def fit(self, X, y=None) -> "Model":
        """No-op — model is pre-trained.  Validates input shape and returns self."""
        _coerce(X, self.n_features_in_)
        return self

    def get_params(self, deep: bool = True) -> dict:
        """Return estimator parameters (empty — loaded models have no hyperparameters)."""
        return {}

    def set_params(self, **params) -> "Model":
        """Set estimator parameters (no-op)."""
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _is_per_feature(self) -> bool:
        """Return True when model uses multiple 1-D per-feature inputs."""
        return len(self._inputs) > 1

    def _predict_raw(self, arr: np.ndarray, output_specs: List[OutputSpec]) -> np.ndarray:
        """Dispatch to named or single-tensor prediction based on input layout."""
        filter_names = [s.name for s in output_specs]
        if self._is_per_feature():
            inputs = {spec.name: arr[:, i:i+1] for i, spec in enumerate(self._inputs)}
        else:
            in_name = self._inputs[0].name if self._inputs else "input"
            inputs = {in_name: arr}
        result = _ext.model_predict(self._ptr, inputs, filter_names)
        return _dict_to_array(result, output_specs)

    def predict(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "scipy.sparse.spmatrix", Tensor, list],
    ) -> np.ndarray:
        """Thread-safe batch inference (sklearn-compatible).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Accepts numpy arrays, pandas DataFrames (including mixed-type),
            scipy sparse matrices, Python lists, or :class:`Tensor`.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` for single-output models,
            ``(n_samples, n_outputs)`` for multi-output models.
        """
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame) and _df_has_strings(X):
                pred_specs = self._pred_specs()
                col_names, col_data = _prepare_col_data(X)
                result = _ext.model_predict_columns(
                    self._ptr, col_names, col_data, [s.name for s in pred_specs])
                return _maybe_squeeze(_dict_to_array(result, pred_specs), True, 1)
        except ImportError:
            pass
        arr = _coerce(X, self.n_features_in_)
        pred_specs = self._pred_specs()
        return _maybe_squeeze(self._predict_raw(arr, pred_specs), True, 1)

    def _pred_specs(self) -> "List[OutputSpec]":
        """Return output specs for prediction (y_pred / role=Prediction)."""
        by_role = [s for s in self._outputs if s.role == OutputRole.Prediction]
        if by_role:
            return by_role
        by_name = [s for s in self._outputs if s.name in ("y_pred", "prediction")]
        return by_name or self._outputs[:1]

    def _prob_specs(self) -> "List[OutputSpec]":
        """Return output specs for probability (y_prob / role=Probability)."""
        by_role = [s for s in self._outputs if s.role == OutputRole.Probability]
        if by_role:
            return by_role
        by_name = [s for s in self._outputs if s.name in ("y_prob", "probability")]
        return by_name or self._outputs

    def predict_proba(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "scipy.sparse.spmatrix", Tensor, list],
    ) -> np.ndarray:
        """Thread-safe probability inference (sklearn-compatible).

        Returns
        -------
        np.ndarray
            Always 2-D: shape ``(n_samples, n_outputs)``.
        """
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame) and _df_has_strings(X):
                prob_specs = self._prob_specs()
                col_names, col_data = _prepare_col_data(X)
                result = _ext.model_predict_columns(
                    self._ptr, col_names, col_data, [s.name for s in prob_specs])
                r = _dict_to_array(result, prob_specs)
                return r if r.ndim == 2 else r[:, np.newaxis]
        except ImportError:
            pass
        arr = _coerce(X, self.n_features_in_)
        prob_specs = self._prob_specs()
        result = self._predict_raw(arr, prob_specs)
        return result if result.ndim == 2 else result[:, np.newaxis]

    def __repr__(self) -> str:
        return (
            f"OMLEModel(n_features_in={self.n_features_in_}, "
            f"n_outputs={self.num_outputs})"
        )


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class Session:
    """Per-thread execution context — not thread-safe.

    Owns mutable execution buffers reused across :meth:`run` calls.
    Create via :meth:`Model.create_session`.
    """

    def __init__(self, ptr: int, model: Model) -> None:
        self._ptr   = ptr
        self._model = model
        self._init_fast_path()
        self._init_col_fast_path()

    def _init_fast_path(self) -> None:
        """Cache names and pre-allocate output buffers for the single-call C fast path."""
        if self._model._is_per_feature() or not self._model._inputs or not self._model._outputs:
            self._fast_path = False
            return
        spec  = self._model._outputs[0]
        shape = spec.shape
        if len(shape) >= 2 and shape[-1] > 0:
            n_out = int(shape[-1])
        elif len(shape) <= 1:
            n_out = 1
        else:
            self._fast_path = False
            return
        self._n_out_cols = n_out
        self._in_name_b  = (self._model._inputs[0].name or "input").encode()
        self._out_name_b = (spec.name or "output").encode()
        self._out_buf: Dict[int, np.ndarray] = {}
        self._fast_path  = True

    def _init_col_fast_path(self) -> None:
        """Cache column layout for per-feature mixed-type models (e.g. adult pipeline).

        Avoids rebuilding col_names and re-checking dtypes on every predict call.
        Uses a single combined C call (bind+run+write) instead of 4 round-trips.
        Pre-registers column schema in C++ to eliminate per-call name string building.
        """
        self._col_fast_path = False
        if not self._model._is_per_feature() or not self._model._inputs or not self._model._outputs:
            return
        # Find the y_prob output (matching predict_proba logic)
        prob_outs = [s for s in self._model._outputs if s.name == "y_prob"] or list(self._model._outputs)
        if len(prob_outs) != 1:
            return
        out_spec = prob_outs[0]
        shape = out_spec.shape
        if len(shape) >= 2 and shape[-1] > 0:
            n_out = int(shape[-1])
        elif len(shape) <= 1:
            n_out = 1
        else:
            return
        self._col_str_mask: List[bool] = [s.dtype == DataType.String for s in self._model._inputs]
        self._col_names_list: List[str] = [s.name for s in self._model._inputs]
        self._col_out_name_b: bytes = (out_spec.name or "output").encode()
        self._col_n_out_cols: int = n_out
        self._col_out_buf: Dict[int, np.ndarray] = {}
        # Pre-register column schema in C++ session (eliminates per-call name building)
        col_types = [1 if is_str else 0 for is_str in self._col_str_mask]
        out_name = out_spec.name or "output"
        try:
            _ext.session_register_columns(self._ptr, self._col_names_list, col_types, out_name)
            self._use_registered = True
        except Exception:
            self._use_registered = False
        self._col_fast_path = True

    def _resolve_col_positions(self, df) -> "list | None":
        """Return per-column (block_idx, pos_in_block) tuples, cached by column tuple.

        Pandas BlockManager stores same-dtype columns in 2D blocks (shape: n_cols×n_rows).
        Accessing block.values[pos_in_block] is a view — no Series/copy overhead.
        """
        try:
            cols_key = tuple(df.columns)
        except Exception:
            return None
        if cols_key == getattr(self, "_cached_col_key", None):
            return self._cached_col_positions
        try:
            mgr = df._mgr
            blocks = mgr.blocks
            # Build col_index -> (block_idx, pos_in_block) map
            col_map: dict = {}
            for bi, b in enumerate(blocks):
                for j, loc in enumerate(b.mgr_locs):
                    col_map[int(loc)] = (bi, j)
            col_to_idx = {c: i for i, c in enumerate(df.columns)}
            positions = [col_map[col_to_idx[c]] for c in self._col_names_list]
            self._cached_col_key = cols_key
            self._cached_col_positions = positions
            return positions
        except Exception:
            return None

    def _build_col_data_fast(self, df) -> list:
        """Extract column arrays via BlockManager views — avoids per-column Series creation.

        block.values has shape (n_cols_in_block, n_rows); indexing it gives a 1D view.
        Two block accesses replace 14 Series creations, dropping ~20 µs to ~1.5 µs.
        """
        col_positions = self._resolve_col_positions(df)
        if col_positions is not None:
            try:
                block_vals = [b.values for b in df._mgr.blocks]
                return [block_vals[bi][pos] for bi, pos in col_positions]
            except Exception:
                pass
        return [df[col].values for col in self._col_names_list]

    def _fast_predict(self, arr: np.ndarray) -> np.ndarray:
        """Single-call predict: zero-copy input view, direct-write output buffer."""
        n = arr.shape[0]
        if n not in self._out_buf:
            self._out_buf[n] = np.empty((n, self._n_out_cols), dtype=np.float32)
        out = self._out_buf[n]
        _ext.session_predict_f32(self._ptr, self._in_name_b, arr, self._out_name_b, out)
        return out

    def __del__(self) -> None:
        if getattr(self, "_ptr", None):
            _ext.free_session(self._ptr)
            self._ptr = None

    @property
    def num_inputs(self) -> int:
        return _ext.session_num_inputs(self._ptr)

    @property
    def num_outputs(self) -> int:
        return _ext.session_num_outputs(self._ptr)

    def predict(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "scipy.sparse.spmatrix", Tensor, list],
    ) -> np.ndarray:
        """Single-threaded inference — sklearn-compatible, not thread-safe.

        Returns
        -------
        np.ndarray
            Shape ``(n_samples,)`` for single-output, ``(n_samples, n_outputs)``
            for multi-output.
        """
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame) and _df_has_strings(X):
                pred_specs = self._model._pred_specs()
                col_names, col_data = _prepare_col_data(X)
                _ext.session_bind_columns(self._ptr, col_names, col_data)
                _ext.session_run(self._ptr)
                cols = [_ext.tensor_read(_ext.session_get_output(self._ptr, s.name))
                        for s in pred_specs]
                result = np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]
                return _maybe_squeeze(result, True, self._model.num_outputs)
        except ImportError:
            pass
        arr = _coerce(X, self._model.n_features_in_)
        if self._fast_path:
            return _maybe_squeeze(self._fast_predict(arr), True, self._model.num_outputs)
        in_name = self._model.inputs[0].name if self._model.inputs else "input"
        pred_specs = self._model._pred_specs()
        result = _session_run_array(self._ptr, arr, pred_specs, in_name)
        return _maybe_squeeze(result, True, self._model.num_outputs)

    def predict_proba(
        self,
        X: Union[np.ndarray, "pd.DataFrame", "scipy.sparse.spmatrix", Tensor, list],
    ) -> np.ndarray:
        """Single-threaded probability inference — always returns 2-D array."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                if self._col_fast_path:
                    n = len(X)
                    if n not in self._col_out_buf:
                        self._col_out_buf[n] = np.empty(
                            (n, self._col_n_out_cols), dtype=np.float32)
                    out = self._col_out_buf[n]
                    col_data = self._build_col_data_fast(X)
                    if self._use_registered:
                        _ext.session_predict_registered_f32(self._ptr, col_data, out)
                    else:
                        _ext.session_predict_columns_f32(
                            self._ptr, self._col_names_list, col_data,
                            self._col_out_name_b, out)
                    return out if out.ndim == 2 else out[:, np.newaxis]
                if _df_has_strings(X):
                    prob_specs = self._model._prob_specs()
                    col_names, col_data = _prepare_col_data(X)
                    _ext.session_bind_columns(self._ptr, col_names, col_data)
                    _ext.session_run(self._ptr)
                    cols = [_ext.tensor_read(_ext.session_get_output(self._ptr, s.name))
                            for s in prob_specs]
                    result = np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]
                    return result if result.ndim == 2 else result[:, np.newaxis]
        except ImportError:
            pass
        arr = _coerce(X, self._model.n_features_in_)
        in_name = self._model.inputs[0].name if self._model.inputs else "input"
        prob_specs = self._model._prob_specs()
        result = _session_run_array(self._ptr, arr, prob_specs, in_name)
        return result if result.ndim == 2 else result[:, np.newaxis]

    def run(
        self,
        X: Union[np.ndarray, "pd.DataFrame", Tensor, list],
        *,
        squeeze: bool = True,
    ) -> Tensor:
        """Run inference and return a :class:`Tensor` (omle-native API).

        Parameters
        ----------
        X : array-like or Tensor
        squeeze : bool
            When True and single-output, return shape ``(n_samples,)``.
        """
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame) and _df_has_strings(X):
                col_names, col_data = _prepare_col_data(X)
                _ext.session_bind_columns(self._ptr, col_names, col_data)
                _ext.session_run(self._ptr)
                cols = [_ext.tensor_read(_ext.session_get_output(self._ptr, s.name))
                        for s in self._model.outputs]
                result = np.concatenate(cols, axis=1) if len(cols) > 1 else cols[0]
                result = _maybe_squeeze(result, squeeze, self._model.num_outputs)
                name = self._model.outputs[0].name if self._model.outputs else "output"
                t = object.__new__(Tensor)
                t._arr  = result
                t.name  = name
                t.dtype = DataType.Float32
                return t
        except ImportError:
            pass
        arr     = _coerce(X, self._model.n_features_in_)
        in_name = self._model.inputs[0].name if self._model.inputs else "input"
        result  = _session_run_array(self._ptr, arr, self._model.outputs, in_name)
        result  = _maybe_squeeze(result, squeeze, self._model.num_outputs)
        name    = self._model.outputs[0].name if self._model.outputs else "output"
        t = object.__new__(Tensor)
        t._arr  = result
        t.name  = name
        t.dtype = DataType.Float32
        return t

    def __repr__(self) -> str:
        return f"<omle_runtime.Session features={self.num_inputs} outputs={self.num_outputs}>"


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def load(path: str, *, n_threads: int = 1, min_parallel_rows: int = 64) -> Model:
    """Load an OMLE model from a .omle file.

    Parameters
    ----------
    path : str
        Path to the protobuf binary model file.
    n_threads : int
        Worker threads for ``model.predict()`` (0 = auto).
    min_parallel_rows : int
        Minimum rows per thread before parallelism activates; a batch is split
        only once it holds ``n_threads * min_parallel_rows`` rows.
    """
    return Model.load(path, n_threads=n_threads, min_parallel_rows=min_parallel_rows)


def load_bytes(data: bytes, *, n_threads: int = 1, min_parallel_rows: int = 64) -> Model:
    """Load an OMLE model from a bytes object."""
    return Model.load_bytes(data, n_threads=n_threads, min_parallel_rows=min_parallel_rows)


__all__ = [
    "Tensor",
    "Model",
    "Session",
    "InputSpec",
    "OutputSpec",
    "DataType",
    "OutputRole",
    "load",
    "load_bytes",
]
