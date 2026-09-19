# omle-runtime

Fast inference for classical ML models, with a scikit-learn-style API.

`omle-runtime` loads `.omle` models and scores them through a native C++ runtime
with pybind11 bindings. Wheels are self-contained — protobuf and Abseil are
linked in statically, so there is nothing to install alongside.

```python
import numpy as np
import omle_runtime as omr

model = omr.load("model.omle", n_threads=4)   # thread-safe
X = np.random.randn(1000, 10).astype(np.float32)

scores = model.predict(X)         # (n_samples,)
proba  = model.predict_proba(X)   # (n_samples, n_outputs)
```

`Model` implements the scikit-learn estimator interface (`predict`,
`predict_proba`, `fit`, `get_params`, `set_params`, `n_features_in_`,
`feature_names_in_`), so it can be the final step of a `Pipeline`:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
pipe.predict(X)
```

For repeated scoring on a thread, create a session once and reuse it:

```python
session = model.create_session()   # one per thread
scores = session.predict(X)
```

## Related packages

- [`omle`](https://pypi.org/project/omle/) — the model IR and converters
- [`omle-spark`](https://pypi.org/project/omle-spark/) — PySpark transformer

## License

Apache-2.0
