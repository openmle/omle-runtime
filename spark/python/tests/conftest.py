"""Pytest fixtures for the omle-spark Python integration tests.

Requires:
  - omle-spark JAR (built with `sbt package`)
  - omleruntime Python package (for native-match tests)
  - PySpark 3.x

Tests are skipped automatically when the JAR is not found.
"""

import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE         = Path(__file__).parent
_SPARK_ROOT   = _HERE.parent.parent            # omle-runtime/spark/
_OMLE_ROOT    = _SPARK_ROOT.parent             # omle-runtime/
_RESOURCES    = _SPARK_ROOT / "src" / "test" / "resources"
_NATIVE_LIB   = _OMLE_ROOT / "python" / "omleruntime"
_RUNTIME_JAR  = _OMLE_ROOT / "java" / "target" / "omle-runtime-0.1.0.jar"
_JNA_JAR_GLOB = (
    list(Path.home().glob("Library/Caches/Coursier/**/jna/jna/*/jna-[0-9]*.jar")) or
    list(Path.home().glob(".ivy2/**/net.java.dev.jna/jna/*/jars/jna-*.jar"))
)
_JNA_JAR      = _JNA_JAR_GLOB[0] if _JNA_JAR_GLOB else None

_JAR_GLOB   = [
    p for p in ((_SPARK_ROOT / "target").glob("scala-*/omle-spark_*.jar")
                if (_SPARK_ROOT / "target").exists() else [])
    if not p.name.endswith(("-javadoc.jar", "-sources.jar"))
]
_JAR_PATH   = _JAR_GLOB[0] if _JAR_GLOB else None

REGR_MODEL_PATH  = _RESOURCES / "test_model_2f.omle"
CLASS_MODEL_PATH = _RESOURCES / "test_model_3class.omle"

# ---------------------------------------------------------------------------
# Skip condition
# ---------------------------------------------------------------------------

_NO_JAR = _JAR_PATH is None or not _JAR_PATH.exists()
skip_no_jar = pytest.mark.skipif(
    _NO_JAR,
    reason=(
        "omle-spark JAR not found. "
        "Build with: cd spark && sbt package"
    ),
)

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Local SparkSession with the omle-spark JAR on the classpath."""
    if _NO_JAR:
        pytest.skip("omle-spark JAR not found")

    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder
        .master("local[2]")
        .appName("omle-spark-python-tests")
        .config("spark.jars", ",".join(str(j) for j in [_JAR_PATH, _RUNTIME_JAR, _JNA_JAR] if j))
        .config("spark.driver.extraJavaOptions",
                f"-Djna.library.path={_NATIVE_LIB}"
                " --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
                " --add-opens=java.base/java.nio=ALL-UNNAMED"
                " --add-opens=java.base/java.lang=ALL-UNNAMED"
                " --add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
                " --add-opens=java.base/java.util=ALL-UNNAMED")
        .config("spark.executor.extraJavaOptions",
                f"-Djna.library.path={_NATIVE_LIB}"
                " --add-opens=java.base/sun.nio.ch=ALL-UNNAMED"
                " --add-opens=java.base/java.nio=ALL-UNNAMED"
                " --add-opens=java.base/java.lang=ALL-UNNAMED"
                " --add-opens=java.base/java.lang.invoke=ALL-UNNAMED"
                " --add-opens=java.base/java.util=ALL-UNNAMED")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
    )
    session = builder.getOrCreate()
    yield session
    session.stop()


# ---------------------------------------------------------------------------
# Model file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def regr_model_path():
    """Path to the 2-feature regression model (.omle)."""
    assert REGR_MODEL_PATH.exists(), f"Missing fixture: {REGR_MODEL_PATH}"
    return str(REGR_MODEL_PATH)


@pytest.fixture(scope="session")
def class_model_path():
    """Path to the 3-class classifier model (.omle)."""
    assert CLASS_MODEL_PATH.exists(), f"Missing fixture: {CLASS_MODEL_PATH}"
    return str(CLASS_MODEL_PATH)


# ---------------------------------------------------------------------------
# Native omleruntime models (for match tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def native_regr_model(regr_model_path):
    """Loaded omleruntime Model for the regression fixture."""
    import sys
    sys.path.insert(0, str(_NATIVE_LIB.parent))
    import omleruntime as omr
    return omr.load(regr_model_path)


@pytest.fixture(scope="session")
def native_class_model(class_model_path):
    """Loaded omleruntime Model for the 3-class classifier fixture."""
    import sys
    sys.path.insert(0, str(_NATIVE_LIB.parent))
    import omleruntime as omr
    return omr.load(class_model_path)
