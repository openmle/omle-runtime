"""Pytest fixtures for the omle-spark Python integration tests.

Requires:
  - omle-spark JAR (built with `sbt package`)
  - omleruntime Python package (for native-match tests)
  - PySpark 3.x

Tests are skipped automatically when the JAR is not found.
"""

import os
import re
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
def _newest_runtime_jar():
    """The omle-runtime jar, whatever version it was built as.

    Both JVM projects take their version from the git tag now, so the filename
    is not knowable here. Sources and javadoc jars are excluded — a release
    build produces those alongside the real one.
    """
    target = _OMLE_ROOT / "java" / "target"
    jars = [
        p for p in (target.glob("omle-runtime-*.jar") if target.is_dir() else [])
        if not p.name.endswith(("-sources.jar", "-javadoc.jar"))
    ]
    # Newest by mtime, not by name: version strings do not sort lexically
    # (0.10.0 < 0.2.0), and what is wanted is simply the jar just built.
    return max(jars, key=lambda p: p.stat().st_mtime) if jars else None


_RUNTIME_JAR = _newest_runtime_jar()


def _jna_search_roots():
    """Dependency caches that may hold the JNA jar, most reliable first.

    The Maven repository comes first because the Java bindings declare JNA in
    java/pom.xml and every workflow that reaches these tests runs `mvn package`
    first, so the jar is always there at the version the pom pins. The Coursier
    caches are the sbt side of the same dependency; their location is
    platform-specific, and only globbing the macOS one is why this used to find
    nothing on Linux and leave JNA off the Spark driver classpath.
    """
    home = Path.home()
    roots = []
    if os.environ.get("COURSIER_CACHE"):
        roots.append(Path(os.environ["COURSIER_CACHE"]))
    if os.environ.get("LOCALAPPDATA"):
        roots.append(Path(os.environ["LOCALAPPDATA"]) / "Coursier" / "Cache")
    roots += [
        home / ".m2" / "repository",               # Maven (all platforms)
        home / ".cache" / "coursier",              # Coursier, Linux
        home / "Library" / "Caches" / "Coursier",  # Coursier, macOS
        home / ".ivy2",                            # sbt before Coursier
    ]
    return roots


def _jna_version_key(jar):
    """Sort key from the jar's filename, so the newest jar wins.

    A developer cache accumulates several JNA versions and plain glob order is
    filesystem order, so taking the first hit can put a years-old jna on the
    classpath. The version is read from the filename rather than the parent
    directory because Ivy's cache keeps every revision in one jars/ directory,
    which would otherwise make all candidates compare equal.
    """
    version = jar.stem.split("-", 1)[-1]        # jna-5.14.0 -> 5.14.0
    return tuple(
        int(m.group()) if (m := re.match(r"\d+", part)) else 0
        for part in version.split(".")
    )


def _find_jna_jar():
    patterns = (
        "**/jna/jna/*/jna-[0-9]*.jar",                    # Maven / Coursier
        "**/net.java.dev.jna/jna/*/jars/jna-[0-9]*.jar",  # Ivy, local repo
        "**/net.java.dev.jna/jna/jars/jna-[0-9]*.jar",    # Ivy, resolved cache
    )
    for root in _jna_search_roots():
        if not root.is_dir():
            continue
        found = [
            j for pattern in patterns for j in root.glob(pattern)
            if not j.name.endswith(("-sources.jar", "-javadoc.jar"))
        ]
        if found:
            return max(found, key=_jna_version_key)
    return None


_JNA_JAR = _find_jna_jar()

def _pyspark_scala_version():
    """Scala binary version of the installed PySpark, e.g. "2.12" or "2.13".

    PyPI ships Spark 3.x built with Scala 2.12 and Spark 4.x with Scala 2.13,
    and the two are binary-incompatible. Reading it from the bundled jar name
    is exact, where inferring it from the PySpark version number would be a
    guess that silently rots when upstream changes its build.
    """
    try:
        import pyspark
    except ImportError:
        return None
    jars = Path(pyspark.__file__).parent / "jars"
    for jar in jars.glob("spark-core_*.jar"):
        # spark-core_2.13-4.0.2.jar -> 2.13
        return jar.name.split("_", 1)[1].split("-", 1)[0]
    return None


def _find_spark_jar():
    """The omle-spark jar matching the running PySpark, and why if there isn't one.

    Returns (jar_or_None, reason_or_None). Picking the wrong one here is worse
    than finding nothing: a Scala 2.12 jar on a 2.13 Spark loads fine and then
    dies inside transform() with a NoSuchMethodError naming an internal Scala
    runtime method, which points nowhere near the actual mismatch.
    """
    target = _SPARK_ROOT / "target"
    jars = [
        p for p in (target.glob("scala-*/omle-spark_*.jar") if target.is_dir() else [])
        if not p.name.endswith(("-javadoc.jar", "-sources.jar"))
    ]
    if not jars:
        return None, "omle-spark JAR not found. Build with: cd spark && sbt +package"

    want = _pyspark_scala_version()
    if want is None:
        return jars[0], None

    # Newest by mtime: versions come from the git tag now, so target/ commonly
    # holds jars from earlier builds and glob order would pick between them at
    # random.
    matched = sorted(
        (p for p in jars if f"_{want}-" in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matched:
        have = ", ".join(sorted(p.name for p in jars))
        return None, (
            f"no omle-spark JAR for Scala {want}, which this PySpark "
            f"({'.'.join(str(v) for v in _pyspark_version())}) is built with. "
            f"Found: {have}. Build it with: cd spark && sbt +package"
        )
    return matched[0], None


def _pyspark_version():
    try:
        import pyspark
        return tuple(pyspark.__version__.split("."))
    except Exception:
        return ("unknown",)


_JAR_PATH, _JAR_SKIP_REASON = _find_spark_jar()

REGR_MODEL_PATH  = _RESOURCES / "test_model_2f.omle"
CLASS_MODEL_PATH = _RESOURCES / "test_model_3class.omle"

# ---------------------------------------------------------------------------
# Skip condition
# ---------------------------------------------------------------------------

_NO_JAR = _JAR_PATH is None or not _JAR_PATH.exists()
skip_no_jar = pytest.mark.skipif(
    _NO_JAR,
    reason=_JAR_SKIP_REASON or "omle-spark JAR not found",
)

# ---------------------------------------------------------------------------
# SparkSession
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def spark():
    """Local SparkSession with the omle-spark JAR on the classpath."""
    if _NO_JAR:
        pytest.skip(_JAR_SKIP_REASON or "omle-spark JAR not found")

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

def _import_omleruntime():
    """Import the native package, or skip — it needs a compiled extension.

    omleruntime is only importable once omle_ext has been built, which takes
    -DBUILD_PYTHON=ON. Without this the missing extension surfaces as a fixture
    ERROR carrying the package's whole module docstring, which reads like a
    failure of the code under test rather than a build that skipped a target.
    """
    import sys
    sys.path.insert(0, str(_NATIVE_LIB.parent))
    try:
        import omleruntime as omr
    except ImportError as exc:
        pytest.skip(
            "omleruntime is not importable, so the Spark output cannot be "
            f"compared against the native runtime ({exc}). Build it with: "
            "cmake -S . -B build -DBUILD_PYTHON=ON && "
            "cmake --build build --target omle_ext"
        )
    return omr


@pytest.fixture(scope="session")
def native_regr_model(regr_model_path):
    """Loaded omleruntime Model for the regression fixture."""
    return _import_omleruntime().load(regr_model_path)


@pytest.fixture(scope="session")
def native_class_model(class_model_path):
    """Loaded omleruntime Model for the 3-class classifier fixture."""
    return _import_omleruntime().load(class_model_path)
