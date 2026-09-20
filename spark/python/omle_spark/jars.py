"""Locate the JARs bundled with this package.

``pip install omle-spark`` ships everything the JVM side needs: the Scala
transformer, cross-built for both Scala versions PySpark uses, and the
omle-runtime JAR, which carries the native ``libomleruntime`` for every
supported platform at JNA's own resource paths. Nothing has to be built with
``sbt``, and no ``-Djna.library.path`` is needed on driver or executors —
``spark.jars`` ships the JARs to the executors and JNA extracts the right
native library from the classpath on each one.

JARs cannot be added to a JVM that is already running, so they have to be named
when the session is built:

    import omle_spark
    spark = (SparkSession.builder
             .config("spark.jars", omle_spark.jars_classpath())
             .getOrCreate())

Calling this after ``getOrCreate()`` has no effect; the transform would then
fail with ``'JavaPackage' object is not callable``, which is what PySpark
reports for any missing class and says nothing about the cause.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

__all__ = ["jars", "jars_classpath", "scala_version"]

_JAR_DIR = Path(__file__).resolve().parent / "jars"


def scala_version() -> Optional[str]:
    """Scala binary version the installed PySpark was built with.

    Read from the name of PySpark's own ``spark-core`` JAR rather than inferred
    from the PySpark version. PyPI ships Spark 3.x built with Scala 2.12 and
    4.x with 2.13; the two are binary-incompatible, and loading the wrong one
    does not fail at load time — the class is simply absent.

    Returns None when PySpark is not installed or its layout is unrecognised.
    """
    try:
        import pyspark
    except ImportError:
        return None
    jar_dir = Path(pyspark.__file__).parent / "jars"
    for jar in jar_dir.glob("spark-core_*.jar"):
        return jar.name.split("_", 1)[1].split("-", 1)[0]
    return None


def jars() -> List[str]:
    """Absolute paths to the JARs this package ships, for the running PySpark.

    Returns the omle-spark JAR matching PySpark's Scala version, plus the
    omle-runtime JAR and JNA. Raises if the wheel carries no JAR for that Scala
    version, which is a packaging fault rather than a user error — a wheel
    built correctly holds both.
    """
    if not _JAR_DIR.is_dir():
        raise RuntimeError(
            f"omle-spark ships no JARs: {_JAR_DIR} is missing. This wheel was "
            "built without stage_jars.py having run; reinstall from PyPI or "
            "see spark/python/README.md for building from source."
        )

    runtime = sorted(_JAR_DIR.glob("omle-runtime-*.jar"))
    spark_all = sorted(_JAR_DIR.glob("omle-spark_*.jar"))
    # JNA is a transitive Maven dependency of omle-runtime, but `spark.jars`
    # resolves nothing transitively, so it ships and is listed explicitly.
    # Without it the first call fails with NoClassDefFoundError:
    # com/sun/jna/Library.
    jna = sorted(p for p in _JAR_DIR.glob("jna-*.jar")
                 if not p.name.startswith("jna-platform"))

    want = scala_version()
    if want is None:
        raise RuntimeError(
            "Cannot determine PySpark's Scala version — is pyspark installed? "
            "omle-spark ships one JAR per Scala version and has to pick the "
            "matching one."
        )

    matched = [p for p in spark_all if f"_{want}-" in p.name]
    if not matched:
        have = ", ".join(sorted({p.name.split("_")[1].split("-")[0]
                                 for p in spark_all})) or "none"
        raise RuntimeError(
            f"omle-spark has no JAR for Scala {want} (PySpark's version); "
            f"this wheel carries: {have}."
        )

    return [str(p) for p in matched + runtime + jna]


def jars_classpath() -> str:
    """`jars()` joined for `spark.jars`, which takes a comma-separated list."""
    return ",".join(jars())
