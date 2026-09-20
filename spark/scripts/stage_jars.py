#!/usr/bin/env python3
"""Copy the built JARs into the omle-spark Python package before the wheel build.

The wheel ships four JARs so that `pip install omle-spark` is sufficient on a
real cluster:

  omle-spark_2.12-<v>.jar   Scala transformer for PySpark 3.x
  omle-spark_2.13-<v>.jar   Scala transformer for PySpark 4.x
  omle-runtime-<v>.jar      Java bindings + native libomleruntime per platform
  jna-<v>.jar               JNA, which omle-runtime loads the native lib through

JNA has to be shipped explicitly. It is a declared Maven dependency of
omle-runtime, but `spark.jars` puts bare JARs on the classpath and resolves no
transitive dependencies, so without it the first call fails with
`NoClassDefFoundError: com/sun/jna/Library`.

Both Scala builds are required. They are binary-incompatible and PyPI ships
Spark 3.x built against 2.12 and 4.x against 2.13, so a wheel carrying only one
silently fails on the other with `'JavaPackage' object is not callable`.

The omle-runtime JAR must be the one release.yml assembles, with the native
libraries staged under JNA's resource prefixes (linux-x86-64, darwin-aarch64,
...). JNA then extracts the right one from the classpath on each executor,
which is what removes the -Djna.library.path requirement. A JAR built by a
plain `mvn package` has no natives in it and will load on the driver only if
the library happens to be findable there; --require-natives fails the build
rather than shipping that.

Run from anywhere:

    python spark/scripts/stage_jars.py [--require-natives]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
SPARK_TARGET = REPO / "spark" / "target"
JAVA_TARGET = REPO / "java" / "target"
DEST = REPO / "spark" / "python" / "omle_spark" / "jars"

# The directory names JNA looks under inside a JAR; see release.yml, which
# stages the libraries at exactly these paths.
JNA_PREFIXES = ("linux-x86-64", "linux-aarch64", "darwin-aarch64", "win32-x86-64")

# sbt and Maven both leave sources/javadoc artifacts next to the real JAR.
_AUX = ("-sources.jar", "-javadoc.jar")


def _newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda p: p.stat().st_mtime)


def _find_spark_jars() -> dict[str, Path]:
    """One omle-spark JAR per Scala version, newest wins."""
    found: dict[str, Path] = {}
    if not SPARK_TARGET.is_dir():
        return found
    for jar in SPARK_TARGET.glob("scala-*/omle-spark_*.jar"):
        if jar.name.endswith(_AUX):
            continue
        scala = jar.name.split("_", 1)[1].split("-", 1)[0]
        found.setdefault(scala, []).append(jar)  # type: ignore[arg-type]
    return {k: _newest(v) for k, v in found.items()}  # type: ignore[arg-type]


def _find_jna_jar() -> Path | None:
    """The JNA jar Maven resolved for the Java build.

    `mvn dependency:copy-dependencies -DoutputDirectory=target/deps` puts it
    there; falling back to the local repository keeps a plain `mvn package`
    checkout working for development.
    """
    for cand in (JAVA_TARGET / "deps", JAVA_TARGET):
        if cand.is_dir():
            jars = [p for p in cand.glob("jna-*.jar")
                    if not p.name.startswith("jna-platform")
                    and not p.name.endswith(_AUX)]
            if jars:
                return _newest(jars)
    return None


def _find_runtime_jar() -> Path | None:
    if not JAVA_TARGET.is_dir():
        return None
    jars = [p for p in JAVA_TARGET.glob("omle-runtime-*.jar")
            if not p.name.endswith(_AUX)]
    return _newest(jars) if jars else None


def _natives_in(jar: Path) -> list[str]:
    with zipfile.ZipFile(jar) as z:
        names = z.namelist()
    return [n for n in names
            if n.startswith(JNA_PREFIXES) and n.rsplit("/", 1)[-1]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--require-natives", action="store_true",
        help="fail unless the omle-runtime JAR carries a native library for "
             "every platform (what a published wheel needs)",
    )
    args = ap.parse_args()

    spark_jars = _find_spark_jars()
    missing = [v for v in ("2.12", "2.13") if v not in spark_jars]
    if missing:
        print(f"::error::no omle-spark JAR for Scala {', '.join(missing)} — "
              f"run `sbt +package` in {REPO / 'spark'}", file=sys.stderr)
        return 1

    runtime = _find_runtime_jar()
    if runtime is None:
        print(f"::error::no omle-runtime JAR in {JAVA_TARGET} — run "
              "`mvn package` in java/, or download the release artifact",
              file=sys.stderr)
        return 1

    jna = _find_jna_jar()
    if jna is None:
        print("::error::no JNA jar found. Run, in java/:\n"
              "  mvn -q dependency:copy-dependencies -DoutputDirectory=target/deps\n"
              "Without it the JVM side fails with "
              "NoClassDefFoundError: com/sun/jna/Library.", file=sys.stderr)
        return 1

    natives = _natives_in(runtime)
    if natives:
        print(f"{runtime.name} carries {len(natives)} native librar"
              f"{'y' if len(natives) == 1 else 'ies'}:")
        for n in sorted(natives):
            print(f"    {n}")
    if args.require_natives:
        have = {n.split("/", 1)[0] for n in natives}
        absent = [p for p in JNA_PREFIXES if p not in have]
        if absent:
            print(f"::error::{runtime.name} has no native library for: "
                  f"{', '.join(absent)}. A wheel built from this would work "
                  "only where libomleruntime is already installed. Use the "
                  "JAR from release.yml's `jar` job.", file=sys.stderr)
            return 1

    # Wipe first: stale JARs from an earlier version would otherwise ship
    # alongside the current ones, and jars() would put both on the classpath.
    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)

    for jar in list(spark_jars.values()) + [runtime, jna]:
        shutil.copy2(jar, DEST / jar.name)
        print(f"staged {jar.name}  ({jar.stat().st_size / 1024:.0f} KB)")

    total = sum(p.stat().st_size for p in DEST.glob("*.jar"))
    print(f"\n{DEST.relative_to(REPO)}: {len(list(DEST.glob('*.jar')))} JARs, "
          f"{total / 1048576:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
