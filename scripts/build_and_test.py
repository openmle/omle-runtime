#!/usr/bin/env python3
"""Build and test all omle-runtime components.

Components (in dependency order):
  cpp      — C++ static library + unit tests  (cmake)
  python   — Python shared library (libomleruntime.dylib) + Python tests
  java     — Java bindings jar + JUnit tests  (maven)
  spark    — Spark Scala jar + ScalaTest      (sbt)
  spark-py — PySpark Python package + pytest

Usage:
  python scripts/build_and_test.py                 # all components
  python scripts/build_and_test.py cpp python      # selected
  python scripts/build_and_test.py --no-test       # build only, skip tests
  python scripts/build_and_test.py --build-type Debug
"""

import argparse
import os
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
BOLD  = "\033[1m"
GREEN = "\033[32m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"


def header(msg: str) -> None:
    width = 70
    print(f"\n{BOLD}{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {msg}{RESET}")
    print(f"{BOLD}{CYAN}{'─' * width}{RESET}\n")


def run(cmd: list, cwd: Optional[Path] = None, env: Optional[dict] = None) -> None:
    display = " ".join(str(c) for c in cmd)
    print(f"  $ {display}")
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, cwd=cwd, env=merged_env)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {display}")


def find_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            f"'{name}' not found on PATH. Please install it and retry."
        )
    return path


# ---------------------------------------------------------------------------
# Component builders
# ---------------------------------------------------------------------------

def build_cpp(build_type: str, run_tests: bool) -> None:
    """Build C++ static library and optionally run ctest."""
    header("C++ — cmake build + ctest")

    cmake   = find_executable("cmake")
    build_dir = ROOT / "build"

    run([cmake, ROOT,
         f"-DCMAKE_BUILD_TYPE={build_type}",
         "-DBUILD_TESTS=ON",
         "-DBUILD_PYTHON=OFF",
         "-B", build_dir])
    run([cmake, "--build", build_dir, "--parallel"])

    if run_tests:
        run([find_executable("ctest"), "--output-on-failure"], cwd=build_dir)


def build_python(build_type: str, run_tests: bool) -> None:
    """Build libomleruntime shared library and run Python pytest."""
    header("Python — cmake shared library + pytest")

    cmake     = find_executable("cmake")
    build_dir = ROOT / "build-python"

    run([cmake, ROOT,
         f"-DCMAKE_BUILD_TYPE={build_type}",
         "-DBUILD_PYTHON=ON",
         "-DBUILD_TESTS=ON",
         "-B", build_dir])
    run([cmake, "--build", build_dir, "--parallel"])

    if run_tests:
        python = sys.executable
        omle_src = ROOT.parent / "omle" / "src"
        env = {"PYTHONPATH": str(omle_src)}
        run([python, "-m", "pytest", "tests/", "-v"],
            cwd=ROOT / "python",
            env=env)


def build_java(run_tests: bool) -> None:
    """Build Java jar with Maven and optionally run JUnit tests."""
    header("Java — mvn package / test")

    mvn = find_executable("mvn")
    goal = "test" if run_tests else "package"

    # Java tests load the shared library via JNA; point it at the built dylib.
    lib_dir = ROOT / "python" / "omle_runtime"
    run([mvn, goal,
         f"-Djna.library.path={lib_dir}",
         "--no-transfer-progress"],
        cwd=ROOT / "java")


def build_spark(run_tests: bool) -> None:
    """Build Spark Scala jar with sbt and optionally run ScalaTest."""
    header("Spark Scala — sbt package / test")

    # Java jar must exist first (sbt unmanagedJars references it).
    # Version comes from the git tag, so match by prefix rather than by name.
    _jars = [
        p for p in (ROOT / "java" / "target").glob("omle-runtime-*.jar")
        if not p.name.endswith(("-sources.jar", "-javadoc.jar"))
    ]
    java_jar = (max(_jars, key=lambda p: p.stat().st_mtime) if _jars
                else ROOT / "java" / "target" / "omle-runtime.jar")
    if not java_jar.exists():
        print(f"  Java jar not found at {java_jar}; building Java first.")
        build_java(run_tests=False)

    sbt = find_executable("sbt")
    lib_dir = ROOT / "python" / "omle_runtime"
    task = "test" if run_tests else "package"

    run([sbt,
         f"-Djna.library.path={lib_dir}",
         task],
        cwd=ROOT / "spark")


def build_spark_python(run_tests: bool) -> None:
    """Install PySpark Python package in dev mode and run pytest."""
    header("Spark Python — pip install + pytest")

    python    = sys.executable
    spark_py  = ROOT / "spark" / "python"
    omle_src = ROOT.parent / "omle" / "src"

    run([python, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=spark_py)

    if run_tests:
        env = {"PYTHONPATH": str(omle_src)}
        run([python, "-m", "pytest", "tests/", "-v"],
            cwd=spark_py,
            env=env)


# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

ALL_COMPONENTS = ["cpp", "python", "java", "spark", "spark-py"]

BUILDERS = {
    "cpp":      lambda bt, t: build_cpp(bt, t),
    "python":   lambda bt, t: build_python(bt, t),
    "java":     lambda bt, t: build_java(t),
    "spark":    lambda bt, t: build_spark(t),
    "spark-py": lambda bt, t: build_spark_python(t),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "components",
        nargs="*",
        choices=ALL_COMPONENTS + [[]],
        default=[],
        metavar="COMPONENT",
        help=f"Components to build: {', '.join(ALL_COMPONENTS)} (default: all)",
    )
    parser.add_argument(
        "--no-test",
        action="store_true",
        help="Build without running tests",
    )
    parser.add_argument(
        "--build-type",
        default="Release",
        choices=["Release", "Debug", "RelWithDebInfo"],
        help="CMake build type (default: Release)",
    )
    args = parser.parse_args()

    components = args.components or ALL_COMPONENTS
    run_tests  = not args.no_test

    results: dict[str, tuple[str, float]] = {}

    for name in components:
        t0 = time.monotonic()
        try:
            BUILDERS[name](args.build_type, run_tests)
            results[name] = ("ok", time.monotonic() - t0)
        except Exception as exc:
            results[name] = ("fail", time.monotonic() - t0)
            print(f"\n{RED}ERROR in '{name}': {exc}{RESET}\n", file=sys.stderr)

    # Summary
    width = 70
    print(f"\n{BOLD}{'─' * width}{RESET}")
    print(f"{BOLD}  Build summary{RESET}")
    print(f"{BOLD}{'─' * width}{RESET}")
    all_ok = True
    for name in components:
        status, elapsed = results[name]
        icon   = f"{GREEN}✓{RESET}" if status == "ok" else f"{RED}✗{RESET}"
        label  = f"{GREEN}PASSED{RESET}" if status == "ok" else f"{RED}FAILED{RESET}"
        print(f"  {icon}  {name:<12} {label}  ({elapsed:.1f}s)")
        if status != "ok":
            all_ok = False
    print(f"{BOLD}{'─' * width}{RESET}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
