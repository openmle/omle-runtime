#!/usr/bin/env python3
"""Remove previously built extension modules from the Python package directory.

CMake writes omle_ext into python/omle_runtime/ (the source tree, not the build
tree), because the in-place developer workflow and the JVM bindings both expect
the library and the extension to live there. scikit-build-core's
wheel.packages then copies that directory verbatim into the wheel.

That is fine for one build and wrong for several. cibuildwheel builds cp310,
cp311, ... one after another in the same checkout, and each leaves behind an
extension module with its own ABI tag, so the second wheel ships two extensions,
the third ships three, and the last ships one for every interpreter — modules it
cannot load, inflating every wheel after the first.

Run from cibuildwheel's before-build hook, once per wheel.
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "python" / "omle_runtime"

# Only the extension modules. The shared library is left alone: it has one name
# per platform, every interpreter overwrites the same file, and the wheel needs
# it.
PATTERNS = ("omle_ext*.so", "omle_ext*.pyd", "omle_ext*.dylib")


def main() -> int:
    if not PACKAGE_DIR.is_dir():
        print(f"clean_ext: nothing to do, {PACKAGE_DIR} does not exist")
        return 0

    removed = []
    for pattern in PATTERNS:
        for path in PACKAGE_DIR.glob(pattern):
            path.unlink()
            removed.append(path.name)

    if removed:
        print(f"clean_ext: removed {len(removed)} stale extension module(s):")
        for name in sorted(removed):
            print(f"  {name}")
    else:
        print("clean_ext: no stale extension modules")
    return 0


if __name__ == "__main__":
    sys.exit(main())
