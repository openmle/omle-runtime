#!/usr/bin/env python3
"""Reject compiler-specific constructs that MSVC cannot parse.

We have no Windows machine in the normal development loop, so MSVC-only
breakage is otherwise found by pushing to CI and waiting. Every construct
listed here has already cost at least one red Windows build:

  __restrict__      MSVC spells it __restrict, and rejects the GNU form with a
                    bare syntax error that then derails the rest of the parse --
                    which is why a single __restrict__ in post_transform.h
                    produced a wall of unrelated errors in simd_traits.h.
  __builtin_memcpy  no MSVC equivalent (C3861).
  __builtin_expect  no MSVC equivalent.
  [[unlikely]]      C++20; MSVC warns C5051 at every use under /std:c++17.

The replacements all live in include/omle/port.h, which is exempt because it is
where the per-compiler spellings are defined.

Usage:  python3 scripts/check_portability.py [root]
Exit code 1 if anything is found, so it can gate CI.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# (pattern, what to use instead). Patterns are regexes.
RULES: list[tuple[str, str]] = [
    (r"__restrict__", "OMLE_RESTRICT (include/omle/port.h)"),
    (r"__builtin_expect\b", "OMLE_EXPECT_TRUE / OMLE_EXPECT_FALSE"),
    (r"__builtin_memcpy\b", "std::memcpy"),
    (r"__builtin_(?!expect\b|memcpy\b)[a-z_0-9]+", "a portable equivalent"),
    (r"\[\[(?:un)?likely\]\]", "OMLE_UNLIKELY"),
    (r"__attribute__\s*\(", "a portable equivalent, or guard it on __GNUC__"),
    (r"__PRETTY_FUNCTION__", "__func__"),
    (r"^\s*#\s*warning\b", "a comment, or #pragma message"),
]

# port.h defines the portable spellings, so it necessarily names the raw ones.
EXEMPT = {"include/omle/port.h"}

# Whole-file checks that a single-line regex cannot express.
def check_windows_h(rel: str, text: str) -> list[str]:
    """<windows.h> defines min/max as macros unless NOMINMAX precedes it.

    Every std::min / std::max the translation unit can see then fails to parse
    with C2589, including ones in headers it merely includes.
    """
    lines = text.splitlines()
    inc = re.compile(r"^\s*#\s*include\s*<windows\.h>")
    dfn = re.compile(r"^\s*#\s*define\s+NOMINMAX\b")
    # Both must be matched as directives: this file's own comments mention
    # <windows.h> and NOMINMAX, and neither should satisfy the check.
    line = next((i for i, l in enumerate(lines, 1) if inc.match(l)), 0)
    if not line:
        return []
    if any(dfn.match(l) for l in lines[: line - 1]):
        return []
    return [f"{rel}:{line}: <windows.h> included without a preceding "
            f"#define NOMINMAX -- it defines min/max as macros and breaks "
            f"std::min / std::max"]


def check_gtest_fixture_names(rel: str, text: str) -> list[str]:
    """A fixture member named T (or U/V) collides with gtest internals.

    testing::internal::SuiteApiResolver<T> inherits the fixture, and MSVC's
    lookup then resolves the template parameter to the inherited member,
    applying sizeof to a function type (C2070).
    """
    if "TEST_F" not in text and "::testing::Test" not in text:
        return []
    out = []
    for i, line in enumerate(text.splitlines(), 1):
        m = re.match(r"\s+[A-Za-z_][A-Za-z_0-9:<>, ]*\s+([TUV])\s*\(", line)
        if m:
            out.append(f"{rel}:{i}: test fixture member named "
                       f"{m.group(1)!r} shadows gtest's template parameter "
                       f"-- give it a longer name")
    return out


WHOLE_FILE_CHECKS = [check_windows_h, check_gtest_fixture_names]

SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cxx"}
SKIP_DIRS = {"build", "build-asan", "bdbg", "bx86", ".git", "target", "third_party"}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    compiled = [(re.compile(p, re.MULTILINE), hint) for p, hint in RULES]

    findings: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.suffix not in SUFFIXES or not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        rel = path.relative_to(root).as_posix()
        if rel in EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for check in WHOLE_FILE_CHECKS:
            findings.extend(check(rel, text))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern, hint in compiled:
                m = pattern.search(line)
                if m:
                    findings.append(
                        f"{rel}:{lineno}: {m.group(0)!r} is not portable to MSVC "
                        f"-- use {hint}"
                    )

    if findings:
        print("Non-portable constructs found:\n", file=sys.stderr)
        for f in findings:
            print("  " + f, file=sys.stderr)
        print(
            f"\n{len(findings)} issue(s). These break the Windows build; the "
            f"portable spellings are in include/omle/port.h.",
            file=sys.stderr,
        )
        return 1

    print("portability check: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
