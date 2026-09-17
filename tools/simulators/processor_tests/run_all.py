#!/usr/bin/env python3
"""Run all individual processor integration tests."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TESTS = [
    "test_chunking_vectorizing.py",
    "test_metadata_extraction.py",
    "test_template_extraction.py",
    "test_reference_extraction.py",
    "test_conversion_rendering.py",
    "test_intelligence_extraction.py",
]

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "http", "auto"), default="auto")
    parser.add_argument("--base-url", default="http://localhost:3100")
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--only", nargs="*", help="Run subset, e.g. chunking metadata")
    args = parser.parse_args()

    selected = TESTS
    if args.only:
        tokens = {t.lower().replace("-", "_") for t in args.only}
        selected = [name for name in TESTS if any(tok in name for tok in tokens)]
        if not selected:
            print("No tests matched --only filter", file=sys.stderr)
            return 2

    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    failures = 0
    for script in selected:
        path = TESTS_DIR / script
        mode = args.mode
        if mode == "auto":
            mode = "http" if "chunking_vectorizing" in script else "direct"
        cmd = [
            sys.executable,
            str(path),
            "--mode",
            mode,
            "--base-url",
            args.base_url,
        ]
        if args.keep:
            cmd.append("--keep")
        print(f"\n=== Running {script} ({mode}) ===", flush=True)
        proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
        if proc.returncode != 0:
            failures += 1

    print(f"\n=== Summary: {len(selected) - failures}/{len(selected)} passed ===")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
