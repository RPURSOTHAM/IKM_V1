#!/usr/bin/env python3
"""Test intelligence_extraction processor — secure intelligence extraction."""

from __future__ import annotations

import argparse

from src.features.document_processing.shared_processor.types import ProcessorType
from src.simulators.processor_tests.common import DEFAULT_PROCESSOR_URL, run_processor_test


def _validate(ctx, result) -> list[str]:
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "http"), default="direct")
    parser.add_argument("--base-url", default=DEFAULT_PROCESSOR_URL)
    args = parser.parse_args()
    return run_processor_test(
        ProcessorType.INTELLIGENCE_EXTRACTION.value,
        mode=args.mode,
        base_url=args.base_url,
        validate_fn=_validate,
        expect_failure=False,
        keep_artifacts=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
