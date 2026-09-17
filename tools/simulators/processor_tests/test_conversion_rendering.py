#!/usr/bin/env python3
"""Test conversion_for_rendering processor — Redis HTML render cache."""

from __future__ import annotations

import argparse

from src.features.document_processing.shared_processor.types import ProcessorType
from src.simulators.processor_tests.common import (
    DEFAULT_PROCESSOR_URL,
    TestContext,
    run_processor_test,
    validate_redis,
)


def _validate(ctx: TestContext, result) -> list[str]:
    errors = validate_redis(document_id=ctx.document_id)
    if hasattr(result, "artifacts"):
        blocks = int((result.artifacts or {}).get("block_count") or 0)
        if blocks <= 0:
            errors.append("conversion_for_rendering loaded zero document blocks")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "http"), default="direct")
    parser.add_argument("--base-url", default=DEFAULT_PROCESSOR_URL)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    args = parser.parse_args()
    return run_processor_test(
        ProcessorType.CONVERSION_FOR_RENDERING.value,
        mode=args.mode,
        base_url=args.base_url,
        validate_fn=_validate,
        keep_artifacts=args.keep,
        wait_timeout=args.wait_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
