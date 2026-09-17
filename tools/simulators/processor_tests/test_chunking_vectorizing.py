#!/usr/bin/env python3
"""Test chunking_vectorizing processor — Weaviate indexing with local embedding model."""

from __future__ import annotations

import argparse
import sys

from src.features.document_processing.shared_processor.types import ProcessorType
from src.simulators.processor_tests.common import (
    DEFAULT_PROCESSOR_URL,
    TestContext,
    run_processor_test,
    validate_weaviate,
)


def _validate(ctx: TestContext, result) -> list[str]:
    return validate_weaviate(collection_name=ctx.collection_name, document_name=ctx.document_name, min_objects=1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "http"), default="direct")
    parser.add_argument("--base-url", default=DEFAULT_PROCESSOR_URL)
    parser.add_argument("--keep", action="store_true", help="Do not delete test artifacts")
    parser.add_argument("--wait-timeout", type=float, default=900.0)
    args = parser.parse_args()
    return run_processor_test(
        ProcessorType.CHUNKING_VECTORIZING.value,
        mode=args.mode,
        base_url=args.base_url,
        validate_fn=_validate,
        keep_artifacts=args.keep,
        wait_timeout=args.wait_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
