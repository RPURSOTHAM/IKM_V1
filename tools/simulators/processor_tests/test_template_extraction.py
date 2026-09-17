#!/usr/bin/env python3
"""Test template_extraction processor — Neo4j document skeleton artifact."""

from __future__ import annotations

import argparse

from src.features.document_processing.shared_processor.types import ProcessorType
from src.simulators.processor_tests.common import (
    DEFAULT_PROCESSOR_URL,
    TestContext,
    run_processor_test,
    validate_neo4j,
)


def _validate(ctx: TestContext, result) -> list[str]:
    errors = validate_neo4j(
        document_id=ctx.document_id,
        processor_type=ctx.processor_type,
        payload_contains="sections",
    )
    if hasattr(result, "artifacts"):
        sections = int((result.artifacts or {}).get("skeleton_sections") or 0)
        if sections <= 0:
            errors.append("template_extraction produced zero skeleton sections")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "http"), default="direct")
    parser.add_argument("--base-url", default=DEFAULT_PROCESSOR_URL)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    args = parser.parse_args()
    return run_processor_test(
        ProcessorType.TEMPLATE_EXTRACTION.value,
        mode=args.mode,
        base_url=args.base_url,
        validate_fn=_validate,
        keep_artifacts=args.keep,
        wait_timeout=args.wait_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
