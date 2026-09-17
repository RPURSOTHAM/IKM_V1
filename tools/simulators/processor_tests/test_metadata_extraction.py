#!/usr/bin/env python3
"""Test metadata_extraction processor — Neo4j artifact with extracted fields."""

from __future__ import annotations

import argparse
import sys

from src.features.document_processing.shared_processor.types import ProcessorType
from src.simulators.processor_tests.common import (
    DEFAULT_PROCESSOR_URL,
    TestContext,
    run_processor_test,
    validate_neo4j,
)


METADATA_FIELDS = [
    {"field_name": "Equipment_Name", "field_label": "Equipment Name", "data_type": "string"},
    {"field_name": "Document_Number", "field_label": "Document Number", "data_type": "string"},
]


def _validate(ctx: TestContext, result) -> list[str]:
    errors = validate_neo4j(
        document_id=ctx.document_id,
        processor_type=ctx.processor_type,
        payload_contains="Centrifuge",
    )
    if hasattr(result, "artifacts"):
        fields = (result.artifacts or {}).get("extracted_fields") or []
        if not any(f.get("field_name") == "Equipment_Name" and f.get("value") for f in fields):
            errors.append("Equipment_Name was not extracted from fixture")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "http"), default="direct")
    parser.add_argument("--base-url", default=DEFAULT_PROCESSOR_URL)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--wait-timeout", type=float, default=300.0)
    args = parser.parse_args()
    return run_processor_test(
        ProcessorType.METADATA_EXTRACTION.value,
        mode=args.mode,
        base_url=args.base_url,
        validate_fn=_validate,
        metadata_fields=METADATA_FIELDS,
        keep_artifacts=args.keep,
        wait_timeout=args.wait_timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
