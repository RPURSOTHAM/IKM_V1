"""CLI for standalone template extraction testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    """Allow ``python -m template_compliance`` from document_optimizer/."""
    here = Path(__file__).resolve().parent
    parent = here.parent
    for candidate in (str(parent), str(here)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="template_compliance",
        description="Extract DOCX (and later PDF) templates into shared JSON.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="Extract one template file to JSON")
    extract.add_argument("path", type=Path, help="Path to .docx (or .pdf stub)")
    extract.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write JSON to this path (default: template_compliance/out/<stem>_template.json)",
    )
    extract.add_argument(
        "--pretty",
        action="store_true",
        help="Also print JSON to stdout",
    )
    extract.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent (default: 2)",
    )

    sub.add_parser("schema", help="Print schema_version and top-level keys")
    return parser


def main(argv: list[str] | None = None) -> int:
    _ensure_import_path()
    from .api import extract_template_to_json
    from .exceptions import TemplateComplianceError
    from .schema import SCHEMA_VERSION, empty_template_document

    args = build_parser().parse_args(argv)

    if args.command == "schema":
        doc = empty_template_document()
        print(json.dumps({"schema_version": SCHEMA_VERSION, "keys": list(doc.keys())}, indent=2))
        return 0

    if args.command == "extract":
        try:
            output = args.output
            if output is None:
                out_dir = Path(__file__).resolve().parent / "out"
                out_dir.mkdir(parents=True, exist_ok=True)
                output = out_dir / f"{args.path.stem}_template.json"
            payload = extract_template_to_json(args.path, output, indent=args.indent)
            document = payload.get("document") if isinstance(payload.get("document"), dict) else payload
            print(f"Wrote {output}")
            sig = document.get("signature_table") or {}
            print(
                "summary:",
                f"sections={len(document.get('sections') or [])}",
                f"total_tables={int(document.get('total_tables') or 0)}",
                f"signature={bool(sig.get('columns') and sig.get('rows'))}",
                f"total_images={int(document.get('total_images') or 0)}",
                f"toc={bool((document.get('toc') or {}).get('present'))}",
                f"logo={bool((document.get('logo') or {}).get('present'))}",
            )
            if args.pretty:
                print(json.dumps(payload, ensure_ascii=False, indent=args.indent))
            return 0
        except TemplateComplianceError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # pragma: no cover
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
