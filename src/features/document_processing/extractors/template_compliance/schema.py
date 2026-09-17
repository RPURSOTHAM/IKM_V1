"""Common JSON schema for DOCX and PDF template extraction."""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "1.0"


def empty_template_document(
    *,
    source: str = "",
    source_format: str = "",
    document_name: str = "",
) -> dict[str, Any]:
    """Return a blank document matching the shared extract schema."""
    return {
        "document": {
            "file_name": document_name or "",
            "file_type": source_format or "",
            "metadata": {
                "document_type": "",
                "title": "",
                "document_no": "",
                "version_no": "",
                "effective_date": "",
                "review_date": "",
            },
            "signature_table": {
                "title": "",
                "columns": [],
                "rows": [],
            },
            "header": [],
            "footer": [],
            "logo": {
                "present": False,
                "location": "",
            },
            "fonts": {
                "header": {
                    "name": "",
                    "size": "",
                },
                "footer": {
                    "name": "",
                    "size": "",
                },
                "section_heading": {
                    "name": "",
                    "size": "",
                },
                "section_content": {
                    "name": "",
                    "size": "",
                },
                "minimum_line_spacing": "",
            },
            "toc": {
                "present": False,
            },
            "page": {
                "present": False,
            },
            "sections": [],
            "repeated_fields": [],
            "total_images": 0,
            "total_tables": 0,
        }
    }
