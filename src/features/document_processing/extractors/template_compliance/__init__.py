"""Template Compliance extraction component (Document Optimizer)."""

from __future__ import annotations

from .adapter import compliance_to_legacy_template
from .api import extract_template, extract_template_to_json
from .schema import SCHEMA_VERSION, empty_template_document

__all__ = [
    "SCHEMA_VERSION",
    "compliance_to_legacy_template",
    "empty_template_document",
    "extract_template",
    "extract_template_to_json",
]

__version__ = "1.0.0"
