"""Default key-field definitions seeded on repository Basic document type creation."""

from __future__ import annotations

from typing import Any

from src.features.document_types.domain.models import KeyFieldType

DEFAULT_BASIC_KEY_FIELDS: tuple[tuple[str, str, bool, Any, str], ...] = (
    ("document_title", KeyFieldType.STRING.value, True, None, "Document title"),
    ("document_number", KeyFieldType.STRING.value, False, None, "Document number"),
    ("version", KeyFieldType.STRING.value, False, None, "Version"),
    ("effective_date", KeyFieldType.DATE.value, False, None, "Effective date"),
)

BASIC_TYPE_NAME = "Basic"
