"""Metadata extraction utilities."""

from src.features.document_processing.metadata.field_discovery import (
    discover_structured_fields,
    fields_from_label_value_pairs,
    merge_extracted_with_discovered,
    normalize_discovered_field_name,
)
from src.features.document_processing.metadata.field_extractor import extract_metadata_fields

__all__ = [
    "discover_structured_fields",
    "extract_metadata_fields",
    "fields_from_label_value_pairs",
    "merge_extracted_with_discovered",
    "normalize_discovered_field_name",
]
