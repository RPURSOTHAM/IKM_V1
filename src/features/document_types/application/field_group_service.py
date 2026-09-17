"""Default metadata field group configuration for seeded system fields.

Canonical group names and sequence numbers for the Base Document Type.
Applied to PostgreSQL on every startup via DocumentTypeStore.sync_foundational_metadata_fields().
"""

from __future__ import annotations

DEFAULT_FIELD_GROUP = "general"
DEFAULT_GROUP_LABEL = "General"
DEFAULT_GROUP_SEQUENCE = 100
DEFAULT_FIELD_SEQUENCE = 100

# Base Document Type — foundational field groups (key, label, group_sequence)
BASE_DOCUMENT_FIELD_GROUPS: tuple[tuple[str, str, int], ...] = (
    ("document_identity", "Document Identity", 1),
    ("document_details", "Document Details", 2),
    ("document_lifecycle", "Document Lifecycle", 3),
)

# field_name, display_label, data_type, required, field_group, group_label, group_sequence, field_sequence
FOUNDATIONAL_FIELD_DEFINITIONS: tuple[tuple[str, str, str, bool, str, str, int, int], ...] = (
    ("id", "ID", "string", True, "document_identity", "Document Identity", 1, 1),
    ("document_uuid", "Document UUID", "string", True, "document_identity", "Document Identity", 1, 2),
    ("document_name", "Document Name", "string", True, "document_identity", "Document Identity", 1, 3),
    ("document_version", "Document Version", "string", False, "document_identity", "Document Identity", 1, 4),
    ("checksum_value", "Checksum Value", "string", False, "document_identity", "Document Identity", 1, 5),
    ("document_type", "Document Type", "string", True, "document_details", "Document Details", 2, 1),
    ("document_classification", "Document Classification", "string", False, "document_details", "Document Details", 2, 2),
    ("title", "Title", "string", False, "document_details", "Document Details", 2, 3),
    ("description", "Description", "text", False, "document_details", "Document Details", 2, 4),
    ("language", "Language", "string", False, "document_details", "Document Details", 2, 5),
    ("status", "Status", "string", True, "document_lifecycle", "Document Lifecycle", 3, 1),
    ("uploaded_date", "Uploaded Date", "datetime", True, "document_lifecycle", "Document Lifecycle", 3, 2),
    ("uploaded_by", "Uploaded By", "string", True, "document_lifecycle", "Document Lifecycle", 3, 3),
)
