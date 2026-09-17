"""Domain constants for Repository Management (Phase 3.1).

Leaf module: no imports from store, service, HTTP, Weaviate, or MySQL.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Lifecycle status values (Phase 2 Domain Model — no DELETED status;
# hard delete is an operation, not a lifecycle state).
# ---------------------------------------------------------------------------

STATUS_CONFIGURING = "configuring"
STATUS_APPROVED = "approved"
STATUS_ACTIVE = "active"
STATUS_ARCHIVED = "archived"

REPOSITORY_STATUS_VALUES: frozenset[str] = frozenset(
    {
        STATUS_CONFIGURING,
        STATUS_APPROVED,
        STATUS_ACTIVE,
        STATUS_ARCHIVED,
    }
)

DEFAULT_REPOSITORY_STATUS = STATUS_CONFIGURING

# ---------------------------------------------------------------------------
# Repository name constraints (Phase 2 validation rules)
# ---------------------------------------------------------------------------

REPOSITORY_NAME_MAX_LENGTH = 128
REPOSITORY_NAME_MIN_LENGTH = 1
REPOSITORY_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

# Names that must never be accepted even if they match the character pattern.
RESERVED_REPOSITORY_NAMES: frozenset[str] = frozenset(
    {
        "null",
        "none",
        "undefined",
        "admin",
        "root",
        "system",
        "default",
        "weaviate",
        "neo4j",
        "mysql",
        "api",
        "repositories",
    }
)

# ---------------------------------------------------------------------------
# Lifecycle transitions (Phase 2). Soft-delete == archived. No DELETED status.
# ---------------------------------------------------------------------------

ALLOWED_STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    STATUS_CONFIGURING: frozenset({STATUS_APPROVED, STATUS_ACTIVE}),
    STATUS_APPROVED: frozenset({STATUS_ACTIVE}),
    STATUS_ACTIVE: frozenset({STATUS_ARCHIVED}),
    STATUS_ARCHIVED: frozenset({STATUS_ACTIVE}),  # reactivation allowed by Phase 2
}

ACTIVATION_SOURCE_STATUSES: frozenset[str] = frozenset({STATUS_CONFIGURING, STATUS_APPROVED})

# ---------------------------------------------------------------------------
# Settings business bounds (shared with business validators; not resolution)
# ---------------------------------------------------------------------------

MIN_CHUNK_SIZE = 200
MAX_CHUNK_SIZE = 2000
ALLOWED_RETRIEVAL_SEARCH_MODES: frozenset[str] = frozenset(
    {"hybrid", "vector", "lexical", "bm25", "keyword"}
)

# Identity / binding fields that must never be patched via settings updates.
IMMUTABLE_REPOSITORY_FIELDS: frozenset[str] = frozenset(
    {
        "repository_id",
        "name",
        "weaviate_collection",
    }
)

# Processing / retrieval settings that become immutable after activation lock.
LOCKED_SETTINGS_KEYS: frozenset[str] = frozenset(
    {
        "embedding_model",
        "chunk_size",
        "chunk_overlap",
        "chunking_strategy",
        "chunking_config",
        "indexing_strategy",
        "retrieval_search_mode",
        "lexical_composition",
        "reranking",
        "metadata_extraction",
        "key_field_extraction",
        "key_field_extraction_enabled",
        "key_fields",
        "folder_hierarchy",
        "template_extraction",
        "reference_document_extraction",
        "citation_retainment",
        "extraction_model",
        "validation_enabled",
        "validation_confidence_threshold",
    }
)

# ---------------------------------------------------------------------------
# Owner / id
# ---------------------------------------------------------------------------

REPOSITORY_OWNER_ID_MAX_LENGTH = 256
REPOSITORY_ID_UUID_VERSIONS = frozenset({4})  # informational; validation accepts any RFC-4122 UUID
