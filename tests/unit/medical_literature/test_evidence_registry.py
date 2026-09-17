"""Unit tests for evidence import registry uniqueness and converters."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest

from src.features.medical_literature.application.converters import evidence_provenance_metadata
from src.features.medical_literature.infrastructure.evidence_repository import EvidenceImportRecord, EvidenceImportStore


def test_provenance_metadata_shape() -> None:
    meta = evidence_provenance_metadata(
        source="pubmed",
        source_record_id="123",
        repository_id="11111111-1111-1111-1111-111111111111",
        published_date="2024-01-01",
        external_url="https://pubmed.ncbi.nlm.nih.gov/123/",
        imported_at="2026-01-01T00:00:00Z",
    )
    assert meta["source_type"] == "external_evidence"
    assert meta["source"] == "pubmed"
    assert meta["source_record_id"] == "123"


def test_imported_ids_batch_tuple_rows() -> None:
    store = EvidenceImportStore.__new__(EvidenceImportStore)
    store._schema_ready = True
    store.ensure_schema = MagicMock()  # type: ignore[method-assign]

    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchall(self):
            return [("123",), ("456",)]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    store._conn = MagicMock(return_value=_Conn())  # type: ignore[method-assign]
    ids = EvidenceImportStore.imported_ids(
        store,
        repository_id=str(uuid.uuid4()),
        source="pubmed",
        record_ids=["123", "456", "789"],
    )
    assert ids == {"123", "456"}


def test_imported_ids_realdict_rows() -> None:
    store = EvidenceImportStore.__new__(EvidenceImportStore)
    store._schema_ready = True
    store.ensure_schema = MagicMock()  # type: ignore[method-assign]

    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchall(self):
            return [{"source_record_id": "123"}, {"source_record_id": "456"}]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    store._conn = MagicMock(return_value=_Conn())  # type: ignore[method-assign]
    ids = EvidenceImportStore.imported_ids(
        store,
        repository_id=str(uuid.uuid4()),
        source="pubmed",
        record_ids=["123", "456"],
    )
    assert ids == {"123", "456"}


def test_row_to_record_tuple() -> None:
    rid = uuid.uuid4()
    repo = uuid.uuid4()
    doc = uuid.uuid4()
    row = (
        rid,
        repo,
        "pubmed",
        "123",
        "2024",
        doc,
        datetime(2026, 1, 1),
        "imported",
        "https://example",
        {"title": "t"},
    )
    record = EvidenceImportStore._row_to_record(row)
    assert isinstance(record, EvidenceImportRecord)
    assert record.source_record_id == "123"
    assert record.metadata["title"] == "t"


def test_row_to_record_realdict() -> None:
    rid = str(uuid.uuid4())
    repo = str(uuid.uuid4())
    doc = str(uuid.uuid4())
    row = {
        "id": rid,
        "repository_id": repo,
        "source": "pubmed",
        "source_record_id": "999",
        "published_date": "2024",
        "document_id": doc,
        "imported_at": datetime(2026, 1, 1),
        "status": "imported",
        "external_url": "https://example",
        "metadata": {"title": "dict-row"},
    }
    record = EvidenceImportStore._row_to_record(row)
    assert record.id == rid
    assert record.repository_id == repo
    assert record.document_id == doc
    assert record.source_record_id == "999"
    assert record.metadata["title"] == "dict-row"


def test_get_uses_realdict_row() -> None:
    store = EvidenceImportStore.__new__(EvidenceImportStore)
    store._schema_ready = True
    store.ensure_schema = MagicMock()  # type: ignore[method-assign]
    repo = str(uuid.uuid4())
    doc = str(uuid.uuid4())
    rid = str(uuid.uuid4())

    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchone(self):
            return {
                "id": rid,
                "repository_id": repo,
                "source": "pubmed",
                "source_record_id": "37397787",
                "published_date": "2023",
                "document_id": doc,
                "imported_at": datetime(2026, 1, 1),
                "status": "imported",
                "external_url": None,
                "metadata": {},
            }

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    store._conn = MagicMock(return_value=_Conn())  # type: ignore[method-assign]
    record = EvidenceImportStore.get(
        store,
        repository_id=repo,
        source="pubmed",
        source_record_id="37397787",
    )
    assert record is not None
    assert record.document_id == doc
    assert record.source_record_id == "37397787"


def test_list_for_repository_realdict() -> None:
    store = EvidenceImportStore.__new__(EvidenceImportStore)
    store._schema_ready = True
    store.ensure_schema = MagicMock()  # type: ignore[method-assign]
    repo = str(uuid.uuid4())

    class _Cur:
        def execute(self, *a, **k):
            return None

        def fetchall(self):
            return [
                {
                    "id": str(uuid.uuid4()),
                    "repository_id": repo,
                    "source": "pubmed",
                    "source_record_id": "1",
                    "published_date": None,
                    "document_id": str(uuid.uuid4()),
                    "imported_at": datetime(2026, 1, 1),
                    "status": "imported",
                    "external_url": None,
                    "metadata": {},
                }
            ]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    store._conn = MagicMock(return_value=_Conn())  # type: ignore[method-assign]
    rows = EvidenceImportStore.list_for_repository(store, repository_id=repo)
    assert len(rows) == 1
    assert rows[0].source_record_id == "1"
