"""Persistent evidence import registry (Postgres)."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from src.infrastructure.database.postgres import PostgresConnectionParams, connect

logger = logging.getLogger(__name__)


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value) if value else {}
    return {}


@dataclass
class EvidenceImportRecord:
    id: str
    repository_id: str
    source: str
    source_record_id: str
    published_date: str | None
    document_id: str | None
    imported_at: datetime
    status: str
    external_url: str | None
    metadata: dict[str, Any]


class EvidenceImportStore:
    """Tracks imported external evidence per repository to prevent duplicates."""

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params
        self._schema_ready = False

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        with connect(self._params) as conn:
            yield conn

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS evidence_import_registry (
                        id UUID PRIMARY KEY,
                        repository_id UUID NOT NULL,
                        source TEXT NOT NULL,
                        source_record_id TEXT NOT NULL,
                        published_date TEXT NULL,
                        document_id UUID NULL,
                        imported_at TIMESTAMP WITHOUT TIME ZONE NOT NULL,
                        status TEXT NOT NULL DEFAULT 'imported',
                        external_url TEXT NULL,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        CONSTRAINT uq_evidence_import_repo_source_record
                            UNIQUE (repository_id, source, source_record_id)
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_evidence_import_repo
                    ON evidence_import_registry (repository_id)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_evidence_import_document
                    ON evidence_import_registry (document_id)
                    WHERE document_id IS NOT NULL
                    """
                )
            conn.commit()
        self._schema_ready = True
        logger.info("evidence_import_registry schema ensured")

    def get(
        self,
        *,
        repository_id: str,
        source: str,
        source_record_id: str,
    ) -> EvidenceImportRecord | None:
        self.ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, repository_id, source, source_record_id, published_date,
                           document_id, imported_at, status, external_url, metadata
                    FROM evidence_import_registry
                    WHERE repository_id = %s::uuid
                      AND source = %s
                      AND source_record_id = %s
                    """,
                    (repository_id, source, source_record_id),
                )
                row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_for_repository(
        self,
        *,
        repository_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> list[EvidenceImportRecord]:
        self.ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, repository_id, source, source_record_id, published_date,
                           document_id, imported_at, status, external_url, metadata
                    FROM evidence_import_registry
                    WHERE repository_id = %s::uuid
                    ORDER BY imported_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    (repository_id, limit, offset),
                )
                rows = cur.fetchall()
        return [self._row_to_record(r) for r in rows if r]

    def imported_ids(
        self,
        *,
        repository_id: str,
        source: str,
        record_ids: list[str],
    ) -> set[str]:
        if not record_ids:
            return set()
        self.ensure_schema()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source_record_id
                    FROM evidence_import_registry
                    WHERE repository_id = %s::uuid
                      AND source = %s
                      AND source_record_id = ANY(%s)
                    """,
                    (repository_id, source, list(record_ids)),
                )
                rows = cur.fetchall()
        out: set[str] = set()
        for row in rows:
            if not row:
                continue
            value = _row_get(row, "source_record_id", 0)
            if value is not None and str(value).strip():
                out.add(str(value))
        return out

    def insert(
        self,
        *,
        repository_id: str,
        source: str,
        source_record_id: str,
        document_id: str,
        published_date: str | None = None,
        external_url: str | None = None,
        status: str = "imported",
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceImportRecord:
        self.ensure_schema()
        record_id = str(uuid.uuid4())
        imported_at = _utc_naive()
        meta = metadata or {}
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO evidence_import_registry (
                        id, repository_id, source, source_record_id, published_date,
                        document_id, imported_at, status, external_url, metadata
                    ) VALUES (
                        %s::uuid, %s::uuid, %s, %s, %s,
                        %s::uuid, %s, %s, %s, %s::jsonb
                    )
                    RETURNING id, repository_id, source, source_record_id, published_date,
                              document_id, imported_at, status, external_url, metadata
                    """,
                    (
                        record_id,
                        repository_id,
                        source,
                        source_record_id,
                        published_date,
                        document_id,
                        imported_at,
                        status,
                        external_url,
                        json.dumps(meta),
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        assert row is not None
        return self._row_to_record(row)

    @staticmethod
    def _row_to_record(row: Any) -> EvidenceImportRecord:
        """Map a RealDictCursor (or tuple) row to EvidenceImportRecord."""
        published = _row_get(row, "published_date", 4)
        document_id = _row_get(row, "document_id", 5)
        external_url = _row_get(row, "external_url", 8)
        return EvidenceImportRecord(
            id=str(_row_get(row, "id", 0)),
            repository_id=str(_row_get(row, "repository_id", 1)),
            source=str(_row_get(row, "source", 2)),
            source_record_id=str(_row_get(row, "source_record_id", 3)),
            published_date=str(published) if published is not None else None,
            document_id=str(document_id) if document_id is not None else None,
            imported_at=_parse_dt(_row_get(row, "imported_at", 6)) or _utc_naive(),
            status=str(_row_get(row, "status", 7)),
            external_url=str(external_url) if external_url is not None else None,
            metadata=_parse_json(_row_get(row, "metadata", 9)),
        )


def _row_get(row: Any, key: str, index: int) -> Any:
    """Read a column from RealDictCursor mappings or legacy tuple rows."""
    if row is None:
        return None
    if isinstance(row, dict):
        return row.get(key)
    # psycopg2 RealDictRow supports both mapping and sequence protocols, but
    # integer keys raise KeyError — prefer mapping access when available.
    if hasattr(row, "keys") and key in row.keys():  # noqa: SIM118
        return row[key]
    try:
        return row[index]
    except (KeyError, IndexError, TypeError):
        return None
