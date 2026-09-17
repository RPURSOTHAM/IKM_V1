"""PostgreSQL persistence for Phase 5 document human reviews."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from src.features.human_review.domain.review_exceptions import ValidationError
from src.infrastructure.database.postgres import PostgresConnectionParams, connect

_logger = logging.getLogger(__name__)


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _is_invalid_metadata_edit_field(
    field_name: str | None,
    *,
    allowed_field_names: set[str] | frozenset[str] | None = None,
) -> bool:
    name = str(field_name or "").strip()
    if not name:
        return True
    if name.lower().startswith("additionalprop"):
        return True
    if allowed_field_names is not None and name not in allowed_field_names:
        return True
    return False


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None) if value.tzinfo else value
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() + "Z"


def _row_get(row: Mapping[str, Any] | Any, key: str, index: int | None = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    if index is not None:
        return row[index]
    raise TypeError(f"Unsupported row type for key {key!r}")


class DocumentReviewStore:
    CREATE_DOCUMENT_REVIEW = """
    CREATE TABLE IF NOT EXISTS document_review (
      review_id CHAR(36) NOT NULL,
      document_id CHAR(36) NOT NULL,
      repository_id CHAR(36) NULL,
      document_type_id CHAR(36) NULL,
      validation_status VARCHAR(32) NOT NULL,
      review_status VARCHAR(32) NOT NULL,
      assigned_to VARCHAR(256) NULL,
      comment TEXT NULL,
      version INT NOT NULL DEFAULT 1,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      decided_at TIMESTAMPTZ NULL,
      decided_by VARCHAR(256) NULL,
      PRIMARY KEY (review_id)
    );
    """

    CREATE_DOCUMENT_REVIEW_AUDIT = """
    CREATE TABLE IF NOT EXISTS document_review_audit (
      audit_id CHAR(36) NOT NULL,
      review_id CHAR(36) NULL,
      document_id CHAR(36) NOT NULL,
      field_name VARCHAR(128) NULL,
      old_value JSON NULL,
      new_value JSON NULL,
      action VARCHAR(64) NOT NULL,
      modified_by VARCHAR(256) NULL,
      modified_at TIMESTAMPTZ NOT NULL,
      reason TEXT NULL,
      comment TEXT NULL,
      PRIMARY KEY (audit_id)
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_document_review_document ON document_review (document_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_review_repo_status ON document_review (repository_id, review_status)",
        "CREATE INDEX IF NOT EXISTS idx_document_review_assigned ON document_review (assigned_to)",
        "CREATE INDEX IF NOT EXISTS idx_document_review_doc_type ON document_review (document_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_review_updated ON document_review (updated_at)",
        "CREATE INDEX IF NOT EXISTS idx_review_audit_review ON document_review_audit (review_id)",
        "CREATE INDEX IF NOT EXISTS idx_review_audit_document ON document_review_audit (document_id)",
        "CREATE INDEX IF NOT EXISTS idx_review_audit_modified ON document_review_audit (modified_at)",
    )

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = connect(self._params, autocommit=True)
        try:
            yield conn
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_DOCUMENT_REVIEW)
                cur.execute(self.CREATE_DOCUMENT_REVIEW_AUDIT)
                for idx_sql in self.CREATE_INDEXES:
                    cur.execute(idx_sql)

    def _row_to_review(self, row: Any) -> dict[str, Any]:
        repo_id = _row_get(row, "repository_id", 2)
        doc_type_id = _row_get(row, "document_type_id", 3)
        assigned = _row_get(row, "assigned_to", 6)
        comment = _row_get(row, "comment", 7)
        decided_by = _row_get(row, "decided_by", 12)
        return {
            "review_id": str(_row_get(row, "review_id", 0)),
            "document_id": str(_row_get(row, "document_id", 1)),
            "repository_id": str(repo_id) if repo_id else None,
            "document_type_id": str(doc_type_id) if doc_type_id else None,
            "validation_status": str(_row_get(row, "validation_status", 4)),
            "review_status": str(_row_get(row, "review_status", 5)),
            "assigned_to": str(assigned) if assigned else None,
            "comment": str(comment) if comment is not None else None,
            "version": int(_row_get(row, "version", 8) or 1),
            "created_at": _iso(_parse_dt(_row_get(row, "created_at", 9))),
            "updated_at": _iso(_parse_dt(_row_get(row, "updated_at", 10))),
            "decided_at": _iso(_parse_dt(_row_get(row, "decided_at", 11))),
            "decided_by": str(decided_by) if decided_by else None,
        }

    def insert_review(
        self,
        *,
        document_id: str,
        repository_id: str | None,
        document_type_id: str | None,
        validation_status: str,
        review_status: str = "PENDING",
        assigned_to: str | None = None,
        comment: str | None = None,
    ) -> dict[str, Any]:
        review_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO document_review (
          review_id, document_id, repository_id, document_type_id, validation_status,
          review_status, assigned_to, comment, version, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        review_id,
                        document_id,
                        repository_id,
                        document_type_id,
                        validation_status,
                        review_status,
                        assigned_to,
                        comment,
                        1,
                        now,
                        now,
                    ),
                )
        record = self.get_review(review_id)
        assert record is not None
        return record

    def get_review(self, review_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT review_id, document_id, repository_id, document_type_id, validation_status,
               review_status, assigned_to, comment, version, created_at, updated_at,
               decided_at, decided_by
        FROM document_review WHERE review_id = %s LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (review_id,))
                row = cur.fetchone()
        return self._row_to_review(row) if row else None

    def get_open_review_for_document(self, document_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT review_id, document_id, repository_id, document_type_id, validation_status,
               review_status, assigned_to, comment, version, created_at, updated_at,
               decided_at, decided_by
        FROM document_review
        WHERE document_id = %s
          AND review_status IN ('PENDING', 'IN_REVIEW', 'CHANGES_REQUESTED', 'REOPENED')
        ORDER BY created_at DESC
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
        return self._row_to_review(row) if row else None

    def get_latest_review_for_document(self, document_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT review_id, document_id, repository_id, document_type_id, validation_status,
               review_status, assigned_to, comment, version, created_at, updated_at,
               decided_at, decided_by
        FROM document_review
        WHERE document_id = %s
        ORDER BY created_at DESC
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
        return self._row_to_review(row) if row else None

    def list_reviews(
        self,
        *,
        repository_id: str | None = None,
        status: str | None = None,
        document_type_id: str | None = None,
        assigned_to: str | None = None,
        document_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if repository_id:
            clauses.append("repository_id = %s")
            params.append(repository_id)
        if status:
            clauses.append("review_status = %s")
            params.append(status)
        if document_type_id:
            clauses.append("document_type_id = %s")
            params.append(document_type_id)
        if assigned_to:
            clauses.append("assigned_to = %s")
            params.append(assigned_to)
        if document_id:
            clauses.append("document_id = %s")
            params.append(document_id)
        sql = f"""
        SELECT review_id, document_id, repository_id, document_type_id, validation_status,
               review_status, assigned_to, comment, version, created_at, updated_at,
               decided_at, decided_by
        FROM document_review
        WHERE {' AND '.join(clauses)}
        ORDER BY updated_at DESC
        LIMIT %s OFFSET %s
        """
        params.extend([int(limit), int(offset)])
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_review(row) for row in rows]

    def update_review(
        self,
        review_id: str,
        *,
        expected_version: int | None = None,
        review_status: str | None = None,
        assigned_to: str | None = None,
        validation_status: str | None = None,
        comment: str | None = None,
        decided_by: str | None = None,
        clear_decision: bool = False,
        set_decision: bool = False,
    ) -> dict[str, Any] | None:
        current = self.get_review(review_id)
        if current is None:
            return None
        if expected_version is not None and int(current.get("version") or 1) != int(expected_version):
            from src.features.human_review.domain.review_exceptions import ConflictError

            raise ConflictError(
                "Review was modified by another user. Refresh and retry.",
                details={"review_id": review_id, "expected_version": expected_version, "current_version": current.get("version")},
            )

        fields: list[str] = ["updated_at = %s", "version = version + 1"]
        params: list[Any] = [_utc_naive()]
        if review_status is not None:
            fields.append("review_status = %s")
            params.append(review_status)
        if assigned_to is not None:
            fields.append("assigned_to = %s")
            params.append(assigned_to)
        if validation_status is not None:
            fields.append("validation_status = %s")
            params.append(validation_status)
        if comment is not None:
            fields.append("comment = %s")
            params.append(comment)
        if set_decision:
            fields.append("decided_at = %s")
            params.append(_utc_naive())
            fields.append("decided_by = %s")
            params.append(decided_by)
        if clear_decision:
            fields.append("decided_at = NULL")
            fields.append("decided_by = NULL")
        params.append(review_id)
        sql = f"UPDATE document_review SET {', '.join(fields)} WHERE review_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        return self.get_review(review_id)

    def insert_audit(
        self,
        *,
        document_id: str,
        action: str,
        review_id: str | None = None,
        field_name: str | None = None,
        old_value: Any = None,
        new_value: Any = None,
        modified_by: str | None = None,
        reason: str | None = None,
        comment: str | None = None,
        allowed_field_names: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any] | None:
        # Safety net: never persist Swagger placeholders / unknown metadata edit fields.
        if str(action or "").strip() == "metadata_edit":
            if _is_invalid_metadata_edit_field(field_name, allowed_field_names=allowed_field_names):
                _logger.warning(
                    "Rejected metadata_edit audit for document_id=%s field_name=%s",
                    document_id,
                    field_name,
                )
                raise ValidationError(
                    "Unknown metadata fields",
                    details={"fields": [str(field_name or "").strip()] if str(field_name or "").strip() else []},
                )

        audit_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO document_review_audit (
          audit_id, review_id, document_id, field_name, old_value, new_value,
          action, modified_by, modified_at, reason, comment
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        audit_id,
                        review_id,
                        document_id,
                        field_name,
                        json.dumps(old_value, default=str) if old_value is not None else None,
                        json.dumps(new_value, default=str) if new_value is not None else None,
                        action,
                        modified_by,
                        now,
                        reason,
                        comment,
                    ),
                )
        return {
            "audit_id": audit_id,
            "review_id": review_id,
            "document_id": document_id,
            "field_name": field_name,
            "old_value": old_value,
            "new_value": new_value,
            "action": action,
            "modified_by": modified_by,
            "modified_at": _iso(now),
            "reason": reason,
            "comment": comment,
        }

    def _parse_audit_row(self, row: Any) -> dict[str, Any]:
        old_raw = _row_get(row, "old_value", 4)
        new_raw = _row_get(row, "new_value", 5)
        try:
            old_value = json.loads(old_raw) if isinstance(old_raw, (str, bytes)) else old_raw
        except Exception:
            old_value = old_raw
        try:
            new_value = json.loads(new_raw) if isinstance(new_raw, (str, bytes)) else new_raw
        except Exception:
            new_value = new_raw
        review_id = _row_get(row, "review_id", 1)
        field_name = _row_get(row, "field_name", 3)
        modified_by = _row_get(row, "modified_by", 7)
        reason = _row_get(row, "reason", 9)
        comment = _row_get(row, "comment", 10)
        return {
            "audit_id": str(_row_get(row, "audit_id", 0)),
            "review_id": str(review_id) if review_id else None,
            "document_id": str(_row_get(row, "document_id", 2)),
            "field_name": str(field_name) if field_name else None,
            "old_value": old_value,
            "new_value": new_value,
            "action": str(_row_get(row, "action", 6)),
            "modified_by": str(modified_by) if modified_by else None,
            "modified_at": _iso(_parse_dt(_row_get(row, "modified_at", 8))),
            "reason": str(reason) if reason is not None else None,
            "comment": str(comment) if comment is not None else None,
        }

    def list_audit_for_document(self, document_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        sql = """
        SELECT audit_id, review_id, document_id, field_name, old_value, new_value,
               action, modified_by, modified_at, reason, comment
        FROM document_review_audit
        WHERE document_id = %s
        ORDER BY modified_at ASC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, int(limit)))
                rows = cur.fetchall()
        return [self._parse_audit_row(row) for row in rows]

    def list_audit_for_review(self, review_id: str, *, limit: int = 500) -> list[dict[str, Any]]:
        sql = """
        SELECT audit_id, review_id, document_id, field_name, old_value, new_value,
               action, modified_by, modified_at, reason, comment
        FROM document_review_audit
        WHERE review_id = %s
        ORDER BY modified_at ASC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (review_id, int(limit)))
                rows = cur.fetchall()
        return [self._parse_audit_row(row) for row in rows]


_store: DocumentReviewStore | None = None


def get_document_review_store() -> DocumentReviewStore | None:
    global _store
    if _store is not None:
        return _store
    from src.features.human_review.configuration.review_config import postgres_params_for_document_review

    params = postgres_params_for_document_review()
    if params is None:
        return None
    _store = DocumentReviewStore(params)
    _store.ensure_schema()
    return _store
