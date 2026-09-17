"""PostgreSQL persistence for repository-scoped logical folders."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from src.features.logical_folders.domain.exceptions import LogicalFolderError, LogicalFolderNotFound
from src.infrastructure.database.postgres import (
    PostgresConnectionParams,
    column_exists,
    connect,
    ensure_column,
    is_unique_violation,
)

logger = logging.getLogger(__name__)


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _row_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    return dict(row)


class LogicalFolderStore:
    CREATE_FOLDER = """
    CREATE TABLE IF NOT EXISTS repository_logical_folder (
      folder_id CHAR(36) NOT NULL,
      repository_id CHAR(36) NOT NULL,
      parent_folder_id CHAR(36) NULL,
      name VARCHAR(256) NOT NULL,
      path VARCHAR(2048) NOT NULL,
      metadata JSONB NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (folder_id)
    );
    """

    CREATE_ASSIGNMENT = """
    CREATE TABLE IF NOT EXISTS repository_document_folder (
      document_id CHAR(36) NOT NULL,
      folder_id CHAR(36) NOT NULL,
      repository_id CHAR(36) NOT NULL,
      assigned_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (document_id)
    );
    """

    UNIQUE_FOLDER = """
    CREATE UNIQUE INDEX IF NOT EXISTS uk_logical_folder_repo_parent_name
      ON repository_logical_folder (repository_id, parent_folder_id, name)
      NULLS NOT DISTINCT
    """

    INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_logical_folder_repo ON repository_logical_folder (repository_id)",
        "CREATE INDEX IF NOT EXISTS idx_logical_folder_parent ON repository_logical_folder (parent_folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_folder_folder ON repository_document_folder (folder_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_folder_repo ON repository_document_folder (repository_id)",
    )

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params

    def _connect(self, *, autocommit: bool) -> Any:
        return connect(self._params, autocommit=autocommit)

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = self._connect(autocommit=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        conn = self._connect(autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                logger.debug("Logical folder rollback failed", exc_info=True)
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(self.CREATE_FOLDER)
                cur.execute(self.CREATE_ASSIGNMENT)
                if not column_exists(cur, "repository_logical_folder", "path"):
                    ensure_column(cur, "repository_logical_folder", "path", "VARCHAR(2048) NOT NULL DEFAULT ''")
                if not column_exists(cur, "repository_logical_folder", "metadata"):
                    ensure_column(cur, "repository_logical_folder", "metadata", "JSONB NULL")
                if column_exists(cur, "repository_logical_folder", "field_id"):
                    cur.execute(
                        "ALTER TABLE repository_logical_folder ALTER COLUMN field_id DROP NOT NULL"
                    )
                cur.execute(self.UNIQUE_FOLDER)
                for sql in self.INDEXES:
                    cur.execute(sql)

    def get_by_id(self, folder_id: str, *, repository_id: str | None = None) -> dict[str, Any] | None:
        sql = """
        SELECT folder_id, repository_id, parent_folder_id, name, path, metadata, created_at, updated_at
        FROM repository_logical_folder
        WHERE folder_id = %s
        """
        params: tuple[Any, ...] = (folder_id,)
        if repository_id:
            sql += " AND repository_id = %s"
            params = (folder_id, repository_id)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        return self._folder_row(row) if row else None

    def list_by_repository(self, repository_id: str) -> list[dict[str, Any]]:
        sql = """
        SELECT folder_id, repository_id, parent_folder_id, name, path, metadata, created_at, updated_at
        FROM repository_logical_folder
        WHERE repository_id = %s
        ORDER BY path ASC, name ASC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id,))
                rows = cur.fetchall() or []
        return [self._folder_row(row) for row in rows]

    def get_child(self, repository_id: str, parent_folder_id: str | None, name: str) -> dict[str, Any] | None:
        if parent_folder_id is None:
            sql = """
            SELECT folder_id, repository_id, parent_folder_id, name, path, metadata, created_at, updated_at
            FROM repository_logical_folder
            WHERE repository_id = %s AND parent_folder_id IS NULL AND name = %s
            """
            params: tuple[Any, ...] = (repository_id, name)
        else:
            sql = """
            SELECT folder_id, repository_id, parent_folder_id, name, path, metadata, created_at, updated_at
            FROM repository_logical_folder
            WHERE repository_id = %s AND parent_folder_id = %s AND name = %s
            """
            params = (repository_id, parent_folder_id, name)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        return self._folder_row(row) if row else None

    def get_or_create_child(
        self,
        *,
        repository_id: str,
        parent_folder_id: str | None,
        name: str,
        path: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        existing = self.get_child(repository_id, parent_folder_id, name)
        if existing is not None:
            return existing
        folder_id = str(uuid.uuid4())
        now = _utc_naive()
        payload = json.dumps(metadata or {})
        insert_sql = """
        INSERT INTO repository_logical_folder (
          folder_id, repository_id, parent_folder_id, name, path, metadata, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
        """
        try:
            with self._transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        insert_sql,
                        (folder_id, repository_id, parent_folder_id, name, path, payload, now, now),
                    )
        except Exception as exc:
            if not is_unique_violation(exc):
                raise LogicalFolderError(
                    "Failed to persist logical folder.",
                    details={"error": str(exc), "repository_id": repository_id},
                ) from exc
        reused = self.get_child(repository_id, parent_folder_id, name)
        if reused is None:
            raise LogicalFolderError(
                "Logical folder insert did not produce a row.",
                details={"repository_id": repository_id, "name": name},
            )
        return reused

    def assign_document(
        self,
        *,
        document_id: str,
        folder_id: str,
        repository_id: str,
    ) -> dict[str, Any]:
        now = _utc_naive()
        sql = """
        INSERT INTO repository_document_folder (document_id, folder_id, repository_id, assigned_at)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE
          SET folder_id = EXCLUDED.folder_id,
              repository_id = EXCLUDED.repository_id,
              assigned_at = EXCLUDED.assigned_at
        RETURNING document_id, folder_id, repository_id, assigned_at
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, folder_id, repository_id, now))
                row = cur.fetchone()
        data = _row_dict(row)
        return {
            "document_id": str(data.get("document_id") or document_id),
            "folder_id": str(data.get("folder_id") or folder_id),
            "repository_id": str(data.get("repository_id") or repository_id),
            "assigned_at": data.get("assigned_at"),
        }

    def get_assignment(self, document_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT d.document_id, d.folder_id, d.repository_id, d.assigned_at,
               f.name, f.path, f.parent_folder_id
        FROM repository_document_folder d
        JOIN repository_logical_folder f ON f.folder_id = d.folder_id
        WHERE d.document_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
        if not row:
            return None
        data = _row_dict(row)
        return {
            "document_id": str(data.get("document_id")),
            "folder_id": str(data.get("folder_id")),
            "repository_id": str(data.get("repository_id")),
            "assigned_at": data.get("assigned_at"),
            "name": data.get("name"),
            "path": data.get("path"),
            "parent_folder_id": data.get("parent_folder_id"),
        }

    def list_document_ids(self, repository_id: str, folder_id: str) -> list[str]:
        folder = self.get_by_id(folder_id, repository_id=repository_id)
        if folder is None:
            raise LogicalFolderNotFound(
                "Logical folder not found.",
                details={"repository_id": repository_id, "folder_id": folder_id},
            )
        sql = """
        SELECT document_id
        FROM repository_document_folder
        WHERE repository_id = %s AND folder_id = %s
        ORDER BY assigned_at ASC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id, folder_id))
                rows = cur.fetchall() or []
        ids: list[str] = []
        for row in rows:
            data = _row_dict(row)
            doc_id = str(data.get("document_id") or "").strip()
            if doc_id:
                ids.append(doc_id)
        return ids

    @staticmethod
    def _folder_row(row: Any) -> dict[str, Any]:
        data = _row_dict(row)
        metadata = data.get("metadata")
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        parent = data.get("parent_folder_id")
        return {
            "folder_id": str(data.get("folder_id")),
            "repository_id": str(data.get("repository_id")),
            "parent_folder_id": str(parent) if parent else None,
            "name": str(data.get("name") or ""),
            "path": str(data.get("path") or ""),
            "metadata": metadata,
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
        }


_store: LogicalFolderStore | None = None


def get_logical_folder_store() -> LogicalFolderStore | None:
    global _store
    if _store is not None:
        return _store
    from src.features.repositories.configuration.repository_config import postgres_params_for_repositories

    params = postgres_params_for_repositories()
    if params is None:
        return None
    candidate = LogicalFolderStore(params)
    candidate.ensure_schema()
    _store = candidate
    return _store


def reset_logical_folder_store_for_tests() -> None:
    global _store
    _store = None


def get_logical_folder_store_fresh() -> LogicalFolderStore | None:
    reset_logical_folder_store_for_tests()
    return get_logical_folder_store()
