"""PostgreSQL persistence for Repository domain objects (Phase 3.2).

Uses the project's shared ``src.infrastructure.database.postgres`` connection helpers (same pattern as
other DMS relational stores).

Store responsibilities:
  - persist / load Repository and RepositorySettings
  - map rows ↔ domain models
  - transactions for atomic repository + settings writes

Store must NOT: validate lifecycle, resolve settings, call Weaviate, HTTP, or services.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from src.features.repositories.domain.constants import (
    DEFAULT_REPOSITORY_STATUS,
    STATUS_ARCHIVED,
)
from src.features.repositories.domain.repository_exceptions import (
    ConstraintViolation,
    DatabaseUnavailable,
    DuplicateRepository,
    RepositoryNotFound,
)
from src.features.repositories.domain.repository import RepositoryRecord, RepositorySettings
from src.infrastructure.database.postgres import (
    PostgresConnectionParams,
    column_exists,
    connect,
    ensure_column,
    index_exists,
    is_unique_violation,
)

logger = logging.getLogger(__name__)

# Legacy pattern kept for validate_name() backward compatibility with Service callers.
# Phase 3.1 domain validators are the authoritative name rules; Store should not
# invent new business validation.
NAME_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_json(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        parsed = json.loads(value) if value else {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _row_get(row: Mapping[str, Any] | Any, key: str, index: int | None = None) -> Any:
    """Support RealDictCursor rows and legacy tuple-style access."""
    if isinstance(row, Mapping):
        return row.get(key)
    if index is not None:
        return row[index]
    raise TypeError(f"Unsupported row type for key {key!r}")


class RepositoryStore:
    CREATE_REPOSITORY = """
    CREATE TABLE IF NOT EXISTS repository (
      repository_id CHAR(36) NOT NULL,
      name VARCHAR(256) NOT NULL,
      owner_user_id VARCHAR(256) NOT NULL,
      weaviate_collection VARCHAR(256) NOT NULL,
      default_tenant_id VARCHAR(256) NULL,
      status VARCHAR(32) NOT NULL DEFAULT 'configuring',
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (repository_id),
      CONSTRAINT uk_repository_name UNIQUE (name)
    );
    """

    CREATE_SETTINGS = """
    CREATE TABLE IF NOT EXISTS repository_settings (
      repository_id CHAR(36) NOT NULL,
      settings JSON NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (repository_id)
    );
    """

    CREATE_DOCUMENT_LINK = """
    CREATE TABLE IF NOT EXISTS repository_document (
      document_id CHAR(36) NOT NULL,
      repository_id CHAR(36) NOT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (document_id)
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_repository_owner ON repository (owner_user_id)",
        "CREATE INDEX IF NOT EXISTS idx_repository_status ON repository (status)",
        "CREATE INDEX IF NOT EXISTS idx_repository_document_repo ON repository_document (repository_id)",
    )

    REPOSITORY_SELECT = """
        SELECT repository_id, name, owner_user_id, owner_user_name, weaviate_collection, default_tenant_id,
               status, created_at, updated_at, settings_locked_at
        FROM repository
    """

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params

    def _connect(self, *, autocommit: bool) -> Any:
        try:
            return connect(self._params, autocommit=autocommit)
        except Exception as exc:
            raise DatabaseUnavailable(
                "Unable to connect to repository PostgreSQL database.",
                details={"error": str(exc)},
            ) from exc

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        """Autocommit connection for single-statement operations."""
        conn = self._connect(autocommit=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        """Explicit transaction: commit on success, rollback on failure."""
        conn = self._connect(autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                logger.debug("Repository store rollback failed", exc_info=True)
            raise
        finally:
            conn.close()

    def _execute(self, cur: Any, sql: str, params: tuple[Any, ...] | list[Any] | None = None) -> None:
        try:
            cur.execute(sql, params or ())
        except Exception as exc:
            if is_unique_violation(exc):
                raise DuplicateRepository(
                    "Repository uniqueness constraint violated.",
                    details={"error": str(exc)},
                ) from exc
            if isinstance(exc, (DuplicateRepository, RepositoryNotFound, DatabaseUnavailable, ConstraintViolation)):
                raise
            raise DatabaseUnavailable(
                "Repository PostgreSQL statement failed.",
                details={"error": str(exc)},
            ) from exc

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                for ddl in (self.CREATE_REPOSITORY, self.CREATE_SETTINGS, self.CREATE_DOCUMENT_LINK):
                    self._execute(cur, ddl)
                for idx_sql in self.CREATE_INDEXES:
                    self._execute(cur, idx_sql)
                self._ensure_repository_columns(cur)
                self._ensure_repository_indexes(cur)
                self._backfill_settings_locks(cur)

    def _ensure_repository_columns(self, cur: Any) -> None:
        if not column_exists(cur, "repository", "settings_locked_at"):
            ensure_column(cur, "repository", "settings_locked_at", "TIMESTAMPTZ NULL")
        if not column_exists(cur, "repository", "owner_user_name"):
            ensure_column(cur, "repository", "owner_user_name", "VARCHAR(256) NULL")

    @staticmethod
    def _backfill_settings_locks(cur: Any) -> None:
        cur.execute(
            """
            UPDATE repository
            SET settings_locked_at = updated_at
            WHERE status = 'active' AND settings_locked_at IS NULL
            """
        )
        cur.execute(
            """
            UPDATE repository r
            SET settings_locked_at = (
                SELECT MIN(d.created_at)
                FROM repository_document d
                WHERE d.repository_id = r.repository_id
            )
            WHERE settings_locked_at IS NULL
              AND EXISTS (
                SELECT 1 FROM repository_document d
                WHERE d.repository_id = r.repository_id LIMIT 1
              )
            """
        )

    def _ensure_repository_indexes(self, cur: Any) -> None:
        if not index_exists(cur, "uk_repository_collection"):
            cur.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS uk_repository_collection ON repository (weaviate_collection)"
            )

    def is_settings_locked(self, repository_id: str) -> bool:
        record = self.get_by_id(repository_id)
        return record is not None and record.settings_locked_at is not None

    def lock_settings(self, repository_id: str) -> None:
        now = _utc_naive()
        sql = """
        UPDATE repository SET settings_locked_at = %s, updated_at = %s
        WHERE repository_id = %s AND settings_locked_at IS NULL
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (now, now, repository_id))

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.debug("Repository store ping failed", exc_info=True)
            return False

    @staticmethod
    def validate_name(name: str) -> str:
        """Legacy trim helper for Service callers.

        Prefer Phase 3.1 ``validators.validate_repository_name`` for domain rules.
        Kept to avoid breaking existing Service / test fakes in this milestone.
        """
        trimmed = (name or "").strip()
        if not trimmed or len(trimmed) > 256:
            raise ValueError("Repository name is required and must be at most 256 characters.")
        if not NAME_PATTERN.match(trimmed):
            raise ValueError(
                "Repository name must start with alphanumeric and contain only letters, numbers, hyphens, or underscores."
            )
        return trimmed

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_by_id(self, repository_id: str) -> RepositoryRecord | None:
        sql = f"{self.REPOSITORY_SELECT} WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (repository_id,))
                row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def get_repository_by_id(self, repository_id: str) -> RepositoryRecord | None:
        """Alias for get_by_id (Phase 3.2 naming)."""
        return self.get_by_id(repository_id)

    def require_by_id(self, repository_id: str) -> RepositoryRecord:
        record = self.get_by_id(repository_id)
        if record is None:
            raise RepositoryNotFound(
                "Repository not found.",
                details={"repository_id": repository_id},
            )
        return record

    def get_by_name(self, name: str) -> RepositoryRecord | None:
        sql = f"{self.REPOSITORY_SELECT} WHERE name = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (name,))
                row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def get_repository_by_name(self, name: str) -> RepositoryRecord | None:
        return self.get_by_name(name)

    def get_by_collection(self, weaviate_collection: str) -> RepositoryRecord | None:
        sql = f"{self.REPOSITORY_SELECT} WHERE weaviate_collection = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (weaviate_collection,))
                row = cur.fetchone()
        return self._row_to_record(row) if row else None

    def list_repositories(
        self,
        *,
        owner_user_id: str | None = None,
        status: str | None = None,
    ) -> list[RepositoryRecord]:
        clauses = ["1=1"]
        params: list[Any] = []
        if owner_user_id:
            clauses.append("owner_user_id = %s")
            params.append(owner_user_id)
        if status:
            clauses.append("status = %s")
            params.append(status)
        sql = f"""
        {self.REPOSITORY_SELECT}
        WHERE {' AND '.join(clauses)}
        ORDER BY name
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, params)
                rows = cur.fetchall()
        return [self._row_to_record(row) for row in rows]

    # ------------------------------------------------------------------
    # Writes — repository
    # ------------------------------------------------------------------

    def insert_repository(
        self,
        *,
        name: str,
        owner_user_id: str,
        owner_user_name: str | None = None,
        weaviate_collection: str,
        default_tenant_id: str | None,
        status: str = DEFAULT_REPOSITORY_STATUS,
        repository_id: str | None = None,
    ) -> RepositoryRecord:
        repo_id = str(repository_id or uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO repository (
          repository_id, name, owner_user_id, owner_user_name, weaviate_collection, default_tenant_id,
          status, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(
                    cur,
                    sql,
                    (
                        repo_id,
                        name,
                        owner_user_id,
                        owner_user_name,
                        weaviate_collection,
                        default_tenant_id,
                        status,
                        now,
                        now,
                    ),
                )
        return RepositoryRecord(
            repository_id=repo_id,
            name=name,
            owner_user_id=owner_user_id,
            weaviate_collection=weaviate_collection,
            default_tenant_id=default_tenant_id,
            status=status,
            created_at=now,
            updated_at=now,
            settings_locked_at=None,
            owner_user_name=owner_user_name,
        )

    def create_repository_with_settings(
        self,
        *,
        name: str,
        owner_user_id: str,
        weaviate_collection: str,
        settings: dict[str, Any] | None = None,
        owner_user_name: str | None = None,
        default_tenant_id: str | None = None,
        status: str = DEFAULT_REPOSITORY_STATUS,
        repository_id: str | None = None,
        settings_locked_at: datetime | None = None,
    ) -> tuple[RepositoryRecord, RepositorySettings]:
        """Atomically insert repository + settings rows (single transaction)."""
        repo_id = str(repository_id or uuid.uuid4())
        now = _utc_naive()
        settings_payload = dict(settings or {})
        insert_repo_sql = """
        INSERT INTO repository (
          repository_id, name, owner_user_id, owner_user_name, weaviate_collection, default_tenant_id,
          status, created_at, updated_at, settings_locked_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        insert_settings_sql = """
        INSERT INTO repository_settings (repository_id, settings, updated_at)
        VALUES (%s, %s, %s)
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                self._execute(
                    cur,
                    insert_repo_sql,
                    (
                        repo_id,
                        name,
                        owner_user_id,
                        owner_user_name,
                        weaviate_collection,
                        default_tenant_id,
                        status,
                        now,
                        now,
                        settings_locked_at,
                    ),
                )
                self._execute(
                    cur,
                    insert_settings_sql,
                    (repo_id, json.dumps(settings_payload), now),
                )

        record = RepositoryRecord(
            repository_id=repo_id,
            name=name,
            owner_user_id=owner_user_id,
            weaviate_collection=weaviate_collection,
            default_tenant_id=default_tenant_id,
            status=status,
            created_at=now,
            updated_at=now,
            settings_locked_at=settings_locked_at,
            owner_user_name=owner_user_name,
        )
        return record, RepositorySettings(
            repository_id=repo_id,
            settings=settings_payload,
            updated_at=now,
        )

    def update_repository(
        self,
        repository_id: str,
        *,
        owner_user_name: str | None | object = ...,
        default_tenant_id: str | None | object = ...,
        weaviate_collection: str | object = ...,
        settings_locked_at: datetime | None | object = ...,
    ) -> RepositoryRecord:
        """Persist mutable repository columns. Does not enforce lifecycle policy."""
        assignments: list[str] = []
        params: list[Any] = []
        if owner_user_name is not ...:
            assignments.append("owner_user_name = %s")
            params.append(owner_user_name)
        if default_tenant_id is not ...:
            assignments.append("default_tenant_id = %s")
            params.append(default_tenant_id)
        if weaviate_collection is not ...:
            assignments.append("weaviate_collection = %s")
            params.append(weaviate_collection)
        if settings_locked_at is not ...:
            assignments.append("settings_locked_at = %s")
            params.append(settings_locked_at)
        if not assignments:
            return self.require_by_id(repository_id)

        now = _utc_naive()
        assignments.append("updated_at = %s")
        params.append(now)
        params.append(repository_id)
        sql = f"UPDATE repository SET {', '.join(assignments)} WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, params)
                if cur.rowcount == 0 and self.get_by_id(repository_id) is None:
                    raise RepositoryNotFound(
                        "Repository not found.",
                        details={"repository_id": repository_id},
                    )
        return self.require_by_id(repository_id)

    def update_repository_status(self, repository_id: str, status: str) -> RepositoryRecord:
        """Persist status value only — no transition validation (Service responsibility)."""
        now = _utc_naive()
        sql = "UPDATE repository SET status = %s, updated_at = %s WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (status, now, repository_id))
                if cur.rowcount == 0 and self.get_by_id(repository_id) is None:
                    raise RepositoryNotFound(
                        "Repository not found.",
                        details={"repository_id": repository_id},
                    )
        return self.require_by_id(repository_id)

    def archive_repository(self, repository_id: str) -> RepositoryRecord:
        """Soft-retire: set status=archived (Phase 2 soft-delete strategy)."""
        return self.update_repository_status(repository_id, STATUS_ARCHIVED)

    def soft_delete_repository(self, repository_id: str) -> RepositoryRecord:
        """Alias for archive_repository (soft delete via status)."""
        return self.archive_repository(repository_id)

    def update_weaviate_collection(self, repository_id: str, weaviate_collection: str) -> RepositoryRecord:
        return self.update_repository(repository_id, weaviate_collection=weaviate_collection)

    def update_owner_user_name(self, repository_id: str, owner_user_name: str | None) -> None:
        self.update_repository(repository_id, owner_user_name=owner_user_name)

    # ------------------------------------------------------------------
    # Settings persistence
    # ------------------------------------------------------------------

    def get_settings(self, repository_id: str) -> dict[str, Any]:
        sql = "SELECT settings FROM repository_settings WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (repository_id,))
                row = cur.fetchone()
        if not row:
            return {}
        return _parse_json(_row_get(row, "settings", 0))

    def load_settings(self, repository_id: str) -> RepositorySettings:
        """Load settings as a domain object (empty map if no row)."""
        sql = "SELECT settings, updated_at FROM repository_settings WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (repository_id,))
                row = cur.fetchone()
        if not row:
            return RepositorySettings(repository_id=repository_id, settings={}, updated_at=None)
        updated = _row_get(row, "updated_at", 1)
        return RepositorySettings(
            repository_id=repository_id,
            settings=_parse_json(_row_get(row, "settings", 0)),
            updated_at=_parse_dt(updated) if updated is not None else None,
        )

    def upsert_settings(self, repository_id: str, settings: dict[str, Any]) -> dict[str, Any]:
        now = _utc_naive()
        sql = """
        INSERT INTO repository_settings (repository_id, settings, updated_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (repository_id) DO UPDATE
          SET settings = EXCLUDED.settings, updated_at = EXCLUDED.updated_at
        """
        payload = json.dumps(settings)
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (repository_id, payload, now))
                self._execute(
                    cur,
                    "UPDATE repository SET updated_at = %s WHERE repository_id = %s",
                    (now, repository_id),
                )
        return settings

    def save_settings(self, settings: RepositorySettings) -> RepositorySettings:
        """Persist a RepositorySettings domain object."""
        now = _utc_naive()
        stored = self.upsert_settings(settings.repository_id, settings.copy_settings())
        return RepositorySettings(
            repository_id=settings.repository_id,
            settings=stored,
            updated_at=now,
        )

    def persist_settings(self, repository_id: str, settings: dict[str, Any]) -> RepositorySettings:
        """Persist raw settings map and return domain object."""
        now = _utc_naive()
        stored = self.upsert_settings(repository_id, settings)
        return RepositorySettings(repository_id=repository_id, settings=stored, updated_at=now)

    # ------------------------------------------------------------------
    # Document links / hard delete
    # ------------------------------------------------------------------

    def count_documents(self, repository_id: str) -> int:
        sql = "SELECT COUNT(*) AS cnt FROM repository_document WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (repository_id,))
                row = cur.fetchone()
        if not row:
            return 0
        return int(_row_get(row, "cnt", 0))

    def list_document_ids(self, repository_id: str, *, limit: int = 500) -> list[str]:
        sql = """
        SELECT document_id FROM repository_document
        WHERE repository_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (repository_id, min(max(limit, 1), 5000)))
                rows = cur.fetchall()
        return [str(_row_get(row, "document_id", 0)) for row in rows]

    def get_repository_id_for_document(self, document_id: str) -> str | None:
        sql = """
        SELECT repository_id FROM repository_document
        WHERE document_id = %s
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (document_id,))
                row = cur.fetchone()
        return str(_row_get(row, "repository_id", 0)) if row else None

    def link_document(self, document_id: str, repository_id: str) -> bool:
        """Link document to repository. Returns True when this was the first document."""
        now = _utc_naive()
        was_first = self.count_documents(repository_id) == 0
        sql = """
        INSERT INTO repository_document (document_id, repository_id, created_at)
        VALUES (%s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET repository_id = EXCLUDED.repository_id
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (document_id, repository_id, now))
        return was_first

    def delete_repository(self, repository_id: str) -> bool:
        """Hard-delete repository row and related settings/document links.

        Caller (Service) must enforce document_count == 0. Store does not check.
        """
        with self._transaction() as conn:
            with conn.cursor() as cur:
                self._execute(cur, "DELETE FROM repository_document WHERE repository_id = %s", (repository_id,))
                self._execute(cur, "DELETE FROM repository_settings WHERE repository_id = %s", (repository_id,))
                self._execute(cur, "DELETE FROM repository WHERE repository_id = %s", (repository_id,))
                return cur.rowcount > 0

    def unlink_document(self, document_id: str) -> bool:
        sql = "DELETE FROM repository_document WHERE document_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                self._execute(cur, sql, (document_id,))
                return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_record(row: Any) -> RepositoryRecord:
        locked_raw = _row_get(row, "settings_locked_at", 9)
        locked_at = _parse_dt(locked_raw) if locked_raw is not None else None
        owner_raw = _row_get(row, "owner_user_name", 3)
        owner_user_name = str(owner_raw).strip() if owner_raw is not None else None
        if owner_user_name == "":
            owner_user_name = None
        default_tenant = _row_get(row, "default_tenant_id", 5)
        return RepositoryRecord(
            repository_id=str(_row_get(row, "repository_id", 0)),
            name=str(_row_get(row, "name", 1)),
            owner_user_id=str(_row_get(row, "owner_user_id", 2)),
            weaviate_collection=str(_row_get(row, "weaviate_collection", 4)),
            default_tenant_id=str(default_tenant) if default_tenant is not None else None,
            status=str(_row_get(row, "status", 6)),
            created_at=_parse_dt(_row_get(row, "created_at", 7)),
            updated_at=_parse_dt(_row_get(row, "updated_at", 8)),
            settings_locked_at=locked_at,
            owner_user_name=owner_user_name,
        )


_store: RepositoryStore | None = None


def get_repository_store() -> RepositoryStore | None:
    global _store
    if _store is not None:
        return _store
    from src.features.repositories.configuration.repository_config import postgres_params_for_repositories

    params = postgres_params_for_repositories()
    if params is None:
        return None
    candidate = RepositoryStore(params)
    candidate.ensure_schema()
    _store = candidate
    return _store
