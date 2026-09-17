from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from src.features.configuration.configuration.postgres_config import postgres_params_for_platform_config
from src.infrastructure.database.postgres import PostgresConnectionParams, connect

logger = logging.getLogger(__name__)


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list, bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _row_get(row: Mapping[str, Any] | Any, key: str, index: int | None = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    if index is not None:
        return row[index]
    raise TypeError(f"Unsupported row type for key {key!r}")


@dataclass
class ConfigEntry:
    namespace: str
    key: str
    value: Any
    value_type: str
    reload_policy: str
    description: str
    is_sensitive: bool
    updated_at: datetime
    updated_by: str


@dataclass
class AuditEntry:
    audit_id: str
    namespace: str
    config_key: str
    action: str
    old_value: Any
    new_value: Any
    changed_by: str
    changed_at: datetime
    request_id: str | None
    change_comment: str | None


class PlatformConfigStore:
    CREATE_CONFIG = """
    CREATE TABLE IF NOT EXISTS platform_config (
      namespace VARCHAR(64) NOT NULL,
      config_key VARCHAR(128) NOT NULL,
      value_json JSON NOT NULL,
      value_type VARCHAR(32) NOT NULL,
      reload_policy VARCHAR(16) NOT NULL,
      description TEXT NULL,
      is_sensitive BOOLEAN NOT NULL DEFAULT FALSE,
      updated_at TIMESTAMPTZ NOT NULL,
      updated_by VARCHAR(256) NOT NULL,
      PRIMARY KEY (namespace, config_key)
    );
    """

    CREATE_META = """
    CREATE TABLE IF NOT EXISTS platform_config_meta (
      id SMALLINT NOT NULL PRIMARY KEY,
      config_version BIGINT NOT NULL DEFAULT 0,
      updated_at TIMESTAMPTZ NOT NULL
    );
    """

    CREATE_AUDIT = """
    CREATE TABLE IF NOT EXISTS platform_config_audit (
      audit_id CHAR(36) NOT NULL,
      namespace VARCHAR(64) NOT NULL,
      config_key VARCHAR(128) NOT NULL,
      action VARCHAR(16) NOT NULL,
      old_value_json JSON NULL,
      new_value_json JSON NULL,
      changed_by VARCHAR(256) NOT NULL,
      changed_at TIMESTAMPTZ NOT NULL,
      request_id VARCHAR(64) NULL,
      change_comment VARCHAR(512) NULL,
      PRIMARY KEY (audit_id)
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_platform_config_audit_ns ON platform_config_audit (namespace)",
        "CREATE INDEX IF NOT EXISTS idx_platform_config_audit_at ON platform_config_audit (changed_at)",
    )

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = connect(self._params, autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                for ddl in (self.CREATE_CONFIG, self.CREATE_META, self.CREATE_AUDIT):
                    cur.execute(ddl)
                for idx_sql in self.CREATE_INDEXES:
                    cur.execute(idx_sql)
                cur.execute(
                    """
                    INSERT INTO platform_config_meta (id, config_version, updated_at)
                    VALUES (1, 0, %s)
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (_utc_naive(),),
                )

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.debug("Platform config store ping failed", exc_info=True)
            return False

    def get_meta(self) -> tuple[int, datetime]:
        sql = "SELECT config_version, updated_at FROM platform_config_meta WHERE id = 1"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        if not row:
            return 0, _utc_naive()
        return int(_row_get(row, "config_version", 0)), _parse_dt(_row_get(row, "updated_at", 1))

    def list_entries(self, namespace: str | None = None) -> list[ConfigEntry]:
        clauses = ["1=1"]
        params: list[Any] = []
        if namespace:
            clauses.append("namespace = %s")
            params.append(namespace)
        sql = f"""
        SELECT namespace, config_key, value_json, value_type, reload_policy,
               description, is_sensitive, updated_at, updated_by
        FROM platform_config WHERE {' AND '.join(clauses)}
        ORDER BY namespace, config_key
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_entry(row) for row in rows]

    def get_entry(self, namespace: str, key: str) -> ConfigEntry | None:
        sql = """
        SELECT namespace, config_key, value_json, value_type, reload_policy,
               description, is_sensitive, updated_at, updated_by
        FROM platform_config WHERE namespace = %s AND config_key = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (namespace, key))
                row = cur.fetchone()
        return self._row_to_entry(row) if row else None

    def upsert_entry(
        self,
        *,
        namespace: str,
        key: str,
        value: Any,
        value_type: str,
        reload_policy: str,
        description: str,
        is_sensitive: bool,
        updated_by: str,
        audit_action: str = "update",
        old_value: Any = None,
        request_id: str | None = None,
        change_comment: str | None = None,
    ) -> int:
        now = _utc_naive()
        payload = json.dumps(value)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO platform_config (
                      namespace, config_key, value_json, value_type, reload_policy,
                      description, is_sensitive, updated_at, updated_by
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (namespace, config_key) DO UPDATE SET
                      value_json = EXCLUDED.value_json,
                      value_type = EXCLUDED.value_type,
                      reload_policy = EXCLUDED.reload_policy,
                      description = EXCLUDED.description,
                      is_sensitive = EXCLUDED.is_sensitive,
                      updated_at = EXCLUDED.updated_at,
                      updated_by = EXCLUDED.updated_by
                    """,
                    (
                        namespace,
                        key,
                        payload,
                        value_type,
                        reload_policy,
                        description,
                        is_sensitive,
                        now,
                        updated_by,
                    ),
                )
                cur.execute(
                    "UPDATE platform_config_meta SET config_version = config_version + 1, updated_at = %s WHERE id = 1",
                    (now,),
                )
                cur.execute("SELECT config_version FROM platform_config_meta WHERE id = 1")
                version_row = cur.fetchone()
                version = int(_row_get(version_row, "config_version", 0)) if version_row else 0
                cur.execute(
                    """
                    INSERT INTO platform_config_audit (
                      audit_id, namespace, config_key, action, old_value_json, new_value_json,
                      changed_by, changed_at, request_id, change_comment
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        namespace,
                        key,
                        audit_action,
                        json.dumps(old_value) if old_value is not None else None,
                        json.dumps(value),
                        updated_by,
                        now,
                        request_id,
                        change_comment,
                    ),
                )
        return version

    def delete_entry(
        self,
        namespace: str,
        key: str,
        *,
        updated_by: str,
        old_value: Any,
        request_id: str | None = None,
        change_comment: str | None = None,
    ) -> int:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM platform_config WHERE namespace = %s AND config_key = %s",
                    (namespace, key),
                )
                cur.execute(
                    "UPDATE platform_config_meta SET config_version = config_version + 1, updated_at = %s WHERE id = 1",
                    (now,),
                )
                cur.execute("SELECT config_version FROM platform_config_meta WHERE id = 1")
                version_row = cur.fetchone()
                version = int(_row_get(version_row, "config_version", 0)) if version_row else 0
                cur.execute(
                    """
                    INSERT INTO platform_config_audit (
                      audit_id, namespace, config_key, action, old_value_json, new_value_json,
                      changed_by, changed_at, request_id, change_comment
                    ) VALUES (%s, %s, %s, %s, %s, NULL, %s, %s, %s, %s)
                    """,
                    (
                        str(uuid.uuid4()),
                        namespace,
                        key,
                        "delete",
                        json.dumps(old_value),
                        updated_by,
                        now,
                        request_id,
                        change_comment,
                    ),
                )
        return version

    def list_audit(
        self,
        *,
        namespace: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
    ) -> list[AuditEntry]:
        clauses = ["1=1"]
        params: list[Any] = []
        if namespace:
            clauses.append("namespace = %s")
            params.append(namespace)
        if since:
            clauses.append("changed_at >= %s")
            params.append(since)
        params.append(min(max(limit, 1), 500))
        sql = f"""
        SELECT audit_id, namespace, config_key, action, old_value_json, new_value_json,
               changed_by, changed_at, request_id, change_comment
        FROM platform_config_audit
        WHERE {' AND '.join(clauses)}
        ORDER BY changed_at DESC
        LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [
            AuditEntry(
                audit_id=str(_row_get(row, "audit_id", 0)),
                namespace=str(_row_get(row, "namespace", 1)),
                config_key=str(_row_get(row, "config_key", 2)),
                action=str(_row_get(row, "action", 3)),
                old_value=_parse_json(_row_get(row, "old_value_json", 4)),
                new_value=_parse_json(_row_get(row, "new_value_json", 5)),
                changed_by=str(_row_get(row, "changed_by", 6)),
                changed_at=_parse_dt(_row_get(row, "changed_at", 7)),
                request_id=(
                    str(_row_get(row, "request_id", 8))
                    if _row_get(row, "request_id", 8) is not None
                    else None
                ),
                change_comment=(
                    str(_row_get(row, "change_comment", 9))
                    if _row_get(row, "change_comment", 9) is not None
                    else None
                ),
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_entry(row: Any) -> ConfigEntry:
        return ConfigEntry(
            namespace=str(_row_get(row, "namespace", 0)),
            key=str(_row_get(row, "config_key", 1)),
            value=_parse_json(_row_get(row, "value_json", 2)),
            value_type=str(_row_get(row, "value_type", 3)),
            reload_policy=str(_row_get(row, "reload_policy", 4)),
            description=str(_row_get(row, "description", 5) or ""),
            is_sensitive=bool(_row_get(row, "is_sensitive", 6)),
            updated_at=_parse_dt(_row_get(row, "updated_at", 7)),
            updated_by=str(_row_get(row, "updated_by", 8)),
        )


_store: PlatformConfigStore | None = None


def get_platform_config_store() -> PlatformConfigStore | None:
    global _store
    if _store is not None:
        return _store
    params = postgres_params_for_platform_config()
    if params is None:
        return None
    _store = PlatformConfigStore(params)
    _store.ensure_schema()
    return _store
