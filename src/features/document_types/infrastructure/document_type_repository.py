from __future__ import annotations

import json
import logging
import re
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from src.features.document_types.domain.models import (
    DocumentTypeRecord,
    KeyFieldDefinitionRecord,
    MetadataDataType,
    MetadataFieldRecord,
    MetadataFieldsBundle,
)
from src.infrastructure.database.postgres import (
    PostgresConnectionParams,
    column_exists,
    connect,
    ensure_column,
    index_exists,
)

logger = logging.getLogger(__name__)

FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,127}$")
FIELD_GROUP_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _count_value(row: Any) -> int:
    if row is None:
        return 0
    if isinstance(row, dict):
        if "cnt" in row:
            return int(row["cnt"])
        return int(next(iter(row.values())))
    return int(row[0])


class DocumentTypeStore:
    CREATE_DOCUMENT_TYPE = """
    CREATE TABLE IF NOT EXISTS document_type (
      document_type_id CHAR(36) NOT NULL,
      name VARCHAR(256) NOT NULL,
      code VARCHAR(64) NULL,
      description TEXT NULL,
      parent_document_type_id CHAR(36) NULL,
      depth_level SMALLINT NOT NULL,
      is_system BOOLEAN NOT NULL DEFAULT FALSE,
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      created_by VARCHAR(256) NULL,
      updated_by VARCHAR(256) NULL,
      PRIMARY KEY (document_type_id),
      CONSTRAINT uk_document_type_parent_name UNIQUE (parent_document_type_id, name)
    );
    """

    CREATE_RELATIONSHIP = """
    CREATE TABLE IF NOT EXISTS document_type_relationship (
      relationship_id CHAR(36) NOT NULL,
      parent_document_type_id CHAR(36) NOT NULL,
      child_document_type_id CHAR(36) NOT NULL,
      depth_level SMALLINT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (relationship_id),
      CONSTRAINT uk_document_type_rel_child UNIQUE (child_document_type_id)
    );
    """

    CREATE_METADATA_FIELD = """
    CREATE TABLE IF NOT EXISTS metadata_field (
      metadata_field_id CHAR(36) NOT NULL,
      document_type_id CHAR(36) NOT NULL,
      field_name VARCHAR(128) NOT NULL,
      display_label VARCHAR(256) NOT NULL,
      data_type VARCHAR(32) NOT NULL,
      required BOOLEAN NOT NULL DEFAULT FALSE,
      default_value JSON NULL,
      enum_values JSON NULL,
      max_length INT NULL,
      field_group VARCHAR(128) NOT NULL DEFAULT 'general',
      group_label VARCHAR(256) NOT NULL DEFAULT 'General',
      group_sequence INT NOT NULL DEFAULT 100,
      field_sequence INT NOT NULL DEFAULT 100,
      is_system BOOLEAN NOT NULL DEFAULT FALSE,
      is_active BOOLEAN NOT NULL DEFAULT TRUE,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (metadata_field_id),
      CONSTRAINT uk_metadata_field_type_name UNIQUE (document_type_id, field_name)
    );
    """

    CREATE_DOCUMENT_INSTANCE = """
    CREATE TABLE IF NOT EXISTS document_instance (
      document_id CHAR(36) NOT NULL,
      document_type_id CHAR(36) NOT NULL,
      repository_id CHAR(36) NULL,
      tenant_id VARCHAR(256) NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (document_id)
    );
    """

    CREATE_METADATA_VALUE = """
    CREATE TABLE IF NOT EXISTS document_metadata_value (
      document_id CHAR(36) NOT NULL,
      metadata_field_id CHAR(36) NOT NULL,
      value JSON NULL,
      PRIMARY KEY (document_id, metadata_field_id)
    );
    """

    CREATE_KEY_FIELD_DEFINITION = """
    CREATE TABLE IF NOT EXISTS key_field_definition (
      key_field_id CHAR(36) NOT NULL,
      document_type_id CHAR(36) NOT NULL,
      field_name VARCHAR(128) NOT NULL,
      field_type VARCHAR(32) NOT NULL,
      required BOOLEAN NOT NULL DEFAULT FALSE,
      default_value JSON NULL,
      description TEXT NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (key_field_id),
      CONSTRAINT uk_key_field_type_name UNIQUE (document_type_id, field_name)
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_document_type_parent ON document_type (parent_document_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_type_depth ON document_type (depth_level)",
        "CREATE INDEX IF NOT EXISTS idx_document_type_rel_parent ON document_type_relationship (parent_document_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_field_type ON metadata_field (document_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_metadata_field_group_order ON metadata_field (document_type_id, group_sequence, field_sequence)",
        "CREATE INDEX IF NOT EXISTS idx_document_instance_type ON document_instance (document_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_instance_repository ON document_instance (repository_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_instance_tenant ON document_instance (tenant_id)",
        "CREATE INDEX IF NOT EXISTS idx_document_metadata_field ON document_metadata_value (metadata_field_id)",
        "CREATE INDEX IF NOT EXISTS idx_key_field_document_type ON key_field_definition (document_type_id)",
        "CREATE INDEX IF NOT EXISTS idx_key_field_name ON key_field_definition (field_name)",
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

    @contextmanager
    def _conn_transaction(self) -> Iterator[Any]:
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
                for ddl in (
                    self.CREATE_DOCUMENT_TYPE,
                    self.CREATE_RELATIONSHIP,
                    self.CREATE_METADATA_FIELD,
                    self.CREATE_DOCUMENT_INSTANCE,
                    self.CREATE_METADATA_VALUE,
                    self.CREATE_KEY_FIELD_DEFINITION,
                ):
                    cur.execute(ddl)
                for index_ddl in self.CREATE_INDEXES:
                    cur.execute(index_ddl)
                self._ensure_document_type_columns(cur)
                self._ensure_document_type_indexes(cur)
                self._ensure_metadata_field_columns(cur)
                self._ensure_document_instance_columns(cur)
            self.sync_foundational_metadata_fields()

    def get_system_base_type_id(self) -> str | None:
        sql = """
        SELECT document_type_id FROM document_type
        WHERE is_system = TRUE AND depth_level = 0
        ORDER BY created_at
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        return str(row["document_type_id"]) if row else None

    def sync_foundational_metadata_fields(self, base_type_id: str | None = None) -> int:
        """Apply canonical grouping, sequencing, and system flags to Base Document metadata."""
        from src.features.document_types.application.field_group_service import FOUNDATIONAL_FIELD_DEFINITIONS

        base_type_id = base_type_id or self.get_system_base_type_id()
        if base_type_id is None:
            return 0
        updated = 0
        with self._conn() as conn:
            with conn.cursor() as cur:
                for (
                    field_name,
                    display_label,
                    data_type,
                    required,
                    field_group,
                    group_label,
                    group_sequence,
                    field_sequence,
                ) in FOUNDATIONAL_FIELD_DEFINITIONS:
                    cur.execute(
                        """
                        UPDATE metadata_field
                        SET display_label = %s,
                            data_type = %s,
                            required = %s,
                            field_group = %s,
                            group_label = %s,
                            group_sequence = %s,
                            field_sequence = %s,
                            is_system = TRUE,
                            is_active = TRUE,
                            updated_at = NOW()
                        WHERE document_type_id = %s AND field_name = %s
                        """,
                        (
                            display_label,
                            data_type,
                            required,
                            field_group,
                            group_label,
                            group_sequence,
                            field_sequence,
                            base_type_id,
                            field_name,
                        ),
                    )
                    updated += int(cur.rowcount)
        if updated:
            logger.info(
                "Synchronized %s foundational metadata field row(s) on Base Document Type.",
                updated,
            )
        return updated

    def _ensure_document_type_columns(self, cur: Any) -> None:
        ensure_column(cur, "document_type", "repository_id", "CHAR(36) NULL")

    def _ensure_document_type_indexes(self, cur: Any) -> None:
        if not index_exists(cur, "idx_document_type_repository"):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_type_repository ON document_type (repository_id)"
            )
        if not index_exists(cur, "uk_document_type_repo_parent_name"):
            cur.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uk_document_type_repo_parent_name
                ON document_type (repository_id, parent_document_type_id, name)
                """
            )

    def _ensure_metadata_field_columns(self, cur: Any) -> None:
        columns = {
            "field_group": "VARCHAR(128) NOT NULL DEFAULT 'general'",
            "group_label": "VARCHAR(256) NOT NULL DEFAULT 'General'",
            "group_sequence": "INT NOT NULL DEFAULT 100",
            "field_sequence": "INT NOT NULL DEFAULT 100",
        }
        for column_name, ddl_suffix in columns.items():
            if not column_exists(cur, "metadata_field", column_name):
                ensure_column(cur, "metadata_field", column_name, ddl_suffix)

    def _ensure_document_instance_columns(self, cur: Any) -> None:
        columns = {
            "repository_id": "CHAR(36) NULL",
            "tenant_id": "VARCHAR(256) NULL",
            "updated_at": "TIMESTAMPTZ NOT NULL DEFAULT NOW()",
        }
        for column_name, ddl_suffix in columns.items():
            if not column_exists(cur, "document_instance", column_name):
                ensure_column(cur, "document_instance", column_name, ddl_suffix)

        if not index_exists(cur, "idx_document_instance_repository"):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_instance_repository ON document_instance (repository_id)"
            )
        if not index_exists(cur, "idx_document_instance_tenant"):
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_document_instance_tenant ON document_instance (tenant_id)"
            )

    def upsert_document_instance(
        self,
        document_id: str,
        document_type_id: str,
        *,
        repository_id: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        now = _utc_naive()
        sql = """
        INSERT INTO document_instance (document_id, document_type_id, repository_id, tenant_id, created_at, updated_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (document_id) DO UPDATE SET
          document_type_id = EXCLUDED.document_type_id,
          repository_id = COALESCE(EXCLUDED.repository_id, document_instance.repository_id),
          tenant_id = COALESCE(EXCLUDED.tenant_id, document_instance.tenant_id),
          updated_at = EXCLUDED.updated_at
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id, document_type_id, repository_id, tenant_id, now, now))

    def get_document_instance(self, document_id: str) -> dict[str, Any] | None:
        sql = """
        SELECT document_id, document_type_id, repository_id, tenant_id, created_at, updated_at
        FROM document_instance
        WHERE document_id = %s
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_id,))
                row = cur.fetchone()
        if not row:
            return None
        return {
            "document_id": str(row["document_id"]),
            "document_type_id": str(row["document_type_id"]),
            "repository_id": str(row["repository_id"]) if row["repository_id"] else None,
            "tenant_id": str(row["tenant_id"]) if row["tenant_id"] else None,
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.debug("Document type store ping failed", exc_info=True)
            return False

    def count_types(self) -> int:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS cnt FROM document_type")
                row = cur.fetchone()
        return _count_value(row)

    def get_type_by_id(self, document_type_id: str) -> DocumentTypeRecord | None:
        sql = """
        SELECT document_type_id, repository_id, name, code, description, parent_document_type_id,
               depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        FROM document_type WHERE document_type_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id,))
                row = cur.fetchone()
        return self._row_to_type(row) if row else None

    def list_types(
        self,
        *,
        is_active: bool | None = None,
        depth_level: int | None = None,
    ) -> list[DocumentTypeRecord]:
        clauses = ["1=1"]
        params: list[Any] = []
        if is_active is not None:
            clauses.append("is_active = %s")
            params.append(is_active)
        if depth_level is not None:
            clauses.append("depth_level = %s")
            params.append(depth_level)
        sql = f"""
        SELECT document_type_id, repository_id, name, code, description, parent_document_type_id,
               depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        FROM document_type WHERE {' AND '.join(clauses)}
        ORDER BY depth_level, name
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_to_type(row) for row in rows]

    def list_types_for_repository(self, repository_id: str) -> list[DocumentTypeRecord]:
        sql = """
        SELECT document_type_id, repository_id, name, code, description, parent_document_type_id,
               depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        FROM document_type
        WHERE repository_id = %s
        ORDER BY depth_level, name
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id,))
                rows = cur.fetchall()
        return [self._row_to_type(row) for row in rows]

    def get_basic_type_for_repository(self, repository_id: str) -> DocumentTypeRecord | None:
        sql = """
        SELECT document_type_id, repository_id, name, code, description, parent_document_type_id,
               depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        FROM document_type
        WHERE repository_id = %s AND is_system = TRUE AND parent_document_type_id IS NULL
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id,))
                row = cur.fetchone()
        return self._row_to_type(row) if row else None

    def list_children(self, parent_id: str) -> list[DocumentTypeRecord]:
        sql = """
        SELECT document_type_id, repository_id, name, code, description, parent_document_type_id,
               depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        FROM document_type WHERE parent_document_type_id = %s
        ORDER BY name
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (parent_id,))
                rows = cur.fetchall()
        return [self._row_to_type(row) for row in rows]

    def insert_type(
        self,
        *,
        repository_id: str | None,
        name: str,
        code: str | None,
        description: str | None,
        parent_document_type_id: str | None,
        depth_level: int,
        is_system: bool,
        is_active: bool,
        created_by: str | None = None,
    ) -> DocumentTypeRecord:
        type_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO document_type (
          document_type_id, repository_id, name, code, description, parent_document_type_id,
          depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        type_id,
                        repository_id,
                        name,
                        code,
                        description,
                        parent_document_type_id,
                        depth_level,
                        is_system,
                        is_active,
                        now,
                        now,
                        created_by,
                        created_by,
                    ),
                )
        return DocumentTypeRecord(
            document_type_id=type_id,
            repository_id=repository_id,
            name=name,
            code=code,
            description=description,
            parent_document_type_id=parent_document_type_id,
            depth_level=depth_level,
            is_system=is_system,
            is_active=is_active,
            created_at=now,
            updated_at=now,
            created_by=created_by,
            updated_by=created_by,
        )

    def bootstrap_repository_document_types(
        self,
        repository_id: str,
        *,
        created_by: str | None = None,
    ) -> DocumentTypeRecord:
        from src.features.document_types.application.basic_fields import BASIC_TYPE_NAME, DEFAULT_BASIC_KEY_FIELDS

        type_id = str(uuid.uuid4())
        now = _utc_naive()
        sql_type = """
        INSERT INTO document_type (
          document_type_id, repository_id, name, code, description, parent_document_type_id,
          depth_level, is_system, is_active, created_at, updated_at, created_by, updated_by
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        sql_field = """
        INSERT INTO key_field_definition (
          key_field_id, document_type_id, field_name, field_type, required,
          default_value, description, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn_transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql_type,
                    (
                        type_id,
                        repository_id,
                        BASIC_TYPE_NAME,
                        None,
                        "Default root document type for the repository.",
                        None,
                        0,
                        True,
                        True,
                        now,
                        now,
                        created_by,
                        created_by,
                    ),
                )
                for field_name, field_type, required, default_value, description in DEFAULT_BASIC_KEY_FIELDS:
                    cur.execute(
                        sql_field,
                        (
                            str(uuid.uuid4()),
                            type_id,
                            field_name,
                            field_type,
                            required,
                            json.dumps(default_value) if default_value is not None else None,
                            description,
                            now,
                            now,
                        ),
                    )
        record = self.get_type_by_id(type_id)
        assert record is not None
        return record

    def insert_relationship(self, parent_id: str, child_id: str, depth_level: int) -> None:
        rel_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO document_type_relationship (
          relationship_id, parent_document_type_id, child_document_type_id, depth_level, created_at
        ) VALUES (%s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (rel_id, parent_id, child_id, depth_level, now))

    def update_type(
        self,
        document_type_id: str,
        *,
        name: str | None = None,
        code: str | None = None,
        description: str | None = None,
        is_active: bool | None = None,
        updated_by: str | None = None,
    ) -> DocumentTypeRecord:
        existing = self.get_type_by_id(document_type_id)
        if existing is None:
            raise KeyError(document_type_id)
        now = _utc_naive()
        fields: list[str] = ["updated_at = %s"]
        params: list[Any] = [now]
        if name is not None:
            fields.append("name = %s")
            params.append(name)
        if code is not None:
            fields.append("code = %s")
            params.append(code)
        if description is not None:
            fields.append("description = %s")
            params.append(description)
        if is_active is not None:
            fields.append("is_active = %s")
            params.append(is_active)
        if updated_by is not None:
            fields.append("updated_by = %s")
            params.append(updated_by)
        params.append(document_type_id)
        sql = f"UPDATE document_type SET {', '.join(fields)} WHERE document_type_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        updated = self.get_type_by_id(document_type_id)
        assert updated is not None
        return updated

    def delete_type(self, document_type_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM key_field_definition WHERE document_type_id = %s", (document_type_id,))
                cur.execute("DELETE FROM metadata_field WHERE document_type_id = %s", (document_type_id,))
                cur.execute(
                    "DELETE FROM document_type_relationship WHERE child_document_type_id = %s",
                    (document_type_id,),
                )
                cur.execute("DELETE FROM document_type WHERE document_type_id = %s", (document_type_id,))

    def count_documents_for_type(self, document_type_id: str) -> int:
        sql = "SELECT COUNT(*) AS cnt FROM document_instance WHERE document_type_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id,))
                row = cur.fetchone()
        return _count_value(row)

    def count_children(self, document_type_id: str) -> int:
        sql = "SELECT COUNT(*) AS cnt FROM document_type WHERE parent_document_type_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id,))
                row = cur.fetchone()
        return _count_value(row)

    def sibling_name_exists(
        self,
        repository_id: str,
        parent_id: str | None,
        name: str,
        *,
        exclude_id: str | None = None,
    ) -> bool:
        if parent_id is None:
            sql = """
            SELECT COUNT(*) AS cnt FROM document_type
            WHERE repository_id = %s AND parent_document_type_id IS NULL AND name = %s
            """
            params: list[Any] = [repository_id, name]
        else:
            sql = """
            SELECT COUNT(*) AS cnt FROM document_type
            WHERE repository_id = %s AND parent_document_type_id = %s AND name = %s
            """
            params = [repository_id, parent_id, name]
        if exclude_id:
            sql += " AND document_type_id <> %s"
            params.append(exclude_id)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                row = cur.fetchone()
        return _count_value(row) > 0

    def list_fields_for_type(self, document_type_id: str) -> list[MetadataFieldRecord]:
        sql = """
        SELECT metadata_field_id, document_type_id, field_name, display_label, data_type,
               required, default_value, enum_values, max_length, field_group, group_label,
               group_sequence, field_sequence, is_system, is_active, created_at, updated_at
        FROM metadata_field WHERE document_type_id = %s
        ORDER BY group_sequence, field_sequence, field_name
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id,))
                rows = cur.fetchall()
        return [self._row_to_field(row) for row in rows]

    def get_field_by_id(self, metadata_field_id: str) -> MetadataFieldRecord | None:
        sql = """
        SELECT metadata_field_id, document_type_id, field_name, display_label, data_type,
               required, default_value, enum_values, max_length, field_group, group_label,
               group_sequence, field_sequence, is_system, is_active, created_at, updated_at
        FROM metadata_field WHERE metadata_field_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (metadata_field_id,))
                row = cur.fetchone()
        return self._row_to_field(row) if row else None

    def insert_metadata_field(
        self,
        *,
        document_type_id: str,
        field_name: str,
        display_label: str,
        data_type: str,
        required: bool,
        default_value: Any,
        enum_values: list[str] | None,
        max_length: int | None,
        field_group: str,
        group_label: str,
        group_sequence: int,
        field_sequence: int,
        is_system: bool = False,
    ) -> MetadataFieldRecord:
        field_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO metadata_field (
          metadata_field_id, document_type_id, field_name, display_label, data_type,
          required, default_value, enum_values, max_length, field_group, group_label,
          group_sequence, field_sequence, is_system, is_active, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, TRUE, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        field_id,
                        document_type_id,
                        field_name,
                        display_label,
                        data_type,
                        required,
                        json.dumps(default_value) if default_value is not None else None,
                        json.dumps(enum_values) if enum_values is not None else None,
                        max_length,
                        field_group,
                        group_label,
                        group_sequence,
                        field_sequence,
                        is_system,
                        now,
                        now,
                    ),
                )
        field = self.get_field_by_id(field_id)
        assert field is not None
        return field

    def update_metadata_field(
        self,
        metadata_field_id: str,
        *,
        display_label: str | None = None,
        required: bool | None = None,
        default_value: Any | None = None,
        enum_values: list[str] | None = None,
        max_length: int | None = None,
        field_group: str | None = None,
        group_label: str | None = None,
        group_sequence: int | None = None,
        field_sequence: int | None = None,
        is_active: bool | None = None,
        unset_default: bool = False,
    ) -> MetadataFieldRecord:
        now = _utc_naive()
        fields = ["updated_at = %s"]
        params: list[Any] = [now]
        if display_label is not None:
            fields.append("display_label = %s")
            params.append(display_label)
        if required is not None:
            fields.append("required = %s")
            params.append(required)
        if unset_default or default_value is not None:
            fields.append("default_value = %s")
            params.append(json.dumps(default_value) if default_value is not None else None)
        if enum_values is not None:
            fields.append("enum_values = %s")
            params.append(json.dumps(enum_values))
        if max_length is not None:
            fields.append("max_length = %s")
            params.append(max_length)
        if field_group is not None:
            fields.append("field_group = %s")
            params.append(field_group)
        if group_label is not None:
            fields.append("group_label = %s")
            params.append(group_label)
        if group_sequence is not None:
            fields.append("group_sequence = %s")
            params.append(group_sequence)
        if field_sequence is not None:
            fields.append("field_sequence = %s")
            params.append(field_sequence)
        if is_active is not None:
            fields.append("is_active = %s")
            params.append(is_active)
        params.append(metadata_field_id)
        sql = f"UPDATE metadata_field SET {', '.join(fields)} WHERE metadata_field_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        field = self.get_field_by_id(metadata_field_id)
        assert field is not None
        return field

    def delete_metadata_field(self, metadata_field_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM metadata_field WHERE metadata_field_id = %s", (metadata_field_id,))

    def count_metadata_value_usage(self, metadata_field_id: str) -> int:
        sql = """
        SELECT COUNT(*) AS cnt FROM document_metadata_value
        WHERE metadata_field_id = %s AND value IS NOT NULL
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (metadata_field_id,))
                row = cur.fetchone()
        return _count_value(row)

    def field_name_on_type(self, document_type_id: str, field_name: str) -> bool:
        sql = """
        SELECT COUNT(*) AS cnt FROM metadata_field
        WHERE document_type_id = %s AND field_name = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id, field_name))
                row = cur.fetchone()
        return _count_value(row) > 0

    def resolve_metadata_bundle(self, document_type_id: str) -> MetadataFieldsBundle:
        chain = self._ancestor_chain(document_type_id)
        own: list[MetadataFieldRecord] = []
        inherited: list[MetadataFieldRecord] = []
        effective_by_name: dict[str, MetadataFieldRecord] = {}

        for ancestor_id in reversed(chain):
            fields = self.list_fields_for_type(ancestor_id)
            for field in fields:
                if ancestor_id == document_type_id:
                    own.append(field)
                else:
                    inherited.append(
                        MetadataFieldRecord(
                            metadata_field_id=field.metadata_field_id,
                            document_type_id=field.document_type_id,
                            field_name=field.field_name,
                            display_label=field.display_label,
                            data_type=field.data_type,
                            required=field.required,
                            default_value=field.default_value,
                            enum_values=field.enum_values,
                            max_length=field.max_length,
                            field_group=field.field_group,
                            group_label=field.group_label,
                            group_sequence=field.group_sequence,
                            field_sequence=field.field_sequence,
                            is_system=field.is_system,
                            is_active=field.is_active,
                            created_at=field.created_at,
                            updated_at=field.updated_at,
                            inherited_from_document_type_id=ancestor_id,
                        )
                    )
                effective_by_name[field.field_name] = field

        effective = sorted(effective_by_name.values(), key=lambda f: (f.group_sequence, f.field_sequence, f.field_name))
        return MetadataFieldsBundle(own=own, inherited=inherited, effective=effective)

    def effective_field_names(self, document_type_id: str) -> set[str]:
        bundle = self.resolve_metadata_bundle(document_type_id)
        return {f.field_name for f in bundle.effective}

    def delete_document_data(self, document_id: str) -> dict[str, Any]:
        """Remove typed metadata rows for a document."""
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM document_metadata_value WHERE document_id = %s",
                    (document_id,),
                )
                metadata_values_deleted = int(cur.rowcount)
                cur.execute(
                    "DELETE FROM document_instance WHERE document_id = %s",
                    (document_id,),
                )
                instances_deleted = int(cur.rowcount)
        return {
            "metadata_values_deleted": metadata_values_deleted,
            "instances_deleted": instances_deleted,
            "deleted": metadata_values_deleted > 0 or instances_deleted > 0,
        }

    def _ancestor_chain(self, document_type_id: str) -> list[str]:
        return self.ancestor_chain(document_type_id)

    def ancestor_chain(self, document_type_id: str) -> list[str]:
        chain: list[str] = []
        current_id: str | None = document_type_id
        seen: set[str] = set()
        while current_id:
            if current_id in seen:
                break
            seen.add(current_id)
            chain.append(current_id)
            record = self.get_type_by_id(current_id)
            if record is None or record.parent_document_type_id is None:
                break
            if record.parent_document_type_id == current_id:
                break
            current_id = record.parent_document_type_id
        return chain

    def has_circular_ancestor_chain(self, document_type_id: str) -> bool:
        seen: set[str] = set()
        current_id: str | None = document_type_id
        while current_id:
            if current_id in seen:
                return True
            seen.add(current_id)
            record = self.get_type_by_id(current_id)
            if record is None or record.parent_document_type_id is None:
                return False
            current_id = record.parent_document_type_id
        return False

    def list_key_fields_for_type(self, document_type_id: str) -> list[KeyFieldDefinitionRecord]:
        sql = """
        SELECT key_field_id, document_type_id, field_name, field_type, required,
               default_value, description, created_at, updated_at
        FROM key_field_definition
        WHERE document_type_id = %s
        ORDER BY field_name
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id,))
                rows = cur.fetchall()
        return [self._row_to_key_field(row) for row in rows]

    def get_key_field_by_id(self, field_id: str) -> KeyFieldDefinitionRecord | None:
        sql = """
        SELECT key_field_id, document_type_id, field_name, field_type, required,
               default_value, description, created_at, updated_at
        FROM key_field_definition
        WHERE key_field_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (field_id,))
                row = cur.fetchone()
        return self._row_to_key_field(row) if row else None

    def insert_key_field(
        self,
        *,
        document_type_id: str,
        field_name: str,
        field_type: str,
        required: bool,
        default_value: Any,
        description: str | None,
    ) -> KeyFieldDefinitionRecord:
        field_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO key_field_definition (
          key_field_id, document_type_id, field_name, field_type, required,
          default_value, description, created_at, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        field_id,
                        document_type_id,
                        field_name,
                        field_type,
                        required,
                        json.dumps(default_value) if default_value is not None else None,
                        description,
                        now,
                        now,
                    ),
                )
        field = self.get_key_field_by_id(field_id)
        assert field is not None
        return field

    def update_key_field(
        self,
        field_id: str,
        *,
        field_type: str | None = None,
        required: bool | None = None,
        default_value: Any | None = None,
        description: str | None = None,
        unset_default: bool = False,
    ) -> KeyFieldDefinitionRecord:
        now = _utc_naive()
        fields = ["updated_at = %s"]
        params: list[Any] = [now]
        if field_type is not None:
            fields.append("field_type = %s")
            params.append(field_type)
        if required is not None:
            fields.append("required = %s")
            params.append(required)
        if unset_default or default_value is not None:
            fields.append("default_value = %s")
            params.append(json.dumps(default_value) if default_value is not None else None)
        if description is not None:
            fields.append("description = %s")
            params.append(description)
        params.append(field_id)
        sql = f"UPDATE key_field_definition SET {', '.join(fields)} WHERE key_field_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
        field = self.get_key_field_by_id(field_id)
        assert field is not None
        return field

    def delete_key_field(self, field_id: str) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM key_field_definition WHERE key_field_id = %s", (field_id,))

    def key_field_name_on_type(self, document_type_id: str, field_name: str) -> bool:
        sql = """
        SELECT COUNT(*) AS cnt FROM key_field_definition
        WHERE document_type_id = %s AND field_name = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id, field_name))
                row = cur.fetchone()
        return _count_value(row) > 0

    @staticmethod
    def validate_field_name(field_name: str) -> None:
        if not FIELD_NAME_PATTERN.match(field_name):
            raise ValueError("field_name must match ^[a-z][a-z0-9_]{0,127}$")

    @staticmethod
    def validate_field_group(field_group: str) -> None:
        if not FIELD_GROUP_PATTERN.match(field_group):
            raise ValueError("field_group must match ^[a-z][a-z0-9_]{0,63}$")

    @staticmethod
    def validate_sequence(value: int, *, field_name: str) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer.")

    def get_group_definition(
        self,
        document_type_id: str,
        field_group: str,
    ) -> tuple[str, int] | None:
        sql = """
        SELECT group_label, group_sequence
        FROM metadata_field
        WHERE document_type_id = %s AND field_group = %s
        LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (document_type_id, field_group))
                row = cur.fetchone()
        if not row:
            return None
        return str(row["group_label"]), int(row["group_sequence"])

    @staticmethod
    def validate_data_type(data_type: str, enum_values: list[str] | None) -> None:
        if data_type not in MetadataDataType.values():
            raise ValueError(f"data_type must be one of: {', '.join(sorted(MetadataDataType.values()))}")
        if data_type == MetadataDataType.ENUM.value and not enum_values:
            raise ValueError("enum_values is required when data_type=enum")

    @staticmethod
    def _row_to_type(row: Any) -> DocumentTypeRecord:
        return DocumentTypeRecord(
            document_type_id=str(row["document_type_id"]),
            repository_id=str(row["repository_id"]) if row["repository_id"] is not None else None,
            name=str(row["name"]),
            code=row["code"],
            description=row["description"],
            parent_document_type_id=(
                str(row["parent_document_type_id"]) if row["parent_document_type_id"] is not None else None
            ),
            depth_level=int(row["depth_level"]),
            is_system=bool(row["is_system"]),
            is_active=bool(row["is_active"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
            created_by=row["created_by"],
            updated_by=row["updated_by"],
        )

    @staticmethod
    def _row_to_key_field(row: Any) -> KeyFieldDefinitionRecord:
        return KeyFieldDefinitionRecord(
            id=str(row["key_field_id"]),
            document_type_id=str(row["document_type_id"]),
            field_name=str(row["field_name"]),
            field_type=str(row["field_type"]),
            required=bool(row["required"]),
            default_value=_parse_json(row["default_value"]),
            description=row["description"],
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )

    @staticmethod
    def _row_to_field(row: Any) -> MetadataFieldRecord:
        enum_values = _parse_json(row["enum_values"])
        return MetadataFieldRecord(
            metadata_field_id=str(row["metadata_field_id"]),
            document_type_id=str(row["document_type_id"]),
            field_name=str(row["field_name"]),
            display_label=str(row["display_label"]),
            data_type=str(row["data_type"]),
            required=bool(row["required"]),
            default_value=_parse_json(row["default_value"]),
            enum_values=list(enum_values) if enum_values is not None else None,
            max_length=int(row["max_length"]) if row["max_length"] is not None else None,
            field_group=str(row["field_group"]),
            group_label=str(row["group_label"]),
            group_sequence=int(row["group_sequence"]),
            field_sequence=int(row["field_sequence"]),
            is_system=bool(row["is_system"]),
            is_active=bool(row["is_active"]),
            created_at=_parse_dt(row["created_at"]),
            updated_at=_parse_dt(row["updated_at"]),
        )


_store: DocumentTypeStore | None = None


def get_document_type_store() -> DocumentTypeStore | None:
    global _store
    if _store is not None:
        return _store
    from src.features.document_types.configuration.document_type_config import postgres_params_for_document_types

    params = postgres_params_for_document_types()
    if params is None:
        return None
    _store = DocumentTypeStore(params)
    _store.ensure_schema()
    return _store
