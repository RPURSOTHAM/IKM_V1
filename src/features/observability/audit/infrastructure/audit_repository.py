"""PostgreSQL persistence for audit events."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from src.features.observability.audit.domain.audit_models import (
    BUSINESS_EVENT_TYPE_PREFIXES,
    AuditEventRecord,
    AuditQuery,
)
from src.features.observability.infrastructure.db import ObservabilityDatabase, get_observability_db

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class AuditRepository:
    def __init__(self, db: ObservabilityDatabase) -> None:
        self._db = db

    def insert(self, event: AuditEventRecord) -> int | None:
        sql = """
        INSERT INTO audit_events (
          timestamp, request_id, correlation_id, application_name, user_id, event_type,
          category, action, source,
          entity_type, entity_id,
          repository_name, document_name,
          status, duration_ms,
          changes, metadata, exception, exception_type, stacktrace
        ) VALUES (%s,%s,%s,%s,%s,%s, %s,%s,%s, %s,%s, %s,%s, %s,%s, %s,%s,%s,%s,%s)
        RETURNING id
        """
        changes_json = json.dumps(event.changes) if event.changes is not None else None
        metadata_json = json.dumps(event.metadata) if event.metadata is not None else None
        ts = event.timestamp or _utc_now()
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        ts,
                        event.request_id,
                        event.correlation_id,
                        event.application_name,
                        event.user_id,
                        event.event_type,
                        event.category,
                        event.action,
                        event.source,
                        event.entity_type,
                        event.entity_id,
                        event.repository_name,
                        event.document_name,
                        event.status,
                        event.duration_ms,
                        changes_json,
                        metadata_json,
                        event.exception,
                        event.exception_type,
                        event.stacktrace,
                    ),
                )
                row = cur.fetchone()
                return int(row[0]) if row else None

    def get_by_id(self, event_id: int) -> dict[str, Any] | None:
        sql = "SELECT * FROM audit_events WHERE id = %s"
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (event_id,))
                row = cur.fetchone()
                if not row:
                    return None
                cols = [d[0] for d in cur.description]
                return self._normalize(dict(zip(cols, row)))

    def list_events(self, query: AuditQuery) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        args: list[Any] = []

        if query.search:
            clauses.append(
                "(event_type ILIKE %s OR entity_type ILIKE %s OR entity_id ILIKE %s "
                "OR exception ILIKE %s OR metadata::text ILIKE %s)"
            )
            pattern = f"%{query.search}%"
            args.extend([pattern] * 5)
        if query.date_from:
            clauses.append("timestamp >= %s")
            args.append(query.date_from)
        if query.date_to:
            clauses.append("timestamp <= %s")
            args.append(query.date_to)
        if query.request_id:
            clauses.append("request_id = %s")
            args.append(query.request_id)
        if query.correlation_id:
            clauses.append("correlation_id = %s")
            args.append(query.correlation_id)
        if query.application_name:
            clauses.append("application_name = %s")
            args.append(query.application_name)
        if query.repository_id:
            clauses.append("(entity_id = %s OR metadata->>'repository_id' = %s)")
            args.extend([query.repository_id, query.repository_id])
        if query.status:
            clauses.append("status = %s")
            args.append(query.status)
        if query.event_type:
            clauses.append("event_type = %s")
            args.append(query.event_type)
        if query.entity_type:
            clauses.append("entity_type = %s")
            args.append(query.entity_type)
        if query.entity_id:
            clauses.append("entity_id = %s")
            args.append(query.entity_id)
        if query.has_changes is True:
            clauses.append("changes IS NOT NULL AND changes::text != '[]'")
        elif query.has_changes is False:
            clauses.append("(changes IS NULL OR changes::text = '[]')")
        if query.category:
            clauses.append("category = %s")
            args.append(query.category)
        if query.action:
            clauses.append("action = %s")
            args.append(query.action)
        if query.source:
            clauses.append("source = %s")
            args.append(query.source)
        if query.business_only:
            prefix_checks = " OR ".join("event_type ILIKE %s" for _ in BUSINESS_EVENT_TYPE_PREFIXES)
            clauses.append(f"({prefix_checks})")
            args.extend(f"{prefix}%" for prefix in BUSINESS_EVENT_TYPE_PREFIXES)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        count_sql = f"SELECT COUNT(*) FROM audit_events {where}"
        offset = max(0, (max(1, query.page) - 1) * max(1, min(query.page_size, 500)))
        limit = max(1, min(query.page_size, 500))
        list_sql = f"""
        SELECT * FROM audit_events {where}
        ORDER BY timestamp DESC, id DESC
        OFFSET %s LIMIT %s
        """
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(count_sql, tuple(args))
                total = int(cur.fetchone()[0])
                cur.execute(list_sql, tuple(args) + (offset, limit))
                cols = [d[0] for d in cur.description]
                rows = [self._normalize(dict(zip(cols, row))) for row in cur.fetchall()]
        return rows, total

    def application_usage(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        application_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Aggregate lightweight business usage directly from audit events."""
        clauses: list[str] = ["event_type = 'APPLICATION_REQUEST'"]
        args: list[Any] = []
        if date_from:
            clauses.append("timestamp >= %s")
            args.append(date_from)
        if date_to:
            clauses.append("timestamp <= %s")
            args.append(date_to)
        if application_name:
            clauses.append("COALESCE(application_name, 'unknown') = %s")
            args.append(application_name)
        where = " AND ".join(clauses)
        sql = f"""
        SELECT COALESCE(application_name, 'unknown') AS application_name,
               COUNT(*)::int AS total_requests,
               COUNT(*) FILTER (
                 WHERE action = 'document_upload' AND status = 'success'
               )::int AS documents_uploaded,
               COUNT(*) FILTER (
                 WHERE action = 'search' AND status = 'success'
               )::int AS search_requests,
               COUNT(*) FILTER (
                 WHERE action = 'ai_query' AND status = 'success'
               )::int AS ai_queries,
               COUNT(*) FILTER (
                 WHERE action = 'document_delete' AND status = 'success'
               )::int AS documents_deleted,
               COUNT(*) FILTER (WHERE status = 'failure')::int AS failed_requests,
               COUNT(DISTINCT user_id) FILTER (
                 WHERE user_id IS NOT NULL
                   AND user_id NOT IN ('scheduler', 'processor', 'consumer-api-key', 'admin-api-key')
               )::int AS active_users
        FROM audit_events
        WHERE {where}
        GROUP BY COALESCE(application_name, 'unknown')
        ORDER BY total_requests DESC, application_name
        """
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def list_changes(self, query: AuditQuery) -> tuple[list[dict[str, Any]], int]:
        query.has_changes = True
        return self.list_events(query)

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        for key in ("changes", "metadata"):
            val = out.get(key)
            if isinstance(val, str):
                try:
                    out[key] = json.loads(val)
                except json.JSONDecodeError:
                    pass
        ts = out.get("timestamp")
        if hasattr(ts, "isoformat"):
            out["timestamp"] = ts.isoformat()
        return out


_repo: AuditRepository | None = None


def get_audit_repository() -> AuditRepository | None:
    global _repo
    db = get_observability_db()
    if db is None:
        return None
    if _repo is None:
        _repo = AuditRepository(db)
    return _repo


def reset_audit_repository_for_tests() -> None:
    global _repo
    _repo = None
