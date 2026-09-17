"""PostgreSQL persistence for metrics events and daily aggregates."""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from typing import Any

from src.features.observability.infrastructure.db import ObservabilityDatabase, get_observability_db
from src.features.observability.metrics.domain.metrics_models import DailyMetricRecord, MetricEventRecord, MetricsQuery

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class MetricsRepository:
    def __init__(self, db: ObservabilityDatabase) -> None:
        self._db = db

    def insert_event(self, event: MetricEventRecord) -> int | None:
        sql = """
        INSERT INTO metrics_events (
          timestamp, request_id, correlation_id, metric_name,
          duration_ms, status, repository_id, document_id, metadata
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING id
        """
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
                        event.metric_name,
                        event.duration_ms,
                        event.status,
                        event.repository_id,
                        event.document_id,
                        metadata_json,
                    ),
                )
                row = cur.fetchone()
                return int(row[0]) if row else None

    def list_events(self, query: MetricsQuery) -> tuple[list[dict[str, Any]], int]:
        clauses: list[str] = []
        args: list[Any] = []
        if query.metric_name:
            clauses.append("metric_name = %s")
            args.append(query.metric_name)
        if query.metric_names:
            placeholders = ",".join(["%s"] * len(query.metric_names))
            clauses.append(f"metric_name IN ({placeholders})")
            args.extend(query.metric_names)
        if query.exclude_metric_names:
            placeholders = ",".join(["%s"] * len(query.exclude_metric_names))
            clauses.append(f"metric_name NOT IN ({placeholders})")
            args.extend(query.exclude_metric_names)
        if query.date_from:
            clauses.append("timestamp >= %s")
            args.append(query.date_from)
        if query.date_to:
            clauses.append("timestamp <= %s")
            args.append(query.date_to)
        if query.request_id:
            clauses.append("request_id = %s")
            args.append(query.request_id)
        if query.repository_id:
            clauses.append("repository_id = %s")
            args.append(query.repository_id)
        if query.status:
            clauses.append("status = %s")
            args.append(query.status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        offset = max(0, (max(1, query.page) - 1) * max(1, min(query.page_size, 500)))
        limit = max(1, min(query.page_size, 500))
        count_sql = f"SELECT COUNT(*) FROM metrics_events {where}"
        list_sql = f"""
        SELECT * FROM metrics_events {where}
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

    def upsert_daily(self, record: DailyMetricRecord) -> None:
        sql = """
        INSERT INTO daily_metrics (
          date, metric_name, count, avg_duration, p50, p95, p99, max_duration, failure_count
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
        ON CONFLICT (date, metric_name) DO UPDATE SET
          count = EXCLUDED.count,
          avg_duration = EXCLUDED.avg_duration,
          p50 = EXCLUDED.p50,
          p95 = EXCLUDED.p95,
          p99 = EXCLUDED.p99,
          max_duration = EXCLUDED.max_duration,
          failure_count = EXCLUDED.failure_count
        """
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        record.date,
                        record.metric_name,
                        record.count,
                        record.avg_duration,
                        record.p50,
                        record.p95,
                        record.p99,
                        record.max_duration,
                        record.failure_count,
                    ),
                )

    def list_daily(
        self,
        *,
        metric_name: str | None = None,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        if metric_name:
            clauses.append("metric_name = %s")
            args.append(metric_name)
        if date_from:
            clauses.append("date >= %s")
            args.append(date_from)
        if date_to:
            clauses.append("date <= %s")
            args.append(date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM daily_metrics {where} ORDER BY date DESC, metric_name"
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d[0] for d in cur.description]
                return [self._normalize_daily(dict(zip(cols, row))) for row in cur.fetchall()]

    def fetch_durations_for_day(self, day: date, metric_name: str | None = None) -> list[tuple[float, str]]:
        clauses = ["timestamp::date = %s", "duration_ms IS NOT NULL"]
        args: list[Any] = [day]
        if metric_name:
            clauses.append("metric_name = %s")
            args.append(metric_name)
        where = " AND ".join(clauses)
        sql = f"SELECT duration_ms, status FROM metrics_events WHERE {where}"
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                return [(float(r[0]), str(r[1])) for r in cur.fetchall()]

    def fetch_status_counts(
        self,
        *,
        metric_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        repository_id: str | None = None,
    ) -> dict[str, int]:
        clauses: list[str] = []
        args: list[Any] = []
        if metric_name:
            clauses.append("metric_name = %s")
            args.append(metric_name)
        if date_from:
            clauses.append("timestamp >= %s")
            args.append(date_from)
        if date_to:
            clauses.append("timestamp <= %s")
            args.append(date_to)
        if repository_id:
            clauses.append("repository_id = %s")
            args.append(repository_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT status, COUNT(*) FROM metrics_events {where} GROUP BY status"
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                return {str(row[0]): int(row[1]) for row in cur.fetchall()}

    def _date_clauses(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        prefix: str = "",
    ) -> tuple[list[str], list[Any]]:
        clauses: list[str] = []
        args: list[Any] = []
        col = f"{prefix}timestamp" if prefix else "timestamp"
        if date_from:
            clauses.append(f"{col} >= %s")
            args.append(date_from)
        if date_to:
            clauses.append(f"{col} <= %s")
            args.append(date_to)
        return clauses, args

    def aggregate_by_metric_name(
        self,
        *,
        metric_names: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        if metric_names:
            placeholders = ",".join(["%s"] * len(metric_names))
            clauses.append(f"metric_name IN ({placeholders})")
            args.extend(metric_names)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT metric_name,
               COUNT(*)::int AS count,
               COUNT(*) FILTER (WHERE status = 'success')::int AS success_count,
               COUNT(*) FILTER (WHERE status = 'failure')::int AS failure_count,
               AVG(duration_ms) AS avg_duration_ms,
               MIN(duration_ms) AS min_duration_ms,
               MAX(duration_ms) AS max_duration_ms
        FROM metrics_events
        {where}
        GROUP BY metric_name
        ORDER BY metric_name
        """
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    def _metadata_numeric_expr(self, key: str, fallback_keys: list[str] | None = None) -> str:
        keys = [key] + (fallback_keys or [])
        parts = [
            f"NULLIF(metadata->>'{k}', '')::double precision"
            for k in keys
        ]
        return f"COALESCE({', '.join(parts)})"

    def avg_metadata_numeric(
        self,
        key: str,
        *,
        metric_names: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        fallback_keys: list[str] | None = None,
    ) -> float | None:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        expr = self._metadata_numeric_expr(key, fallback_keys)
        clauses.append(f"{expr} IS NOT NULL")
        if metric_names:
            placeholders = ",".join(["%s"] * len(metric_names))
            clauses.append(f"metric_name IN ({placeholders})")
            args.extend(metric_names)
        where = " AND ".join(clauses)
        sql = f"SELECT AVG({expr}) FROM metrics_events WHERE {where}"
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                row = cur.fetchone()
                return float(row[0]) if row and row[0] is not None else None

    def sum_metadata_numeric(
        self,
        key: str,
        *,
        metric_names: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        fallback_keys: list[str] | None = None,
    ) -> float | None:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        expr = self._metadata_numeric_expr(key, fallback_keys)
        clauses.append(f"{expr} IS NOT NULL")
        if metric_names:
            placeholders = ",".join(["%s"] * len(metric_names))
            clauses.append(f"metric_name IN ({placeholders})")
            args.extend(metric_names)
        where = " AND ".join(clauses)
        sql = f"SELECT SUM({expr}) FROM metrics_events WHERE {where}"
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                row = cur.fetchone()
                return float(row[0]) if row and row[0] is not None else None

    def top_metadata_values(
        self,
        key: str,
        *,
        metric_names: list[str] | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        fallback_keys: list[str] | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        keys = [key] + (fallback_keys or [])
        coalesce = "COALESCE(" + ", ".join(f"NULLIF(metadata->>'{k}', '')" for k in keys) + ")"
        clauses.append(f"{coalesce} IS NOT NULL")
        if metric_names:
            placeholders = ",".join(["%s"] * len(metric_names))
            clauses.append(f"metric_name IN ({placeholders})")
            args.extend(metric_names)
        where = " AND ".join(clauses)
        sql = f"""
        SELECT {coalesce} AS value, COUNT(*)::int AS count
        FROM metrics_events
        WHERE {where}
        GROUP BY value
        ORDER BY count DESC
        LIMIT %s
        """
        args.append(max(1, min(limit, 50)))
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                return [{"value": row[0], "count": int(row[1])} for row in cur.fetchall()]

    def group_failures_by_metadata(
        self,
        key: str,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        clauses.append("status = 'failure'")
        expr = f"COALESCE(NULLIF(metadata->>'{key}', ''), 'unknown')"
        where = " AND ".join(clauses)
        sql = f"""
        SELECT {expr} AS value, COUNT(*)::int AS count
        FROM metrics_events
        WHERE {where}
        GROUP BY value
        ORDER BY count DESC
        LIMIT %s
        """
        args.append(max(1, min(limit, 50)))
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                return [{"value": row[0], "count": int(row[1])} for row in cur.fetchall()]

    def fetch_recent_events(
        self,
        *,
        metric_names: list[str] | None = None,
        status: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        if metric_names:
            placeholders = ",".join(["%s"] * len(metric_names))
            clauses.append(f"metric_name IN ({placeholders})")
            args.extend(metric_names)
        if status:
            clauses.append("status = %s")
            args.append(status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT * FROM metrics_events {where}
        ORDER BY timestamp DESC, id DESC
        LIMIT %s
        """
        args.append(max(1, min(limit, 100)))
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d[0] for d in cur.description]
                return [self._normalize(dict(zip(cols, row))) for row in cur.fetchall()]

    def fetch_hourly_timeline(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        clauses, args = self._date_clauses(date_from=date_from, date_to=date_to)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT date_trunc('hour', timestamp) AS hour,
               COUNT(*)::int AS total,
               COUNT(*) FILTER (WHERE status = 'success')::int AS success,
               COUNT(*) FILTER (WHERE status = 'failure')::int AS failure
        FROM metrics_events
        {where}
        GROUP BY hour
        ORDER BY hour
        """
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                rows = []
                for hour, total, success, failure in cur.fetchall():
                    ts = hour.isoformat() if hasattr(hour, "isoformat") else str(hour)
                    rows.append({"hour": ts, "total": total, "success": success, "failure": failure})
                return rows

    def top_slow_operations(
        self,
        *,
        limit: int = 10,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        clauses = ["duration_ms IS NOT NULL"]
        args: list[Any] = []
        if date_from:
            clauses.append("timestamp >= %s")
            args.append(date_from)
        if date_to:
            clauses.append("timestamp <= %s")
            args.append(date_to)
        where = " AND ".join(clauses)
        sql = f"""
        SELECT metric_name,
               COUNT(*) AS count,
               AVG(duration_ms) AS avg_duration,
               MAX(duration_ms) AS max_duration,
               PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY duration_ms) AS p95
        FROM metrics_events
        WHERE {where}
        GROUP BY metric_name
        ORDER BY avg_duration DESC
        LIMIT %s
        """
        args.append(max(1, min(limit, 100)))
        with self._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(args))
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]

    @staticmethod
    def _normalize(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        meta = out.get("metadata")
        if isinstance(meta, str):
            try:
                out["metadata"] = json.loads(meta)
            except json.JSONDecodeError:
                pass
        ts = out.get("timestamp")
        if hasattr(ts, "isoformat"):
            out["timestamp"] = ts.isoformat()
        return out

    @staticmethod
    def _normalize_daily(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row)
        d = out.get("date")
        if hasattr(d, "isoformat"):
            out["date"] = d.isoformat()
        return out


_repo: MetricsRepository | None = None


def get_metrics_repository() -> MetricsRepository | None:
    global _repo
    db = get_observability_db()
    if db is None:
        return None
    if _repo is None:
        _repo = MetricsRepository(db)
    return _repo


def reset_metrics_repository_for_tests() -> None:
    global _repo
    _repo = None
