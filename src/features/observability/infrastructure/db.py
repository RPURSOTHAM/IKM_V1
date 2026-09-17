import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class ObservabilityCursorWrapper:
    def __init__(self, cursor, on_query):
        self._cursor = cursor
        self._on_query = on_query

    def execute(self, query, vars=None):
        self._on_query(query)
        return self._cursor.execute(query, vars)

    def executemany(self, query, vars_list):
        self._on_query(query)
        return self._cursor.executemany(query, vars_list)

    def __getattr__(self, name):
        return getattr(self._cursor, name)

    def __iter__(self):
        return iter(self._cursor)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._cursor.__exit__(exc_type, exc_val, exc_tb)


class ObservabilityConnectionWrapper:
    def __init__(self, conn, on_query):
        self._conn = conn
        self._on_query = on_query

    def cursor(self, *args, **kwargs):
        cursor = self._conn.cursor(*args, **kwargs)
        return ObservabilityCursorWrapper(cursor, self._on_query)

    def __getattr__(self, name):
        return getattr(self._conn, name)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._conn.__exit__(exc_type, exc_val, exc_tb)


_store_initialized = False


@dataclass(frozen=True)
class PostgresParams:
    host: str
    port: int
    user: str
    password: str
    database: str


def postgres_params_from_env() -> PostgresParams:
    """Resolve Postgres params for observability storage.

    Preference order per field:
      OBSERVABILITY_POSTGRES_* → DOCUMENT_JOBS_POSTGRES_* → POSTGRES_* → defaults
    Defaults: host localhost (via shared helper), port 5432, user postgres, db rag_builder.
    """
    try:
        from src.infrastructure.database.postgres import postgres_params_from_env as shared_postgres_params_from_env

        shared = shared_postgres_params_from_env(
            host_keys=("OBSERVABILITY_POSTGRES_HOST",),
            port_keys=("OBSERVABILITY_POSTGRES_PORT",),
            user_keys=("OBSERVABILITY_POSTGRES_USER",),
            password_keys=("OBSERVABILITY_POSTGRES_PASSWORD",),
            database_keys=("OBSERVABILITY_POSTGRES_DB", "OBSERVABILITY_POSTGRES_DATABASE"),
        )
        return PostgresParams(
            host=shared.host,
            port=shared.port,
            user=shared.user,
            password=shared.password,
            database=shared.database,
        )
    except Exception:
        # Local fallback if shared helper is unavailable.
        def _env_first(*keys: str, default: str = "") -> str:
            for key in keys:
                val = os.getenv(key)
                if val is not None and str(val).strip():
                    return str(val).strip()
            return default

        return PostgresParams(
            host=_env_first(
                "OBSERVABILITY_POSTGRES_HOST",
                "DOCUMENT_JOBS_POSTGRES_HOST",
                "POSTGRES_HOST",
                default="localhost",
            ),
            port=int(
                _env_first(
                    "OBSERVABILITY_POSTGRES_PORT",
                    "DOCUMENT_JOBS_POSTGRES_PORT",
                    "POSTGRES_PORT",
                    default="5432",
                )
            ),
            user=_env_first(
                "OBSERVABILITY_POSTGRES_USER",
                "DOCUMENT_JOBS_POSTGRES_USER",
                "POSTGRES_USER",
                default="postgres",
            ),
            password=_env_first(
                "OBSERVABILITY_POSTGRES_PASSWORD",
                "DOCUMENT_JOBS_POSTGRES_PASSWORD",
                "POSTGRES_PASSWORD",
                default="",
            ),
            database=_env_first(
                "OBSERVABILITY_POSTGRES_DB",
                "OBSERVABILITY_POSTGRES_DATABASE",
                "DOCUMENT_JOBS_POSTGRES_DATABASE",
                "POSTGRES_DB",
                default="rag_builder",
            ),
        )


class ObservabilityDatabase:
    """Schema management and connection pooling for observability tables."""

    AUDIT_EVENTS_SQL = """
    CREATE TABLE IF NOT EXISTS audit_events (
      id BIGSERIAL PRIMARY KEY,
      timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      request_id VARCHAR(64),
      correlation_id VARCHAR(64),
      application_name VARCHAR(128),
      user_id VARCHAR(128),
      event_type VARCHAR(128) NOT NULL,
      entity_type VARCHAR(128),
      entity_id VARCHAR(128),
      status VARCHAR(32) NOT NULL,
      changes JSONB,
      metadata JSONB,
      exception TEXT,
      stacktrace TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp ON audit_events (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_audit_events_request_id ON audit_events (request_id);
    CREATE INDEX IF NOT EXISTS idx_audit_events_correlation_id ON audit_events (correlation_id);
    CREATE INDEX IF NOT EXISTS idx_audit_events_event_type ON audit_events (event_type);
    CREATE INDEX IF NOT EXISTS idx_audit_events_entity ON audit_events (entity_type, entity_id);
    CREATE INDEX IF NOT EXISTS idx_audit_events_status ON audit_events (status);
  """

    # Backward-compatible: adds new columns if they don't exist yet.
    # Each statement is executed independently so partial failures don't block others.
    AUDIT_EVENTS_ALTER_STATEMENTS: list[str] = [
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS category VARCHAR(64)",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS application_name VARCHAR(128)",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS action VARCHAR(64)",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS source VARCHAR(32)",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS duration_ms DOUBLE PRECISION",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS exception_type VARCHAR(256)",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS repository_name VARCHAR(256)",
        "ALTER TABLE audit_events ADD COLUMN IF NOT EXISTS document_name VARCHAR(512)",
        "CREATE INDEX IF NOT EXISTS idx_audit_events_category ON audit_events (category)",
        "CREATE INDEX IF NOT EXISTS idx_audit_events_application_name ON audit_events (application_name)",
        "CREATE INDEX IF NOT EXISTS idx_audit_events_source ON audit_events (source)",
    ]

    METRICS_EVENTS_SQL = """
    CREATE TABLE IF NOT EXISTS metrics_events (
      id BIGSERIAL PRIMARY KEY,
      timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      request_id VARCHAR(64),
      correlation_id VARCHAR(64),
      metric_name VARCHAR(128) NOT NULL,
      duration_ms DOUBLE PRECISION,
      status VARCHAR(32) NOT NULL,
      repository_id VARCHAR(128),
      document_id VARCHAR(128),
      metadata JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_metrics_events_timestamp ON metrics_events (timestamp DESC);
    CREATE INDEX IF NOT EXISTS idx_metrics_events_metric_name ON metrics_events (metric_name);
    CREATE INDEX IF NOT EXISTS idx_metrics_events_request_id ON metrics_events (request_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_events_repository_id ON metrics_events (repository_id);
    CREATE INDEX IF NOT EXISTS idx_metrics_events_status ON metrics_events (status);
  """

    DAILY_METRICS_SQL = """
    CREATE TABLE IF NOT EXISTS daily_metrics (
      id BIGSERIAL PRIMARY KEY,
      date DATE NOT NULL,
      metric_name VARCHAR(128) NOT NULL,
      count BIGINT NOT NULL DEFAULT 0,
      avg_duration DOUBLE PRECISION,
      p50 DOUBLE PRECISION,
      p95 DOUBLE PRECISION,
      p99 DOUBLE PRECISION,
      max_duration DOUBLE PRECISION,
      failure_count BIGINT NOT NULL DEFAULT 0,
      UNIQUE (date, metric_name)
    );
    CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics (date DESC);
    CREATE INDEX IF NOT EXISTS idx_daily_metrics_metric_name ON daily_metrics (metric_name);
  """

    def __init__(self, params: PostgresParams) -> None:
        import psycopg2

        self._psycopg2 = psycopg2
        self._params = params

    @contextmanager
    def connection(self) -> Iterator[Any]:
        import time
        from src.features.observability.metrics.application.metrics_service import is_in_metrics_persist, get_metrics_service

        started = time.perf_counter()
        conn = self._psycopg2.connect(
            host=self._params.host,
            port=self._params.port,
            user=self._params.user,
            password=self._params.password,
            dbname=self._params.database,
        )
        status = "success"
        queries = []
        def on_query(sql):
            queries.append(str(sql))

        wrapped_conn = ObservabilityConnectionWrapper(conn, on_query)
        try:
            yield wrapped_conn
            conn.commit()
        except Exception as exc:
            status = "failure"
            conn.rollback()
            if not is_in_metrics_persist():
                try:
                    duration_ms = (time.perf_counter() - started) * 1000.0
                    get_metrics_service().record(
                        "postgresql_error",
                        status="failure",
                        metadata={
                            "category": "storage",
                            "component": "postgresql",
                            "operation": "query",
                            "query_duration_ms": duration_ms,
                            "duration_ms": duration_ms,
                            "status": "failure",
                            "error": str(exc),
                        }
                    )
                except Exception:
                    pass
            raise exc
        finally:
            conn.close()
            duration_ms = (time.perf_counter() - started) * 1000.0
            if not is_in_metrics_persist():
                query_count = 0
                insert_count = 0
                update_count = 0
                delete_count = 0
                for q in queries:
                    q_upper = q.strip().upper()
                    if q_upper.startswith("SELECT"):
                        query_count += 1
                    elif q_upper.startswith("INSERT"):
                        insert_count += 1
                    elif q_upper.startswith("UPDATE"):
                        update_count += 1
                    elif q_upper.startswith("DELETE") or q_upper.startswith("TRUNCATE"):
                        delete_count += 1

                db_errors = 1 if status == "failure" else 0
                try:
                    get_metrics_service().record(
                        "postgresql_query",
                        duration_ms=duration_ms,
                        status=status,
                        metadata={
                            "category": "storage",
                            "component": "postgresql",
                            "operation": "query",
                            "query_duration_ms": duration_ms,
                            "duration_ms": duration_ms,
                            "status": status,
                            "query_count": query_count,
                            "insert_count": insert_count,
                            "update_count": update_count,
                            "delete_count": delete_count,
                            "database_errors": db_errors,
                        }
                    )
                except Exception:
                    pass

    def ensure_schema(self) -> None:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(self.AUDIT_EVENTS_SQL)
                cur.execute(self.METRICS_EVENTS_SQL)
                cur.execute(self.DAILY_METRICS_SQL)
        # Run each ALTER TABLE / CREATE INDEX independently so that an
        # already-existing column doesn't block the others.
        for stmt in self.AUDIT_EVENTS_ALTER_STATEMENTS:
            try:
                with self.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(stmt)
            except Exception:
                logger.debug("Schema migration skipped (already applied?): %s", stmt, exc_info=True)

    def ping(self) -> bool:
        try:
            with self.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            logger.debug("Observability PostgreSQL ping failed", exc_info=True)
            return False


_db: ObservabilityDatabase | None = None


def get_observability_db() -> ObservabilityDatabase | None:
    global _db, _store_initialized
    if _db is None:
        try:
            _db = ObservabilityDatabase(postgres_params_from_env())
            _db.ensure_schema()
            _store_initialized = True
        except Exception:
            logger.debug("Observability database unavailable", exc_info=True)
            return None
    return _db


def reset_observability_db_for_tests() -> None:
    global _db, _store_initialized
    _db = None
    _store_initialized = False
