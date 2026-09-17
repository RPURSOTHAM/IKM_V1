"""PostgreSQL connection helpers for DMS relational persistence.

All application stores (document jobs, document types, repositories, platform
config/security, document review) share this module. Connection parameters are
resolved from environment variables — nothing is hardcoded at call sites.

Resolution order for each field:
  1. Service-specific ``*_POSTGRES_*`` (when callers pass overrides)
  2. Shared ``DOCUMENT_JOBS_POSTGRES_*`` / ``POSTGRES_*``
  3. Legacy ``DOCUMENT_JOBS_MYSQL_*`` (migration compatibility)
  4. Deployment-mode host defaults + safe port/user/database defaults
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Default relational database name for the DMS platform (overridable via env).
_DEFAULT_DATABASE = "rag_builder"
_DEFAULT_PORT = 5432
_DEFAULT_USER = "postgres"


@dataclass(frozen=True)
class PostgresConnectionParams:
    host: str
    port: int
    user: str
    password: str
    database: str


# Backward-compatible alias while callers migrate off the MySQL name.
MySQLConnectionParams = PostgresConnectionParams


def _env_first(*keys: str, default: str = "") -> str:
    for key in keys:
        raw = os.getenv(key)
        if raw is not None and str(raw).strip() != "":
            return str(raw).strip()
    return default


def _default_host() -> str:
    try:
        from src.shared.networking.hosts import detect_deployment_mode, infra_hosts_for_mode

        hosts = infra_hosts_for_mode(detect_deployment_mode())
        return hosts.get("postgres") or hosts.get("mysql") or "localhost"
    except Exception:
        return "localhost"


def postgres_params_from_env(
    *,
    host_keys: tuple[str, ...] = (),
    port_keys: tuple[str, ...] = (),
    user_keys: tuple[str, ...] = (),
    password_keys: tuple[str, ...] = (),
    database_keys: tuple[str, ...] = (),
) -> PostgresConnectionParams:
    """Build connection params from env, with optional service-specific key overrides."""
    host = _env_first(
        *host_keys,
        "DOCUMENT_JOBS_POSTGRES_HOST",
        "POSTGRES_HOST",
        "DOCUMENT_JOBS_MYSQL_HOST",
        default="",
    )
    if not host:
        host = _default_host()

    # Prefer Postgres ports only — never inherit MySQL's 3306 as the PG listen port.
    port_raw = _env_first(
        *port_keys,
        "DOCUMENT_JOBS_POSTGRES_PORT",
        "POSTGRES_PORT",
        default=str(_DEFAULT_PORT),
    )

    user = _env_first(
        *user_keys,
        "DOCUMENT_JOBS_POSTGRES_USER",
        "POSTGRES_USER",
        "DOCUMENT_JOBS_MYSQL_USER",
        default=_DEFAULT_USER,
    )
    password = _env_first(
        *password_keys,
        "DOCUMENT_JOBS_POSTGRES_PASSWORD",
        "POSTGRES_PASSWORD",
        "DOCUMENT_JOBS_MYSQL_PASSWORD",
        default="",
    )
    database = _env_first(
        *database_keys,
        "DOCUMENT_JOBS_POSTGRES_DATABASE",
        "POSTGRES_DB",
        "DOCUMENT_JOBS_MYSQL_DATABASE",
        default=_DEFAULT_DATABASE,
    )
    return PostgresConnectionParams(
        host=host,
        port=int(port_raw or _DEFAULT_PORT),
        user=user,
        password=password,
        database=database,
    )


# Backward-compatible alias.
mysql_params_from_env = postgres_params_from_env


def connect(params: PostgresConnectionParams, *, autocommit: bool = True) -> Any:
    """Open a psycopg2 connection using the given params."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(
        host=params.host,
        port=params.port,
        user=params.user,
        password=params.password,
        dbname=params.database,
        cursor_factory=RealDictCursor,
    )
    conn.autocommit = autocommit
    return conn


@contextmanager
def connection(params: PostgresConnectionParams, *, autocommit: bool = True) -> Iterator[Any]:
    conn = connect(params, autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()


def is_unique_violation(exc: BaseException) -> bool:
    """True when ``exc`` is a PostgreSQL unique-constraint violation."""
    try:
        from psycopg2 import errorcodes
        from psycopg2 import errors as pg_errors

        if isinstance(exc, pg_errors.UniqueViolation):
            return True
        pgcode = getattr(exc, "pgcode", None)
        return pgcode == errorcodes.UNIQUE_VIOLATION
    except Exception:
        # Fallback: message sniff for drivers not exposing pgcode.
        text = str(exc).lower()
        return "unique" in text and ("violat" in text or "duplicate" in text)


def column_exists(cur: Any, table: str, column: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = %s
          AND column_name = %s
        LIMIT 1
        """,
        (table, column),
    )
    return cur.fetchone() is not None


def index_exists(cur: Any, index_name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname = %s
        LIMIT 1
        """,
        (index_name,),
    )
    return cur.fetchone() is not None


def ensure_column(cur: Any, table: str, column: str, ddl_suffix: str) -> None:
    """Add a column when missing (PostgreSQL ``IF NOT EXISTS``)."""
    cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {ddl_suffix}")
