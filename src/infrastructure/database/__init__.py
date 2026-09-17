"""Shared persistence helpers (PostgreSQL document jobs and connection utilities)."""

from src.infrastructure.database.document_jobs import (
    DocumentJobStore,
    MySQLConnectionParams,
    PostgresConnectionParams,
    effective_job_status,
    get_document_job_store,
    mysql_params_from_env,
    postgres_params_from_env,
    reset_document_job_store_for_tests,
)
from src.infrastructure.database.postgres import (
    column_exists,
    connect,
    connection,
    ensure_column,
    index_exists,
    is_unique_violation,
)

__all__ = [
    "DocumentJobStore",
    "MySQLConnectionParams",
    "PostgresConnectionParams",
    "column_exists",
    "connect",
    "connection",
    "effective_job_status",
    "ensure_column",
    "get_document_job_store",
    "index_exists",
    "is_unique_violation",
    "mysql_params_from_env",
    "postgres_params_from_env",
    "reset_document_job_store_for_tests",
]
