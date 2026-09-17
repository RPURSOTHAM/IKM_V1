from __future__ import annotations

from src.infrastructure.database.postgres import PostgresConnectionParams, postgres_params_from_env


def postgres_params_for_document_types() -> PostgresConnectionParams | None:
    """Resolve PostgreSQL connection params for document type tables."""
    return postgres_params_from_env(
        host_keys=("DOCUMENT_TYPE_POSTGRES_HOST", "DOCUMENT_TYPE_MYSQL_HOST"),
        port_keys=("DOCUMENT_TYPE_POSTGRES_PORT", "DOCUMENT_TYPE_MYSQL_PORT"),
        user_keys=("DOCUMENT_TYPE_POSTGRES_USER", "DOCUMENT_TYPE_MYSQL_USER"),
        password_keys=("DOCUMENT_TYPE_POSTGRES_PASSWORD", "DOCUMENT_TYPE_MYSQL_PASSWORD"),
        database_keys=("DOCUMENT_TYPE_POSTGRES_DATABASE", "DOCUMENT_TYPE_MYSQL_DATABASE"),
    )


# Backward-compatible alias.
mysql_params_for_document_types = postgres_params_for_document_types
