from __future__ import annotations

from src.infrastructure.database.postgres import PostgresConnectionParams, postgres_params_from_env


def postgres_params_for_document_review() -> PostgresConnectionParams | None:
    return postgres_params_from_env(
        host_keys=("DOCUMENT_REVIEW_POSTGRES_HOST", "DOCUMENT_REVIEW_MYSQL_HOST"),
        port_keys=("DOCUMENT_REVIEW_POSTGRES_PORT", "DOCUMENT_REVIEW_MYSQL_PORT"),
        user_keys=("DOCUMENT_REVIEW_POSTGRES_USER", "DOCUMENT_REVIEW_MYSQL_USER"),
        password_keys=("DOCUMENT_REVIEW_POSTGRES_PASSWORD", "DOCUMENT_REVIEW_MYSQL_PASSWORD"),
        database_keys=("DOCUMENT_REVIEW_POSTGRES_DATABASE", "DOCUMENT_REVIEW_MYSQL_DATABASE"),
    )


# Backward-compatible alias.
mysql_params_for_document_review = postgres_params_for_document_review
