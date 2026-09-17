from __future__ import annotations

from src.infrastructure.database.postgres import PostgresConnectionParams, postgres_params_from_env


def postgres_params_for_repositories() -> PostgresConnectionParams | None:
    return postgres_params_from_env(
        host_keys=("REPOSITORY_POSTGRES_HOST", "REPOSITORY_MYSQL_HOST"),
        port_keys=("REPOSITORY_POSTGRES_PORT", "REPOSITORY_MYSQL_PORT"),
        user_keys=("REPOSITORY_POSTGRES_USER", "REPOSITORY_MYSQL_USER"),
        password_keys=("REPOSITORY_POSTGRES_PASSWORD", "REPOSITORY_MYSQL_PASSWORD"),
        database_keys=("REPOSITORY_POSTGRES_DATABASE", "REPOSITORY_MYSQL_DATABASE"),
    )


# Backward-compatible alias.
mysql_params_for_repositories = postgres_params_for_repositories
