from __future__ import annotations

import os

from src.infrastructure.database.postgres import PostgresConnectionParams, postgres_params_from_env


def postgres_params_for_platform_config() -> PostgresConnectionParams | None:
    return postgres_params_from_env(
        host_keys=("PLATFORM_CONFIG_POSTGRES_HOST", "PLATFORM_CONFIG_MYSQL_HOST"),
        port_keys=("PLATFORM_CONFIG_POSTGRES_PORT", "PLATFORM_CONFIG_MYSQL_PORT"),
        user_keys=("PLATFORM_CONFIG_POSTGRES_USER", "PLATFORM_CONFIG_MYSQL_USER"),
        password_keys=("PLATFORM_CONFIG_POSTGRES_PASSWORD", "PLATFORM_CONFIG_MYSQL_PASSWORD"),
        database_keys=("PLATFORM_CONFIG_POSTGRES_DATABASE", "PLATFORM_CONFIG_MYSQL_DATABASE"),
    )


# Backward-compatible alias.
mysql_params_for_platform_config = postgres_params_for_platform_config


REFRESH_INTERVAL_SEC = float(os.getenv("PLATFORM_CONFIG_REFRESH_INTERVAL_SEC", "5"))
