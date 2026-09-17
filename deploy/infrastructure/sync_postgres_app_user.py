"""Ensure the PostgreSQL app role/database match DOCUMENT_JOBS_POSTGRES_* (dev / Docker).

Usage (from repo root, with infrastructure up):

    python src/dependencies/sync_postgres_app_user.py

Requires ``psycopg2`` and a reachable Postgres (superuser credentials via
``POSTGRES_USER`` / ``POSTGRES_PASSWORD``, typically the compose bootstrap user).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _load_dotenv() -> None:
    here = Path(__file__).resolve().parent
    candidates = (
        here.parent / "application" / ".env",
        here / ".env",
    )
    env_path = next((p for p in candidates if p.is_file()), None)
    if env_path is None:
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, _, value = text.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def main() -> int:
    _load_dotenv()
    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError:
        print("psycopg2 is required: pip install psycopg2-binary", file=sys.stderr)
        return 1

    admin_user = os.getenv("POSTGRES_USER") or "postgres"
    admin_password = os.getenv("POSTGRES_PASSWORD") or "postgres_password"
    host = os.getenv("DOCUMENT_JOBS_POSTGRES_HOST") or os.getenv("POSTGRES_HOST") or "localhost"
    port = int(os.getenv("DOCUMENT_JOBS_POSTGRES_PORT") or os.getenv("POSTGRES_PORT") or "5432")
    database = (
        os.getenv("DOCUMENT_JOBS_POSTGRES_DATABASE")
        or os.getenv("POSTGRES_DB")
        or "rag_builder"
    )
    app_user = os.getenv("DOCUMENT_JOBS_POSTGRES_USER") or admin_user
    app_pw = os.getenv("DOCUMENT_JOBS_POSTGRES_PASSWORD") or admin_password

    conn = psycopg2.connect(
        host=host,
        port=port,
        user=admin_user,
        password=admin_password,
        dbname="postgres",
    )
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
            if cur.fetchone() is None:
                cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
                print(f"Created database {database!r}.")

            cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (app_user,))
            if cur.fetchone() is None:
                cur.execute(
                    sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD %s").format(sql.Identifier(app_user)),
                    (app_pw,),
                )
                print(f"Created role {app_user!r}.")
            else:
                cur.execute(
                    sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD %s").format(sql.Identifier(app_user)),
                    (app_pw,),
                )
                print(f"Updated role {app_user!r} password to match DOCUMENT_JOBS_POSTGRES_PASSWORD.")

            cur.execute(
                sql.SQL("GRANT ALL PRIVILEGES ON DATABASE {} TO {}").format(
                    sql.Identifier(database),
                    sql.Identifier(app_user),
                )
            )
    finally:
        conn.close()

    # Grant schema privileges inside the app database.
    app_conn = psycopg2.connect(
        host=host,
        port=port,
        user=admin_user,
        password=admin_password,
        dbname=database,
    )
    app_conn.autocommit = True
    try:
        with app_conn.cursor() as cur:
            cur.execute(sql.SQL("GRANT ALL ON SCHEMA public TO {}").format(sql.Identifier(app_user)))
            cur.execute(
                sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO {}").format(
                    sql.Identifier(app_user)
                )
            )
    finally:
        app_conn.close()

    print(f"PostgreSQL ready: host={host} db={database} user={app_user}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
