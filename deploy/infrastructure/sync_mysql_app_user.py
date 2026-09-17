"""Set MySQL app user password to match ``DOCUMENT_JOBS_MYSQL_PASSWORD`` (dev / Docker).

The official MySQL image only applies ``MYSQL_PASSWORD`` on first database init. An existing
``mysql_data`` volume keeps the old password, which causes PyMySQL error 1045 from the API.

Run from repo root (requires Docker and a running ``rag_mysql`` container):

    python src/dependencies/sync_mysql_app_user.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _load_env(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in {'"', "'"}:
            val = val[1:-1]
        out[key] = val
    return out


def _sql_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "''")


def main() -> int:
    deps_dir = Path(__file__).resolve().parent
    env_path = deps_dir.parent / "application" / ".env"
    if not env_path.is_file():
        env_path = deps_dir / ".env"
    env = _load_env(env_path)
    root_pw = env.get("MYSQL_ROOT_PASSWORD") or "rag_root_password"
    app_user = env.get("DOCUMENT_JOBS_MYSQL_USER") or "rag"
    app_pw = env.get("DOCUMENT_JOBS_MYSQL_PASSWORD") or "rag_password"
    container = env.get("MYSQL_CONTAINER_NAME") or "rag_mysql"

    sql = (
        f"ALTER USER '{_sql_escape(app_user)}'@'%' IDENTIFIED BY '{_sql_escape(app_pw)}'; "
        "FLUSH PRIVILEGES;"
    )
    cmd = ["docker", "exec", container, "mysql", "-uroot", f"-p{root_pw}", "-e", sql]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(
            r.stderr
            or r.stdout
            or f"docker exec failed (code {r.returncode}). Is container {container!r} running?\n"
        )
        sys.stderr.write(
            "If root login fails, MYSQL_ROOT_PASSWORD in src/dependencies/.env must match "
            "the password set when this MySQL data volume was first created.\n"
        )
        return r.returncode or 1
    print(f"Updated MySQL user {app_user!r}@'%' password to match DOCUMENT_JOBS_MYSQL_PASSWORD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
