from __future__ import annotations

import json
import logging
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Mapping

from src.infrastructure.database.postgres import PostgresConnectionParams, connect

logger = logging.getLogger(__name__)

REPOSITORY_ROLES = frozenset({"owner", "delegate", "contributor", "consumer"})
PLATFORM_ROLES = frozenset({"administrator", "platform_owner"})


def _utc_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _parse_json(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return json.loads(value) if value else None
    return value


def _row_get(row: Mapping[str, Any] | Any, key: str, index: int | None = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key)
    if index is not None:
        return row[index]
    raise TypeError(f"Unsupported row type for key {key!r}")


@dataclass
class PlatformUser:
    user_id: str
    display_name: str | None
    platform_role: str | None
    status: str
    must_change_password: bool
    password_hash: str
    created_at: datetime
    updated_at: datetime
    created_by: str | None


@dataclass
class RegistrationRequest:
    request_id: str
    requested_by: str
    proposed_name: str
    description: str | None
    requested_settings: dict[str, Any] | None
    status: str
    reviewed_by: str | None
    review_notes: str | None
    repository_id: str | None
    created_at: datetime
    reviewed_at: datetime | None


@dataclass
class RoleBinding:
    binding_id: str
    repository_id: str
    user_id: str
    role: str
    granted_by: str
    granted_at: datetime
    revoked_at: datetime | None


class PlatformSecurityStore:
    CREATE_USER = """
    CREATE TABLE IF NOT EXISTS platform_user (
      user_id VARCHAR(128) NOT NULL,
      password_hash VARCHAR(256) NOT NULL,
      display_name VARCHAR(256) NULL,
      platform_role VARCHAR(32) NULL,
      status VARCHAR(16) NOT NULL DEFAULT 'active',
      must_change_password BOOLEAN NOT NULL DEFAULT FALSE,
      failed_login_count INT NOT NULL DEFAULT 0,
      locked_until TIMESTAMPTZ NULL,
      created_at TIMESTAMPTZ NOT NULL,
      updated_at TIMESTAMPTZ NOT NULL,
      created_by VARCHAR(128) NULL,
      PRIMARY KEY (user_id)
    );
    """

    CREATE_REVOKED_TOKEN = """
    CREATE TABLE IF NOT EXISTS platform_revoked_token (
      jti CHAR(36) NOT NULL,
      revoked_at TIMESTAMPTZ NOT NULL,
      PRIMARY KEY (jti)
    );
    """

    CREATE_REGISTRATION = """
    CREATE TABLE IF NOT EXISTS repository_registration_request (
      request_id CHAR(36) NOT NULL,
      requested_by VARCHAR(128) NOT NULL,
      proposed_name VARCHAR(256) NOT NULL,
      description TEXT NULL,
      requested_settings JSON NULL,
      status VARCHAR(16) NOT NULL DEFAULT 'pending',
      reviewed_by VARCHAR(128) NULL,
      review_notes TEXT NULL,
      repository_id CHAR(36) NULL,
      created_at TIMESTAMPTZ NOT NULL,
      reviewed_at TIMESTAMPTZ NULL,
      PRIMARY KEY (request_id)
    );
    """

    CREATE_BINDING = """
    CREATE TABLE IF NOT EXISTS repository_role_binding (
      binding_id CHAR(36) NOT NULL,
      repository_id CHAR(36) NOT NULL,
      user_id VARCHAR(128) NOT NULL,
      role VARCHAR(32) NOT NULL,
      granted_by VARCHAR(128) NOT NULL,
      granted_at TIMESTAMPTZ NOT NULL,
      revoked_at TIMESTAMPTZ NULL,
      PRIMARY KEY (binding_id)
    );
    """

    CREATE_AUDIT = """
    CREATE TABLE IF NOT EXISTS platform_audit_event (
      audit_id CHAR(36) NOT NULL,
      event_time TIMESTAMPTZ NOT NULL,
      event_category VARCHAR(64) NOT NULL,
      event_type VARCHAR(128) NOT NULL,
      actor_user_id VARCHAR(128) NULL,
      actor_platform_role VARCHAR(32) NULL,
      actor_repository_role VARCHAR(32) NULL,
      repository_id CHAR(36) NULL,
      document_id CHAR(36) NULL,
      resource_type VARCHAR(64) NULL,
      resource_id VARCHAR(256) NULL,
      action VARCHAR(16) NOT NULL,
      outcome VARCHAR(16) NOT NULL,
      old_value_json JSON NULL,
      new_value_json JSON NULL,
      change_reason VARCHAR(512) NULL,
      request_id VARCHAR(64) NULL,
      client_ip VARCHAR(64) NULL,
      service_component VARCHAR(64) NOT NULL DEFAULT 'dms_api',
      PRIMARY KEY (audit_id)
    );
    """

    CREATE_INDEXES = (
        "CREATE INDEX IF NOT EXISTS idx_platform_user_role ON platform_user (platform_role)",
        "CREATE INDEX IF NOT EXISTS idx_platform_user_status ON platform_user (status)",
        "CREATE INDEX IF NOT EXISTS idx_repo_reg_status ON repository_registration_request (status)",
        "CREATE INDEX IF NOT EXISTS idx_repo_reg_requester ON repository_registration_request (requested_by)",
        "CREATE INDEX IF NOT EXISTS idx_repo_role_repo ON repository_role_binding (repository_id)",
        "CREATE INDEX IF NOT EXISTS idx_repo_role_user ON repository_role_binding (user_id)",
        "CREATE INDEX IF NOT EXISTS idx_repo_role_active ON repository_role_binding (repository_id, user_id, role)",
        "CREATE INDEX IF NOT EXISTS idx_audit_time ON platform_audit_event (event_time)",
        "CREATE INDEX IF NOT EXISTS idx_audit_actor ON platform_audit_event (actor_user_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_repo ON platform_audit_event (repository_id)",
        "CREATE INDEX IF NOT EXISTS idx_audit_category ON platform_audit_event (event_category)",
    )

    def __init__(self, params: PostgresConnectionParams) -> None:
        self._params = params

    @contextmanager
    def _conn(self) -> Iterator[Any]:
        conn = connect(self._params, autocommit=False)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._conn() as conn:
            with conn.cursor() as cur:
                for ddl in (
                    self.CREATE_USER,
                    self.CREATE_REVOKED_TOKEN,
                    self.CREATE_REGISTRATION,
                    self.CREATE_BINDING,
                    self.CREATE_AUDIT,
                ):
                    cur.execute(ddl)
                for idx_sql in self.CREATE_INDEXES:
                    cur.execute(idx_sql)

    def ping(self) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            return True
        except Exception:
            return False

    def has_administrator(self) -> bool:
        sql = "SELECT COUNT(*) AS cnt FROM platform_user WHERE platform_role = 'administrator'"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                row = cur.fetchone()
        return int(_row_get(row, "cnt", 0)) > 0

    def get_user(self, user_id: str) -> PlatformUser | None:
        sql = """
        SELECT user_id, password_hash, display_name, platform_role, status,
               must_change_password, created_at, updated_at, created_by
        FROM platform_user WHERE user_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                row = cur.fetchone()
        return self._row_user(row) if row else None

    def insert_user(
        self,
        *,
        user_id: str,
        password_hash: str,
        platform_role: str | None,
        display_name: str | None,
        must_change_password: bool,
        created_by: str | None,
    ) -> PlatformUser:
        now = _utc_naive()
        sql = """
        INSERT INTO platform_user (
          user_id, password_hash, display_name, platform_role, status,
          must_change_password, created_at, updated_at, created_by
        ) VALUES (%s, %s, %s, %s, 'active', %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (user_id, password_hash, display_name, platform_role, must_change_password, now, now, created_by),
                )
        user = self.get_user(user_id)
        assert user is not None
        return user

    def update_password(self, user_id: str, password_hash: str, *, must_change_password: bool = False) -> None:
        now = _utc_naive()
        sql = """
        UPDATE platform_user SET password_hash=%s, must_change_password=%s, updated_at=%s
        WHERE user_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (password_hash, must_change_password, now, user_id))

    def update_user_status(self, user_id: str, status: str) -> None:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE platform_user SET status=%s, updated_at=%s WHERE user_id=%s",
                    (status, now, user_id),
                )

    def update_display_name(self, user_id: str, display_name: str | None) -> None:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE platform_user SET display_name=%s, updated_at=%s WHERE user_id=%s",
                    (display_name, now, user_id),
                )

    def upsert_external_display_profile(
        self,
        *,
        user_id: str,
        display_name: str,
        created_by: str | None = None,
    ) -> None:
        uid = str(user_id or "").strip()
        name = str(display_name or "").strip()
        if not uid or not name:
            return

        existing = self.get_user(uid)
        if existing is not None:
            if name != str(existing.display_name or "").strip():
                self.update_display_name(uid, name)
            return

        import secrets

        from src.features.authentication.application.password_service import hash_password

        self.insert_user(
            user_id=uid,
            password_hash=hash_password(secrets.token_urlsafe(48)),
            platform_role=None,
            display_name=name,
            must_change_password=False,
            created_by=created_by,
        )
        self.update_user_status(uid, "disabled")

    def set_platform_role(self, user_id: str, platform_role: str | None) -> None:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE platform_user SET platform_role=%s, updated_at=%s WHERE user_id=%s",
                    (platform_role, now, user_id),
                )

    def record_login_failure(self, user_id: str, *, lock_after: int = 5, lock_minutes: int = 15) -> None:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE platform_user SET failed_login_count = failed_login_count + 1, updated_at=%s WHERE user_id=%s",
                    (now, user_id),
                )
                cur.execute("SELECT failed_login_count FROM platform_user WHERE user_id=%s", (user_id,))
                row = cur.fetchone()
                if row and int(_row_get(row, "failed_login_count", 0)) >= lock_after:
                    from datetime import timedelta

                    locked = now + timedelta(minutes=lock_minutes)
                    cur.execute(
                        "UPDATE platform_user SET status='locked', locked_until=%s WHERE user_id=%s",
                        (locked, user_id),
                    )

    def clear_login_failures(self, user_id: str) -> None:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE platform_user SET failed_login_count=0, status='active', locked_until=NULL, updated_at=%s
                    WHERE user_id=%s
                    """,
                    (now, user_id),
                )

    def list_users(self) -> list[PlatformUser]:
        sql = """
        SELECT user_id, password_hash, display_name, platform_role, status,
               must_change_password, created_at, updated_at, created_by
        FROM platform_user ORDER BY user_id
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                rows = cur.fetchall()
        return [self._row_user(row) for row in rows]

    def get_user_display_names(self, user_ids: set[str]) -> dict[str, str]:
        ids = {str(user_id).strip() for user_id in user_ids if str(user_id or "").strip()}
        if not ids:
            return {}

        placeholders = ", ".join(["%s"] * len(ids))
        sql = f"""
        SELECT user_id, display_name
        FROM platform_user
        WHERE user_id IN ({placeholders})
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(ids))
                rows = cur.fetchall()

        names: dict[str, str] = {}
        for row in rows:
            user_id = str(_row_get(row, "user_id", 0)).strip()
            display_raw = _row_get(row, "display_name", 1)
            display_name = str(display_raw).strip() if display_raw is not None else ""
            if user_id and display_name:
                names[user_id] = display_name
        return names

    def revoke_token(self, jti: str) -> None:
        now = _utc_naive()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO platform_revoked_token (jti, revoked_at) VALUES (%s, %s)
                    ON CONFLICT (jti) DO NOTHING
                    """,
                    (jti, now),
                )

    def is_token_revoked(self, jti: str) -> bool:
        try:
            with self._conn() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1 AS one FROM platform_revoked_token WHERE jti=%s", (jti,))
                    return cur.fetchone() is not None
        except Exception as exc:
            # Fail-open on revoke lookup so auth does not 500 when Postgres is down.
            _logger = __import__("logging").getLogger(__name__)
            _logger.warning("token revocation check skipped (store unavailable): %s", exc)
            return False

    def insert_registration_request(
        self,
        *,
        requested_by: str,
        proposed_name: str,
        description: str | None,
        requested_settings: dict[str, Any] | None,
    ) -> RegistrationRequest:
        request_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO repository_registration_request (
          request_id, requested_by, proposed_name, description, requested_settings,
          status, created_at
        ) VALUES (%s, %s, %s, %s, %s, 'pending', %s)
        """
        payload = json.dumps(requested_settings) if requested_settings else None
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (request_id, requested_by, proposed_name, description, payload, now))
        req = self.get_registration_request(request_id)
        assert req is not None
        return req

    def get_registration_request(self, request_id: str) -> RegistrationRequest | None:
        sql = """
        SELECT request_id, requested_by, proposed_name, description, requested_settings,
               status, reviewed_by, review_notes, repository_id, created_at, reviewed_at
        FROM repository_registration_request WHERE request_id = %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (request_id,))
                row = cur.fetchone()
        return self._row_registration(row) if row else None

    def list_registration_requests(self, *, status: str | None = None) -> list[RegistrationRequest]:
        clauses = ["1=1"]
        params: list[Any] = []
        if status:
            clauses.append("status = %s")
            params.append(status)
        sql = f"""
        SELECT request_id, requested_by, proposed_name, description, requested_settings,
               status, reviewed_by, review_notes, repository_id, created_at, reviewed_at
        FROM repository_registration_request WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_registration(row) for row in rows]

    def update_registration_request(
        self,
        request_id: str,
        *,
        status: str,
        reviewed_by: str,
        review_notes: str | None,
        repository_id: str | None = None,
    ) -> RegistrationRequest:
        now = _utc_naive()
        sql = """
        UPDATE repository_registration_request SET
          status=%s, reviewed_by=%s, review_notes=%s, repository_id=%s, reviewed_at=%s
        WHERE request_id=%s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (status, reviewed_by, review_notes, repository_id, now, request_id))
        req = self.get_registration_request(request_id)
        assert req is not None
        return req

    def grant_repository_role(
        self,
        *,
        repository_id: str,
        user_id: str,
        role: str,
        granted_by: str,
    ) -> RoleBinding:
        if role not in REPOSITORY_ROLES:
            raise ValueError(f"Invalid repository role: {role}")
        binding_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO repository_role_binding (
          binding_id, repository_id, user_id, role, granted_by, granted_at
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE repository_role_binding SET revoked_at=%s WHERE repository_id=%s AND user_id=%s AND role=%s AND revoked_at IS NULL",
                    (now, repository_id, user_id, role),
                )
                cur.execute(sql, (binding_id, repository_id, user_id, role, granted_by, now))
        binding = self.get_active_binding(repository_id, user_id, role)
        assert binding is not None
        return binding

    def delete_repository_bindings(self, repository_id: str) -> None:
        sql = "DELETE FROM repository_role_binding WHERE repository_id = %s"
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id,))

    def revoke_repository_role(self, repository_id: str, user_id: str, role: str) -> bool:
        now = _utc_naive()
        sql = """
        UPDATE repository_role_binding SET revoked_at=%s
        WHERE repository_id=%s AND user_id=%s AND role=%s AND revoked_at IS NULL
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, repository_id, user_id, role))
                return cur.rowcount > 0

    def revoke_all_repository_bindings(self, repository_id: str) -> int:
        now = _utc_naive()
        sql = """
        UPDATE repository_role_binding SET revoked_at=%s
        WHERE repository_id=%s AND revoked_at IS NULL
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (now, repository_id))
                return int(cur.rowcount)

    def get_active_binding(self, repository_id: str, user_id: str, role: str) -> RoleBinding | None:
        sql = """
        SELECT binding_id, repository_id, user_id, role, granted_by, granted_at, revoked_at
        FROM repository_role_binding
        WHERE repository_id=%s AND user_id=%s AND role=%s AND revoked_at IS NULL
        ORDER BY granted_at DESC LIMIT 1
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id, user_id, role))
                row = cur.fetchone()
        return self._row_binding(row) if row else None

    def highest_repository_role(self, repository_id: str, user_id: str) -> str | None:
        order = {"owner": 4, "delegate": 3, "contributor": 2, "consumer": 1}
        sql = """
        SELECT role FROM repository_role_binding
        WHERE repository_id=%s AND user_id=%s AND revoked_at IS NULL
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id, user_id))
                rows = cur.fetchall()
        if not rows:
            return None
        roles = [str(_row_get(r, "role", 0)) for r in rows]
        return max(roles, key=lambda r: order.get(r, 0))

    def list_user_repository_ids(self, user_id: str) -> list[str]:
        sql = """
        SELECT DISTINCT repository_id
        FROM repository_role_binding
        WHERE user_id=%s AND revoked_at IS NULL
        ORDER BY repository_id
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (user_id,))
                rows = cur.fetchall()
        return [str(_row_get(row, "repository_id", 0)) for row in rows if _row_get(row, "repository_id", 0)]

    def list_repository_members(self, repository_id: str) -> list[RoleBinding]:
        sql = """
        SELECT binding_id, repository_id, user_id, role, granted_by, granted_at, revoked_at
        FROM repository_role_binding
        WHERE repository_id=%s AND revoked_at IS NULL
        ORDER BY role, user_id
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (repository_id,))
                rows = cur.fetchall()
        return [self._row_binding(row) for row in rows]

    def insert_audit_event(self, event: dict[str, Any]) -> str:
        audit_id = str(uuid.uuid4())
        now = _utc_naive()
        sql = """
        INSERT INTO platform_audit_event (
          audit_id, event_time, event_category, event_type, actor_user_id, actor_platform_role,
          actor_repository_role, repository_id, document_id, resource_type, resource_id,
          action, outcome, old_value_json, new_value_json, change_reason, request_id,
          client_ip, service_component
        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        audit_id,
                        now,
                        event["event_category"],
                        event["event_type"],
                        event.get("actor_user_id"),
                        event.get("actor_platform_role"),
                        event.get("actor_repository_role"),
                        event.get("repository_id"),
                        event.get("document_id"),
                        event.get("resource_type"),
                        event.get("resource_id"),
                        event["action"],
                        event["outcome"],
                        json.dumps(event.get("old_value_json")) if event.get("old_value_json") is not None else None,
                        json.dumps(event.get("new_value_json")) if event.get("new_value_json") is not None else None,
                        event.get("change_reason"),
                        event.get("request_id"),
                        event.get("client_ip"),
                        event.get("service_component", "dms_api"),
                    ),
                )
        return audit_id

    def list_audit_events(
        self,
        *,
        repository_id: str | None = None,
        category: str | None = None,
        event_type: str | None = None,
        document_id: str | None = None,
        actor_user_id: str | None = None,
        since: datetime | None = None,
        limit: int = 50,
        ascending: bool = False,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if repository_id:
            clauses.append("repository_id = %s")
            params.append(repository_id)
        if category:
            clauses.append("event_category = %s")
            params.append(category)
        if event_type:
            clauses.append("event_type = %s")
            params.append(event_type)
        if document_id:
            clauses.append("document_id = %s")
            params.append(document_id)
        if actor_user_id:
            clauses.append("actor_user_id = %s")
            params.append(actor_user_id)
        if since:
            clauses.append("event_time >= %s")
            params.append(since)
        params.append(min(max(limit, 1), 500))
        order = "ASC" if ascending else "DESC"
        sql = f"""
        SELECT audit_id, event_time, event_category, event_type, actor_user_id, actor_platform_role,
               actor_repository_role, repository_id, document_id, resource_type, resource_id,
               action, outcome, old_value_json, new_value_json, change_reason, request_id,
               client_ip, service_component
        FROM platform_audit_event WHERE {' AND '.join(clauses)}
        ORDER BY event_time {order} LIMIT %s
        """
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [self._row_audit(row) for row in rows]

    @staticmethod
    def _row_user(row: Any) -> PlatformUser:
        return PlatformUser(
            user_id=str(_row_get(row, "user_id", 0)),
            password_hash=str(_row_get(row, "password_hash", 1)),
            display_name=(
                str(_row_get(row, "display_name", 2))
                if _row_get(row, "display_name", 2) is not None
                else None
            ),
            platform_role=(
                str(_row_get(row, "platform_role", 3))
                if _row_get(row, "platform_role", 3) is not None
                else None
            ),
            status=str(_row_get(row, "status", 4)),
            must_change_password=bool(_row_get(row, "must_change_password", 5)),
            created_at=_parse_dt(_row_get(row, "created_at", 6)),
            updated_at=_parse_dt(_row_get(row, "updated_at", 7)),
            created_by=(
                str(_row_get(row, "created_by", 8))
                if _row_get(row, "created_by", 8) is not None
                else None
            ),
        )

    @staticmethod
    def _row_registration(row: Any) -> RegistrationRequest:
        return RegistrationRequest(
            request_id=str(_row_get(row, "request_id", 0)),
            requested_by=str(_row_get(row, "requested_by", 1)),
            proposed_name=str(_row_get(row, "proposed_name", 2)),
            description=(
                str(_row_get(row, "description", 3))
                if _row_get(row, "description", 3) is not None
                else None
            ),
            requested_settings=_parse_json(_row_get(row, "requested_settings", 4)),
            status=str(_row_get(row, "status", 5)),
            reviewed_by=(
                str(_row_get(row, "reviewed_by", 6))
                if _row_get(row, "reviewed_by", 6) is not None
                else None
            ),
            review_notes=(
                str(_row_get(row, "review_notes", 7))
                if _row_get(row, "review_notes", 7) is not None
                else None
            ),
            repository_id=(
                str(_row_get(row, "repository_id", 8))
                if _row_get(row, "repository_id", 8) is not None
                else None
            ),
            created_at=_parse_dt(_row_get(row, "created_at", 9)),
            reviewed_at=(
                _parse_dt(_row_get(row, "reviewed_at", 10))
                if _row_get(row, "reviewed_at", 10) is not None
                else None
            ),
        )

    @staticmethod
    def _row_binding(row: Any) -> RoleBinding:
        return RoleBinding(
            binding_id=str(_row_get(row, "binding_id", 0)),
            repository_id=str(_row_get(row, "repository_id", 1)),
            user_id=str(_row_get(row, "user_id", 2)),
            role=str(_row_get(row, "role", 3)),
            granted_by=str(_row_get(row, "granted_by", 4)),
            granted_at=_parse_dt(_row_get(row, "granted_at", 5)),
            revoked_at=(
                _parse_dt(_row_get(row, "revoked_at", 6))
                if _row_get(row, "revoked_at", 6) is not None
                else None
            ),
        )

    @staticmethod
    def _row_audit(row: Any) -> dict[str, Any]:
        return {
            "audit_id": str(_row_get(row, "audit_id", 0)),
            "event_time": _parse_dt(_row_get(row, "event_time", 1)).isoformat(),
            "event_category": str(_row_get(row, "event_category", 2)),
            "event_type": str(_row_get(row, "event_type", 3)),
            "actor_user_id": (
                str(_row_get(row, "actor_user_id", 4))
                if _row_get(row, "actor_user_id", 4) is not None
                else None
            ),
            "actor_platform_role": (
                str(_row_get(row, "actor_platform_role", 5))
                if _row_get(row, "actor_platform_role", 5) is not None
                else None
            ),
            "actor_repository_role": (
                str(_row_get(row, "actor_repository_role", 6))
                if _row_get(row, "actor_repository_role", 6) is not None
                else None
            ),
            "repository_id": (
                str(_row_get(row, "repository_id", 7))
                if _row_get(row, "repository_id", 7) is not None
                else None
            ),
            "document_id": (
                str(_row_get(row, "document_id", 8))
                if _row_get(row, "document_id", 8) is not None
                else None
            ),
            "resource_type": (
                str(_row_get(row, "resource_type", 9))
                if _row_get(row, "resource_type", 9) is not None
                else None
            ),
            "resource_id": (
                str(_row_get(row, "resource_id", 10))
                if _row_get(row, "resource_id", 10) is not None
                else None
            ),
            "action": str(_row_get(row, "action", 11)),
            "outcome": str(_row_get(row, "outcome", 12)),
            "old_value_json": _parse_json(_row_get(row, "old_value_json", 13)),
            "new_value_json": _parse_json(_row_get(row, "new_value_json", 14)),
            "change_reason": (
                str(_row_get(row, "change_reason", 15))
                if _row_get(row, "change_reason", 15) is not None
                else None
            ),
            "request_id": (
                str(_row_get(row, "request_id", 16))
                if _row_get(row, "request_id", 16) is not None
                else None
            ),
            "client_ip": (
                str(_row_get(row, "client_ip", 17))
                if _row_get(row, "client_ip", 17) is not None
                else None
            ),
            "service_component": str(_row_get(row, "service_component", 18)),
        }


_store: PlatformSecurityStore | None = None


def postgres_params_for_security():
    from src.features.configuration.configuration.postgres_config import postgres_params_for_platform_config

    return postgres_params_for_platform_config()


# Backward-compatible alias.
mysql_params_for_security = postgres_params_for_security


def get_platform_security_store() -> PlatformSecurityStore | None:
    global _store
    if _store is not None:
        return _store
    params = postgres_params_for_security()
    if params is None:
        return None
    _store = PlatformSecurityStore(params)
    _store.ensure_schema()
    return _store
