from __future__ import annotations

import os


def jwt_signing_key() -> str:
    key = (os.getenv("JWT_SIGNING_KEY") or os.getenv("PLATFORM_JWT_SECRET") or "").strip()
    if not key:
        key = (os.getenv("CONSUMER_API_ADMIN_KEYS") or os.getenv("RAG_API_ADMIN_KEYS") or "dev-insecure-change-me").split(",")[0].strip()
    return key


def jwt_issuer() -> str:
    return (os.getenv("JWT_ISSUER") or "dms-platform-engine").strip()


def jwt_access_ttl_seconds() -> int:
    return int(os.getenv("JWT_ACCESS_TTL_SECONDS", "3600"))


def bootstrap_admin_user() -> str:
    return (os.getenv("PLATFORM_BOOTSTRAP_ADMIN_USER") or "admin").strip()


def bootstrap_admin_password() -> str | None:
    raw = os.getenv("PLATFORM_BOOTSTRAP_ADMIN_PASSWORD")
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


def security_mode() -> str:
    return (os.getenv("PLATFORM_SECURITY_MODE") or "dual").strip().lower()
