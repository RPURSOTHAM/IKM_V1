"""Resolve and bind multi-user audit context for observability events."""

from __future__ import annotations

from typing import Any

# Standard metadata keys written on every audit event when available.
USER_METADATA_KEYS = (
    "application_name",
    "user_id",
    "username",
    "email",
    "display_name",
    "role",
    "authentication_provider",
    "client_ip",
    "user_agent",
    "session_id",
    "request_id",
    "correlation_id",
)

PLATFORM_ROLE_LABELS: dict[str, str] = {
    "administrator": "Admin",
    "platform_owner": "Admin",
    "contributor": "Editor",
    "consumer": "Viewer",
    "delegate": "Editor",
}

REPOSITORY_ROLE_LABELS: dict[str, str] = {
    "owner": "Admin",
    "delegate": "Editor",
    "contributor": "Editor",
    "consumer": "Viewer",
}

AUTH_METHOD_LABELS: dict[str, str] = {
    "jwt": "Local",
    "api_key": "API Key",
    "sso": "SSO",
    "oauth": "OAuth",
}

API_KEY_USER_PROFILES: dict[str, dict[str, str]] = {
    "admin-api-key": {
        "username": "admin-api-key",
        "display_name": "Admin API Key",
        "role": "Admin",
        "authentication_provider": "API Key",
    },
    "consumer-api-key": {
        "username": "consumer-api-key",
        "display_name": "Consumer API Key",
        "role": "Viewer",
        "authentication_provider": "API Key",
    },
}


def _role_label(platform_role: str | None, repository_role: str | None) -> str | None:
    if repository_role:
        label = REPOSITORY_ROLE_LABELS.get(str(repository_role).strip().lower())
        if label:
            return label
    if platform_role:
        return PLATFORM_ROLE_LABELS.get(str(platform_role).strip().lower(), platform_role.title())
    return None


def _auth_provider_label(auth_method: str | None) -> str | None:
    if not auth_method:
        return None
    return AUTH_METHOD_LABELS.get(str(auth_method).strip().lower(), auth_method.title())


def _lookup_platform_user(user_id: str) -> dict[str, Any] | None:
    try:
        from src.features.users.infrastructure.user_repository import get_platform_security_store

        store = get_platform_security_store()
        if store is None:
            return None
        user = store.get_user(user_id)
        if user is None:
            return None
        return {
            "display_name": user.display_name,
            "platform_role": user.platform_role,
            "email": None,
        }
    except Exception:
        return None


def _lookup_repository_role(user_id: str, repository_id: str | None) -> str | None:
    if not repository_id or not user_id:
        return None
    try:
        from src.features.users.infrastructure.user_repository import get_platform_security_store

        store = get_platform_security_store()
        if store is None:
            return None
        return store.highest_repository_role(repository_id, user_id)
    except Exception:
        return None


def resolve_user_audit_context(
    *,
    user_id: str | None = None,
    auth_method: str | None = None,
    platform_role: str | None = None,
    repository_id: str | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    correlation_id: str | None = None,
    email: str | None = None,
    username: str | None = None,
    display_name: str | None = None,
    role: str | None = None,
    authentication_provider: str | None = None,
) -> dict[str, Any]:
    """Build a normalized user-context dict for audit metadata."""
    uid = (user_id or "").strip() or None
    ctx: dict[str, Any] = {}

    if uid and uid in API_KEY_USER_PROFILES:
        profile = API_KEY_USER_PROFILES[uid]
        ctx.update(
            {
                "user_id": uid,
                "username": profile.get("username", uid),
                "display_name": profile.get("display_name"),
                "role": profile.get("role"),
                "authentication_provider": profile.get("authentication_provider"),
                "email": None,
            }
        )
    elif uid:
        profile = _lookup_platform_user(uid) or {}
        repo_role = _lookup_repository_role(uid, repository_id)
        ctx["user_id"] = uid
        ctx["username"] = username or uid
        ctx["email"] = email or profile.get("email")
        ctx["display_name"] = display_name or profile.get("display_name") or uid
        ctx["role"] = role or _role_label(platform_role or profile.get("platform_role"), repo_role)
        ctx["authentication_provider"] = authentication_provider or _auth_provider_label(auth_method)
    else:
        ctx["user_id"] = None

    if client_ip:
        ctx["client_ip"] = client_ip
    if user_agent:
        ctx["user_agent"] = user_agent
    if session_id:
        ctx["session_id"] = session_id
    if request_id:
        ctx["request_id"] = request_id
    if correlation_id:
        ctx["correlation_id"] = correlation_id

    return {k: v for k, v in ctx.items() if v is not None and str(v).strip() != ""}


def enrich_audit_metadata(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge request/user context into audit metadata without overwriting explicit values."""
    from src.features.observability.middleware.request_context import current_context

    base = dict(metadata or {})
    ctx = current_context()
    extra = dict(ctx.metadata or {})
    application_name = base.get("application_name") or ctx.application_name or extra.get("application_name")
    if application_name:
        base["application_name"] = str(application_name)

    user_id = base.get("user_id") or ctx.user_id or extra.get("user_id")
    user_ctx = resolve_user_audit_context(
        user_id=str(user_id) if user_id else None,
        auth_method=extra.get("auth_method") or base.get("auth_method"),
        platform_role=extra.get("platform_role") or base.get("platform_role"),
        repository_id=base.get("repository_id") or ctx.repository_id or extra.get("repository_id"),
        client_ip=base.get("client_ip") or extra.get("client_ip"),
        user_agent=base.get("user_agent") or extra.get("user_agent"),
        session_id=base.get("session_id") or extra.get("session_id"),
        request_id=base.get("request_id") or ctx.request_id,
        correlation_id=base.get("correlation_id") or ctx.correlation_id,
        email=base.get("email") or extra.get("email"),
        username=base.get("username") or extra.get("username"),
        display_name=base.get("display_name") or extra.get("display_name"),
        role=base.get("role") or extra.get("role"),
        authentication_provider=base.get("authentication_provider") or extra.get("authentication_provider"),
    )

    for key in USER_METADATA_KEYS:
        if key in user_ctx and key not in base:
            base[key] = user_ctx[key]

    if user_ctx:
        nested_user = {k: user_ctx[k] for k in USER_METADATA_KEYS if k in user_ctx}
        if nested_user:
            existing = dict(base.get("user") or {})
            for k, v in nested_user.items():
                existing.setdefault(k, v)
            base["user"] = existing

    if ctx.repository_id and "repository_id" not in base:
        base["repository_id"] = ctx.repository_id
    if ctx.document_id and "document_id" not in base:
        base["document_id"] = ctx.document_id

    ctx_meta = dict(ctx.metadata or {})
    for key in ("component", "operation", "source"):
        if key in ctx_meta and key not in base:
            base[key] = ctx_meta[key]

    return base


def bind_audit_user_context(request: Any, user: Any | None) -> None:
    """Bind authenticated user and HTTP client metadata into observability context."""
    from src.features.observability.middleware.request_context import bind_context, current_context

    metadata: dict[str, Any] = dict(current_context().metadata or {})
    forwarded_user_id = None
    application_name = current_context().application_name

    if request is not None:
        client_ip = getattr(request, "client", None)
        if client_ip is not None and getattr(client_ip, "host", None):
            metadata["client_ip"] = client_ip.host
        headers = getattr(request, "headers", None)
        if headers is not None:
            application_name = (
                headers.get("x-application-name")
                or headers.get("x-application-id")
                or application_name
                or "unknown"
            )
            metadata["application_name"] = str(application_name).strip()[:128]
            forwarded_user_id = (headers.get("x-user-id") or "").strip() or None
            ua = headers.get("user-agent")
            if ua:
                metadata["user_agent"] = ua
            session_id = headers.get("x-session-id") or headers.get("X-Session-ID")
            if session_id:
                metadata["session_id"] = str(session_id).strip()

    user_id = None
    if user is not None:
        authenticated_user_id = getattr(user, "user_id", None)
        is_api_key = bool(
            getattr(user, "is_admin_api_key", False)
            or getattr(user, "is_consumer_api_key", False)
        )
        # Integrated applications authenticate with an API key and forward
        # their end-user identity. JWT identity always remains authoritative.
        user_id = forwarded_user_id if is_api_key and forwarded_user_id else authenticated_user_id
        metadata["auth_method"] = getattr(user, "auth_method", None)
        metadata["platform_role"] = getattr(user, "platform_role", None)
        if getattr(user, "is_admin_api_key", False):
            metadata["role"] = "Admin"
        elif getattr(user, "is_consumer_api_key", False):
            metadata["role"] = "Viewer"

    else:
        user_id = forwarded_user_id or current_context().user_id

    bind_context(
        application_name=str(application_name or "unknown"),
        user_id=str(user_id) if user_id else None,
        metadata=metadata,
    )
