"""Shared authenticated actor helpers for retrieval unit tests."""

from __future__ import annotations

from types import SimpleNamespace

from src.application.consumer_api.context import RequestActor

ADMIN_ACTOR = RequestActor(
    user_id="admin",
    auth_method="jwt",
    platform_role="administrator",
    is_platform_admin=True,
)

SECURITY_ALLOW = SimpleNamespace(check_retrieval_access=lambda *_a, **_k: "administrator")
