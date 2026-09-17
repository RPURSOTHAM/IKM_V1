from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.features.authentication.domain.authentication_exceptions import AuthorizationError
from src.features.users.infrastructure.user_repository import PlatformSecurityStore

REPOSITORY_ROLE_RANK = {
    "consumer": 1,
    "contributor": 2,
    "delegate": 3,
    "owner": 4,
}


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: str
    platform_role: str | None
    auth_method: str  # jwt | api_key
    must_change_password: bool = False
    is_admin_api_key: bool = False
    is_consumer_api_key: bool = False

    @property
    def is_platform_admin(self) -> bool:
        if self.is_admin_api_key:
            return True
        return self.platform_role in {"administrator", "platform_owner"}

    @property
    def is_administrator(self) -> bool:
        if self.is_admin_api_key:
            return True
        return self.platform_role == "administrator"


def require_platform_admin(user: AuthenticatedUser) -> None:
    if not user.is_platform_admin:
        raise AuthorizationError("Platform administrator or owner role required")


def require_administrator(user: AuthenticatedUser) -> None:
    if not user.is_administrator:
        raise AuthorizationError("Platform administrator role required")


ensure_platform_admin = require_platform_admin
ensure_administrator = require_administrator


def require_repository_role(
    store: PlatformSecurityStore,
    user: AuthenticatedUser,
    repository_id: str,
    minimum_role: str,
) -> str:
    if user.is_platform_admin:
        return user.platform_role or "administrator"

    role = store.highest_repository_role(repository_id, user.user_id)
    if role is None:
        raise AuthorizationError("No repository access for this user")

    min_rank = REPOSITORY_ROLE_RANK.get(minimum_role, 0)
    user_rank = REPOSITORY_ROLE_RANK.get(role, 0)
    if user_rank < min_rank:
        raise AuthorizationError(f"Requires repository role {minimum_role} or higher")
    return role


ensure_repository_role = require_repository_role


def can_grant_repository_role(granter: AuthenticatedUser, role: str) -> None:
    if granter.is_platform_admin:
        return
    if role == "delegate":
        raise AuthorizationError("Only platform administrators can grant delegate role")
    raise AuthorizationError("Only repository owners or platform admins can grant roles")


def user_to_audit_actor(user: AuthenticatedUser, repository_role: str | None = None) -> dict[str, Any]:
    return {
        "actor_user_id": user.user_id,
        "actor_platform_role": user.platform_role,
        "actor_repository_role": repository_role,
    }
