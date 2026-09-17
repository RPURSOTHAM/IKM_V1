"""Auth + access control tests for retrieval service boundaries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.application.consumer_api.context import RequestActor
from src.features.authorization.application.authorization_service import AuthenticatedUser
from src.features.authentication.domain.authentication_exceptions import AuthorizationError
from src.features.users.application.user_service import PlatformSecurityService


def test_admin_user_passes_retrieval_access() -> None:
    service = PlatformSecurityService()
    actor = AuthenticatedUser(user_id="admin", platform_role="administrator", auth_method="jwt")
    with patch.object(service, "_require_store"), patch.object(
        service,
        "_repos",
        return_value=MagicMock(get_repository=lambda _rid: {"status": "active"}),
    ):
        role = service.check_retrieval_access(actor, "11111111-1111-1111-1111-111111111111")
    assert role in {"administrator", "platform_owner", "api_key"} or role == "administrator"


def test_unauthorized_repository_access_raises_403() -> None:
    service = PlatformSecurityService()
    actor = AuthenticatedUser(user_id="alice", platform_role="standard_user", auth_method="jwt")
    store = MagicMock()
    store.highest_repository_role.return_value = None
    with patch.object(service, "_require_store", return_value=store), patch.object(
        service,
        "_repos",
        return_value=MagicMock(get_repository=lambda _rid: {"status": "active"}),
    ):
        with pytest.raises(AuthorizationError, match="No repository access"):
            service.check_retrieval_access(actor, "11111111-1111-1111-1111-111111111111")


def test_repository_owner_passes_retrieval_access() -> None:
    service = PlatformSecurityService()
    actor = AuthenticatedUser(user_id="owner1", platform_role="standard_user", auth_method="jwt")
    store = MagicMock()
    store.highest_repository_role.return_value = "owner"
    with patch.object(service, "_require_store", return_value=store), patch.object(
        service,
        "_repos",
        return_value=MagicMock(get_repository=lambda _rid: {"status": "active"}),
    ):
        role = service.check_retrieval_access(actor, "11111111-1111-1111-1111-111111111111")
    assert role == "owner"


def test_request_actor_duck_types_for_admin_check() -> None:
    """RequestActor from middleware must satisfy check_retrieval_access duck typing."""
    service = PlatformSecurityService()
    actor = RequestActor(
        user_id="admin",
        auth_method="jwt",
        platform_role="administrator",
        is_platform_admin=True,
    )
    with patch.object(service, "_require_store"), patch.object(
        service,
        "_repos",
        return_value=MagicMock(get_repository=lambda _rid: {"status": "active"}),
    ):
        role = service.check_retrieval_access(actor, "11111111-1111-1111-1111-111111111111")  # type: ignore[arg-type]
    assert role == "administrator"
