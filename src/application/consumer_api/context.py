from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass


@dataclass(frozen=True)
class RequestActor:
    user_id: str = "anonymous"
    auth_method: str = "anonymous"
    platform_role: str | None = None
    is_platform_admin: bool = False
    is_admin_api_key: bool = False
    is_consumer_api_key: bool = False
    must_change_password: bool = False


_request_id: ContextVar[str | None] = ContextVar("dms_request_id", default=None)
request_id_ctx = _request_id
_current_user: ContextVar[RequestActor | None] = ContextVar("dms_current_user", default=None)


def get_request_id() -> str | None:
    return _request_id.get()


def set_request_id(request_id: str | None):
    return _request_id.set(request_id)


def get_current_user_from_context() -> RequestActor | None:
    return _current_user.get()


def set_current_user_in_context(user: RequestActor | None) -> Token:
    return _current_user.set(user)


def reset_current_user_in_context(token: Token) -> None:
    _current_user.reset(token)
