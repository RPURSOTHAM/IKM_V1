from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from src.features.authentication.configuration.authentication_config import jwt_access_ttl_seconds, jwt_issuer, jwt_signing_key
from src.features.authentication.domain.authentication_exceptions import AuthenticationError
from src.features.users.infrastructure.user_repository import PlatformSecurityStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JwtService:
    def __init__(self, store: PlatformSecurityStore | None = None) -> None:
        self._store = store

    def issue_token(
        self,
        *,
        user_id: str,
        platform_role: str | None,
        must_change_password: bool = False,
    ) -> dict[str, Any]:
        now = _utc_now()
        ttl = jwt_access_ttl_seconds()
        jti = str(uuid.uuid4())
        payload = {
            "sub": user_id,
            "iss": jwt_issuer(),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=ttl)).timestamp()),
            "jti": jti,
            "platform_role": platform_role,
            "must_change_password": must_change_password,
        }
        token = jwt.encode(payload, jwt_signing_key(), algorithm="HS256")
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": ttl,
            "must_change_password": must_change_password,
        }

    def decode_token(self, token: str) -> dict[str, Any]:
        try:
            payload = jwt.decode(
                token,
                jwt_signing_key(),
                algorithms=["HS256"],
                issuer=jwt_issuer(),
                options={"require": ["exp", "sub", "iss", "jti"]},
            )
        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError("Token expired") from exc
        except jwt.InvalidTokenError as exc:
            raise AuthenticationError("Invalid token") from exc

        jti = str(payload.get("jti", ""))
        if self._store and jti and self._store.is_token_revoked(jti):
            raise AuthenticationError("Token revoked")
        return payload

    def revoke_token(self, token: str) -> None:
        payload = self.decode_token(token)
        jti = str(payload.get("jti", ""))
        if self._store and jti:
            self._store.revoke_token(jti)


def looks_like_jwt(value: str) -> bool:
    parts = value.split(".")
    return len(parts) == 3 and all(parts)
