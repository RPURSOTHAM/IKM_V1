"""Unit tests for JWT retrieval auth middleware."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.application.consumer_api.auth_middleware import (
    JwtAuthMiddleware,
    actor_from_jwt_payload,
    authenticate_bearer_token,
    requires_api_auth,
    requires_retrieval_auth,
)
from src.application.consumer_api.context import get_current_user_from_context
from src.features.authentication.domain.authentication_exceptions import AuthenticationError
from src.features.authentication.application.token_service import JwtService


def test_requires_retrieval_auth_patterns() -> None:
    assert requires_retrieval_auth("POST", "/api/v1/retrieve") is True
    assert requires_retrieval_auth("POST", "/api/v1/retrieve/") is True
    assert requires_retrieval_auth("GET", "/api/v1/retrieve/documents") is True
    assert requires_retrieval_auth("POST", "/api/v1/repositories/abc/search/bm25") is True
    assert requires_retrieval_auth("GET", "/api/v1/repositories/abc/documents/def/chunks") is True
    assert requires_retrieval_auth("GET", "/api/v1/documents/def/chunks") is True
    assert requires_retrieval_auth("GET", "/health") is False
    assert requires_retrieval_auth("POST", "/api/v1/auth/token") is False
    assert requires_retrieval_auth("GET", "/api/v1/repositories") is False


def test_requires_api_auth_covers_platform_routes() -> None:
    assert requires_api_auth("POST", "/api/v1/documents/upload") is True
    assert requires_api_auth("GET", "/api/v1/documents/abc") is True
    assert requires_api_auth("POST", "/api/v1/platform/users") is True
    assert requires_api_auth("POST", "/api/v1/repositories/abc/access") is True
    assert requires_api_auth("POST", "/api/v1/chat") is True
    assert requires_api_auth("POST", "/api/v1/auth/token") is False
    assert requires_api_auth("GET", "/health") is False
    assert requires_api_auth("OPTIONS", "/api/v1/retrieve") is False


def _issue(user_id: str = "admin", *, platform_role: str = "administrator", exp_delta: int = 3600) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iss": "dms-platform-engine",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=exp_delta)).timestamp()),
        "jti": "test-jti-1",
        "platform_role": platform_role,
        "must_change_password": False,
    }
    with patch("src.features.authentication.application.token_service.jwt_signing_key", return_value="test-secret"), patch(
        "src.features.authentication.application.token_service.jwt_issuer", return_value="dms-platform-engine"
    ):
        return jwt.encode(payload, "test-secret", algorithm="HS256")


def test_valid_jwt_builds_actor() -> None:
    token = _issue("admin")
    with patch("src.features.authentication.application.token_service.jwt_signing_key", return_value="test-secret"), patch(
        "src.features.authentication.application.token_service.jwt_issuer", return_value="dms-platform-engine"
    ), patch(
        "src.features.users.infrastructure.user_repository.get_platform_security_store", return_value=None
    ):
        actor = authenticate_bearer_token(f"Bearer {token}")
    assert actor.user_id == "admin"
    assert actor.auth_method == "jwt"
    assert actor.is_platform_admin is True
    assert actor.platform_role == "administrator"


def test_expired_jwt_rejected() -> None:
    token = _issue("admin", exp_delta=-10)
    with patch("src.features.authentication.application.token_service.jwt_signing_key", return_value="test-secret"), patch(
        "src.features.authentication.application.token_service.jwt_issuer", return_value="dms-platform-engine"
    ), patch(
        "src.features.users.infrastructure.user_repository.get_platform_security_store", return_value=None
    ):
        with pytest.raises(AuthenticationError, match="Token expired|expired"):
            authenticate_bearer_token(f"Bearer {token}")


def test_malformed_jwt_rejected() -> None:
    with pytest.raises(AuthenticationError):
        authenticate_bearer_token("Bearer not-a-jwt")


def test_invalid_signature_rejected() -> None:
    token = _issue("admin")
    with patch("src.features.authentication.application.token_service.jwt_signing_key", return_value="other-secret"), patch(
        "src.features.authentication.application.token_service.jwt_issuer", return_value="dms-platform-engine"
    ), patch(
        "src.features.users.infrastructure.user_repository.get_platform_security_store", return_value=None
    ):
        with pytest.raises(AuthenticationError):
            authenticate_bearer_token(f"Bearer {token}")


def test_missing_authorization_header() -> None:
    with pytest.raises(AuthenticationError, match="Authentication required"):
        authenticate_bearer_token(None)


def test_actor_from_jwt_payload_standard_user() -> None:
    actor = actor_from_jwt_payload({"sub": "alice", "platform_role": "standard_user"})
    assert actor.user_id == "alice"
    assert actor.is_platform_admin is False
    assert actor.platform_role == "standard_user"


def _build_app() -> TestClient:
    app = FastAPI()
    app.add_middleware(JwtAuthMiddleware)

    @app.post("/api/v1/retrieve")
    def retrieve():
        user = get_current_user_from_context()
        return {"user_id": None if user is None else user.user_id}

    @app.post("/api/v1/documents/upload")
    def upload():
        user = get_current_user_from_context()
        return {"user_id": None if user is None else user.user_id}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return TestClient(app)


def test_middleware_missing_header_returns_401() -> None:
    client = _build_app()
    response = client.post("/api/v1/retrieve", json={"query": "x"})
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "unauthorized"
    assert "Authentication required" in body["error"]["message"]


def test_middleware_anonymous_health_allowed() -> None:
    client = _build_app()
    assert client.get("/health").status_code == 200


def test_middleware_valid_jwt_allows_retrieve() -> None:
    token = _issue("admin")
    client = _build_app()
    with patch("src.features.authentication.application.token_service.jwt_signing_key", return_value="test-secret"), patch(
        "src.features.authentication.application.token_service.jwt_issuer", return_value="dms-platform-engine"
    ), patch(
        "src.features.users.infrastructure.user_repository.get_platform_security_store", return_value=None
    ):
        response = client.post(
            "/api/v1/retrieve",
            json={"query": "x"},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 200
    assert response.json()["user_id"] == "admin"


def test_middleware_missing_header_rejects_upload() -> None:
    client = _build_app()
    response = client.post("/api/v1/documents/upload")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthorized"
