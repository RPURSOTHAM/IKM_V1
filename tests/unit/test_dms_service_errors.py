"""Unit tests for DMS structured error handling."""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.application.consumer_api.context import request_id_ctx
from src.application.consumer_api.errors.handlers import register_exception_handlers
from src.shared.errors import (
    COMPONENT_DOCUMENT_PREVIEW,
    DmsServiceError,
    raise_client_error,
    raise_service_error,
)


def test_raise_service_error_logs_and_raises(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.ERROR, logger="dms_service.errors")
    token = request_id_ctx.set("req-test-1")
    try:
        with pytest.raises(DmsServiceError) as exc_info:
            raise_service_error(
                COMPONENT_DOCUMENT_PREVIEW,
                code="preview_conversion_failed",
                http_status=500,
                user_message="Document preview could not be generated.",
                reason="conversion exploded",
                cause=RuntimeError("boom"),
            )
    finally:
        request_id_ctx.reset(token)

    err = exc_info.value
    assert err.user_message == "Document preview could not be generated."
    assert err.code == "preview_conversion_failed"
    assert "component=document_preview" in caplog.text
    assert "request_id=req-test-1" in caplog.text
    assert "conversion exploded" in caplog.text


def test_raise_client_error_uses_warning_level(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dms_service.errors")
    with pytest.raises(DmsServiceError):
        raise_client_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="unsupported_media_type",
            http_status=415,
            user_message="Unsupported format.",
            reason="bad suffix",
        )
    assert "component=document_preview" in caplog.text


def test_api_error_response_shape() -> None:
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise_service_error(
            COMPONENT_DOCUMENT_PREVIEW,
            code="preview_conversion_failed",
            http_status=500,
            user_message="Document preview could not be generated.",
            reason="test failure",
        )

    client = TestClient(app)
    response = client.get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "preview_conversion_failed"
    assert body["error"]["message"] == "Document preview could not be generated."


def test_http_exception_is_logged_and_user_friendly(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger="dms_service.errors")
    app = FastAPI()
    register_exception_handlers(app)

    @app.get("/missing")
    def missing() -> None:
        raise HTTPException(status_code=404, detail="Document not found")

    client = TestClient(app)
    response = client.get("/missing")
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "Document not found"
    assert "component=consumer_api_service" in caplog.text
