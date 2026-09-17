"""HTTP middleware capturing request lifecycle metrics and context."""

from __future__ import annotations

import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.features.observability.audit.application.audit_service import get_audit_service
from src.features.observability.metrics.application.metrics_service import get_metrics_service
from src.features.observability.middleware.request_context import (
    bind_context,
    current_context,
    reset_context,
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    request_header = "X-Request-ID"
    correlation_header = "X-Correlation-ID"
    application_headers = ("X-Application-Name", "X-Application-ID")

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        request_id = (request.headers.get(self.request_header) or "").strip() or str(uuid.uuid4())
        correlation_id = (request.headers.get(self.correlation_header) or "").strip() or request_id
        application_name = self._extract_application_name(request)
        user_id = self._extract_user_id(request)
        repository_id = self._extract_path_param(request, "repository_id")
        document_id = self._extract_path_param(request, "document_id")

        client_ip = self._client_ip(request)
        user_agent = (request.headers.get("user-agent") or "").strip() or None
        session_id = (request.headers.get("x-session-id") or request.headers.get("X-Session-ID") or "").strip() or None
        request_metadata = {
            k: v
            for k, v in {
                "client_ip": client_ip,
                "user_agent": user_agent,
                "session_id": session_id,
                "application_name": application_name,
            }.items()
            if v
        }
        tokens = bind_context(
            request_id=request_id,
            correlation_id=correlation_id,
            application_name=application_name,
            user_id=user_id,
            repository_id=repository_id,
            document_id=document_id,
            endpoint=request.url.path,
            http_method=request.method,
            metadata=request_metadata,
        )
        dms_token = None
        try:
            from src.application.consumer_api.context import request_id_ctx

            dms_token = request_id_ctx.set(request_id)
        except Exception:
            pass
        status_code = 500
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers[self.request_header] = request_id
            response.headers[self.correlation_header] = correlation_id
            return response
        finally:
            duration_ms = (time.perf_counter() - start) * 1000.0
            status = "success" if 200 <= status_code < 400 else "failure"
            request_size = int(request.headers.get("content-length") or 0)
            response_size = int(response.headers.get("content-length") or 0) if response is not None else 0
            try:
                get_metrics_service().record_http_request(
                    endpoint=request.url.path,
                    http_method=request.method,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    status=status,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    user_id=user_id,
                    repository_id=repository_id,
                    document_id=document_id,
                    request_size=request_size,
                    response_size=response_size,
                )
            except Exception:
                pass
            try:
                resolved_context = current_context()
                action = self._classify_action(request.method, request.url.path)
                get_audit_service().record(
                    "APPLICATION_REQUEST",
                    category="application",
                    action=action,
                    source="API",
                    entity_type="api_resource",
                    entity_id=request.url.path,
                    application_name=resolved_context.application_name or application_name,
                    user_id=resolved_context.user_id or user_id,
                    duration_ms=duration_ms,
                    status=status,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    metadata={
                        "http_method": request.method,
                        "endpoint": request.url.path,
                        "status_code": status_code,
                        "request_size": request_size,
                        "response_size": response_size,
                    },
                )
            except Exception:
                pass

            reset_context(tokens)
            if dms_token is not None:
                try:
                    from src.application.consumer_api.context import request_id_ctx

                    request_id_ctx.reset(dms_token)
                except Exception:
                    pass

    @staticmethod
    def _client_ip(request: Request) -> str | None:
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip()
        if forwarded:
            return forwarded
        client = getattr(request, "client", None)
        if client is not None and getattr(client, "host", None):
            return str(client.host)
        return None

    @staticmethod
    def _extract_user_id(request: Request) -> str | None:
        for header in ("X-User-ID", "X-API-Key-Subject"):
            val = (request.headers.get(header) or "").strip()
            if val:
                return val
        state_user = getattr(request.state, "user_id", None)
        if state_user:
            return str(state_user)
        return None

    @staticmethod
    def _extract_application_name(request: Request) -> str:
        for header in ObservabilityMiddleware.application_headers:
            value = (request.headers.get(header) or "").strip()
            if value:
                return value[:128]
        return "unknown"

    @staticmethod
    def _classify_action(method: str, path: str) -> str:
        normalized = path.rstrip("/").lower()
        if method == "POST" and normalized.endswith("/documents/upload"):
            return "document_upload"
        if method == "DELETE" and "/documents/" in normalized:
            return "document_delete"
        if method == "POST" and any(part in normalized for part in ("/chat", "/query", "/ask")):
            return "ai_query"
        if method == "POST" and "/retrieve" in normalized:
            return "search"
        return f"{method.lower()} {normalized or '/'}"

    @staticmethod
    def _extract_path_param(request: Request, name: str) -> str | None:
        val = request.path_params.get(name) if hasattr(request, "path_params") else None
        if val:
            return str(val)
        qp = request.query_params.get(name)
        return str(qp).strip() if qp else None
