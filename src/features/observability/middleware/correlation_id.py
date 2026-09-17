"""Correlation ID header propagation."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.features.observability.middleware.request_context import correlation_id_var


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    header_name = "X-Correlation-ID"

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming = (request.headers.get(self.header_name) or "").strip()
        correlation_id = incoming or str(uuid.uuid4())
        token = correlation_id_var.set(correlation_id)
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        response.headers[self.header_name] = correlation_id
        return response
