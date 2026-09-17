"""
Shared FastAPI security schemes for OpenAPI / Swagger UI authentication.
"""

from __future__ import annotations

from typing import Any

from fastapi.security import HTTPBearer

# auto_error=False: middleware performs real JWT checks; this dependency only
# documents Bearer auth so Swagger UI attaches Authorization headers.
security_scheme = HTTPBearer(auto_error=False)

_HTTP_METHODS = frozenset({"get", "put", "post", "delete", "options", "head", "patch", "trace"})


def apply_openapi_bearer_security(openapi_schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure Swagger sends Bearer JWTs for every middleware-protected route.

    JwtAuthMiddleware enforces auth for ``/api/v1/*`` (except public auth/token).
    Routes without ``Depends(security_scheme)`` omit ``security`` in OpenAPI, so
    Swagger UI can show "Authorized" globally while still omitting the header on
    those operations — producing 401 Authentication required.
    """
    from src.application.consumer_api.auth_middleware import requires_api_auth

    components = openapi_schema.setdefault("components", {})
    schemes = components.setdefault("securitySchemes", {})
    schemes["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": (
            "Paste the access_token from POST /api/v1/auth/token "
            "(Swagger adds the Bearer prefix)."
        ),
    }

    bearer_requirement = [{"HTTPBearer": []}]
    paths = openapi_schema.get("paths") or {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, operation in path_item.items():
            if method not in _HTTP_METHODS or not isinstance(operation, dict):
                continue
            if not requires_api_auth(method.upper(), path):
                # Public routes (e.g. /api/v1/auth/token) must not require Bearer in Swagger.
                operation.pop("security", None)
                continue
            operation["security"] = bearer_requirement
    return openapi_schema
