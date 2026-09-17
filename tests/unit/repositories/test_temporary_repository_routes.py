from __future__ import annotations

from fastapi import FastAPI

from src.features.repositories.api.repository_routes import router


def test_temporary_repository_routes_are_exposed_before_parameterized_repository_route() -> None:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    schema = app.openapi()
    paths = schema["paths"]

    assert "/api/v1/repositories/temporary" in paths
    assert "post" in paths["/api/v1/repositories/temporary"]
    assert "get" in paths["/api/v1/repositories/temporary"]
    assert "/api/v1/repositories/temporary/cleanup" in paths
    assert "post" in paths["/api/v1/repositories/temporary/cleanup"]
    assert "/api/v1/repositories/temporary/cleanup-status" in paths
    assert "get" in paths["/api/v1/repositories/temporary/cleanup-status"]
    assert "/api/v1/repositories/temporary/documents" in paths
    assert "get" in paths["/api/v1/repositories/temporary/documents"]
