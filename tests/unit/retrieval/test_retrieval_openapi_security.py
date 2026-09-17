"""
Unit tests for Retrieval API authentication and OpenAPI/Swagger configuration.
"""

from __future__ import annotations

from unittest.mock import patch
import pytest
from src.application.consumer_api.main import create_app


def test_openapi_schema_contains_http_bearer_scheme():
    # Arrange & Act: Mock startup functions to prevent DB/infrastructure initializations
    with patch("src.infrastructure.application_support.bootstrap_infra"), \
         patch("src.features.document_processing.shared_processor.deployment.bootstrap_deployment_processors"), \
         patch("src.infrastructure.application_support.HealthRegistry"), \
         patch("src.application.consumer_api.main.settings") as mock_settings:
        
        mock_settings.expose_openapi = True
        mock_settings.api_prefix = "/api/v1"
        
        app = create_app()
        openapi = app.openapi()

    # Assert 1: The security scheme exists in the OpenAPI components
    components = openapi.get("components", {})
    security_schemes = components.get("securitySchemes", {})
    
    assert "HTTPBearer" in security_schemes, "HTTPBearer security scheme missing from components"
    assert security_schemes["HTTPBearer"]["type"] == "http"
    assert security_schemes["HTTPBearer"]["scheme"] == "bearer"

    # Assert 2: Protected paths require HTTPBearer security scheme
    paths = openapi.get("paths", {})
    
    protected_endpoints = [
        ("/api/v1/retrieve", "post"),
        ("/api/v1/retrieve/documents", "get"),
        ("/api/v1/retrieve/documents/search", "post"),
        ("/api/v1/retrieve/documents/{document_id}/download", "get"),
        ("/api/v1/documents/{document_id}/chunks", "get"),
        ("/api/v1/repositories/{repository_id}/search/bm25", "post"),
        ("/api/v1/repositories/{repository_id}/documents/{document_id}/chunks", "get"),
    ]

    for route_path, method in protected_endpoints:
        path_item = paths.get(route_path)
        assert path_item is not None, f"Path {route_path} not found in OpenAPI spec"
        
        operation = path_item.get(method)
        assert operation is not None, f"Method {method} not found for path {route_path}"
        
        security_requirement = operation.get("security")
        assert security_requirement is not None, f"Security requirement missing for {method.upper()} {route_path}"
        
        # Verify the security scheme list contains HTTPBearer
        has_bearer = any("HTTPBearer" in req for req in security_requirement)
        assert has_bearer, f"HTTPBearer security scheme not applied to {method.upper()} {route_path}"
