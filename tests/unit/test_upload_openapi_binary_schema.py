"""OpenAPI multipart upload schema must expose files as binary for Swagger UI."""

from __future__ import annotations

from typing import Any


def _files_schema_from_openapi(schema: dict[str, Any], path: str) -> dict[str, Any]:
    operation = ((schema.get("paths") or {}).get(path) or {}).get("post") or {}
    content = ((operation.get("requestBody") or {}).get("content") or {}).get("multipart/form-data") or {}
    assert "multipart/form-data" in ((operation.get("requestBody") or {}).get("content") or {}), path
    request_schema = content.get("schema") or {}
    ref = str(request_schema.get("$ref") or "")
    assert ref.startswith("#/components/schemas/"), ref
    name = ref.rsplit("/", 1)[-1]
    component = ((schema.get("components") or {}).get("schemas") or {}).get(name) or {}
    files = (component.get("properties") or {}).get("files")
    assert isinstance(files, dict), files
    return files


def test_upload_openapi_files_are_binary_for_swagger_file_picker() -> None:
    from src.application.consumer_api.main import create_app

    app = create_app()
    openapi = app.openapi()

    for path in (
        "/api/v1/documents/upload",
        "/api/v1/documents/upload-by-name",
    ):
        files = _files_schema_from_openapi(openapi, path)
        assert files.get("type") == "array", path
        items = files.get("items") or {}
        assert items.get("type") == "string", path
        assert items.get("format") == "binary", f"{path}: expected format=binary, got {items}"
        assert "contentMediaType" not in items, f"{path}: contentMediaType breaks Swagger file picker"

    by_name = (
        ((openapi.get("paths") or {}).get("/api/v1/documents/upload-by-name") or {}).get("post") or {}
    )
    props = (
        (
            (
                (openapi.get("components") or {}).get("schemas")
                or {}
            ).get(
                str(
                    (
                        (
                            ((by_name.get("requestBody") or {}).get("content") or {})
                            .get("multipart/form-data")
                            or {}
                        ).get("schema")
                        or {}
                    ).get("$ref")
                    or ""
                ).rsplit("/", 1)[-1]
            )
            or {}
        ).get("properties")
        or {}
    )
    for required in ("files", "repository_name"):
        assert required in props
    for optional in ("document_type_name", "collection_name", "tenant_id", "submit_for_processing"):
        assert optional in props
