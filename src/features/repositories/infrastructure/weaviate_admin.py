from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from weaviate.connect import executor as wexec

from src.features.configuration.platform_settings import settings
from src.features.repositories.infrastructure.weaviate_backend import backend
from src.infrastructure.vector_store.shared_weaviate.tenancy import resolve_collection


def repository_status() -> dict[str, Any]:
    """Whether repository admin can talk to Weaviate (no secrets returned)."""
    no_key = not (settings.weaviate_api_key or "").strip()
    return {
        "enabled": backend.enabled,
        "ready": backend.ready,
        "error": None if backend.ready else backend.last_error,
        "weaviate_url": settings.weaviate_url,
        "weaviate_grpc_port": settings.weaviate_grpc_port,
        "api_key_configured": not no_key,
        "hint": (
            "Set WEAVIATE_API_KEY to match your Weaviate server (compose default: weaviate_secret_key)."
            if no_key and backend.enabled
            else None
        ),
    }


def _collection_handle(client: Any, collection_name: str, tenant: str | None):
    col = client.collections.get(collection_name)
    return resolve_collection(col, tenant)


def list_collections() -> dict[str, Any]:
    client = backend.require_client()
    try:
        configs = wexec.result(client.collections.list_all(simple=True))
        names = sorted(configs.keys()) if isinstance(configs, dict) else []
        return {"collections": names, "count": len(names)}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weaviate list collections failed: {exc}") from exc


def list_tenants(collection_name: str) -> dict[str, Any]:
    client = backend.require_client()
    try:
        col = client.collections.get(collection_name)
        tenants = wexec.result(col.tenants.get())
        out = [
            {"name": name, "activity_status": getattr(t, "activity_status", None)}
            for name, t in (tenants or {}).items()
        ]
        return {"collection": collection_name, "tenants": out, "count": len(out)}
    except Exception as exc:
        message = str(exc)
        if "multi-tenancy is not enabled" in message:
            return {
                "collection": collection_name,
                "tenants": [],
                "count": 0,
                "multi_tenancy_enabled": False,
            }
        raise HTTPException(status_code=502, detail=f"Weaviate tenants list failed: {exc}") from exc


def list_unique_documents(
    collection_name: str,
    *,
    tenant: str | None = None,
    limit: int = 5000,
) -> dict[str, Any]:
    client = backend.require_client()
    try:
        col = _collection_handle(client, collection_name, tenant)
        result = wexec.result(col.query.fetch_objects(limit=limit))
        names: set[str] = set()
        for obj in result.objects:
            props = obj.properties or {}
            dn = props.get("doc_name") or props.get("document_name")
            if dn:
                names.add(str(dn))
        sorted_names = sorted(names)
        return {
            "collection": collection_name,
            "tenant": tenant,
            "documents": sorted_names,
            "count": len(sorted_names),
            "truncated": len(result.objects) >= limit,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weaviate document list failed: {exc}") from exc


def _document_chunk_filter(
    *,
    document_id: str | None = None,
    document_name: str | None = None,
) -> Any | None:
    """Build a Weaviate filter for document-scoped chunk queries.

    ``document_id`` matches the processor-written ``document_id`` property.
    ``document_name`` matches either ``doc_name`` or ``document_name`` (original
    filename); intake disk names like ``{uuid}.pdf`` often do not appear there.
    """
    from weaviate.classes.query import Filter

    clauses: list[Any] = []
    if document_name:
        name = str(document_name).strip()
        if name:
            clauses.append(
                Filter.by_property("doc_name").equal(name)
                | Filter.by_property("document_name").equal(name)
            )
    if document_id:
        did = str(document_id).strip()
        if did:
            clauses.append(Filter.by_property("document_id").equal(did))
    if not clauses:
        return None
    combined = clauses[0]
    for clause in clauses[1:]:
        combined = combined | clause
    return combined


def list_chunks_metadata(
    collection_name: str,
    *,
    tenant: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    limit: int = 200,
    offset: int = 0,
    include_text: bool = False,
    include_vector: bool = False,
    include_query_metadata: bool = True,
) -> dict[str, Any]:
    client = backend.require_client()
    try:
        from weaviate.classes.query import Filter, MetadataQuery

        col = _collection_handle(client, collection_name, tenant)
        flt = _document_chunk_filter(document_id=document_id, document_name=document_name)
        qkwargs: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }
        if include_query_metadata:
            qkwargs["return_metadata"] = MetadataQuery(distance=True, score=True)
        if include_vector:
            qkwargs["include_vector"] = True
        if flt is not None:
            qkwargs["filters"] = flt
        result = wexec.result(col.query.fetch_objects(**qkwargs))
        chunks: list[dict[str, Any]] = []
        for obj in result.objects:
            props = dict(obj.properties or {})
            meta = getattr(obj, "metadata", None)
            if not include_text:
                props.pop("text", None)
            entry: dict[str, Any] = {
                "uuid": str(obj.uuid),
                "properties": props,
            }
            if include_vector:
                vector = getattr(obj, "vector", None)
                if isinstance(vector, dict):
                    vector = next((value for value in vector.values() if value is not None), None)
                if vector is not None:
                    if hasattr(vector, "tolist"):
                        entry["vector"] = vector.tolist()
                    else:
                        entry["vector"] = list(vector) if vector else None
            if meta:
                entry["metadata"] = {
                    "distance": getattr(meta, "distance", None),
                    "score": getattr(meta, "score", None),
                }
            chunks.append(entry)
        return {
            "collection": collection_name,
            "tenant": tenant,
            "document_id": document_id,
            "document_name": document_name,
            "offset": offset,
            "limit": limit,
            "include_text": include_text,
            "include_vector": include_vector,
            "returned": len(chunks),
            "chunks": chunks,
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weaviate chunks list failed: {exc}") from exc


def fetch_all_document_chunks(
    collection_name: str,
    *,
    tenant: str | None = None,
    document_id: str | None = None,
    document_name: str | None = None,
    include_text: bool = True,
    page_size: int = 1000,
) -> list[dict[str, Any]]:
    """Fetch every chunk for a document without loading vectors or query metadata."""
    if not document_id and not document_name:
        raise ValueError("document_id or document_name is required for chunk fetch.")
    page_size = max(1, min(page_size, 5000))
    all_chunks: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = list_chunks_metadata(
            collection_name,
            tenant=tenant,
            document_id=document_id,
            document_name=document_name,
            limit=page_size,
            offset=offset,
            include_text=include_text,
            include_vector=False,
            include_query_metadata=False,
        )
        batch = payload.get("chunks") or []
        all_chunks.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return all_chunks


def delete_collection(collection_name: str) -> None:
    client = backend.require_client()
    try:
        wexec.result(client.collections.delete(collection_name))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weaviate delete collection failed: {exc}") from exc


def purge_document_chunks(
    collection_name: str,
    document_name: str | None = None,
    *,
    document_id: str | None = None,
    tenant: str | None = None,
) -> dict[str, Any]:
    """Delete Weaviate chunks for a document without raising HTTP errors."""
    if not document_name and not document_id:
        raise ValueError("document_name or document_id is required")
    if not backend.ready:
        raise RuntimeError(backend.last_error or "Weaviate repository admin is not ready")
    client = backend.require_client()

    col = _collection_handle(client, collection_name, tenant)
    from src.infrastructure.vector_store.shared_weaviate.chunk_index_sync import purge_document_chunks_from_collection

    payload = purge_document_chunks_from_collection(
        col,
        document_name=document_name,
        document_id=document_id,
    )
    return {
        "collection": collection_name,
        "tenant": tenant,
        "document_name": document_name,
        "document_id": document_id,
        "successful": payload.get("successful"),
        "matches": payload.get("matches"),
        "failed": payload.get("failed"),
    }


def delete_document_chunks(
    collection_name: str,
    document_name: str | None = None,
    *,
    document_id: str | None = None,
    tenant: str | None = None,
) -> dict[str, Any]:
    client = backend.require_client()
    try:
        return purge_document_chunks(
            collection_name,
            document_name,
            document_id=document_id,
            tenant=tenant,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weaviate delete document chunks failed: {exc}") from exc
