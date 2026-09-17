from __future__ import annotations

import logging
import json
from typing import List

import numpy as np
import weaviate
from weaviate.auth import Auth
import weaviate.classes.config as wc
from weaviate.util import generate_uuid5

from src.features.document_processing.core.config import get_settings
from src.infrastructure.vector_store.shared_weaviate.tenancy import resolve_collection
from src.features.document_processing.core.logger import log
from src.features.chunking.application.chunking_service import Chunk
from src.features.embeddings.application.embedding_service import EmbeddingValidationError, validate_embedding_vector


_cached_client: weaviate.WeaviateClient | None = None
_cached_client_params: tuple[str, str | None] | None = None


def _close_cached_client() -> None:
    global _cached_client, _cached_client_params
    if _cached_client is None:
        return
    try:
        close_method = getattr(_cached_client, "close", None)
        if callable(close_method):
            close_method()
    except Exception:
        pass
    _cached_client = None
    _cached_client_params = None


def close_weaviate_client() -> None:
    """Close the cached Weaviate client, if one is open."""
    _close_cached_client()


def _is_client_ready(client: weaviate.WeaviateClient) -> bool:
    try:
        ready_method = getattr(client, "is_ready", None)
        if callable(ready_method):
            return ready_method()
        # Fallback: test a simple meta call
        return bool(client.schema.get())
    except Exception:
        return False


def _create_weaviate_client(weaviate_url: str, api_key: str | None = None) -> weaviate.WeaviateClient:
    logging.getLogger("httpx").setLevel(logging.WARNING)

    auth_credentials = Auth.api_key(api_key) if api_key else None

    from weaviate import connect_to_custom

    settings = get_settings()
    host = weaviate_url.split("://")[-1].split(":")[0]
    port_str = weaviate_url.split("://")[-1].split(":")[-1].split("/")[0]
    port = int(port_str) if port_str.isdigit() else 8080
    secure = weaviate_url.startswith("https")

    return connect_to_custom(
        http_host=host,
        http_port=port,
        http_secure=secure,
        grpc_host=host,
        grpc_port=settings.weaviate_grpc_port,
        grpc_secure=secure,
        auth_credentials=auth_credentials,
    )


def get_weaviate_client(weaviate_url: str, api_key: str | None = None) -> weaviate.WeaviateClient:
    global _cached_client, _cached_client_params
    if _cached_client is not None and _cached_client_params == (weaviate_url, api_key):
        if _is_client_ready(_cached_client):
            return _cached_client
        log.warning("Cached Weaviate client is no longer ready, recreating client.")
        _close_cached_client()

    client = _create_weaviate_client(weaviate_url, api_key)
    _cached_client = client
    _cached_client_params = (weaviate_url, api_key)
    return client


def ensure_collection(client: weaviate.WeaviateClient, collection_name: str) -> None:
    compatibility_properties = [
        wc.Property(
            name="document_id",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="repository_id",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="logical_folder_id",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="logical_folder_path",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="heading",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="topic",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="entities",
            data_type=wc.DataType.TEXT_ARRAY,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="sensitivity",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="embedding_version",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="semantic_metadata",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
        wc.Property(
            name="line_start",
            data_type=wc.DataType.INT,
            index_filterable=True,
            index_searchable=False,
        ),
        wc.Property(
            name="line_end",
            data_type=wc.DataType.INT,
            index_filterable=True,
            index_searchable=False,
        ),
        wc.Property(
            name="line_range",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="content_types",
            data_type=wc.DataType.TEXT_ARRAY,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="table_count",
            data_type=wc.DataType.INT,
            index_filterable=True,
            index_searchable=False,
        ),
        wc.Property(
            name="image_count",
            data_type=wc.DataType.INT,
            index_filterable=True,
            index_searchable=False,
        ),
        wc.Property(
            name="extraction_metadata",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
        wc.Property(
            name="citation_anchor",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
        wc.Property(
            name="page_end",
            data_type=wc.DataType.INT,
            index_filterable=True,
            index_searchable=False,
        ),
        wc.Property(
            name="chunk_type",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="strategy_name",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="parent_section",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
        wc.Property(
            name="table_name",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="title",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="document_name",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="source_blocks",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
    ]

    if client.collections.exists(collection_name):
        col = client.collections.get(collection_name)
        try:
            config = col.config.get()
            existing = {prop.name for prop in getattr(config, "properties", [])}
            for prop in compatibility_properties:
                if prop.name not in existing:
                    col.config.add_property(prop)
                    log.info("Added Weaviate property %s to %s.", prop.name, collection_name)
        except Exception:
            log.exception("Failed to ensure line number properties on %s", collection_name)
            raise
        return

    log.info("Creating Weaviate collection: %s", collection_name)
    client.collections.create(
        name=collection_name,
        vector_config=wc.Configure.Vectors.self_provided(),
        inverted_index_config=wc.Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            index_null_state=False,
            index_property_length=False,
        ),
        properties=[
            wc.Property(
                name="chunk_id",
                data_type=wc.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
                tokenization=wc.Tokenization.FIELD,
            ),
            wc.Property(
                name="doc_name",
                data_type=wc.DataType.TEXT,
                index_filterable=True,
                index_searchable=False,
                tokenization=wc.Tokenization.FIELD,
            ),
            wc.Property(
                name="page",
                data_type=wc.DataType.INT,
                index_filterable=True,
                index_searchable=False,
            ),
            *compatibility_properties,
            wc.Property(
                name="section_name",
                data_type=wc.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
            ),
            wc.Property(
                name="section_path",
                data_type=wc.DataType.TEXT,
                index_filterable=False,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
                skip_vectorization=True,
            ),
            wc.Property(
                name="section",
                data_type=wc.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
            ),
            wc.Property(
                name="page_label",
                data_type=wc.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
                tokenization=wc.Tokenization.FIELD,
            ),
            wc.Property(
                name="retrieval_metadata",
                data_type=wc.DataType.TEXT,
                index_filterable=False,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
                skip_vectorization=True,
            ),
            wc.Property(
                name="text",
                data_type=wc.DataType.TEXT,
                index_filterable=False,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
                skip_vectorization=True,
            ),
            wc.Property(
                name="cluster_id",
                data_type=wc.DataType.INT,
                index_filterable=False,
                index_searchable=False,
            ),
            wc.Property(
                name="match_pct",
                data_type=wc.DataType.NUMBER,
                index_filterable=False,
                index_searchable=False,
            ),
            wc.Property(
                name="is_duplicate",
                data_type=wc.DataType.BOOL,
                index_filterable=True,
                index_searchable=False,
            ),
            wc.Property(
                name="category",
                data_type=wc.DataType.TEXT,
                index_filterable=True,
                index_searchable=True,
                tokenization=wc.Tokenization.FIELD,
            ),
            wc.Property(
                name="category_confidence",
                data_type=wc.DataType.NUMBER,
                index_filterable=False,
                index_searchable=False,
            ),
            wc.Property(
                name="category_keywords",
                data_type=wc.DataType.TEXT_ARRAY,
                index_filterable=False,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
            ),
            wc.Property(
                name="category_summary",
                data_type=wc.DataType.TEXT,
                index_filterable=False,
                index_searchable=True,
                tokenization=wc.Tokenization.WORD,
            ),
            wc.Property(
                name="file_size_bytes",
                data_type=wc.DataType.INT,
                index_filterable=True,
                index_searchable=False,
            ),
        ],
    )
    log.info("Collection '%s' created with BM25 + HNSW hybrid schema.", collection_name)


def _ensure_collection_properties(client: weaviate.WeaviateClient, collection_name: str) -> None:
    """Add document-level metadata properties to collections created before schema updates."""
    if not client.collections.exists(collection_name):
        return
    collection = client.collections.get(collection_name)
    try:
        config = collection.config.get()
        existing = {prop.name for prop in (config.properties or [])}
    except Exception as exc:
        log.warning("Could not inspect Weaviate collection '%s' schema: %s", collection_name, exc)
        return

    missing: list[wc.Property] = []
    if "file_size_bytes" not in existing:
        missing.append(
            wc.Property(
                name="file_size_bytes",
                data_type=wc.DataType.INT,
                index_filterable=True,
                index_searchable=False,
            )
        )
    bm25_props = [
        wc.Property(
            name="repository_id",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="section",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="page_label",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="retrieval_metadata",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
            skip_vectorization=True,
        ),
    ]
    for prop in bm25_props:
        if prop.name not in existing:
            missing.append(prop)
    citation_props = [
        wc.Property(
            name="citation_anchor",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
        wc.Property(
            name="page_end",
            data_type=wc.DataType.INT,
            index_filterable=True,
            index_searchable=False,
        ),
        wc.Property(
            name="chunk_type",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="strategy_name",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="parent_section",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
        wc.Property(
            name="table_name",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="title",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=True,
            tokenization=wc.Tokenization.WORD,
        ),
        wc.Property(
            name="document_name",
            data_type=wc.DataType.TEXT,
            index_filterable=True,
            index_searchable=False,
            tokenization=wc.Tokenization.FIELD,
        ),
        wc.Property(
            name="source_blocks",
            data_type=wc.DataType.TEXT,
            index_filterable=False,
            index_searchable=False,
            skip_vectorization=True,
        ),
    ]
    for prop in citation_props:
        if prop.name not in existing:
            missing.append(prop)
    for prop in missing:
        try:
            collection.config.add_property(prop)
            log.info("Added Weaviate property '%s' to collection '%s'.", prop.name, collection_name)
        except Exception as exc:
            log.warning(
                "Could not add Weaviate property '%s' to collection '%s': %s",
                prop.name,
                collection_name,
                exc,
            )


def _document_props_for_chunks(document_metadata: dict | None) -> dict[str, int]:
    """Extract document-level fields replicated on each chunk object."""
    if not document_metadata:
        return {}
    out: dict[str, int] = {}
    raw_size = document_metadata.get("file_size_bytes")
    if raw_size is not None:
        try:
            out["file_size_bytes"] = int(raw_size)
        except (TypeError, ValueError):
            pass
    return out


def _retrieval_metadata_blob(
    *,
    chunk_id: str,
    document_id: str | None,
    repository_id: str | None,
    doc_name: str,
    page: int | None,
    section_name: str | None,
    extraction_metadata: dict | None,
) -> str:
    payload = {
        "chunk_id": chunk_id,
        "document_id": document_id or "",
        "repository_id": repository_id or "",
        "doc_name": doc_name,
        "page": page,
        "section": section_name or "",
    }
    if extraction_metadata:
        payload["extraction"] = extraction_metadata
    return json.dumps(payload, ensure_ascii=False)


def store_in_weaviate(
    chunks: List[Chunk],
    weaviate_url: str,
    collection_name: str,
    api_key: str | None = None,
    tenant_id: str | None = None,
    document_metadata: dict | None = None,
) -> None:
    candidates = [c for c in chunks if c.embedding is not None and not c.is_duplicate]
    if not candidates:
        log.info("No valid chunks to insert into Weaviate.")
        return

    expected_dim: int | None = None
    to_insert: list[Chunk] = []
    for chunk in candidates:
        try:
            vector = validate_embedding_vector(
                chunk.embedding,
                expected_dim=expected_dim,
                context=f"chunk_id={chunk.id}",
            )
        except EmbeddingValidationError as exc:
            raise RuntimeError(f"Rejecting Weaviate write: {exc}") from exc
        if expected_dim is None:
            expected_dim = len(vector)
        chunk.embedding = np.asarray(vector, dtype=np.float32)
        to_insert.append(chunk)

    log.info("Connecting to Weaviate at %s …", weaviate_url)
    try:
        client = get_weaviate_client(weaviate_url, api_key)
        ensure_collection(client, collection_name)
        _ensure_collection_properties(client, collection_name)
        col = client.collections.get(collection_name)
        col = resolve_collection(col, tenant_id)

        doc_level_props = _document_props_for_chunks(document_metadata)
        repository_id = str((document_metadata or {}).get("repository_id") or "").strip() or None
        first_chunk = to_insert[0]
        purge_document_id = str(
            (document_metadata or {}).get("document_id") or first_chunk.document_id or ""
        ).strip() or None
        from src.infrastructure.vector_store.shared_weaviate.chunk_index_sync import purge_document_chunks_from_collection

        purge_document_chunks_from_collection(
            col,
            document_name=first_chunk.doc_name,
            document_id=purge_document_id,
        )

        log.info(
            "Inserting %d object(s) into '%s'%s …",
            len(to_insert),
            collection_name,
            f" tenant={tenant_id!r}" if tenant_id else "",
        )

        total = len(to_insert)
        progress_every = max(1, min(50, total // 10 or 1))
        with col.batch.dynamic() as batch:
            for index, c in enumerate(to_insert, start=1):
                obj_uuid = generate_uuid5(c.id)
                props = {
                    "chunk_id": c.id,
                    "document_id": c.document_id,
                    "document_name": getattr(c, "document_name", None) or c.doc_name,
                    "repository_id": repository_id or "",
                    "logical_folder_id": str((document_metadata or {}).get("logical_folder_id") or ""),
                    "logical_folder_path": str((document_metadata or {}).get("logical_folder_path") or ""),
                    "doc_name": c.doc_name,
                    "page": c.page,
                    "page_end": getattr(c, "page_end", None) or getattr(c, "end_page", None) or c.page,
                    "page_label": str(c.page) if c.page is not None else "",
                    "line_start": c.line_start,
                    "line_end": c.line_end,
                    "line_range": (
                        f"{c.line_start}-{c.line_end}"
                        if c.line_start is not None and c.line_end is not None
                        else ""
                    ),
                    "section_name": c.section_name,
                    "section": c.section_name or c.heading or "",
                    "heading": c.heading or c.section_name,
                    "section_path": c.section_path,
                    "parent_section": getattr(c, "parent_section", "") or "",
                    "title": getattr(c, "title", "") or "",
                    "chunk_type": getattr(c, "chunk_type", None) or "paragraph",
                    "strategy_name": getattr(c, "strategy_name", "") or "",
                    "table_name": getattr(c, "table_name", "") or "",
                    "topic": c.topic,
                    "entities": c.entities,
                    "sensitivity": c.sensitivity,
                    "embedding_version": c.embedding_version,
                    "text": c.normalized_text or c.text,
                    "retrieval_metadata": _retrieval_metadata_blob(
                        chunk_id=c.id,
                        document_id=c.document_id,
                        repository_id=repository_id,
                        doc_name=c.doc_name,
                        page=c.page,
                        section_name=c.section_name,
                        extraction_metadata=c.extraction_metadata,
                    ),
                    "cluster_id": c.cluster_id or 0,
                    "match_pct": float(c.match_pct or 0.0),
                    "is_duplicate": c.is_duplicate,
                    "category": c.category,
                    "category_confidence": float(c.category_confidence or 0.0),
                    "category_keywords": c.category_keywords,
                    "category_summary": c.category_summary,
                    "content_types": c.content_types or ["text"],
                    "table_count": int(c.table_count or 0),
                    "image_count": int(c.image_count or 0),
                    "extraction_metadata": json.dumps(c.extraction_metadata or {}),
                    "source_blocks": json.dumps(getattr(c, "source_blocks", None) or []),
                    "citation_anchor": (
                        json.dumps(c.citation_anchor)
                        if getattr(c, "citation_anchor", None)
                        else ""
                    ),
                    "semantic_metadata": json.dumps(
                        {
                            "coordinates": [
                                {
                                    "page": coordinate.page,
                                    "x0": coordinate.x0,
                                    "y0": coordinate.y0,
                                    "x1": coordinate.x1,
                                    "y1": coordinate.y1,
                                }
                                for coordinate in c.coordinates
                            ],
                            "source": c.source,
                            "schema_version": c.version.semantic_schema_version,
                            "content_version": c.version.content_version,
                            "retrieval": {
                                "retrieval_mode": c.retrieval_metadata.retrieval_mode,
                                "original_score": c.retrieval_metadata.original_score,
                                "fused_score": c.retrieval_metadata.fused_score,
                                "rerank_score": c.retrieval_metadata.rerank_score,
                                "rank": c.retrieval_metadata.rank,
                            },
                        }
                    ),
                    **doc_level_props,
                }
                batch.add_object(
                    properties=props,
                    vector=c.embedding.tolist(),
                    uuid=obj_uuid,
                )
                if index == total or index % progress_every == 0:
                    log.info("Queued %d/%d chunk object(s) for Weaviate insert.", index, total)

        failed = getattr(col.batch, "failed_objects", [])
        if failed:
            log.warning("Weaviate: %d inserts failed.", len(failed))
            raise RuntimeError(f"Weaviate insert failed for {len(failed)} object(s).")
        else:
            log.info("Successfully pushed %d objects to Weaviate.", len(to_insert))

    except Exception as e:
        log.error("Weaviate storage failed: %s", e)
        raise
