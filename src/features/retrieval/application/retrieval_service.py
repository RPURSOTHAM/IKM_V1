from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from src.shared.errors import COMPONENT_RETRIEVAL, DmsServiceError, raise_client_error, raise_service_error
from src.features.logical_folders.domain.exceptions import LogicalFolderError
from src.features.configuration.platform_settings import REPO_ROOT, get_settings
from src.features.retrieval.application.document_catalog import (
    aggregate_indexed_metadata,
    build_catalog_records_index,
    catalog_record_keys,
    matches_indexed_filters,
    rank_documents_by_chunk_scores,
    record_to_catalog_item,
    split_catalog_filters,
)
from src.features.retrieval.schemas.retrieval_schemas import (
    ChunkOut,
    DocumentCatalogItem,
    DocumentListResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    RetrieveRequest,
    RetrieveResponse,
)
from src.infrastructure.vector_store.dms_weaviate_client.client import connect_weaviate
from src.infrastructure.database.document_jobs import get_document_job_store
from src.infrastructure.vector_store.shared_weaviate.tenancy import resolve_collection

_logger = logging.getLogger(__name__)


def _apply_retrieval_dlp(chunks: list[ChunkOut]) -> list[ChunkOut]:
    """Mandatory retrieval-time masking — raw chunk text must not egress unredacted."""
    if not chunks:
        return chunks
    try:
        from src.features.security.dlp.policy_loader import scan_keyword_policies
        from src.features.security.dlp.sensitive_data_detector import mask_text_content
        from src.features.security.audit.security_event_logger import EVENT_EXPORT_SCAN, log_security_event
    except Exception:
        return chunks

    masked: list[ChunkOut] = []
    decisions: list[str] = []
    for chunk in chunks:
        text = chunk.text or ""
        if not text:
            masked.append(chunk)
            continue
        kw = scan_keyword_policies(text)
        if kw.get("action") == "block":
            new_text = "[Content withheld by retrieval DLP.]"
            decisions.append("block")
        else:
            new_text = mask_text_content(text)
            decisions.append("mask" if new_text != text or kw.get("matches") else "allow")
        if new_text != text:
            masked.append(chunk.model_copy(update={"text": new_text}))
        else:
            masked.append(chunk)
    if any(d != "allow" for d in decisions):
        log_security_event(
            EVENT_EXPORT_SCAN,
            decision="mask" if "block" not in decisions else "block",
            reason="retrieval_chunk_dlp",
            severity="medium",
            policy="retrieval_dlp",
            metadata={"chunk_count": len(chunks)},
        )
    return masked


def _build_repository_citation_lookup(repository_id: str) -> dict[str, dict[str, str]]:
    """Map Weaviate doc_name / document_id keys to human-readable citation fields."""
    from src.features.documents.application.document_metadata_service import (
        is_internal_storage_filename,
        resolve_display_filename_from_job,
    )
    from src.features.repositories.application.repository_service import get_repository_service
    from src.features.retrieval.application.repository_gate import load_active_repository_context
    from src.infrastructure.database.document_jobs import get_document_job_store

    lookup: dict[str, dict[str, str]] = {}
    try:
        repo_service = get_repository_service()
        repo_context = load_active_repository_context(repository_id)
        repo_name = str(repo_context.get("name") or "Repository").strip()
        payload = repo_service.list_documents(repository_id, limit=500)
        document_ids = [
            str(entry.get("document_id") or "")
            for entry in payload.get("documents") or []
            if entry.get("document_id")
        ]
        jobs = get_document_job_store().fetch_for_documents(document_ids) if document_ids else {}
    except Exception:
        return lookup

    for entry in payload.get("documents") or []:
        doc_id = str(entry.get("document_id") or "").strip()
        if not doc_id:
            continue
        stored = str(entry.get("document_name") or "").strip()
        display = str(entry.get("original_file_name") or "").strip()
        job = jobs.get(doc_id)
        if not display or is_internal_storage_filename(display, doc_id):
            resolved = resolve_display_filename_from_job(job, document_id=doc_id) if job else None
            display = resolved or display
        if not display or is_internal_storage_filename(display, doc_id):
            continue
        info = {
            "document_id": doc_id,
            "original_file_name": display,
            "repository_name": repo_name,
            "repository_id": repository_id,
        }
        keys = {stored, Path(stored).stem if stored else "", doc_id, display}
        if display:
            keys.add(Path(display).stem)
            keys.add(Path(display).name)
        for key in keys:
            if key:
                lookup[key] = info
                # Case-insensitive alias for filename matching from queries.
                lower = str(key).lower()
                if lower not in lookup:
                    lookup[lower] = info
    return lookup


def _compute_pipeline_stages(search_mode: str) -> list[str]:
    if search_mode == "keyword":
        return ["bm25"]
    if search_mode == "vector":
        return ["embedding", "near_vector"]
    return ["embedding", "hybrid"]


def _model_path_or_name() -> str:
    cfg = get_settings()
    if cfg.retrieval_model_dir:
        candidate = Path(cfg.retrieval_model_dir).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / "src" / "models" / candidate
        snapshots = candidate / "snapshots"
        if snapshots.exists():
            snapshot_dirs = [path for path in snapshots.iterdir() if path.is_dir()]
            if snapshot_dirs:
                return str(snapshot_dirs[0])
        if candidate.exists():
            return str(candidate)
    return cfg.retrieval_model_name


def _resolve_model_path(*, model_name: str | None = None, model_dir: str | None = None) -> str:
    """Resolve a SentenceTransformer path from repository embedding settings."""
    if model_dir:
        candidate = Path(str(model_dir)).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / "src" / "models" / candidate
        snapshots = candidate / "snapshots"
        if snapshots.exists():
            snapshot_dirs = [path for path in snapshots.iterdir() if path.is_dir()]
            if snapshot_dirs:
                return str(snapshot_dirs[0])
        if candidate.exists():
            return str(candidate)
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    return _model_path_or_name()


def _resolve_reranker_model_path(repo_settings: dict[str, Any] | None = None) -> str:
    """Resolve a local or hub reranker model path."""
    from src.features.retrieval.reranking.reranker_model_resolver import resolve_reranker_model_path

    return resolve_reranker_model_path(repo_settings)


def _effective_use_rerank(
    body: RetrieveRequest,
    repo_settings: dict[str, Any] | None,
) -> bool:
    """Request → repository → RERANKER_ENABLED / deployment (see resolve_rerank_enabled)."""
    from src.features.retrieval.reranking.config import resolve_rerank_enabled

    return resolve_rerank_enabled(
        use_rerank=body.use_rerank,
        repo_settings=repo_settings,
    )


def _is_retrieval_schema_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "could not find class" in message or "class not found" in message


def _is_vector_dim_mismatch(exc: Exception) -> bool:
    message = str(exc).lower()
    return "vector lengths don't match" in message or "vector with length" in message


def _is_invalid_near_vector(exc: Exception) -> bool:
    """True when Weaviate rejects near_vector input (null/empty/malformed)."""
    message = str(exc).lower()
    return (
        "nearvector" in message
        or "near_vector" in message
        or "argument 'nearvector'" in message
        or "must not be none" in message
        or "is required" in message and "vector" in message
    )


def _coerce_page(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _citation_fields_from_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Extract page/line citation fields stored on Weaviate chunk objects."""
    try:
        from src.features.citations.application.citation_anchor_builder import parse_citation_anchor
    except Exception:
        parse_citation_anchor = None  # type: ignore[assignment]

    page = _coerce_optional_int(props.get("page"))
    page_end = _coerce_optional_int(props.get("page_end"))
    line_start = _coerce_optional_int(props.get("line_start"))
    line_end = _coerce_optional_int(props.get("line_end"))
    line_range = str(props.get("line_range") or "").strip() or None
    if not line_range and line_start is not None and line_end is not None:
        line_range = f"{line_start}-{line_end}"
    source_blocks: list[Any] = []
    raw_blocks = props.get("source_blocks")
    if isinstance(raw_blocks, list):
        source_blocks = raw_blocks
    elif isinstance(raw_blocks, str) and raw_blocks.strip():
        try:
            parsed = json.loads(raw_blocks)
            if isinstance(parsed, list):
                source_blocks = parsed
        except Exception:
            source_blocks = []
    citation_anchor = None
    if parse_citation_anchor is not None:
        try:
            citation_anchor = parse_citation_anchor(props.get("citation_anchor"))
        except Exception:
            citation_anchor = None
    if citation_anchor:
        page = page or _coerce_optional_int(citation_anchor.get("start_page"))
        page_end = page_end or _coerce_optional_int(citation_anchor.get("end_page"))
        line_start = line_start if line_start is not None else _coerce_optional_int(citation_anchor.get("start_line"))
        line_end = line_end if line_end is not None else _coerce_optional_int(citation_anchor.get("end_line"))
        if not line_range and line_start is not None and line_end is not None:
            line_range = f"{line_start}-{line_end}"
        if not source_blocks:
            anchor_blocks = citation_anchor.get("source_blocks")
            if isinstance(anchor_blocks, list):
                source_blocks = anchor_blocks
    return {
        "page": page,
        "page_end": page_end,
        "line_start": line_start,
        "line_end": line_end,
        "line_range": line_range,
        "citation_anchor": citation_anchor,
        "source_blocks": source_blocks,
    }

def _safe_pipeline_trace(trace: dict[str, Any] | None) -> dict[str, float] | None:
    """Keep only numeric latency entries so RetrieveResponse validation cannot 500."""
    if not trace:
        return None
    cleaned: dict[str, float] = {}
    for key, value in trace.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            cleaned[str(key)] = float(value)
    return cleaned or None


def _log_retrieval_failure(
    *,
    repository_id: str | None,
    query: str,
    collection_name: str | None,
    search_mode: str | None,
    exc: BaseException,
) -> None:
    _logger.exception(
        "Retrieval failed repository_id=%s collection=%s search_mode=%s query=%r error=%s",
        repository_id,
        collection_name,
        search_mode,
        query,
        exc,
    )


def _requested_document_id(filters: dict[str, Any] | None) -> str | None:
    raw = (filters or {}).get("document_id")
    if raw in (None, ""):
        return None
    return str(raw).strip() or None


_QUERY_FILENAME_RE = re.compile(
    r"(?i)\b([A-Za-z0-9][\w.\-]{0,240}\.(?:pdf|docx?|pptx?|txt|html?|xlsx?|csv))\b"
)


def _extract_mentioned_document_tokens(query_text: str) -> list[str]:
    """Extract filename / document tokens mentioned in a query (generic, no hardcoding)."""
    text = str(query_text or "").strip()
    if not text:
        return []
    tokens: list[str] = []
    seen: set[str] = set()
    for match in _QUERY_FILENAME_RE.finditer(text):
        token = match.group(1).strip()
        key = token.lower()
        if token and key not in seen:
            seen.add(key)
            tokens.append(token)
    return tokens


def _resolve_document_id_from_query(
    query_text: str,
    citation_lookup: dict[str, dict[str, str]],
) -> str | None:
    """When the user explicitly names a document that exists in the repo, return its id.

    Only filters when exactly one repository document matches a mentioned filename.
    Never invents ids and never applies when the user did not mention a document.
    """
    if not citation_lookup:
        return None
    mentions = _extract_mentioned_document_tokens(query_text)
    if not mentions:
        return None
    matched_ids: list[str] = []
    seen_ids: set[str] = set()
    for mention in mentions:
        candidates = [
            mention,
            Path(mention).stem,
            mention.lower(),
            Path(mention).stem.lower(),
        ]
        for key in candidates:
            info = citation_lookup.get(key)
            if not info:
                # Case-insensitive scan over lookup keys (filenames only).
                lower_key = key.lower()
                for stored_key, stored_info in citation_lookup.items():
                    if str(stored_key).lower() == lower_key:
                        info = stored_info
                        break
                    display = str(stored_info.get("original_file_name") or "").lower()
                    if display and (
                        display == lower_key
                        or Path(display).stem.lower() == lower_key
                        or Path(display).name.lower() == lower_key
                    ):
                        info = stored_info
                        break
            if info:
                doc_id = str(info.get("document_id") or "").strip()
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    matched_ids.append(doc_id)
                break
    if len(matched_ids) == 1:
        return matched_ids[0]
    return None


def _normalize_retrieval_filters(
    filters: dict[str, Any] | None,
    *,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Merge top-level document_id into filters used for Weaviate queries."""
    merged = dict(filters or {})
    explicit = (document_id or "").strip()
    if explicit:
        merged["document_id"] = explicit
    return merged


def _collect_request_folder_ids(body: Any) -> list[str]:
    """Collect unique logical folder IDs from request fields and filters."""
    values: list[str] = []
    for item in list(getattr(body, "folder_ids", None) or []) + (
        [getattr(body, "folder_id", None)] if getattr(body, "folder_id", None) else []
    ):
        text = str(item).strip()
        if text:
            values.append(text)
    extra = (getattr(body, "filters", None) or {}).get("logical_folder_id") if getattr(body, "filters", None) else None
    if isinstance(extra, (list, tuple, set)):
        values.extend(str(item).strip() for item in extra if str(item).strip())
    elif extra not in (None, ""):
        values.append(str(extra).strip())
    return list(dict.fromkeys(values))


def _folder_filter_requires_repository() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail={
            "code": "folder_filter_requires_repository",
            "message": "folder_id and folder_ids require a single repository_id.",
            "details": {},
        },
    )


def _apply_logical_folder_scope(body: Any) -> Any:
    """Validate selected folders and merge their IDs into exact Weaviate pre-filters."""
    folder_ids = _collect_request_folder_ids(body)
    if not folder_ids:
        return body
    if getattr(body, "repository_ids", None) or getattr(body, "search_all_repositories", False):
        raise _folder_filter_requires_repository()
    repository_id = str(getattr(body, "repository_id", None) or "").strip()
    if not repository_id:
        raise _folder_filter_requires_repository()
    from src.features.logical_folders.application.folder_service import get_logical_folder_service

    unique_ids = get_logical_folder_service().require_folders_in_repository(repository_id, folder_ids)
    merged = dict(getattr(body, "filters", None) or {})
    merged["logical_folder_id"] = unique_ids[0] if len(unique_ids) == 1 else unique_ids
    if merged != dict(getattr(body, "filters", None) or {}):
        return body.model_copy(update={"filters": merged})
    return body


def _build_filter(filters: dict[str, Any]):
    """Build a Weaviate Filter applied before near_vector / BM25 / hybrid candidate fetch."""
    if not filters:
        return None

    from weaviate.classes.query import Filter

    # Keys must match Weaviate collection property names (except document_name → doc_name).
    property_map = {
        "document_id": "document_id",
        "repository_id": "repository_id",
        "logical_folder_id": "logical_folder_id",
        "logical_folder_path": "logical_folder_path",
        "document_name": "doc_name",
        "doc_name": "doc_name",
        "section_name": "section_name",
        "category": "category",
        "is_duplicate": "is_duplicate",
    }
    where = None
    applied: list[str] = []
    ignored: list[str] = []
    for source_name, value in filters.items():
        prop = property_map.get(source_name)
        if prop is None:
            ignored.append(str(source_name))
            continue
        if value in (None, ""):
            continue
        if isinstance(value, (list, tuple, set)):
            values = [item for item in value if item not in (None, "")]
            if not values:
                continue
            clause = Filter.any_of([Filter.by_property(prop).equal(item) for item in values])
        else:
            clause = Filter.by_property(prop).equal(value)
        where = clause if where is None else where & clause
        applied.append(f"{prop} == {value}")
    if applied:
        _logger.info("Applied Weaviate filters: %s", "; ".join(applied))
    if ignored:
        _logger.debug("Ignored unknown retrieval filter keys: %s", ignored)
    return where


def _chunk_document_id(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    return str(meta.get("document_id") or getattr(chunk, "document_id", None) or "").strip()


def _chunk_document_name(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    return str(
        getattr(chunk, "document_name", None)
        or getattr(chunk, "doc_name", None)
        or meta.get("document_name")
        or meta.get("doc_name")
        or ""
    ).strip()


def _chunk_folder_id(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    return str(meta.get("logical_folder_id") or "").strip()


def _log_retrieval_filter_outcome(
    *,
    requested_document_id: str | None,
    chunks: list[Any],
    stage: str,
    requested_folder_ids: list[str] | None = None,
) -> list[Any]:
    """Log and enforce document/folder scopes when a backend returns excess candidates."""
    unique_ids = sorted({doc_id for chunk in chunks if (doc_id := _chunk_document_id(chunk))})
    unique_names = sorted({name for chunk in chunks if (name := _chunk_document_name(chunk))})
    _logger.info(
        "Retrieval filter outcome stage=%s request_document_id=%s request_folder_ids=%s "
        "candidate_count=%s returned_document_ids=%s returned_document_names=%s",
        stage,
        requested_document_id,
        list(requested_folder_ids or []),
        len(chunks),
        unique_ids,
        unique_names,
    )

    allowed_folders = {str(item).strip() for item in (requested_folder_ids or []) if str(item).strip()}
    if not requested_document_id and not allowed_folders:
        return chunks

    kept: list[Any] = []
    dropped_docs = 0
    dropped_folders = 0
    for chunk in chunks:
        if requested_document_id:
            doc_id = _chunk_document_id(chunk)
            if doc_id and doc_id != requested_document_id:
                dropped_docs += 1
                continue
        if allowed_folders and _chunk_folder_id(chunk) not in allowed_folders:
            # Fail closed: a chunk without a matching indexed folder ID must
            # never appear in a user-selected folder result.
            dropped_folders += 1
            continue
        kept.append(chunk)
    if dropped_docs:
        _logger.warning(
            "Dropped %s candidates that did not match request_document_id=%s "
            "(Weaviate pre-filter should have excluded these)",
            dropped_docs,
            requested_document_id,
        )
    if dropped_folders:
        _logger.warning(
            "Dropped %s candidates that did not match request_folder_ids=%s "
            "(Weaviate pre-filter should have excluded these)",
            dropped_folders,
            sorted(allowed_folders),
        )
    return kept


_QUERY_STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "against",
        "also",
        "answer",
        "based",
        "before",
        "being",
        "between",
        "does",
        "effect",
        "affected",
        "have",
        "into",
        "need",
        "only",
        "procedure",
        "question",
        "related",
        "repo",
        "repository",
        "should",
        "show",
        "that",
        "their",
        "there",
        "these",
        "this",
        "what",
        "when",
        "where",
        "which",
        "with",
        "work",
    }
)


def _query_evidence_terms(query_text: str) -> list[str]:
    """Extract evidence tokens from a query, ignoring stopwords and filename mentions."""
    scrubbed = _QUERY_FILENAME_RE.sub(" ", str(query_text or ""))
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z0-9]+", scrubbed.lower()):
        if len(token) < 4 or token in _QUERY_STOPWORDS:
            continue
        if token not in terms:
            terms.append(token)
    return terms


def _require_query_evidence(body: RetrieveRequest) -> bool:
    """Require query-term overlap unless the user already scoped to one document.

    Document-scoped asks often target a table cell whose OCR text lacks the
    question's wording (e.g. a control number). Dropping those chunks hides the
    answer even though the correct document was identified.
    """
    if not body.repository_id:
        return False
    if _requested_document_id(body.filters):
        return False
    return True


def _chunk_evidence_terms(chunk_text: str, query_terms: list[str]) -> list[str]:
    normalized = f" {' '.join(re.findall(r'[A-Za-z0-9]+', chunk_text.lower()))} "
    hits: list[str] = []
    for term in query_terms:
        if f" {term} " in normalized or f" {term.rstrip('s')} " in normalized:
            hits.append(term)
    return hits


def _has_required_query_evidence(chunk_text: str, query_terms: list[str]) -> tuple[bool, list[str]]:
    if not query_terms:
        return True, []
    hits = _chunk_evidence_terms(chunk_text, query_terms)
    required = 1 if len(query_terms) == 1 else 2
    return len(hits) >= required, hits


class RetrievalService:
    def __init__(self, enabled: bool = True, bm25_service: Any | None = None):
        self._bootstrap_enabled = enabled
        self._bm25_service = bm25_service
        self.ready = False
        self.error: str | None = None
        self._client = None
        self._model = None
        self._models: dict[str, Any] = {}
        self._rerankers: dict[str, Any] = {}
        self._model_errors: dict[str, str] = {}
        self._reranker_errors: dict[str, str] = {}
        self._model_error: str | None = None
        self._reranker_error: str | None = None
        self._startup_time = 0.0
        self._requests = 0
        self._last_latency_ms = 0.0

    def _bm25(self) -> Any:
        if self._bm25_service is None:
            from src.features.retrieval.strategies.bm25.bm25_search_service import get_bm25_search_service

            self._bm25_service = get_bm25_search_service()
        return self._bm25_service

    def _retrieval_enabled(self) -> bool:
        if not self._bootstrap_enabled:
            return False
        return get_settings().enable_retrieval

    async def startup(self) -> None:
        if not self._retrieval_enabled():
            self.error = None
            return
        try:
            self._client = connect_weaviate()
            cfg = get_settings()
            if not self._client.is_ready():
                raise RuntimeError(f"Weaviate is not ready at {cfg.weaviate_url}")

            self.ready = True
            self.error = None
            self._startup_time = time.time()
        except Exception as exc:
            self.ready = False
            self.error = str(exc)

    async def shutdown(self) -> None:
        if self._client:
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
        self.ready = False

    def health(self) -> dict[str, Any]:
        if not self._retrieval_enabled():
            return {
                "enabled": False,
                "status": "disabled",
                "ready": False,
                "error": None,
                "hint": "Retrieval is opt-in. Enable via platform config features.enable_retrieval or PATCH /platform/config.",
                "weaviate_url": None,
                "collection": None,
                "model": None,
                "model_ready": False,
                "model_error": None,
            }
        cfg = get_settings()
        return {
            "enabled": True,
            "status": "ready" if self.ready else "degraded",
            "ready": self.ready,
            "error": self.error,
            "weaviate_url": cfg.weaviate_url,
            "collection": cfg.default_collection_name,
            "model": cfg.retrieval_model_name,
            "model_ready": self._model is not None,
            "model_error": self._model_error,
            "reranker_ready": bool(self._rerankers),
            "reranker_error": self._reranker_error,
        }

    def _require_ready(self) -> None:
        if not self._retrieval_enabled():
            raise_client_error(
                COMPONENT_RETRIEVAL,
                code="retrieval_disabled",
                http_status=503,
                user_message="Search is currently disabled for this environment.",
                reason="Retrieval feature flag is disabled.",
            )
        if not self.ready or self._client is None:
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="retrieval_unavailable",
                http_status=503,
                user_message="Search is temporarily unavailable. Please try again shortly.",
                reason=f"Retrieval service not ready: {self.error or 'not ready'}",
            )

    def _get_encoder(self, model_path: str):
        if model_path in self._models:
            return self._models[model_path]
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_path, device="cpu")
            self._models[model_path] = model
            self._model = model
            self._model_error = None
            return model
        except Exception as exc:
            self._model_errors[model_path] = str(exc)
            self._model_error = str(exc)
            return None

    def _query_vector(self, query_text: str, *, model_path: str | None = None) -> list[float] | None:
        path = model_path or _model_path_or_name()
        model = self._get_encoder(path)
        if model is None:
            return None
        return model.encode([query_text], normalize_embeddings=True)[0].tolist()

    def _get_reranker(self, model_path: str):
        if not model_path or not str(model_path).strip():
            _logger.warning("Reranker model unavailable: empty model path (offline local model missing)")
            return None
        if model_path in self._rerankers:
            _logger.debug("reranker_model_cache_hit path=%s", model_path)
            return self._rerankers[model_path]
        try:
            from pathlib import Path

            from sentence_transformers import CrossEncoder

            from src.features.retrieval.reranking.reranker_model_resolver import has_required_model_files

            path_obj = Path(str(model_path))
            offline = any(
                (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}
                for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
            )
            if path_obj.exists():
                if path_obj.is_dir() and not has_required_model_files(path_obj):
                    raise FileNotFoundError(
                        f"Local reranker directory is missing required model files: {model_path}"
                    )
            elif offline:
                raise FileNotFoundError(
                    f"Local reranker model path not found while TRANSFORMERS_OFFLINE=1: {model_path}"
                )
            elif "/" not in str(model_path).replace("\\", "/"):
                # Not a hub id (org/name) and not an existing path.
                raise FileNotFoundError(f"Local reranker model path not found: {model_path}")

            load_started = time.perf_counter()
            model = CrossEncoder(model_path, device="cpu")
            load_ms = (time.perf_counter() - load_started) * 1000.0
            self._rerankers[model_path] = model
            self._reranker_error = None
            _logger.info(
                "reranker_model_loaded path=%s load_ms=%.1f cached=false offline=%s",
                model_path,
                load_ms,
                offline,
            )
            return model
        except Exception as exc:
            self._reranker_errors[model_path] = str(exc)
            self._reranker_error = str(exc)
            _logger.warning("Reranker model unavailable at %s: %s", model_path, exc)
            return None

    def _get_cross_encoder_reranker(self, *, config: Any | None = None) -> Any:
        from src.features.retrieval.reranking import CrossEncoderReranker, RerankerConfig

        return CrossEncoderReranker(self._get_reranker, config=config or RerankerConfig.from_env())

    def clear_reranker_model_cache(self) -> None:
        """Release cached CrossEncoder instances (e.g. after deployment disable)."""
        count = len(self._rerankers)
        self._rerankers.clear()
        self._reranker_errors.clear()
        self._reranker_error = None
        _logger.info("Cleared %d cached reranker model(s).", count)
        try:
            import gc

            gc.collect()
        except Exception:
            pass

    def _rerank_chunks(
        self,
        query_text: str,
        chunks: list[ChunkOut],
        *,
        repo_settings: dict[str, Any] | None,
        top_k_before: int | None = None,
        top_k_after: int | None = None,
    ) -> tuple[list[ChunkOut], bool, Any | None]:
        if not chunks:
            return chunks, False, None

        from src.features.retrieval.reranking.config import RerankerConfig

        config = RerankerConfig.from_env()
        if not config.enabled:
            return chunks, False, None

        reranker = self._get_cross_encoder_reranker(config=config)
        result = reranker.rerank(
            query_text,
            chunks,
            top_k=top_k_after,
            top_k_before=top_k_before,
            repo_settings=repo_settings,
        )
        from src.features.retrieval.reranking.order import finalize_reranked_chunks

        finalized = finalize_reranked_chunks(
            result.chunks,
            top_k=top_k_after,
            rerank_applied=result.applied,
            query=query_text,
        )
        return finalized, result.applied, result.metrics if result.applied else None

    def _apply_retrieval_observability(
        self,
        *,
        body: RetrieveRequest,
        query_text: str,
        chunks: list[ChunkOut],
        strategy: str,
        strategy_metrics: Any | None,
        stage_timer: Any,
        embedding_ms: float,
        rerank_applied: bool,
        rerank_metrics: Any | None,
        latency_ms: float,
        existing_quality_summary: dict[str, Any] | None = None,
        scoring_summary: dict[str, Any] | None = None,
    ) -> tuple[list[ChunkOut], dict[str, float], dict[str, Any], dict[str, Any]]:
        from src.features.retrieval.metrics import (
            RetrievalMetricsCollector,
            export_retrieval_metrics,
        )

        metrics = RetrievalMetricsCollector.build(
            strategy=strategy,
            repository_id=body.repository_id,
            query=query_text,
            top_k_requested=body.resolved_top_k,
            chunks=chunks,
            strategy_metrics=strategy_metrics,
            stage_timer=stage_timer,
            embedding_ms=embedding_ms,
            rerank_metrics=rerank_metrics,
            rerank_applied=rerank_applied,
            total_ms=latency_ms,
        )
        with stage_timer.stage("dlp"):
            dlp_chunks = _apply_retrieval_dlp(chunks)
        with stage_timer.stage("serialization"):
            pass
        metrics.latency.dlp_ms = stage_timer.get("dlp")
        metrics.latency.serialization_ms = stage_timer.get("serialization")
        metrics.latency.total_ms = latency_ms
        metrics.top_k_returned = len(dlp_chunks)

        RetrievalMetricsCollector.log_metrics(metrics)
        export_retrieval_metrics(metrics)
        try:
            from src.features.observability.hooks.retrieval_hooks import record_search_completed

            record_search_completed(
                repository_id=body.repository_id,
                duration_ms=latency_ms,
                result_count=len(dlp_chunks),
                retrieved_chunks=len(dlp_chunks),
                retrieved_documents=metrics.documents_retrieved,
                search_mode=strategy,
            )
        except Exception:
            _logger.debug("Failed to persist retrieval_duration metrics", exc_info=True)
        pipeline_trace, retrieval_summary, quality_summary = RetrievalMetricsCollector.finalize_response_fields(
            metrics,
            debug=body.debug_retrieval,
            existing_quality_summary=existing_quality_summary,
        )
        if scoring_summary:
            retrieval_summary.update(scoring_summary)
        return dlp_chunks, pipeline_trace, retrieval_summary, quality_summary

    def _query_model_path_for_context(
        self,
        body: RetrieveRequest,
        repo_context: dict[str, Any] | None,
    ) -> str:
        if body.model_name or body.model_dir:
            return _resolve_model_path(model_name=body.model_name, model_dir=body.model_dir)
        if repo_context:
            from src.features.repositories.infrastructure.collection_naming import resolve_processor_model_name

            settings = repo_context.get("settings") or {}
            hints = repo_context.get("processing_hints") or {}
            embedding = settings.get("embedding_model") or {}
            model_name = hints.get("model_name") or resolve_processor_model_name(embedding)
            model_dir = hints.get("model_dir") or embedding.get("local_model_dir")
            return _resolve_model_path(
                model_name=str(model_name or ""),
                model_dir=str(model_dir) if model_dir else None,
            )
        return _model_path_or_name()

    def _collection_exists(self, collection_name: str) -> bool:
        if not self._client:
            return False
        try:
            return bool(self._client.collections.exists(collection_name))
        except Exception:
            try:
                names = self._client.collections.list_all()
                return collection_name in set(names)
            except Exception:
                return False

    @staticmethod
    def _empty_retrieve_response(
        body: RetrieveRequest,
        *,
        collection_name: str | None,
        started: float,
        reason: str,
        security: dict[str, Any] | None = None,
        blocked: bool = False,
    ) -> RetrieveResponse:
        _logger.warning("Retrieval skipped for collection %s: %s", collection_name, reason)
        latency_ms = (time.time() - started) * 1000
        is_guard_block = blocked or "prompt blocked" in reason.lower() or "jailbreak" in reason.lower()
        return RetrieveResponse(
            query=body.resolved_query,
            expanded_queries=[],
            sub_queries=[],
            results=[],
            total_candidates=0,
            latency_ms=latency_ms,
            pipeline_stages_executed=["prompt_guard"] if is_guard_block else _compute_pipeline_stages(body.search_mode),
            pipeline_trace={"total_ms": round(latency_ms, 2)},
            retrieval_summary={
                "strategy": body.search_mode or "hybrid",
                "repository_id": body.repository_id,
                "documents_retrieved": 0,
                "pages_retrieved": 0,
                "returned_candidates": 0,
                "candidate_overlap": 0.0,
                "average_complexity": 0.0,
                "average_confidence": 0.0,
                "highest_confidence": 0.0,
                "lowest_confidence": 0.0,
                "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
                "confidence_distribution": {"very_high": 0, "high": 0, "medium": 0, "low": 0},
            },
            grounding_summary={"result_count": 0, "collection": collection_name, "skipped": reason},
            blocked=is_guard_block,
            block_reason=reason if is_guard_block else None,
            security=security
            or (
                {
                    "prompt_guard_executed": True,
                    "action": "block",
                    "pipeline_terminated": True,
                    "reason": reason,
                }
                if is_guard_block
                else None
            ),
        )

    def _collection_name(self, body: RetrieveRequest, repo_context: dict[str, Any] | None = None) -> str:
        cfg = get_settings()
        if repo_context:
            return str(repo_context.get("weaviate_collection") or cfg.default_collection_name)
        value = body.filters.get("collection_name") if body.filters else None
        return str(value or cfg.default_collection_name)

    @staticmethod
    def _empty_document_search_response(
        body: DocumentSearchRequest,
        *,
        query_text: str,
        search_mode: str,
        collection_name: str,
        started: float,
    ) -> DocumentSearchResponse:
        latency_ms = (time.time() - started) * 1000
        return DocumentSearchResponse(
            query=query_text,
            search_mode=search_mode,
            documents=[],
            count=0,
            total_candidates=0,
            latency_ms=round(latency_ms, 1),
            collection_name=collection_name,
            repository_id=body.repository_id,
        )

    def _prepare_query_rewrite(
        self,
        *,
        query_text: str,
        expand_query: bool | None,
    ) -> tuple[str, list[str], list[str], dict[str, Any]]:
        """Run heuristic rewrite/decomposition for default and production paths.

        Returns (effective_query, expanded_queries, sub_queries, rewrite_meta).
        When expansion is disabled → expanded_queries=[].
        When enabled but no rewrite provider produces distinct variants →
        expanded_queries=[] and rewrite_meta.expansion_status='unavailable'
        (never echo the original query as a fake expansion).
        """
        from src.features.retrieval.application.pipeline.stages import HeuristicQueryRewriter
        from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
        from src.features.retrieval.domain.models import PipelineState

        if expand_query is False:
            return query_text, [], [], {
                "expansion_status": "disabled",
                "rewrite_provider_available": False,
            }

        pipeline_cfg = RetrievalPipelineConfig.from_env({"enable_query_rewrite": True})
        state = PipelineState(original_query=query_text)
        stage = HeuristicQueryRewriter()
        result = stage.run(state, pipeline_cfg)
        meta = dict(getattr(result, "metadata", None) or {})
        expansions = [
            q for q in (state.expanded_queries or [])
            if str(q).strip() and str(q).strip().lower() != query_text.strip().lower()
        ]
        if expand_query is True and not expansions:
            meta["expansion_status"] = "unavailable"
        elif expansions:
            meta["expansion_status"] = "expanded"
        else:
            meta["expansion_status"] = (
                "unavailable" if not meta.get("rewrite_provider_available") else "expanded"
            )
        effective = (state.rewritten_query or query_text).strip() or query_text
        return effective, expansions, list(state.sub_queries or []), meta

    def _resolve_repository_context(self, body: RetrieveRequest) -> dict[str, Any] | None:
        if not body.repository_id:
            return None
        from src.features.retrieval.application.repository_gate import load_active_repository_context

        return load_active_repository_context(body.repository_id)

    def _effective_search_mode(self, body: RetrieveRequest, repo_settings: dict[str, Any] | None) -> str:
        requested = body.search_mode
        if requested in {"hybrid", "vector", "keyword"}:
            search_mode = requested
        elif repo_settings:
            configured = str(repo_settings.get("retrieval_search_mode") or "hybrid").strip().lower()
            search_mode = configured if configured in {"hybrid", "vector", "keyword"} else "hybrid"
        else:
            search_mode = "hybrid"
        if repo_settings and not repo_settings.get("lexical_composition", True) and search_mode in {"hybrid", "keyword"}:
            return "vector"
        return search_mode

    def _execute_collection_query(
        self,
        collection,
        *,
        query_text: str,
        query_vector: list[float] | None,
        search_mode: str,
        common: dict[str, Any],
        hybrid_alpha: float,
    ):
        def _bm25_fallback():
            return self._bm25().execute_bm25_on_collection(
                collection,
                query_text,
                limit=int(common.get("limit") or 10),
                filters=common.get("filters"),
                explain_score=bool(
                    common.get("return_metadata")
                    and getattr(common.get("return_metadata"), "explain_score", False)
                ),
            )

        try:
            if search_mode == "keyword" or (search_mode == "hybrid" and query_vector is None):
                return _bm25_fallback()
            if search_mode == "vector":
                if query_vector is None:
                    _logger.warning(
                        "Vector search requested without a query embedding; falling back to BM25"
                    )
                    return _bm25_fallback()
                return collection.query.near_vector(near_vector=query_vector, **common)
            return collection.query.hybrid(
                query=query_text,
                vector=query_vector,
                alpha=hybrid_alpha,
                **common,
            )
        except Exception as exc:
            if search_mode != "keyword" and (
                _is_vector_dim_mismatch(exc) or _is_invalid_near_vector(exc)
            ):
                _logger.warning(
                    "Vector search failed during %s search; falling back to BM25: %s",
                    search_mode,
                    exc,
                )
                return _bm25_fallback()
            raise

    def _should_use_production_pipeline(
        self,
        body: RetrieveRequest,
        *,
        search_mode: str,
        repo_settings: dict[str, Any] | None,
    ) -> bool:
        if body.use_production_pipeline is False:
            return False
        if body.use_production_pipeline is True:
            return True
        return False

    def _run_retrieval_strategy(
        self,
        *,
        body: RetrieveRequest,
        search_mode: str,
        collection: Any,
        collection_name: str,
        query_text: str,
        query_vector: list[float] | None,
        common: dict[str, Any],
        hybrid_alpha: float,
        citation_lookup: dict[str, dict[str, str]],
        query_terms: list[str],
        repo_settings: dict[str, Any] | None,
    ) -> Any:
        from src.features.retrieval.strategies import RetrievalStrategyContext, get_retrieval_strategy
        from src.features.retrieval.configuration.retrieval_config import (
            RetrievalPipelineConfig,
            resolve_candidate_pool_k,
        )

        pipeline_cfg = RetrievalPipelineConfig.from_env(
            {
                "rrf_k": body.rrf_k,
                "candidate_k": body.candidate_k,
                "dense_limit": body.candidate_k,
                "sparse_limit": body.candidate_k,
                "rerank_top_k": body.rerank_top_k,
            }
        )
        candidate_pool_k = resolve_candidate_pool_k(
            final_top_k=body.resolved_top_k,
            rerank_top_k=body.rerank_top_k,
            candidate_k=body.candidate_k,
            pipeline_cfg=pipeline_cfg,
        )
        vector_limit = int(body.candidate_k or pipeline_cfg.dense_limit or candidate_pool_k)
        bm25_limit = int(body.candidate_k or pipeline_cfg.sparse_limit or candidate_pool_k)
        rrf_k = int(body.rrf_k or pipeline_cfg.rrf_k)
        # Explicit alpha/hybrid_alpha must never be silently ignored by RRF.
        # When the caller sets alpha, force alpha-weighted fusion.
        explicit_alpha = body.hybrid_alpha is not None
        use_rrf = bool(pipeline_cfg.enable_hybrid_rrf) and not explicit_alpha

        def fetch_candidates(
            *,
            mode: str,
            query_text: str,
            query_vector: list[float] | None,
            common: dict[str, Any],
        ):
            return self._fetch_mode_candidates(
                collection,
                mode=mode,
                query_text=query_text,
                query_vector=query_vector,
                common=common,
                hybrid_alpha=hybrid_alpha,
                citation_lookup=citation_lookup,
                query_terms=query_terms,
                min_score=body.min_score,
                require_evidence=_require_query_evidence(body),
            )

        ctx = RetrievalStrategyContext(
            repository_id=body.repository_id,
            query=query_text,
            top_k=body.resolved_top_k,
            candidate_k=candidate_pool_k,
            collection=collection,
            query_vector=query_vector,
            common=common,
            hybrid_alpha=hybrid_alpha,
            citation_lookup=citation_lookup,
            query_terms=query_terms,
            min_score=body.min_score,
            require_evidence=_require_query_evidence(body),
            rrf_k=rrf_k,
            vector_limit=vector_limit,
            bm25_limit=bm25_limit,
            use_rrf=use_rrf,
            fetch_candidates=fetch_candidates,
        )
        effective_mode = search_mode
        if search_mode == "hybrid" and query_vector is None:
            effective_mode = "keyword"
        strategy = get_retrieval_strategy(effective_mode)
        return strategy.search(ctx)

    def _chunk_out_from_weaviate_object(
        self,
        obj: Any,
        *,
        idx: int,
        citation_lookup: dict[str, dict[str, str]],
        query_terms: list[str],
        min_score: float,
        require_evidence: bool,
    ) -> ChunkOut | None:
        props = obj.properties or {}
        metadata = getattr(obj, "metadata", None)
        distance = getattr(metadata, "distance", None) if metadata else None
        score = getattr(metadata, "score", None) if metadata else None
        if score is None and distance is not None:
            score = max(0.0, min(1.0, 1.0 - float(distance)))
        if score is None:
            score = max(0.01, 1.0 - (idx * 0.025))
        if float(score) < min_score:
            return None

        chunk_id = str(props.get("chunk_id") or obj.uuid)
        doc_name = str(props.get("doc_name") or "")
        section_name = str(props.get("section_name") or "")
        citation_fields = _citation_fields_from_properties(props)
        page = citation_fields["page"]
        line_start = citation_fields["line_start"]
        line_end = citation_fields["line_end"]
        line_range = citation_fields["line_range"]
        citation_anchor = citation_fields["citation_anchor"]
        source_blocks = citation_fields.get("source_blocks") or []
        text = str(props.get("text") or "")
        has_evidence, evidence_terms = _has_required_query_evidence(text, query_terms)
        if require_evidence and not has_evidence:
            return None
        cite = (
            citation_lookup.get(doc_name)
            or citation_lookup.get(Path(doc_name).stem)
            or {}
        )
        resolved_citation: dict[str, Any] = {}
        try:
            from src.features.citations.resolution.citation_resolver import resolve_citation

            resolved_citation = resolve_citation(
                {
                    **props,
                    "source_blocks": source_blocks or props.get("source_blocks"),
                    "document_name": cite.get("original_file_name") or doc_name,
                    "document_id": cite.get("document_id") or props.get("document_id"),
                },
                document_display_name=cite.get("original_file_name") or doc_name,
                evidence_terms=evidence_terms or query_terms,
                query_text=" ".join(query_terms) if query_terms else None,
            )
            # Prefer precise supporting-block lines/section when resolution narrowed them.
            if resolved_citation.get("line_start") is not None:
                line_start = resolved_citation.get("line_start")
            if resolved_citation.get("line_end") is not None:
                line_end = resolved_citation.get("line_end")
            if resolved_citation.get("page") is not None:
                page = resolved_citation.get("page")
            if resolved_citation.get("line_range"):
                line_range = resolved_citation.get("line_range")
            if resolved_citation.get("section"):
                section_name = str(resolved_citation.get("section") or section_name)
            if resolved_citation.get("source_blocks"):
                source_blocks = list(resolved_citation.get("source_blocks") or source_blocks)
        except Exception:
            _logger.debug("Citation resolution unavailable for chunk %s", chunk_id, exc_info=True)
        page_end = resolved_citation.get("page_end")
        if page_end is None:
            page_end = citation_fields.get("page_end")
        section_path = resolved_citation.get("section_path") or props.get("section_path")
        # Prefer Weaviate chunk property over citation-lookup (request-scoped) value.
        repo_id = props.get("repository_id") or cite.get("repository_id")
        chunk_metadata = {
                "source": doc_name,
                "section": section_name,
                "page": page,
                "page_end": page_end,
                "line_start": line_start,
                "line_end": line_end,
                "line_range": line_range,
                "section_path": section_path,
                "parent_section": props.get("parent_section"),
                "title": resolved_citation.get("title") or props.get("title"),
                "chunk_type": resolved_citation.get("chunk_type") or props.get("chunk_type"),
                "strategy_name": props.get("strategy_name"),
                "table_name": resolved_citation.get("table_name") or props.get("table_name"),
                "category": props.get("category"),
                "category_summary": props.get("category_summary"),
                "file_size_bytes": props.get("file_size_bytes"),
                "document_id": cite.get("document_id") or props.get("document_id"),
                "original_file_name": cite.get("original_file_name"),
                "repository_name": cite.get("repository_name"),
                "repository_id": repo_id,
                "heading": resolved_citation.get("section") or props.get("heading") or section_name,
                "topic": props.get("topic"),
                "entities": props.get("entities") or [],
                "sensitivity": props.get("sensitivity"),
                "embedding_version": props.get("embedding_version"),
                "logical_folder_id": props.get("logical_folder_id"),
                "logical_folder_path": props.get("logical_folder_path"),
                "query_evidence_terms": evidence_terms,
                "citation_anchor": citation_anchor,
                "source_blocks": source_blocks,
                "citation": resolved_citation or None,
                "citation_text": resolved_citation.get("citation_text"),
            }
        from src.features.retrieval.scoring.quality_scorer import enrich_chunk_scores

        highlights, grounding, quality, flags = enrich_chunk_scores(
            text=text,
            retrieval_score=float(score),
            metadata=chunk_metadata,
            query_terms=query_terms,
        )
        return ChunkOut(
            chunk_id=chunk_id,
            id=chunk_id,
            text=text,
            doc_name=doc_name,
            document_name=resolved_citation.get("document_name") or cite.get("original_file_name"),
            section_name=resolved_citation.get("section") or section_name,
            title=resolved_citation.get("title"),
            page=_coerce_page(page),
            page_end=page_end,
            line_start=line_start,
            line_end=line_end,
            line_range=line_range,
            citation_anchor=citation_anchor,
            citation_text=resolved_citation.get("citation_text"),
            score=round(float(score), 4),
            source=doc_name,
            graph_context=[],
            highlight_spans=highlights,
            metadata=chunk_metadata,
            grounding_score=grounding,
            quality_score=quality,
            quality_flags=flags,
            hop=0,
            retrieval_signals={
                "score": round(float(score), 4),
                "distance": round(float(distance), 4) if distance is not None else 0.0,
                "query_evidence_count": float(len(evidence_terms)),
            },
            repository_id=str(repo_id) if repo_id else None,
        )

    def _candidate_from_chunk(self, chunk: ChunkOut) -> Any:
        from src.features.retrieval.domain.models import RetrievalCandidate

        metadata = dict(chunk.metadata or {})
        if chunk.repository_id and not metadata.get("repository_id"):
            metadata["repository_id"] = chunk.repository_id
        return RetrievalCandidate(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            doc_name=chunk.doc_name,
            section_name=chunk.section_name,
            page=chunk.page,
            score=float(chunk.score),
            metadata=metadata,
            retrieval_signals=dict(chunk.retrieval_signals or {}),
        )

    def _chunk_from_candidate(self, candidate: Any, *, query_terms: list[str] | None = None) -> ChunkOut:
        from src.features.retrieval.scoring.quality_scorer import enrich_chunk_scores

        signals = dict(candidate.retrieval_signals or {})
        metadata = dict(candidate.metadata or {})
        text = str(getattr(candidate, "text", "") or "")
        repo_id = metadata.get("repository_id") or getattr(candidate, "repository_id", None)
        if repo_id and not metadata.get("repository_id"):
            metadata["repository_id"] = repo_id
        evidence = metadata.get("query_evidence_terms")
        if not isinstance(evidence, list) and query_terms:
            _, evidence_terms = _has_required_query_evidence(text, query_terms)
            metadata["query_evidence_terms"] = evidence_terms
        highlights, grounding, quality, flags = enrich_chunk_scores(
            text=text,
            retrieval_score=float(candidate.score),
            metadata=metadata,
            query_terms=query_terms or [],
        )
        return ChunkOut(
            chunk_id=str(candidate.chunk_id),
            id=str(candidate.chunk_id),
            text=text,
            doc_name=candidate.doc_name,
            document_name=metadata.get("original_file_name") or metadata.get("document_name"),
            section_name=candidate.section_name,
            title=metadata.get("title"),
            page=_coerce_page(candidate.page),
            page_end=_coerce_optional_int(metadata.get("page_end")),
            line_start=_coerce_optional_int(metadata.get("line_start")),
            line_end=_coerce_optional_int(metadata.get("line_end")),
            line_range=str(metadata.get("line_range") or "").strip() or None,
            citation_anchor=metadata.get("citation_anchor") if isinstance(metadata.get("citation_anchor"), dict) else None,
            citation_text=metadata.get("citation_text"),
            score=round(float(candidate.score), 4),
            source=candidate.doc_name,
            graph_context=[],
            highlight_spans=highlights,
            metadata=metadata,
            grounding_score=grounding,
            quality_score=quality,
            quality_flags=flags,
            hop=0,
            retrieval_signals=signals,
            repository_id=str(repo_id) if repo_id else None,
        )

    def _fetch_mode_candidates(
        self,
        collection,
        *,
        mode: str,
        query_text: str,
        query_vector: list[float] | None,
        common: dict[str, Any],
        hybrid_alpha: float,
        citation_lookup: dict[str, dict[str, str]],
        query_terms: list[str],
        min_score: float,
        require_evidence: bool,
    ) -> list[Any]:
        result = self._execute_collection_query(
            collection,
            query_text=query_text,
            query_vector=query_vector,
            search_mode=mode,
            common=common,
            hybrid_alpha=hybrid_alpha,
        )
        candidates = []
        for idx, obj in enumerate(result.objects or []):
            chunk = self._chunk_out_from_weaviate_object(
                obj,
                idx=idx,
                citation_lookup=citation_lookup,
                query_terms=query_terms,
                min_score=min_score,
                require_evidence=require_evidence,
            )
            if chunk is None:
                continue
            candidate = self._candidate_from_chunk(chunk)
            candidate.source_ranks[mode] = idx + 1
            candidate.source_scores[mode] = float(chunk.score)
            candidates.append(candidate)
        return candidates

    def _rerank_candidates(
        self,
        query_text: str,
        candidates: list[Any],
        config: Any,
        *,
        repo_settings: dict[str, Any] | None,
    ) -> list[Any]:
        self._last_rerank_outcome = {"applied": False, "metrics": None, "reason": "no_candidates"}
        if not candidates:
            return []
        chunks = [self._chunk_from_candidate(item, query_terms=_query_evidence_terms(query_text)) for item in candidates]
        settings = dict(repo_settings or {})
        settings.setdefault("reranker_model", config.preferred_reranker_model)
        from src.features.retrieval.reranking.config import RerankerConfig

        rerank_cfg = RerankerConfig.from_env(
            {
                "enabled": config.enable_rerank,
                "model": config.preferred_reranker_model,
                "top_k_before": config.rerank_top_k,
                "top_k_after": config.final_top_k,
                "score_threshold": config.rerank_score_threshold,
            }
        )
        reranker = self._get_cross_encoder_reranker(config=rerank_cfg)
        result = reranker.rerank(
            query_text,
            chunks,
            top_k=config.final_top_k,
            top_k_before=config.rerank_top_k,
            repo_settings=settings,
        )
        reason = "applied" if result.applied else "model_unavailable_or_disabled"
        self._last_rerank_outcome = {
            "applied": bool(result.applied),
            "metrics": result.metrics,
            "reason": reason,
        }
        if not result.applied:
            return candidates
        from src.features.retrieval.reranking.order import finalize_reranked_chunks

        finalized = finalize_reranked_chunks(
            result.chunks,
            top_k=config.final_top_k,
            rerank_applied=True,
            query=query_text,
        )
        return [self._candidate_from_chunk(chunk) for chunk in finalized]

    def _run_production_pipeline(
        self,
        *,
        body: RetrieveRequest,
        collection,
        collection_name: str,
        query_text: str,
        query_vector: list[float] | None,
        search_mode: str,
        repo_settings: dict[str, Any] | None,
        started: float,
        actor: Any,
    ) -> RetrieveResponse:
        from src.features.retrieval.application.pipeline import (
            CallableHybridRetriever,
            ExistingCrossEncoderReranker,
            build_default_pipeline,
        )
        from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
        from src.features.retrieval.domain.models import PipelineState
        from weaviate.classes.query import MetadataQuery

        cfg = get_settings()
        _effective_alpha = body.hybrid_alpha if body.hybrid_alpha is not None else cfg.hybrid_alpha
        use_rerank = _effective_use_rerank(body, repo_settings)
        # Explicit alpha forces alpha-weighted fusion; do not silently ignore via RRF.
        explicit_alpha = body.hybrid_alpha is not None
        pipeline_cfg = RetrievalPipelineConfig.from_env(
            {
                "hybrid_alpha": _effective_alpha,
                "final_top_k": body.resolved_top_k,
                "min_score": body.min_score,
                "enable_rerank": use_rerank,
                "enable_query_rewrite": True if body.expand_query else False if body.expand_query is False else True,
                "candidate_k": body.candidate_k,
                "rerank_top_k": body.rerank_top_k,
                "rrf_k": body.rrf_k,
            }
        ).with_request(
            top_k=body.resolved_top_k,
            min_score=body.min_score,
            use_rerank=use_rerank,
            expand_query=body.expand_query,
            hybrid_alpha=_effective_alpha,
        )
        if explicit_alpha:
            pipeline_cfg.enable_hybrid_rrf = False

        citation_lookup = (
            _build_repository_citation_lookup(body.repository_id)
            if body.repository_id
            else {}
        )
        query_terms = _query_evidence_terms(query_text)
        requested_document_id = _requested_document_id(body.filters)
        where_filter = _build_filter(body.filters)
        base_common = {
            "limit": pipeline_cfg.candidate_k,
            "filters": where_filter,
            "return_metadata": MetadataQuery(distance=True, score=True, explain_score=body.explain),
        }
        if where_filter is None:
            base_common.pop("filters")
        _logger.info(
            "Production pipeline Weaviate filters request_document_id=%s has_filter=%s",
            requested_document_id,
            where_filter is not None,
        )

        def dense_fn(state, config):
            if search_mode == "keyword":
                return []
            common = dict(base_common)
            common["limit"] = config.dense_limit
            filters = _build_filter(state.filters)
            if filters is not None:
                common["filters"] = filters
            elif "filters" in common:
                common.pop("filters")
            return self._fetch_mode_candidates(
                collection,
                mode="vector",
                query_text=state.rewritten_query or state.original_query,
                query_vector=query_vector,
                common=common,
                hybrid_alpha=config.hybrid_alpha,
                citation_lookup=citation_lookup,
                query_terms=_query_evidence_terms(state.rewritten_query or state.original_query),
                min_score=0.0,
                require_evidence=_require_query_evidence(body),
            )

        def sparse_fn(state, config):
            if search_mode == "vector":
                return []
            common = dict(base_common)
            common["limit"] = config.sparse_limit
            filters = _build_filter(state.filters)
            if filters is not None:
                common["filters"] = filters
            elif "filters" in common:
                common.pop("filters")
            return self._fetch_mode_candidates(
                collection,
                mode="keyword",
                query_text=state.rewritten_query or state.original_query,
                query_vector=None,
                common=common,
                hybrid_alpha=config.hybrid_alpha,
                citation_lookup=citation_lookup,
                query_terms=_query_evidence_terms(state.rewritten_query or state.original_query),
                min_score=0.0,
                require_evidence=_require_query_evidence(body),
            )

        # Vector/keyword production runs must not silently fuse both channels.
        if search_mode in {"vector", "keyword"}:
            pipeline_cfg.enable_hybrid_rrf = False

        state = PipelineState(original_query=query_text, filters=dict(body.filters or {}))
        pipeline = build_default_pipeline(
            hybrid_retriever=CallableHybridRetriever(dense_fn, sparse_fn),
            reranker=ExistingCrossEncoderReranker(
                lambda q, cands, conf: self._rerank_candidates(
                    q, cands, conf, repo_settings=repo_settings
                )
            ),
        )
        state = pipeline.run(state, pipeline_cfg)

        if state.blocked:
            guard_meta = {}
            for stage in state.stages or []:
                if getattr(stage, "stage", "") == "prompt_guard":
                    guard_meta = dict(getattr(stage, "metadata", None) or {})
                    break
            return self._empty_retrieve_response(
                body,
                collection_name=collection_name,
                started=started,
                reason=state.block_reason or "prompt blocked by guard",
                blocked=True,
                security={
                    "prompt_guard_executed": True,
                    "action": "block",
                    "pipeline_terminated": True,
                    "reason": state.block_reason or "prompt blocked by guard",
                    **guard_meta,
                },
            )

        chunks = [
            self._chunk_from_candidate(item, query_terms=_query_evidence_terms(query_text))
            for item in state.final_candidates
        ]
        rerank_outcome_early = getattr(self, "_last_rerank_outcome", None) or {}
        if rerank_outcome_early.get("applied"):
            from src.features.retrieval.reranking.order import finalize_reranked_chunks

            chunks = finalize_reranked_chunks(
                chunks,
                top_k=pipeline_cfg.final_top_k,
                rerank_applied=True,
                query=query_text,
            )
        pre_filter_count = len(chunks)
        chunks = _log_retrieval_filter_outcome(
            requested_document_id=requested_document_id,
            requested_folder_ids=_collect_request_folder_ids(body),
            chunks=chunks,
            stage="production_pipeline",
        )
        _logger.info(
            "Production pipeline candidates dense=%s sparse=%s fused=%s "
            "final_before_doc_check=%s final_after_doc_check=%s",
            len(state.dense_candidates),
            len(state.sparse_candidates),
            len(state.fused_candidates or []),
            pre_filter_count,
            len(chunks),
        )
        if body.repository_id and not chunks:
            return self._empty_retrieve_response(
                body,
                collection_name=collection_name,
                started=started,
                reason="no relevant evidence found in selected repository",
            )

        latency_ms = (time.time() - started) * 1000
        self._requests += 1
        self._last_latency_ms = latency_ms

        if actor and body.repository_id:
            try:
                from src.features.audit.application.security_audit_service import AuditService
                from src.features.users.infrastructure.user_repository import get_platform_security_store

                sec_store = get_platform_security_store()
                if sec_store:
                    AuditService(sec_store).record(
                        event_category="retrieval",
                        event_type="retrieval.search_executed",
                        action="execute",
                        outcome="success",
                        actor=actor,
                        repository_id=body.repository_id,
                        new_value={
                            "search_mode": search_mode,
                            "result_count": len(chunks),
                            "latency_ms": round(latency_ms, 1),
                            "pipeline": "production_rrf",
                        },
                    )
            except Exception:
                pass

        total_candidates = len(state.dense_candidates) + len(state.sparse_candidates)
        stages = list(state.pipeline_stages_executed)
        if "cross_encoder_reranker" in stages and "rerank" not in stages:
            stages.append("rerank")
        from src.features.retrieval.metrics import StageTimer
        from src.features.retrieval.scoring import enrich_retrieval_chunks
        from types import SimpleNamespace

        dense_count = len(state.dense_candidates)
        sparse_count = len(state.sparse_candidates)
        merged_count = len(state.fused_candidates or [])
        shared = max(0, dense_count + sparse_count - merged_count)
        overlap_pct = (shared / max(dense_count, sparse_count, 1)) * 100.0
        chunks, scoring_summary = enrich_retrieval_chunks(
            chunks,
            candidate_overlap_pct=overlap_pct,
            strategy=search_mode,
            repository_id=body.repository_id,
        )

        prod_timer = StageTimer()
        prod_timer.merge({k: float(v) for k, v in state.pipeline_trace.items() if isinstance(v, (int, float))})
        prod_metrics = SimpleNamespace(
            vector_candidates=len(state.dense_candidates),
            bm25_candidates=len(state.sparse_candidates),
            merged_candidates=len(state.fused_candidates or []),
            as_trace=lambda: dict(state.pipeline_trace),
            as_summary=lambda: {
                "vector_candidates": len(state.dense_candidates),
                "bm25_candidates": len(state.sparse_candidates),
                "merged_candidates": len(state.fused_candidates or []),
                "returned_candidates": len(chunks),
            },
        )
        rerank_outcome = getattr(self, "_last_rerank_outcome", None) or {}
        rerank_applied = bool(rerank_outcome.get("applied"))
        rerank_metrics = rerank_outcome.get("metrics")
        # Only advertise rerank stages when the model actually applied scores.
        if not rerank_applied:
            stages = [
                stage
                for stage in stages
                if stage not in {"rerank", "cross_encoder_reranker"}
            ]
            if pipeline_cfg.enable_rerank:
                stages.append("rerank_skipped")
        dlp_chunks, pipeline_trace, retrieval_summary, quality_summary = self._apply_retrieval_observability(
            body=body,
            query_text=query_text,
            chunks=chunks,
            strategy=search_mode,
            strategy_metrics=prod_metrics,
            stage_timer=prod_timer,
            embedding_ms=0.0,
            rerank_applied=rerank_applied,
            rerank_metrics=rerank_metrics,
            latency_ms=latency_ms,
            existing_quality_summary={
                "min_score": body.min_score,
                "returned": len(chunks),
                "rerank_top_k": pipeline_cfg.rerank_top_k,
                "rerank_reason": rerank_outcome.get("reason"),
            },
            scoring_summary=scoring_summary,
        )
        pipeline_trace = _safe_pipeline_trace(pipeline_trace) or pipeline_trace
        intent = state.intent if isinstance(state.intent, dict) else None
        return RetrieveResponse(
            query=query_text,
            domain=body.domain_name,
            complexity_level=body.complexity_level,
            expanded_queries=state.expanded_queries,
            sub_queries=state.sub_queries,
            results=dlp_chunks,
            total_candidates=total_candidates,
            latency_ms=round(latency_ms, 1),
            pipeline_stages_executed=stages,
            pipeline_trace=pipeline_trace,
            retrieval_summary=retrieval_summary,
            grounding_summary={
                "result_count": len(dlp_chunks),
                "collection": collection_name,
                "repository_id": body.repository_id,
                "query_evidence_terms": query_terms,
                "repository_scoped": bool(body.repository_id),
                "dense_count": len(state.dense_candidates),
                "sparse_count": len(state.sparse_candidates),
                "rrf_k": pipeline_cfg.rrf_k,
                "documents_retrieved": retrieval_summary.get("documents_retrieved"),
                "pages_retrieved": retrieval_summary.get("pages_retrieved"),
            },
            quality_summary=quality_summary,
            intent=intent,
            compressed_context=state.compressed_context or None,
            prompt=state.prompt or None,
        )

    def _resolve_retrieval_repository_ids(self, body: RetrieveRequest, actor) -> list[str]:
        if body.repository_id and not body.repository_ids and not body.search_all_repositories:
            return [body.repository_id]

        # Never fail open: search-all / multi-repo requires an authenticated actor.
        if not actor:
            return []

        if body.repository_ids:
            candidate_ids = [str(rid).strip() for rid in body.repository_ids if str(rid).strip()]
        elif body.search_all_repositories:
            from src.features.repositories.application.repository_service import get_repository_service

            payload = get_repository_service().list_repositories(status="active")
            candidate_ids = [
                str(repo.get("repository_id") or "").strip()
                for repo in payload.get("repositories") or []
                if str(repo.get("repository_id") or "").strip()
            ]
        else:
            return []

        from src.features.users.application.user_service import get_platform_security_service

        security = get_platform_security_service()
        allowed: list[str] = []
        for repository_id in candidate_ids:
            try:
                security.check_retrieval_access(actor, repository_id)
            except Exception:
                continue
            allowed.append(repository_id)
        return allowed

    async def _retrieve_merged(
        self,
        body: RetrieveRequest,
        repository_ids: list[str],
        started: float,
    ) -> RetrieveResponse:
        if not repository_ids:
            return self._empty_retrieve_response(
                body,
                collection_name=None,
                started=started,
                reason="no accessible repositories for scoped search",
            )

        per_repo_k = max(3, min(body.resolved_top_k, 40 // max(len(repository_ids), 1)))
        merged: list[ChunkOut] = []
        total_candidates = 0
        for repository_id in repository_ids:
            single_body = body.model_copy(
                update={
                    "repository_id": repository_id,
                    "repository_ids": None,
                    "search_all_repositories": False,
                    "top_k": per_repo_k,
                }
            )
            try:
                response = await self.retrieve(single_body)
            except (HTTPException, DmsServiceError):
                continue
            merged.extend(response.results)
            total_candidates += response.total_candidates

        merged.sort(key=lambda chunk: chunk.score, reverse=True)
        merged = merged[: body.resolved_top_k]
        latency_ms = (time.time() - started) * 1000
        self._requests += 1
        self._last_latency_ms = latency_ms
        return RetrieveResponse(
            query=body.resolved_query,
            domain=body.domain_name,
            complexity_level=body.complexity_level,
            expanded_queries=[],
            sub_queries=[],
            results=_apply_retrieval_dlp(merged),
            total_candidates=total_candidates,
            latency_ms=round(latency_ms, 1),
            pipeline_stages_executed=_compute_pipeline_stages(body.search_mode),
            pipeline_trace={"latency_ms": round(latency_ms, 1)},
            grounding_summary={"result_count": len(merged), "repository_count": len(repository_ids)},
            quality_summary={"min_score": body.min_score, "returned": len(merged)},
        )

    async def retrieve(self, body: RetrieveRequest) -> RetrieveResponse:
        if self._retrieval_enabled() and not self.ready:
            await self.startup()
        self._require_ready()
        started = time.time()

        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.users.application.user_service import get_platform_security_service
        from src.features.retrieval.metrics import StageTimer

        stage_timer = StageTimer()
        embedding_ms = 0.0
        strategy_result = None
        actor = get_current_user_from_context()
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError

        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        requested_folder_ids = _collect_request_folder_ids(body)
        if requested_folder_ids and (
            not body.repository_id or body.repository_ids or body.search_all_repositories
        ):
            raise _folder_filter_requires_repository()
        if (
            not body.repository_id
            and not body.repository_ids
            and not body.search_all_repositories
        ):
            return self._empty_retrieve_response(
                body,
                collection_name=None,
                started=started,
                reason="repository_id is required for repository-scoped retrieval",
            )

        if body.repository_ids or body.search_all_repositories:
            repository_ids = self._resolve_retrieval_repository_ids(body, actor)
            if len(repository_ids) > 1:
                return await self._retrieve_merged(body, repository_ids, started)
            if len(repository_ids) == 1:
                body = body.model_copy(
                    update={
                        "repository_id": repository_ids[0],
                        "repository_ids": None,
                        "search_all_repositories": False,
                    }
                )
            elif body.repository_ids or body.search_all_repositories:
                return self._empty_retrieve_response(
                    body,
                    collection_name=None,
                    started=started,
                    reason="no accessible repositories for scoped search",
                )

        if body.repository_id:
            get_platform_security_service().check_retrieval_access(actor, body.repository_id)

        cfg = get_settings()
        query_text = body.resolved_query
        normalized_filters = _normalize_retrieval_filters(
            body.filters,
            document_id=body.document_id,
        )
        if normalized_filters != dict(body.filters or {}):
            body = body.model_copy(update={"filters": normalized_filters})
        body = _apply_logical_folder_scope(body)
        requested_folder_ids = _collect_request_folder_ids(body)
        requested_document_id = _requested_document_id(body.filters)
        # When the user explicitly names a document present in the repository,
        # scope retrieval to that document_id (generic filename match only).
        if not requested_document_id and body.repository_id:
            early_lookup = _build_repository_citation_lookup(body.repository_id)
            inferred_document_id = _resolve_document_id_from_query(query_text, early_lookup)
            if inferred_document_id:
                scoped_doc_filters = dict(body.filters or {})
                scoped_doc_filters["document_id"] = inferred_document_id
                body = body.model_copy(update={"filters": scoped_doc_filters})
                requested_document_id = inferred_document_id
                _logger.info(
                    "Applied query-mentioned document filter document_id=%s",
                    inferred_document_id,
                )
        repo_context = self._resolve_repository_context(body)
        repo_settings = repo_context.get("settings") if repo_context else None
        collection_name = self._collection_name(body, repo_context)
        search_mode = self._effective_search_mode(body, repo_settings)
        use_rerank = _effective_use_rerank(body, repo_settings)
        if body.use_rerank is None and use_rerank:
            # Reflect resolved default on the request for downstream logging/trace.
            body = body.model_copy(update={"use_rerank": True})
        elif body.use_rerank is None and not use_rerank:
            body = body.model_copy(update={"use_rerank": False})

        # Enforce repository scoping in Weaviate filters before candidate fetch.
        if body.repository_id:
            scoped_filters = dict(body.filters or {})
            scoped_filters.setdefault("repository_id", body.repository_id)
            if scoped_filters != dict(body.filters or {}):
                body = body.model_copy(update={"filters": scoped_filters})

        (
            rewrite_query,
            expanded_queries,
            sub_queries,
            rewrite_meta,
        ) = self._prepare_query_rewrite(query_text=query_text, expand_query=body.expand_query)
        # Use rewritten query for retrieval; keep original on the response.
        effective_query_text = rewrite_query or query_text

        _logger.info(
            "Retrieval request received repository_id=%s document_id=%s collection=%s "
            "search_mode=%s top_k=%s use_rerank=%s filters=%s query=%r",
            body.repository_id,
            requested_document_id,
            collection_name,
            search_mode,
            body.resolved_top_k,
            body.use_rerank,
            body.filters or {},
            query_text,
        )

        if not self._collection_exists(collection_name):
            return self._empty_retrieve_response(
                body,
                collection_name=collection_name,
                started=started,
                reason="collection not found in Weaviate schema",
            )

        query_model_path = self._query_model_path_for_context(body, repo_context)
        tenant_id = repo_context.get("default_tenant_id") if repo_context else None

        try:
            from weaviate.classes.query import MetadataQuery

            collection = self._client.collections.get(collection_name)
            collection = resolve_collection(collection, tenant_id)
            where_filter = _build_filter(body.filters)
            query_vector = None
            if search_mode in {"vector", "hybrid"}:
                embed_started = time.perf_counter()
                query_vector = self._query_vector(effective_query_text, model_path=query_model_path)
                embedding_ms = (time.perf_counter() - embed_started) * 1000.0
                stage_timer.record("embedding", embedding_ms)
                if query_vector is not None and repo_settings:
                    from src.features.repositories.infrastructure.collection_naming import (
                        EmbeddingDimensionMismatchError,
                        assert_embedding_dimension_consistency,
                    )

                    embedding_cfg = repo_settings.get("embedding_model") or {}
                    if isinstance(embedding_cfg, dict) and embedding_cfg:
                        try:
                            assert_embedding_dimension_consistency(
                                embedding_cfg,
                                len(query_vector),
                                context="retrieval query embedding",
                            )
                        except EmbeddingDimensionMismatchError as exc:
                            raise_service_error(
                                COMPONENT_RETRIEVAL,
                                code="embedding_dimension_mismatch",
                                http_status=422,
                                user_message=(
                                    "Search cannot run because the query embedding size does not match "
                                    "this repository's configured model."
                                ),
                                reason=str(exc),
                            )
                _logger.info(
                    "Generated query embedding dimension=%s model_path=%s embedding_ms=%.1f",
                    len(query_vector) if query_vector is not None else None,
                    query_model_path,
                    embedding_ms,
                )
                if search_mode == "vector" and query_vector is None:
                    raise_service_error(
                        COMPONENT_RETRIEVAL,
                        code="embedding_model_unavailable",
                        http_status=503,
                        user_message="Search is temporarily unavailable because the embedding model is not ready.",
                        reason=f"Vector retrieval unavailable: {self._model_error or 'embedding model not loaded'}",
                    )

            if self._should_use_production_pipeline(
                body, search_mode=search_mode, repo_settings=repo_settings
            ):
                try:
                    return self._run_production_pipeline(
                        body=body,
                        collection=collection,
                        collection_name=collection_name,
                        query_text=query_text,
                        query_vector=query_vector,
                        search_mode=search_mode,
                        repo_settings=repo_settings,
                        started=started,
                        actor=actor,
                    )
                except Exception as exc:
                    _logger.warning(
                        "Production retrieval pipeline failed; falling back to retrieval strategy: %s",
                        exc,
                        exc_info=True,
                    )

            from src.features.retrieval.configuration.retrieval_config import (
                RetrievalPipelineConfig,
                resolve_candidate_pool_k,
            )

            strategy_pipeline_cfg = RetrievalPipelineConfig.from_env(
                {
                    "candidate_k": body.candidate_k,
                    "rerank_top_k": body.rerank_top_k,
                }
            )
            candidate_pool_k = resolve_candidate_pool_k(
                final_top_k=body.resolved_top_k,
                rerank_top_k=body.rerank_top_k,
                candidate_k=body.candidate_k,
                pipeline_cfg=strategy_pipeline_cfg,
            )

            _logger.info(
                "Searching Weaviate collection=%s search_type=%s candidate_pool_k=%s final_top_k=%s",
                collection_name,
                search_mode,
                candidate_pool_k,
                body.resolved_top_k,
            )
            common = {
                "limit": candidate_pool_k,
                "filters": where_filter,
                "return_metadata": MetadataQuery(distance=True, score=True, explain_score=body.explain),
            }
            if where_filter is None:
                common.pop("filters")

            citation_lookup = (
                _build_repository_citation_lookup(body.repository_id)
                if body.repository_id
                else {}
            )
            query_terms = _query_evidence_terms(effective_query_text)

            strategy_result = self._run_retrieval_strategy(
                body=body,
                search_mode=search_mode,
                collection=collection,
                collection_name=collection_name,
                query_text=effective_query_text,
                query_vector=query_vector,
                common=common,
                hybrid_alpha=body.hybrid_alpha if body.hybrid_alpha is not None else cfg.hybrid_alpha,
                citation_lookup=citation_lookup,
                query_terms=query_terms,
                repo_settings=repo_settings,
            )
            chunks = list(strategy_result.chunks)
            pipeline_stages = list(strategy_result.pipeline_stages)
            if body.expand_query is not False:
                pipeline_stages = ["query_rewrite", *pipeline_stages]

            # Execute distinct sub-queries on the default pipeline and merge uniquely.
            if sub_queries and search_mode in {"vector", "keyword", "hybrid"}:
                seen_ids = {c.chunk_id for c in chunks}
                for sub in sub_queries:
                    sub_text = str(sub or "").strip()
                    if not sub_text or sub_text.rstrip(" ?.").lower() == effective_query_text.rstrip(" ?.").lower():
                        continue
                    sub_vector = None
                    if search_mode in {"vector", "hybrid"}:
                        sub_vector = self._query_vector(sub_text, model_path=query_model_path)
                    sub_result = self._run_retrieval_strategy(
                        body=body,
                        search_mode=search_mode,
                        collection=collection,
                        collection_name=collection_name,
                        query_text=sub_text,
                        query_vector=sub_vector if search_mode != "keyword" else None,
                        common=common,
                        hybrid_alpha=body.hybrid_alpha if body.hybrid_alpha is not None else cfg.hybrid_alpha,
                        citation_lookup=citation_lookup,
                        query_terms=_query_evidence_terms(sub_text),
                        repo_settings=repo_settings,
                    )
                    for chunk in sub_result.chunks:
                        if chunk.chunk_id in seen_ids:
                            continue
                        seen_ids.add(chunk.chunk_id)
                        chunks.append(chunk)
                chunks = chunks[:candidate_pool_k]
                if "query_decomposition" not in pipeline_stages:
                    pipeline_stages.append("query_decomposition")

            pre_filter_count = len(chunks)
            chunks = _log_retrieval_filter_outcome(
                requested_document_id=requested_document_id,
                requested_folder_ids=requested_folder_ids,
                chunks=chunks,
                stage="strategy_pipeline",
            )
            total_candidates = max(strategy_result.total_candidates, len(chunks))
            stage_timer.merge(strategy_result.metrics.as_trace())
            grounding_summary_extra = {
                **strategy_result.metrics.as_summary(),
                "strategy": strategy_result.strategy,
                "rrf_k": int(body.rrf_k or 60),
                "expansion_status": rewrite_meta.get("expansion_status"),
                "sub_query_count": len(sub_queries),
            }
            _logger.info(
                "Retrieved %d chunks (total_candidates=%d before_doc_check=%s after_doc_check=%s) "
                "from collection=%s",
                len(chunks),
                total_candidates,
                pre_filter_count,
                len(chunks),
                collection_name,
            )
        except HTTPException:
            raise
        except DmsServiceError:
            raise
        except LogicalFolderError:
            raise
        except Exception as exc:
            _log_retrieval_failure(
                repository_id=body.repository_id,
                query=query_text,
                collection_name=collection_name,
                search_mode=search_mode,
                exc=exc,
            )
            if _is_retrieval_schema_error(exc):
                return self._empty_retrieve_response(
                    body,
                    collection_name=collection_name,
                    started=started,
                    reason=str(exc),
                )
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="retrieval_failed",
                http_status=500,
                user_message="Search could not be completed. Please try again later.",
                reason=f"Retrieval query failed for collection={collection_name}",
                cause=exc,
            )

        try:
            rerank_applied = False
            rerank_metrics = None
            if use_rerank:
                from src.features.retrieval.reranking.config import RerankerConfig
                from src.features.retrieval.reranking.reranker_model_resolver import (
                    resolve_reranker_model_path,
                )

                rerank_cfg = RerankerConfig.from_env(
                    {
                        "top_k_before": body.rerank_top_k,
                        "top_k_after": body.resolved_top_k,
                    }
                )
                model_path = resolve_reranker_model_path(repo_settings, config=rerank_cfg)
                candidates_before = len(chunks)
                _logger.info(
                    "Reranker enabled=%s model=%s top_k_before=%s top_k_after=%s "
                    "candidates_before=%s",
                    True,
                    model_path or rerank_cfg.model,
                    body.rerank_top_k or rerank_cfg.top_k_before,
                    body.resolved_top_k,
                    candidates_before,
                )
                chunks, rerank_applied, rerank_metrics = self._rerank_chunks(
                    query_text,
                    chunks,
                    repo_settings=repo_settings,
                    top_k_before=candidate_pool_k,
                    top_k_after=body.resolved_top_k,
                )
                if rerank_metrics is not None:
                    stage_timer.merge(rerank_metrics.as_trace())
                _logger.info(
                    "Reranker completed applied=%s candidates_after=%s rerank_ms=%s",
                    rerank_applied,
                    len(chunks),
                    getattr(rerank_metrics, "rerank_ms", None) if rerank_metrics else None,
                )
            else:
                _logger.info("Reranker enabled=False")

            if not rerank_applied:
                chunks = chunks[: body.resolved_top_k]

            if body.repository_id and not chunks:
                return self._empty_retrieve_response(
                    body,
                    collection_name=collection_name,
                    started=started,
                    reason="no relevant evidence found in selected repository",
                )

            latency_ms = (time.time() - started) * 1000
            self._requests += 1
            self._last_latency_ms = latency_ms

            if actor and body.repository_id:
                try:
                    from src.features.audit.application.security_audit_service import AuditService
                    from src.features.users.infrastructure.user_repository import get_platform_security_store

                    sec_store = get_platform_security_store()
                    if sec_store:
                        AuditService(sec_store).record(
                            event_category="retrieval",
                            event_type="retrieval.search_executed",
                            action="execute",
                            outcome="success",
                            actor=actor,
                            repository_id=body.repository_id,
                            new_value={
                                "search_mode": search_mode,
                                "result_count": len(chunks),
                                "latency_ms": round(latency_ms, 1),
                            },
                        )
                except Exception:
                    pass

            if rerank_applied:
                pipeline_stages.append("rerank")

            overlap_pct = 0.0
            if strategy_result is not None:
                metrics_obj = strategy_result.metrics
                vector_count = int(getattr(metrics_obj, "vector_candidates", 0) or 0)
                bm25_count = int(getattr(metrics_obj, "bm25_candidates", 0) or 0)
                merged_count = int(getattr(metrics_obj, "merged_candidates", 0) or 0)
                shared = max(0, vector_count + bm25_count - merged_count)
                overlap_pct = (shared / max(vector_count, bm25_count, 1)) * 100.0

            from src.features.retrieval.scoring import enrich_retrieval_chunks

            with stage_timer.stage("scoring"):
                chunks, scoring_summary = enrich_retrieval_chunks(
                    chunks,
                    candidate_overlap_pct=overlap_pct,
                    strategy=strategy_result.strategy if strategy_result else search_mode,
                    repository_id=body.repository_id,
                )

            dlp_chunks, pipeline_trace, retrieval_summary, quality_summary = self._apply_retrieval_observability(
                body=body,
                query_text=query_text,
                chunks=chunks,
                strategy=strategy_result.strategy if strategy_result else search_mode,
                strategy_metrics=strategy_result.metrics if strategy_result else None,
                stage_timer=stage_timer,
                embedding_ms=embedding_ms,
                rerank_applied=rerank_applied,
                rerank_metrics=rerank_metrics,
                latency_ms=latency_ms,
                existing_quality_summary={"min_score": body.min_score, "returned": len(chunks)},
                scoring_summary=scoring_summary,
            )
            pipeline_trace = _safe_pipeline_trace(pipeline_trace) or pipeline_trace

            response = RetrieveResponse(
                query=query_text,
                domain=body.domain_name,
                complexity_level=body.complexity_level,
                expanded_queries=expanded_queries,
                sub_queries=sub_queries,
                results=dlp_chunks,
                total_candidates=total_candidates,
                latency_ms=round(latency_ms, 1),
                pipeline_stages_executed=pipeline_stages,
                pipeline_trace=pipeline_trace,
                retrieval_summary={
                    **(retrieval_summary or {}),
                    "expansion_status": rewrite_meta.get("expansion_status"),
                    "sub_query_count": len(sub_queries),
                },
                grounding_summary={
                    "result_count": len(dlp_chunks),
                    "collection": collection_name,
                    "repository_id": body.repository_id,
                    "query_evidence_terms": query_terms,
                    "repository_scoped": bool(body.repository_id),
                    **grounding_summary_extra,
                    "documents_retrieved": retrieval_summary.get("documents_retrieved"),
                    "pages_retrieved": retrieval_summary.get("pages_retrieved"),
                },
                quality_summary=quality_summary,
            )
            _logger.info(
                "Returning top %d chunks response_size=%d repository_id=%s",
                len(dlp_chunks),
                len(dlp_chunks),
                body.repository_id,
            )
            return response
        except HTTPException:
            raise
        except DmsServiceError:
            raise
        except Exception as exc:
            _log_retrieval_failure(
                repository_id=body.repository_id,
                query=query_text,
                collection_name=collection_name,
                search_mode=search_mode,
                exc=exc,
            )
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="retrieval_failed",
                http_status=500,
                user_message="Search could not be completed. Please try again later.",
                reason=f"Retrieval post-processing failed for collection={collection_name}",
                cause=exc,
            )

    def _resolve_collection_context(
        self,
        *,
        repository_id: str | None,
        collection_name: str | None = None,
        tenant_id: str | None = None,
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        cfg = get_settings()
        if not repository_id:
            return (
                str(collection_name or cfg.default_collection_name),
                tenant_id or cfg.default_tenant_id,
                None,
            )
        from src.features.retrieval.application.repository_gate import load_active_repository_context

        repo = load_active_repository_context(repository_id)
        return (
            str(repo.get("weaviate_collection") or cfg.default_collection_name),
            tenant_id if tenant_id is not None else repo.get("default_tenant_id"),
            repo,
        )

    def _check_retrieval_authorization(self, repository_id: str | None) -> None:
        if not repository_id:
            return
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.users.application.user_service import get_platform_security_service

        actor = get_current_user_from_context()
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError

        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        get_platform_security_service().check_retrieval_access(actor, repository_id)

    def _fetch_indexed_stats(
        self,
        collection_name: str,
        tenant_id: str | None,
        *,
        limit: int = 10_000,
    ) -> dict[str, dict[str, Any]]:
        if not self.ready or self._client is None:
            return {}
        try:
            collection = self._client.collections.get(collection_name)
            collection = resolve_collection(collection, tenant_id)
            result = collection.query.fetch_objects(limit=limit)
            return aggregate_indexed_metadata(result.objects)
        except Exception:
            return {}

    def _load_catalog_records(
        self,
        *,
        intake_filters: dict[str, Any],
        metadata_filters: dict[str, Any],
        limit: int,
        offset: int,
    ) -> tuple[list[dict[str, Any]], int]:
        from src.features.documents.application.document_service import DocumentReceiverService

        store = DocumentReceiverService().get_store()
        total = store.count_documents(
            status=intake_filters.get("status"),
            tenant_id=intake_filters.get("tenant_id"),
            collection_name=intake_filters.get("collection_name"),
            repository_id=intake_filters.get("repository_id"),
            document_type=intake_filters.get("document_type"),
            original_file_name=intake_filters.get("original_file_name"),
            metadata_filters=metadata_filters or None,
        )
        records = store.list_documents(
            status=intake_filters.get("status"),
            tenant_id=intake_filters.get("tenant_id"),
            collection_name=intake_filters.get("collection_name"),
            repository_id=intake_filters.get("repository_id"),
            document_type=intake_filters.get("document_type"),
            original_file_name=intake_filters.get("original_file_name"),
            metadata_filters=metadata_filters or None,
            limit=limit,
            offset=offset,
        )
        return records, total

    def list_documents_catalog(
        self,
        *,
        repository_id: str | None = None,
        status: str | None = None,
        tenant_id: str | None = None,
        collection_name: str | None = None,
        document_type: str | None = None,
        original_file_name: str | None = None,
        category: str | None = None,
        indexed: bool | None = None,
        is_duplicate: bool | None = None,
        metadata_filters: dict[str, Any] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentListResponse:
        self._check_retrieval_authorization(repository_id)

        filters = {
            k: v
            for k, v in {
                "status": status,
                "tenant_id": tenant_id,
                "collection_name": collection_name,
                "repository_id": repository_id,
                "document_type": document_type,
                "original_file_name": original_file_name,
                "category": category,
                "indexed": indexed,
                "is_duplicate": is_duplicate,
            }.items()
            if v is not None
        }
        if metadata_filters:
            filters.update(metadata_filters)

        intake_filters, meta_filters, indexed_filters = split_catalog_filters(filters)
        records, total = self._load_catalog_records(
            intake_filters=intake_filters,
            metadata_filters=meta_filters,
            limit=min(max(limit, 1), 500),
            offset=max(offset, 0),
        )

        resolved_collection, resolved_tenant, _ = self._resolve_collection_context(
            repository_id=repository_id,
            collection_name=collection_name or intake_filters.get("collection_name"),
            tenant_id=tenant_id or intake_filters.get("tenant_id"),
        )
        indexed_stats = self._fetch_indexed_stats(resolved_collection, resolved_tenant)

        js = get_document_job_store()
        prefetched = js.fetch_for_documents([r["document_id"] for r in records]) if js else None

        items: list[DocumentCatalogItem] = []
        for record in records:
            job = prefetched.get(record["document_id"]) if prefetched else None
            item = record_to_catalog_item(record, indexed_stats=indexed_stats, job=job)
            if matches_indexed_filters(item, indexed_filters):
                items.append(item)

        return DocumentListResponse(
            documents=items,
            count=len(items),
            total=total,
            limit=limit,
            offset=offset,
            collection_name=resolved_collection,
            repository_id=repository_id,
        )

    async def search_documents_catalog(self, body: DocumentSearchRequest) -> DocumentSearchResponse:
        if self._retrieval_enabled() and not self.ready:
            await self.startup()
        self._require_ready()
        started = time.time()

        self._check_retrieval_authorization(body.repository_id)

        intake_filters, meta_filters, indexed_filters = split_catalog_filters(body.filters)
        if body.repository_id:
            intake_filters["repository_id"] = body.repository_id
        records, _ = self._load_catalog_records(
            intake_filters=intake_filters,
            metadata_filters=meta_filters,
            limit=5000,
            offset=0,
        )
        records_by_key = build_catalog_records_index(records)

        resolved_collection, resolved_tenant, repo_context = self._resolve_collection_context(
            repository_id=body.repository_id,
            collection_name=intake_filters.get("collection_name"),
            tenant_id=intake_filters.get("tenant_id"),
        )

        query_text = body.resolved_query
        search_mode = body.search_mode or "hybrid"
        if repo_context:
            repo_settings = repo_context.get("settings") or {}
            if not repo_settings.get("lexical_composition", True) and search_mode in {"hybrid", "keyword"}:
                search_mode = "vector"

        chunk_limit = min(max(body.top_k * 25, 50), 500)
        try:
            from weaviate.classes.query import MetadataQuery

            if not self._collection_exists(resolved_collection):
                _logger.warning(
                    "Document catalog search skipped for collection %s: collection not found in Weaviate schema",
                    resolved_collection,
                )
                return self._empty_document_search_response(
                    body,
                    query_text=query_text,
                    search_mode=search_mode,
                    collection_name=resolved_collection,
                    started=started,
                )

            collection = self._client.collections.get(resolved_collection)
            collection = resolve_collection(collection, resolved_tenant)

            query_vector = None
            if search_mode in {"vector", "hybrid"}:
                query_vector = self._query_vector(query_text)
                if search_mode == "vector" and query_vector is None:
                    raise_service_error(
                        COMPONENT_RETRIEVAL,
                        code="embedding_model_unavailable",
                        http_status=503,
                        user_message="Search is temporarily unavailable because the embedding model is not ready.",
                        reason=f"Vector retrieval unavailable: {self._model_error or 'embedding model not loaded'}",
                    )

            common = {
                "limit": chunk_limit,
                "return_metadata": MetadataQuery(distance=True, score=True),
            }
            # Enforce repository scoping in Weaviate BEFORE ranking/limiting.
            if body.repository_id:
                repo_filter = _build_filter({"repository_id": body.repository_id})
                if repo_filter is not None:
                    common["filters"] = repo_filter

            if search_mode == "keyword" or (search_mode == "hybrid" and query_vector is None):
                result = self._bm25().execute_bm25_on_collection(
                    collection,
                    query_text,
                    limit=chunk_limit,
                    filters=common.get("filters"),
                )
            elif search_mode == "vector":
                result = collection.query.near_vector(near_vector=query_vector, **common)
            else:
                cfg = get_settings()
                result = collection.query.hybrid(
                    query=query_text,
                    vector=query_vector,
                    alpha=cfg.hybrid_alpha,
                    **common,
                )
        except HTTPException:
            raise
        except DmsServiceError:
            raise
        except Exception as exc:
            raise_service_error(
                COMPONENT_RETRIEVAL,
                code="document_search_failed",
                http_status=500,
                user_message="Document search could not be completed. Please try again later.",
                reason=f"Document search failed for collection={resolved_collection}",
                cause=exc,
            )

        chunk_scores: dict[str, float] = {}
        for idx, obj in enumerate(result.objects):
            props = obj.properties or {}
            doc_keys: list[str] = []
            document_id = str(props.get("document_id") or "").strip()
            if document_id:
                doc_keys.append(document_id)
            doc_name = str(props.get("doc_name") or "").strip()
            if doc_name and doc_name not in doc_keys:
                doc_keys.append(doc_name)
            document_name = str(props.get("document_name") or "").strip()
            if document_name and document_name not in doc_keys:
                doc_keys.append(document_name)
            if not doc_keys:
                continue
            matched_record = None
            matched_key = None
            for key in doc_keys:
                if key in records_by_key:
                    matched_record = records_by_key[key]
                    matched_key = key
                    break
            if matched_record is None or matched_key is None:
                continue
            metadata = getattr(obj, "metadata", None)
            distance = getattr(metadata, "distance", None) if metadata else None
            score = getattr(metadata, "score", None) if metadata else None
            if score is None and distance is not None:
                score = max(0.0, min(1.0, 1.0 - float(distance)))
            if score is None:
                score = max(0.01, 1.0 - (idx * 0.025))
            chunk_scores[matched_key] = max(chunk_scores.get(matched_key, 0.0), float(score))

        ranked = rank_documents_by_chunk_scores(
            chunk_scores,
            records_by_key,
            top_k=body.top_k,
            min_score=body.min_score,
        )

        js = get_document_job_store()
        indexed_stats = self._fetch_indexed_stats(resolved_collection, resolved_tenant)

        documents: list[DocumentCatalogItem] = []
        for record, score in ranked:
            if body.repository_id and str(record.get("repository_id") or "") != str(body.repository_id):
                continue
            job = js.get_by_document_id(record["document_id"]) if js else None
            item = record_to_catalog_item(
                record,
                indexed_stats=indexed_stats,
                job=job,
                relevance_score=round(score, 4),
            )
            if body.repository_id and str(item.repository_id or "") != str(body.repository_id):
                continue
            if matches_indexed_filters(item, indexed_filters):
                documents.append(item)

        latency_ms = (time.time() - started) * 1000
        self._requests += 1
        self._last_latency_ms = latency_ms

        actor = None
        try:
            from src.application.consumer_api.context import get_current_user_from_context

            actor = get_current_user_from_context()
        except Exception:
            pass
        if actor and body.repository_id:
            try:
                from src.features.audit.application.security_audit_service import AuditService
                from src.features.users.infrastructure.user_repository import get_platform_security_store

                sec_store = get_platform_security_store()
                if sec_store:
                    AuditService(sec_store).record(
                        event_category="retrieval",
                        event_type="retrieval.document_search_executed",
                        action="execute",
                        outcome="success",
                        actor=actor,
                        repository_id=body.repository_id,
                        new_value={
                            "search_mode": search_mode,
                            "result_count": len(documents),
                            "latency_ms": round(latency_ms, 1),
                        },
                    )
            except Exception:
                pass

        return DocumentSearchResponse(
            query=query_text,
            search_mode=search_mode,
            documents=documents,
            count=len(documents),
            total_candidates=len(result.objects),
            latency_ms=round(latency_ms, 1),
            collection_name=resolved_collection,
            repository_id=body.repository_id,
        )

    def documents(self) -> dict[str, Any]:
        """Legacy helper — returns document names only."""
        catalog = self.list_documents_catalog(limit=1000, offset=0)
        return {
            "documents": sorted({item.document_name for item in catalog.documents}),
            "count": catalog.count,
        }

    def domains(self) -> dict[str, Any]:
        cfg = get_settings()
        return {"domains": ["default"], "collections": [cfg.default_collection_name]}

    def metrics(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "requests": self._requests,
            "last_latency_ms": round(self._last_latency_ms, 1),
            "model_ready": self._model is not None,
            "model_error": self._model_error,
            "uptime_seconds": round(time.time() - self._startup_time, 1) if self._startup_time else 0.0,
        }

    def get_document_chunks(self, repository_id: str, document_id: str) -> dict[str, Any]:
        """Return every stored chunk for a document within a repository (inspection / debug)."""
        from src.features.documents.application.document_service import DocumentReceiverService
        from src.features.repositories.domain.repository_exceptions import NotFoundError
        from src.features.repositories.application.repository_service import get_repository_service
        from src.features.retrieval.application.document_chunks import fetch_repository_document_chunks
        from src.features.retrieval.application.repository_gate import load_active_repository_context
        from src.features.retrieval.schemas.retrieval_schemas import DocumentChunkItem, DocumentChunksResponse

        started = time.perf_counter()
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError
        from src.features.users.application.user_service import get_platform_security_service

        actor = get_current_user_from_context()
        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        get_platform_security_service().check_retrieval_access(actor, repository_id)

        repo_service = get_repository_service()
        repo_context = load_active_repository_context(repository_id)

        if document_id not in repo_service.collect_document_ids_for_repository(repository_id):
            raise NotFoundError(
                "Document is not linked to this repository.",
                details={"repository_id": repository_id, "document_id": document_id},
            )

        receiver = DocumentReceiverService()
        try:
            record = receiver.get_document_record(document_id)
        except HTTPException as exc:
            if exc.status_code == 404:
                raise NotFoundError("Document not found.", details={"document_id": document_id}) from exc
            raise

        collection_name = str(
            record.get("collection_name") or repo_context.get("weaviate_collection") or ""
        ).strip()
        if not collection_name:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            _logger.info(
                "get_document_chunks repository_id=%s document_id=%s chunk_count=0 elapsed_ms=%.1f",
                repository_id,
                document_id,
                elapsed_ms,
            )
            return DocumentChunksResponse(
                document_id=document_id,
                repository_id=repository_id,
                chunk_count=0,
                chunks=[],
            ).model_dump()

        tenant = record.get("tenant_id") or repo_context.get("default_tenant_id")
        document_name = (
            str(record.get("original_file_name") or record.get("document_name") or "").strip() or None
        )

        chunks = fetch_repository_document_chunks(
            collection_name=collection_name,
            tenant=str(tenant) if tenant else None,
            document_id=document_id,
            document_name=document_name,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        _logger.info(
            "get_document_chunks repository_id=%s document_id=%s chunk_count=%d elapsed_ms=%.1f",
            repository_id,
            document_id,
            len(chunks),
            elapsed_ms,
        )
        response = DocumentChunksResponse(
            document_id=document_id,
            repository_id=repository_id,
            chunk_count=len(chunks),
            chunks=[DocumentChunkItem.model_validate(item) for item in chunks],
        )
        return response.model_dump()

    def download_original_document(self, document_id: str) -> tuple[Path, str, str]:
        """Resolve and authorize download of the stored original file for a document."""
        from src.application.consumer_api.context import get_current_user_from_context
        from src.features.documents.application.document_service import DocumentReceiverService
        from src.features.users.application.user_service import get_platform_security_service

        document_receiver = DocumentReceiverService()
        record = document_receiver.get_document_record(document_id)
        actor = get_current_user_from_context()
        from src.features.authentication.domain.authentication_exceptions import AuthenticationError

        if actor is None or actor.auth_method not in {"jwt", "api_key"} or actor.user_id in {"", "anonymous"}:
            raise AuthenticationError("Authentication required")
        repository_id = record.get("repository_id")
        if repository_id:
            get_platform_security_service().check_retrieval_access(actor, str(repository_id))

        path, filename, media_type = document_receiver.resolve_original_file(document_id)

        # Export-time DLP — downloads must not bypass upload controls.
        from src.features.security.moderation.export_moderator import moderate_export_file

        export = moderate_export_file(
            path,
            document_id=document_id,
            user=getattr(actor, "user_id", None) if actor else None,
        )
        if export.get("status") == "block":
            raise PermissionError(
                export.get("decision") or "Download blocked by export DLP policy."
            )
        if export.get("status") == "mask":
            # Persist redacted bytes beside the original for the HTTP layer to serve.
            redacted_name = str(export.get("filename") or f"{path.stem}.redacted.txt")
            redacted_path = path.parent / redacted_name
            redacted_path.write_bytes(export["content"])
            path = redacted_path
            filename = redacted_name
            media_type = str(export.get("media_type") or "text/plain")

        if actor and actor.auth_method in {"jwt", "api_key"} and actor.user_id != "anonymous":
            try:
                from src.features.audit.application.security_audit_service import AuditService
                from src.features.users.infrastructure.user_repository import get_platform_security_store

                sec_store = get_platform_security_store()
                if sec_store:
                    AuditService(sec_store).record(
                        event_category="retrieval",
                        event_type="retrieval.document_downloaded",
                        action="read",
                        outcome="success",
                        actor=actor,
                        repository_id=str(repository_id) if repository_id else None,
                        document_id=document_id,
                        new_value={
                            "original_file_name": filename,
                            "media_type": media_type,
                            "export_dlp": export.get("status"),
                        },
                    )
            except Exception:
                pass

        return path, filename, media_type


_default_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _default_retrieval_service
    if _default_retrieval_service is None:
        from src.features.retrieval.strategies.bm25.bm25_search_service import get_bm25_search_service

        _default_retrieval_service = RetrievalService(bm25_service=get_bm25_search_service())
    return _default_retrieval_service


def clear_reranker_model_cache() -> None:
    """Clear reranker models on the default retrieval service, if constructed."""
    if _default_retrieval_service is None:
        return
    _default_retrieval_service.clear_reranker_model_cache()
