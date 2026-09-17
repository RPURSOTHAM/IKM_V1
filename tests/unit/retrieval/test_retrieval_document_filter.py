"""Document-level Weaviate filter must be applied before vector/BM25 retrieval."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.features.retrieval.schemas.retrieval_schemas import RetrieveRequest
from src.features.retrieval.application.retrieval_service import (
    _build_filter,
    _apply_logical_folder_scope,
    _collect_request_folder_ids,
    _log_retrieval_filter_outcome,
    _normalize_retrieval_filters,
    _requested_document_id,
)


DOC_A = "5b4f5738-9fa3-45e6-8dde-99b3e066d45d"
DOC_B = "0a0620da-a1d8-40a3-b00b-57ede2a035bf"
REPO_ID = "11111111-1111-1111-1111-111111111111"
FOLDER_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
FOLDER_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


def test_retrieve_request_merges_top_level_document_id_into_filters() -> None:
    body = RetrieveRequest(
        query="what is CAPA?",
        repository_id=REPO_ID,
        document_id=DOC_A,
    )
    assert body.filters.get("document_id") == DOC_A
    assert _requested_document_id(body.filters) == DOC_A


def test_normalize_retrieval_filters_merges_document_id() -> None:
    assert _normalize_retrieval_filters({}, document_id=DOC_A) == {"document_id": DOC_A}
    assert _normalize_retrieval_filters(
        {"doc_name": "GL-CQA-ANN-0427.pdf"},
        document_id=DOC_A,
    ) == {"doc_name": "GL-CQA-ANN-0427.pdf", "document_id": DOC_A}


def test_build_filter_applies_document_id_and_repository_id() -> None:
    pytest.importorskip("weaviate")

    where = _build_filter({"document_id": DOC_A, "repository_id": REPO_ID})
    assert where is not None
    clauses = list(getattr(where, "filters", None) or [where])
    by_target = {str(getattr(c, "target", "")): str(getattr(c, "value", "")) for c in clauses}
    assert by_target.get("document_id") == DOC_A
    assert by_target.get("repository_id") == REPO_ID
    assert str(getattr(where, "operator", "")).endswith("AND") or len(clauses) == 2


def test_build_filter_without_document_id_omits_document_clause() -> None:
    pytest.importorskip("weaviate")
    where = _build_filter({"doc_name": "GL-CQA-ANN-0427.pdf"})
    assert where is not None
    target = str(getattr(where, "target", "") or "")
    assert target == "doc_name"
    assert getattr(where, "value", None) == "GL-CQA-ANN-0427.pdf"


def test_build_filter_invalid_or_empty_document_id_returns_none_clause() -> None:
    assert _build_filter({"document_id": ""}) is None
    assert _build_filter({"document_id": None}) is None
    assert _build_filter({}) is None


def test_build_filter_unknown_keys_ignored_known_keys_kept() -> None:
    pytest.importorskip("weaviate")
    where = _build_filter({"document_id": DOC_A, "unknown_key": "x"})
    assert where is not None
    assert DOC_A in str(where)


def _chunk(doc_id: str, name: str, *, folder_id: str | None = None) -> SimpleNamespace:
    metadata = {"document_id": doc_id, "doc_name": name}
    if folder_id:
        metadata["logical_folder_id"] = folder_id
    return SimpleNamespace(
        doc_name=name,
        document_name=name,
        metadata=metadata,
    )


def test_log_outcome_keeps_all_when_no_document_filter() -> None:
    chunks = [_chunk(DOC_A, "A.pdf"), _chunk(DOC_B, "B.pdf")]
    kept = _log_retrieval_filter_outcome(
        requested_document_id=None,
        chunks=chunks,
        stage="test",
    )
    assert len(kept) == 2


def test_log_outcome_drops_mismatched_document_ids() -> None:
    chunks = [_chunk(DOC_A, "GL-CQA-ANN-0427.pdf"), _chunk(DOC_B, "GL-CQA-ANN-0063.pdf")]
    kept = _log_retrieval_filter_outcome(
        requested_document_id=DOC_A,
        chunks=chunks,
        stage="test",
    )
    assert len(kept) == 1
    assert kept[0].metadata["document_id"] == DOC_A


def test_folder_scope_validates_ids_and_merges_exact_weaviate_filter() -> None:
    body = RetrieveRequest(
        query="find CAPA",
        repository_id=REPO_ID,
        folder_id=FOLDER_A,
        folder_ids=[FOLDER_B, FOLDER_A],
    )
    service = MagicMock()
    service.require_folders_in_repository.return_value = [FOLDER_B, FOLDER_A]
    with patch(
        "src.features.logical_folders.application.folder_service.get_logical_folder_service",
        return_value=service,
    ):
        scoped = _apply_logical_folder_scope(body)

    service.require_folders_in_repository.assert_called_once_with(REPO_ID, [FOLDER_B, FOLDER_A])
    assert _collect_request_folder_ids(scoped) == [FOLDER_B, FOLDER_A]
    assert scoped.filters["logical_folder_id"] == [FOLDER_B, FOLDER_A]


def test_folder_scope_fails_closed_for_nonmatching_or_unassigned_chunk() -> None:
    chunks = [
        _chunk(DOC_A, "A.pdf", folder_id=FOLDER_A),
        _chunk(DOC_B, "B.pdf", folder_id=FOLDER_B),
        _chunk("cccccccc-cccc-cccc-cccc-cccccccccccc", "unassigned.pdf"),
    ]
    kept = _log_retrieval_filter_outcome(
        requested_document_id=None,
        requested_folder_ids=[FOLDER_A],
        chunks=chunks,
        stage="test",
    )
    assert len(kept) == 1
    assert kept[0].metadata["logical_folder_id"] == FOLDER_A


def test_folder_scope_requires_one_repository() -> None:
    from fastapi import HTTPException

    body = RetrieveRequest(query="find CAPA", search_all_repositories=True, folder_id=FOLDER_A)
    with pytest.raises(HTTPException) as exc_info:
        _apply_logical_folder_scope(body)
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail["code"] == "folder_filter_requires_repository"


class _CapturingQuery:
    def __init__(self) -> None:
        self.near_vector_kwargs: dict | None = None
        self.bm25_kwargs: dict | None = None
        self.hybrid_kwargs: dict | None = None

    def near_vector(self, **kwargs):
        self.near_vector_kwargs = kwargs
        return SimpleNamespace(objects=[])

    def bm25(self, **kwargs):
        self.bm25_kwargs = kwargs
        return SimpleNamespace(objects=[])

    def hybrid(self, **kwargs):
        self.hybrid_kwargs = kwargs
        return SimpleNamespace(objects=[])


def test_vector_and_keyword_paths_receive_document_filter() -> None:
    """Case 2: document_id must be present on Weaviate query kwargs before ranking."""
    pytest.importorskip("weaviate")
    from src.features.retrieval.application.retrieval_service import RetrievalService

    where = _build_filter({"document_id": DOC_A})
    assert where is not None

    query = _CapturingQuery()
    collection = SimpleNamespace(query=query)
    service = RetrievalService.__new__(RetrievalService)
    service._bm25 = MagicMock(
        return_value=SimpleNamespace(
            execute_bm25_on_collection=MagicMock(
                side_effect=lambda coll, q, **kw: coll.query.bm25(query=q, **kw)
            )
        )
    )

    common = {"limit": 5, "filters": where}
    service._execute_collection_query(
        collection,
        query_text="CAPA",
        query_vector=[0.1, 0.2],
        search_mode="vector",
        common=dict(common),
        hybrid_alpha=0.5,
    )
    assert query.near_vector_kwargs is not None
    assert query.near_vector_kwargs.get("filters") is where

    service._execute_collection_query(
        collection,
        query_text="CAPA",
        query_vector=None,
        search_mode="keyword",
        common=dict(common),
        hybrid_alpha=0.5,
    )
    assert query.bm25_kwargs is not None
    assert query.bm25_kwargs.get("filters") is where


def test_case_without_document_id_filter_is_absent() -> None:
    """Case 1: without document_id, Weaviate queries must not include a document filter."""
    assert _build_filter({"section_name": "intro"}) is not None
    assert _build_filter({}) is None
    body = RetrieveRequest(query="search all", repository_id=REPO_ID)
    assert "document_id" not in (body.filters or {})


def test_case_invalid_document_id_builds_equal_filter() -> None:
    """Case 3: invalid UUID still becomes an equality filter (Weaviate returns no matches)."""
    pytest.importorskip("weaviate")
    bogus = "00000000-0000-0000-0000-000000000000"
    where = _build_filter({"document_id": bogus})
    assert where is not None
    assert bogus in str(where)
