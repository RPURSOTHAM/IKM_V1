from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.features.document_processing.core.contract import ProcessRequest
from src.features.document_processing.pipeline_tracking import (
    coalesce_setting,
    explain_zero_chunks,
    get_last_failure,
    record_failure,
    validate_chunking_strategy,
    validate_document_file,
)
from src.features.chunking.domain.chunking_strategy import resolve_chunking_strategy


def test_coalesce_setting_skips_blank_strings() -> None:
    assert coalesce_setting("", None, "bge-base-en") == "bge-base-en"
    assert coalesce_setting("   ", "fallback") == "fallback"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sentence", "sentence-based"),
        ("fixed", "fixed-overlap-based"),
        ("section-based", "section-based"),
    ],
)
def test_validate_chunking_strategy_aliases(raw: str, expected: str) -> None:
    assert validate_chunking_strategy(raw) == expected


def test_invalid_chunking_strategy_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported chunking strategy"):
        resolve_chunking_strategy("not-valid")


def test_validate_document_file_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pdf"
    with pytest.raises(FileNotFoundError):
        validate_document_file(missing)


def test_validate_document_file_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "notes.xyz"
    bad.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported document extension"):
        validate_document_file(bad)


def test_explain_zero_chunks_messages() -> None:
    stats = {"paragraph_count": 0, "table_count": 0, "image_count": 3}
    assert "No document blocks" in explain_zero_chunks(stats, had_blocks=False, had_text=False)
    assert "none contained chunkable text" in explain_zero_chunks(stats, had_blocks=True, had_text=False)


def test_record_failure_snapshot() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError as exc:
        record_failure(
            exc,
            stage="chunking",
            document_id="doc-1",
            document_path="/app/documents/a.pdf",
            chunking_strategy="sentence-based",
        )
    payload = get_last_failure()
    assert payload["failed_stage"] == "chunking"
    assert payload["exception_type"] == "RuntimeError"
    assert "boom" in payload["exception_message"]
    assert payload["document_id"] == "doc-1"


def test_process_request_blank_model_fields_become_none() -> None:
    request = ProcessRequest(
        document_id="doc-1",
        document_name="sample.pdf",
        model_name="",
        model_dir="",
        weaviate_url="",
    )
    assert request.model_name is None
    assert request.model_dir is None
    assert request.weaviate_url is None


@patch("src.features.document_processing.pipeline_tracking.ping_neo4j")
@patch("src.features.document_processing.pipeline_tracking.ping_weaviate")
@patch("src.features.document_processing.pipeline_tracking.check_embedding_model_available")
@patch("src.features.document_processing.pipeline_tracking.validate_document_file")
def test_validate_pipeline_request_happy_path(
    mock_validate_doc: MagicMock,
    mock_embedding: MagicMock,
    mock_weaviate: MagicMock,
    mock_neo4j: MagicMock,
    tmp_path: Path,
) -> None:
    from src.features.document_processing.pipeline_tracking import validate_pipeline_request

    doc = tmp_path / "sample.pdf"
    doc.write_bytes(b"%PDF-1.4")
    resolved = validate_pipeline_request(
        document_path=doc,
        chunking_strategy="sentence",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_dir=None,
        weaviate_url="http://weaviate:8080",
        weaviate_api_key=None,
        collection_name="DocumentChunk",
    )
    assert resolved == "sentence-based"
    mock_validate_doc.assert_called_once()
    mock_embedding.assert_called_once()
    mock_weaviate.assert_called_once()
    mock_neo4j.assert_called_once()


def test_process_request_blank_model_fields_fall_through_to_env_default() -> None:
    request = ProcessRequest(
        document_id="doc-1",
        document_name="sample.pdf",
        model_name="",
        model_dir="",
        chunking_strategy="sentence",
    )
    assert request.model_name is None
    assert request.model_dir is None
    assert request.chunking_strategy == "sentence-based"


@patch("src.features.embeddings.application.embedding_service.get_embedding_model")
def test_check_embedding_model_available_requires_name(mock_get_model: MagicMock) -> None:
    from src.features.document_processing.pipeline_tracking import check_embedding_model_available

    with pytest.raises(RuntimeError, match="Embedding model is not configured"):
        check_embedding_model_available("")
    mock_get_model.assert_not_called()
