"""Unit tests for chunking/vectorizing processor and related settings."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.features.document_processing.core.contract import ProcessRequest, RepositorySettings
from src.features.chunking.application.chunking_service import Chunk
from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy
from src.features.document_processing.loaders.component_classification import COMPONENT_IMAGE
from src.features.document_processing.loaders.loader import Block
from src.features.document_processing.processors.chunking_vectorizing import ChunkingVectorizingProcessor
from src.features.document_processing.processors.processing_settings import resolve_chunking_settings


def _text_block(text: str, *, page: int = 1, line_number: int = 1) -> Block:
    return Block(text=text, page=page, line_number=line_number)


def _image_block(text: str = "Image 1 on page 1", *, page: int = 1, line_number: int = 6) -> Block:
    return Block(
        text=text,
        page=page,
        line_number=line_number,
        block_type="image",
        component_type=COMPONENT_IMAGE,
        metadata={"component_type": COMPONENT_IMAGE, "image_index": 1},
    )


def _content_blocks() -> list[Block]:
    return [
        _text_block(
            "This section describes the validation requirements for the autoclave sterilization process.",
            line_number=1,
        ),
        _text_block(
            "Operators must verify cycle parameters and record results in the batch record.",
            line_number=2,
        ),
        _image_block(),
        _text_block(
            "Equipment must be qualified before routine production use under change control.",
            line_number=7,
        ),
    ]


@pytest.mark.parametrize(
    "strategy",
    [
        "section-based",
        "fixed-overlap-based",
        "sentence-based",
        "paragraph-based",
        "hierarchical",
        "sliding-window",
    ],
)
def test_chunking_strategies_exclude_image_blocks(strategy: str) -> None:
    blocks = _content_blocks()
    chunks = chunk_document_with_strategy(
        blocks,
        "sample.pdf",
        strategy=strategy,
        chunk_size=120,
        overlap_sentences=1,
        min_content_words=5,
        chunking_config={"max_sentences_per_chunk": 4, "max_paragraphs_per_chunk": 2},
    )
    assert chunks
    combined = " ".join(chunk.text for chunk in chunks)
    assert "Image 1" not in combined
    image_metadata_count = sum(
        len((chunk.extraction_metadata or {}).get("images") or [])
        for chunk in chunks
    )
    if strategy in {"section-based", "hierarchical"}:
        assert image_metadata_count == 1
    else:
        assert image_metadata_count == 0
        for chunk in chunks:
            assert COMPONENT_IMAGE not in (chunk.content_types or [])


def test_section_chunks_do_not_cross_page_boundaries() -> None:
    blocks = [
        _text_block("1.0 Procedure", page=1, line_number=1),
        _text_block(
            "Operators shall verify the first page setup and record the observations.",
            page=1,
            line_number=2,
        ),
        _text_block(
            "Operators shall continue the second page verification and record the final result.",
            page=2,
            line_number=1,
        ),
    ]

    chunks = chunk_document_with_strategy(
        blocks,
        "sample.docx",
        strategy="section-based",
        chunk_size=200,
        overlap_sentences=0,
        min_content_words=3,
    )

    assert [chunk.page for chunk in chunks] == [1, 2]
    assert chunks[0].line_start == 2
    assert chunks[0].line_end == 2
    assert chunks[1].line_start == 1
    assert chunks[1].line_end == 1


def test_resolve_settings_prefers_request_over_repository() -> None:
    request = ProcessRequest(
        document_id="doc-1",
        document_name="doc.pdf",
        chunk_size=400,
        chunk_overlap_sentences=2,
        chunking_strategy="sentence-based",
        citation_retainment=True,
        model_name="request/model",
        model_dir="request-dir",
        repository_settings=RepositorySettings(
            chunk_size=512,
            chunk_overlap_sentences=5,
            chunking_strategy="paragraph-based",
            citation_retainment=False,
            embedding_model_name="repo/model",
            embedding_model_dir="repo-dir",
        ),
    )
    resolved = resolve_chunking_settings(request)
    assert resolved.chunk_size == 400
    assert resolved.chunk_overlap_sentences == 2
    assert resolved.chunking_strategy == "sentence-based"
    assert resolved.citation_retainment is True
    assert resolved.model_name == "request/model"
    assert resolved.model_dir == "request-dir"


def test_resolve_settings_falls_back_to_repository() -> None:
    request = ProcessRequest(
        document_id="doc-2",
        document_name="doc.pdf",
        repository_settings=RepositorySettings(
            chunk_size=640,
            chunk_overlap_sentences=3,
            chunking_strategy="fixed-overlap-based",
            chunking_config={"min_content_words": 12},
            citation_retainment=False,
            embedding_model_name="BAAI/bge-base-en",
            embedding_model_dir="bge-base-en",
        ),
    )
    resolved = resolve_chunking_settings(request)
    assert resolved.chunk_size == 640
    assert resolved.chunk_overlap_sentences == 3
    assert resolved.chunking_strategy == "fixed-overlap-based"
    assert resolved.min_content_words == 12
    assert resolved.citation_retainment is False
    assert resolved.model_name == "BAAI/bge-base-en"
    assert resolved.model_dir == "bge-base-en"


def test_processor_applies_repository_settings_and_skips_images(tmp_path: Path) -> None:
    document_path = tmp_path / "sample.pdf"
    document_path.write_bytes(b"%PDF-1.4")

    request = ProcessRequest(
        document_id="doc-3",
        document_name="sample.pdf",
        original_file_name="Original Name.pdf",
        collection_name="TestCollection",
        repository_settings=RepositorySettings(
            chunk_size=256,
            chunk_overlap_sentences=1,
            chunking_strategy="fixed-overlap-based",
            citation_retainment=False,
            embedding_model_name="test-model",
            embedding_model_dir="test-dir",
        ),
    )

    produced_chunks = [
        Chunk(
            id="chunk-1",
            doc_name="Original Name.pdf",
            page=2,
            section_name="Document",
            section_path="Document",
            text="Validation content for the autoclave process and batch records.",
            line_start=1,
            line_end=2,
        )
    ]

    processor = ChunkingVectorizingProcessor()
    statuses: list[tuple[str, float | None]] = []

    with (
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.validate_pipeline_request",
            return_value="fixed-overlap-based",
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.load_document_blocks",
            return_value=(_content_blocks(), {"source": "test"}),
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.chunk_document_with_strategy",
            return_value=produced_chunks,
        ) as chunk_mock,
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.embed_chunks",
            side_effect=lambda chunks, *_args, **_kwargs: [setattr(c, "embedding", [0.1, 0.2]) for c in chunks],
        ),
        patch("src.features.document_processing.processors.chunking_vectorizing.store_in_weaviate"),
        patch("src.features.document_processing.processors.chunking_vectorizing.apply_document_classification"),
    ):
        result = processor.run(
            request,
            document_path,
            set_status=lambda name, pct=None: statuses.append((name, pct)),
            check_stop=lambda: False,
        )

    chunk_mock.assert_called_once()
    kwargs = chunk_mock.call_args.kwargs
    assert kwargs["strategy"] == "fixed-overlap-based"
    assert kwargs["chunk_size"] == 256
    assert kwargs["overlap_sentences"] == 1
    passed_blocks = chunk_mock.call_args.args[0]
    assert any(getattr(block, "block_type", "") == "image" for block in passed_blocks)

    assert produced_chunks[0].page is None
    assert produced_chunks[0].line_start is None
    assert produced_chunks[0].line_end is None

    assert result.document_metadata["citation_retainment"] is False
    assert result.document_metadata["chunking_strategy"] == "fixed-overlap-based"
    assert result.document_metadata["embedding_model_name"] == "test-model"
    assert result.artifacts["embedded_chunks"] == 1
    assert ("chunking", 10.0) in statuses
    assert ("embedding", 45.0) in statuses
    assert ("storing_weaviate", 80.0) in statuses


def test_processor_raises_when_only_image_blocks_remain(tmp_path: Path) -> None:
    document_path = tmp_path / "image-only.pdf"
    document_path.write_bytes(b"%PDF-1.4")
    request = ProcessRequest(document_id="doc-4", document_name="image-only.pdf")

    processor = ChunkingVectorizingProcessor()
    with (
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.validate_pipeline_request",
            return_value="section-based",
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.load_document_blocks",
            return_value=([_image_block()], {}),
        ),
    ):
        with pytest.raises(RuntimeError, match="No chunkable content"):
            processor.run(
                request,
                document_path,
                set_status=lambda *_args, **_kwargs: None,
                check_stop=lambda: False,
            )



def test_chunking_processor_does_not_import_neo4j_reference_writes() -> None:
    import inspect

    from src.features.document_processing.processors import chunking_vectorizing as module

    source = inspect.getsource(module)
    assert "neo4j_reference_store" not in source
    assert "save_chunk_references" not in source
    assert "save_document_reference_graph" not in source


def test_semantic_strategy_uses_resolved_embedding_model(tmp_path: Path) -> None:
    document_path = tmp_path / "semantic.pdf"
    document_path.write_bytes(b"%PDF-1.4")
    request = ProcessRequest(
        document_id="doc-5",
        document_name="semantic.pdf",
        chunking_strategy="semantic",
        model_name="semantic-model",
        model_dir="semantic-dir",
    )

    captured: dict[str, str | None] = {}

    def fake_encode(texts, model_name, *, model_dir=None):
        captured["model_name"] = model_name
        captured["model_dir"] = model_dir
        return np.ones((len(texts), 2))

    processor = ChunkingVectorizingProcessor()
    chunk_kwargs: dict = {}

    def capture_chunk_call(*args, **kwargs):
        chunk_kwargs.update(kwargs)
        return [
            Chunk(
                id="chunk-1",
                doc_name="semantic.pdf",
                page=1,
                section_name="Document",
                section_path="Document",
                text="Validation content for the autoclave process and batch records.",
            )
        ]

    with (
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.validate_pipeline_request",
            return_value="semantic",
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.load_document_blocks",
            return_value=(_content_blocks(), {}),
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.encode_texts",
            side_effect=fake_encode,
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.chunk_document_with_strategy",
            side_effect=capture_chunk_call,
        ),
        patch(
            "src.features.document_processing.processors.chunking_vectorizing.embed_chunks",
            side_effect=lambda chunks, *_args, **_kwargs: [setattr(c, "embedding", [0.1]) for c in chunks],
        ),
        patch("src.features.document_processing.processors.chunking_vectorizing.store_in_weaviate"),
        patch("src.features.document_processing.processors.chunking_vectorizing.apply_document_classification"),
    ):
        processor.run(
            request,
            document_path,
            set_status=lambda *_args, **_kwargs: None,
            check_stop=lambda: False,
        )
        embed_fn = chunk_kwargs.get("embed_fn")
        assert embed_fn is not None
        embed_fn(["First sentence.", "Second sentence."])

    assert captured["model_name"] == "semantic-model"
    assert captured["model_dir"] == "semantic-dir"
