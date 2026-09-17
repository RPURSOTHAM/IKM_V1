from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip("weaviate")

from src.features.repositories.infrastructure.collection_naming import (  # noqa: E402
    LOCAL_MODEL_HUB_IDS,
    LOCAL_MODEL_VECTOR_DIMS,
)
from src.infrastructure.document_databases.weaviate_store import store_in_weaviate  # noqa: E402
from src.shared.semantic.models import SemanticChunk  # noqa: E402


def test_bge_m3_is_configurable_without_becoming_the_default() -> None:
    assert LOCAL_MODEL_HUB_IDS["bge-m3"] == "BAAI/bge-m3"
    assert LOCAL_MODEL_VECTOR_DIMS["bge-m3"] == 1024


def test_semantic_metadata_is_stored_with_vector() -> None:
    class FakeBatch:
        failed_objects: list[object] = []

        def __init__(self) -> None:
            self.objects: list[dict] = []

        def dynamic(self) -> "FakeBatch":
            return self

        def __enter__(self) -> "FakeBatch":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def add_object(self, **kwargs: object) -> None:
            self.objects.append(dict(kwargs))

    class FakeCollection:
        def __init__(self) -> None:
            self.batch = FakeBatch()

    collection = FakeCollection()
    client = type(
        "FakeClient",
        (),
        {"collections": type("Collections", (), {"get": lambda self, _name: collection})()},
    )()
    chunk = SemanticChunk(
        id="chunk-1",
        document_id="DOC-123",
        doc_name="procedure.pdf",
        page=2,
        section_name="Preparation",
        heading="Preparation",
        section_path="Procedure > Preparation",
        text="Validate the production batch.",
        topic="Manufacturing",
        entities=["Batch A"],
        sensitivity="low",
        embedding_version="BAAI/bge-m3",
        embedding=np.array([0.1, 0.2], dtype=np.float32),
    )

    with (
        patch(
            "src.infrastructure.document_databases.weaviate_store.get_weaviate_client",
            return_value=client,
        ),
        patch("src.infrastructure.document_databases.weaviate_store.ensure_collection"),
        patch("src.infrastructure.document_databases.weaviate_store._ensure_collection_properties"),
        patch(
            "src.infrastructure.vector_store.shared_weaviate.chunk_index_sync.purge_document_chunks_from_collection"
        ),
    ):
        store_in_weaviate(
            [chunk],
            "http://weaviate:8080",
            "Chunks",
            document_metadata={
                "repository_id": "repo-1",
                "logical_folder_id": "folder-1",
                "logical_folder_path": "site/quality",
            },
        )

    properties = collection.batch.objects[0]["properties"]
    assert properties["chunk_id"] == "chunk-1"
    assert properties["document_id"] == "DOC-123"
    assert properties["heading"] == "Preparation"
    assert properties["topic"] == "Manufacturing"
    assert properties["entities"] == ["Batch A"]
    assert properties["sensitivity"] == "low"
    assert properties["embedding_version"] == "BAAI/bge-m3"
    assert properties["logical_folder_id"] == "folder-1"
    assert properties["logical_folder_path"] == "site/quality"
    assert np.allclose(collection.batch.objects[0]["vector"], [0.1, 0.2])
