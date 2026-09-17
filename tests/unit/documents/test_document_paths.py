from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.networking.document_paths import (
    build_processor_document_path,
    host_path_to_processor_container_path,
)


def test_host_path_to_processor_container_path_maps_repo_documents(tmp_path: Path) -> None:
    host_root = tmp_path / "_documents"
    host_root.mkdir()
    host_file = host_root / "GL-CQA-GOP-0030.pdf"
    host_file.write_bytes(b"%PDF-1.4")

    container_path = host_path_to_processor_container_path(
        host_file,
        host_root=host_root,
        container_root="/app/documents",
    )

    assert container_path == "/app/documents/GL-CQA-GOP-0030.pdf"


def test_build_processor_document_path_includes_debug_fields(tmp_path: Path) -> None:
    host_root = tmp_path / "_documents"
    host_root.mkdir()
    host_file = host_root / "sample.docx"
    host_file.write_bytes(b"docx")

    mapping = build_processor_document_path(
        host_file,
        host_root=host_root,
        container_root="/app/documents",
    )

    assert mapping["saved_host_path"] == str(host_file.resolve())
    assert mapping["container_document_path"] == "/app/documents/sample.docx"
    assert mapping["processor_documents_container_dir"] == "/app/documents"


def test_host_path_to_processor_container_path_rejects_missing_file(tmp_path: Path) -> None:
    host_root = tmp_path / "_documents"
    host_root.mkdir()
    missing = host_root / "missing.pdf"

    with pytest.raises(FileNotFoundError, match="Host file does not exist"):
        host_path_to_processor_container_path(missing, host_root=host_root, container_root="/app/documents")


def test_host_path_to_processor_container_path_rejects_outside_documents_dir(tmp_path: Path) -> None:
    host_root = tmp_path / "_documents"
    host_root.mkdir()
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4")

    with pytest.raises(ValueError, match="outside HOST_DOCUMENTS_DIR"):
        host_path_to_processor_container_path(outside, host_root=host_root, container_root="/app/documents")


def test_container_path_uses_posix_separators_for_nested_relative_paths(tmp_path: Path) -> None:
    host_root = tmp_path / "_documents"
    nested = host_root / "nested"
    nested.mkdir(parents=True)
    host_file = nested / "file.txt"
    host_file.write_text("hello", encoding="utf-8")

    container_path = host_path_to_processor_container_path(
        host_file,
        host_root=host_root,
        container_root="/app/documents",
    )

    assert container_path == "/app/documents/nested/file.txt"
