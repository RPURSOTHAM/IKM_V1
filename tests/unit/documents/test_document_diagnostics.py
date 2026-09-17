from __future__ import annotations

from pathlib import Path

import pytest

from src.features.document_processing.document_diagnostics import (
    reject_windows_host_path,
    resolve_processor_document_path,
)
from src.shared.networking.repo_root import resolve_document_host_path, resolve_repository_root


def test_reject_windows_host_path() -> None:
    with pytest.raises(ValueError, match="cannot read Windows host paths"):
        reject_windows_host_path(r"C:\Users\DELL\Downloads\rag-builder\_documents\file.pdf")


def test_resolve_processor_document_path_finds_file_under_document_root(tmp_path: Path, monkeypatch) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    file_path = documents / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("DOCUMENT_ROOT", str(documents))
    monkeypatch.setenv("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", str(documents))

    resolved, diagnostics = resolve_processor_document_path(
        document_path="/app/documents/sample.pdf",
        document_name="sample.pdf",
    )

    assert resolved == file_path.resolve()
    assert diagnostics["is_file"] is True


def test_resolve_processor_document_path_falls_back_to_document_root_name(tmp_path: Path, monkeypatch) -> None:
    documents = tmp_path / "documents"
    documents.mkdir()
    file_path = documents / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setenv("DOCUMENT_ROOT", str(documents))
    monkeypatch.setenv("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", str(documents))

    resolved, diagnostics = resolve_processor_document_path(
        document_path="/app/documents/sample.pdf",
        document_name="sample.pdf",
    )

    assert resolved == file_path.resolve()
    assert diagnostics["is_file"] is True


def test_resolve_document_host_path_uses_scheduler_env_when_running_in_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCHEDULER_DOCUMENT_HOST_PATH", "C:/Users/DELL/Downloads/rag-builder/_documents")
    monkeypatch.setenv("HOST_REPO_ROOT", "C:/Users/DELL/Downloads/rag-builder")
    monkeypatch.setattr("src.shared.networking.repo_root.running_inside_container", lambda: True)

    resolved = resolve_document_host_path(Path("/app/_documents"))

    assert resolved == "C:/Users/DELL/Downloads/rag-builder/_documents"


def test_resolve_repository_root_uses_parent_when_nested_repo_is_wrong(tmp_path: Path) -> None:
    nested = tmp_path / "rag-builder" / "rag-builder"
    nested.mkdir(parents=True)
    real_docs = tmp_path / "rag-builder" / "_documents"
    real_docs.mkdir()

    resolved = resolve_repository_root(nested)

    assert resolved == (tmp_path / "rag-builder").resolve()
