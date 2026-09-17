"""Map host document upload paths to processor container paths."""

from __future__ import annotations

import os
from pathlib import Path

from src.shared.networking.paths import documents_dir


def host_documents_dir(*, repo_root: Path | None = None) -> Path:
    """Directory on the host where uploaded documents are stored."""
    return resolve_security_documents_dir(repo_root=repo_root)


def resolve_security_documents_dir(*, repo_root: Path | None = None) -> Path:
    """Shared queue/audit storage — works on host Streamlit and in processor containers."""
    return documents_dir(repo_root=repo_root)


def processor_documents_container_dir() -> str:
    """Directory the processor should use to read uploaded documents."""
    configured = (
        os.getenv("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER")
        or os.getenv("SCHEDULER_DOCUMENT_CONTAINER_PATH")
        or "/app/documents"
    ).strip()
    if (os.getenv("DEPLOYMENT_MODE") or "").strip().lower() == "local":
        return str(Path(configured).expanduser().resolve()).rstrip("\\/")
    return configured.replace("\\", "/").rstrip("/")


def host_path_to_processor_container_path(
    host_path: Path | str,
    *,
    host_root: Path | str | None = None,
    container_root: str | None = None,
) -> str:
    """Convert a host upload path to the path the processor container can read."""
    resolved_host = Path(host_path).expanduser().resolve()
    host_documents = Path(host_root or host_documents_dir()).expanduser().resolve()
    container_documents = (container_root or processor_documents_container_dir()).rstrip("/")

    if not resolved_host.is_file():
        raise FileNotFoundError(f"Host file does not exist: {resolved_host}")

    try:
        relative = resolved_host.relative_to(host_documents)
    except ValueError as exc:
        raise ValueError(
            f"Host file {resolved_host} is outside HOST_DOCUMENTS_DIR ({host_documents}). "
            "Save uploads under the shared documents directory mounted into the processor."
        ) from exc

    if (os.getenv("DEPLOYMENT_MODE") or "").strip().lower() == "local":
        return str(Path(container_documents) / relative)
    return f"{container_documents}/{relative.as_posix()}"


def build_processor_document_path(
    host_path: Path | str,
    *,
    host_root: Path | str | None = None,
    container_root: str | None = None,
) -> dict[str, str]:
    """Validate a host upload and return debug-friendly path mapping."""
    resolved_host = Path(host_path).expanduser().resolve()
    container_path = host_path_to_processor_container_path(
        resolved_host,
        host_root=host_root,
        container_root=container_root,
    )
    return {
        "saved_host_path": str(resolved_host),
        "container_document_path": container_path,
        "host_documents_dir": str(Path(host_root or host_documents_dir()).expanduser().resolve()),
        "processor_documents_container_dir": (container_root or processor_documents_container_dir()).rstrip("/"),
    }
