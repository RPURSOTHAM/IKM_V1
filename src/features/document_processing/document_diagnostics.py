"""Processor-side document path resolution and mount diagnostics."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.shared.networking.bind_paths import is_windows_host_path
from src.shared.networking.document_paths import processor_documents_container_dir


def document_root() -> Path:
    return Path(
        os.getenv("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER")
        or os.getenv("DOCUMENT_ROOT")
        or processor_documents_container_dir()
    )


def _list_directory(path: Path, *, limit: int = 200) -> list[str]:
    if not path.exists():
        return []
    try:
        entries = sorted(path.iterdir(), key=lambda item: item.name.lower())
    except OSError:
        return []
    return [entry.name + ("/" if entry.is_dir() else "") for entry in entries[:limit]]


def directory_listing(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "entries": _list_directory(path) if path.is_dir() else [],
    }


def startup_document_diagnostics() -> dict[str, Any]:
    root = document_root()
    app_dir = Path("/app")
    return {
        "cwd": os.getcwd(),
        "DOCUMENT_ROOT": os.getenv("DOCUMENT_ROOT"),
        "PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER": os.getenv(
            "PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", processor_documents_container_dir()
        ),
        "app": directory_listing(app_dir),
        "document_root": directory_listing(root),
        "document_root_exists": root.exists(),
        "document_root_is_dir": root.is_dir(),
    }


def format_startup_document_diagnostics() -> str:
    info = startup_document_diagnostics()
    lines = [
        "Processor document mount diagnostics:",
        f"  Current working directory: {info['cwd']}",
        f"  DOCUMENT_ROOT: {info['DOCUMENT_ROOT']}",
        f"  PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER: {info['PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER']}",
        f"  /app exists: {info['app']['exists']}",
        f"  /app entries: {info['app']['entries']}",
        f"  document root path: {info['document_root']['path']}",
        f"  document root exists: {info['document_root_exists']}",
        f"  document root entries: {info['document_root']['entries']}",
    ]
    return "\n".join(lines)


def reject_windows_host_path(path_text: str) -> None:
    normalized = path_text.strip()
    if is_windows_host_path(normalized) or normalized.startswith(("C:\\", "C:/", "D:\\", "D:/")):
        raise ValueError(
            "Docker processors cannot read Windows host paths. "
            f"Received document_path={path_text!r}. "
            "Streamlit must send a container path such as /app/documents/<filename>."
        )


def resolve_processor_document_path(
    *,
    document_path: str | None,
    document_name: str | None,
) -> tuple[Path, dict[str, Any]]:
    """Resolve and validate the document path inside the processor container."""
    diagnostics: dict[str, Any] = {
        "received_document_path": document_path,
        "document_name": document_name,
        "DOCUMENT_ROOT": os.getenv("DOCUMENT_ROOT"),
        "PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER": os.getenv(
            "PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", processor_documents_container_dir()
        ),
    }

    root = document_root()
    candidates: list[Path] = []

    if document_path:
        reject_windows_host_path(document_path)
        received = Path(document_path)
        if received.is_absolute():
            candidates.append(received)
        else:
            # Queue payloads often send paths relative to DOCUMENT_ROOT
            # (e.g. external_evidence/{repo_id}/file.pdf).
            candidates.append(root / received)
            candidates.append(received)

    if document_name:
        candidates.append(root / document_name)
        if document_path:
            candidates.append(root / Path(document_path).name)

    # Accept legacy mount locations when the configured root differs.
    for legacy_root in (Path("/app/documents"), Path("/app/_documents")):
        if document_name and legacy_root != root:
            candidates.append(legacy_root / document_name)
        if document_path and legacy_root != root:
            candidates.append(legacy_root / Path(document_path).name)

    seen: set[str] = set()
    unique_candidates: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    diagnostics["candidate_paths"] = [str(path) for path in unique_candidates]

    chosen: Path | None = None
    candidate_checks: list[dict[str, Any]] = []
    for candidate in unique_candidates:
        resolved = candidate.expanduser()
        try:
            resolved = resolved.resolve(strict=False)
        except OSError:
            resolved = resolved.absolute()
        exists = resolved.exists()
        is_file = resolved.is_file()
        candidate_checks.append(
            {
                "path": str(resolved),
                "exists": exists,
                "is_file": is_file,
            }
        )
        if exists and is_file:
            chosen = resolved
            break

    diagnostics["candidate_checks"] = candidate_checks
    diagnostics["absolute_resolved_path"] = str(chosen) if chosen else None
    diagnostics["exists"] = bool(chosen and chosen.exists())
    diagnostics["is_file"] = bool(chosen and chosen.is_file())

    if chosen is None:
        diagnostics["app_listing"] = directory_listing(Path("/app"))
        diagnostics["document_root_listing"] = directory_listing(root)
        for legacy_root in (Path("/app/documents"), Path("/app/_documents")):
            if legacy_root != root:
                diagnostics[f"{legacy_root.name}_listing"] = directory_listing(legacy_root)
        raise FileNotFoundError(
            "Document file not found inside the processor container. "
            f"received_document_path={document_path!r}, document_name={document_name!r}, "
            f"DOCUMENT_ROOT={root}. "
            "Verify the host _documents folder is mounted into the processor at the configured container path."
        )

    return chosen, diagnostics


def debug_documents_payload() -> dict[str, Any]:
    root = document_root()
    payload = startup_document_diagnostics()
    payload["files"] = _list_directory(root)
    payload["document_root"] = str(root)
    payload["exists"] = root.exists()
    return payload
