"""Resolve repository root for Docker bind mounts."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.shared.networking.bind_paths import (
    host_repo_root,
    is_windows_host_path,
    normalize_bind_mount_host_path,
    running_inside_container,
)

logger = logging.getLogger(__name__)


def _has_documents_dir(path: Path) -> bool:
    return (path / "data" / "documents").is_dir() or (path / "_documents").is_dir()


def _documents_dir_score(path: Path) -> int:
    if not path.is_dir():
        return -1
    try:
        return sum(
            1
            for entry in path.iterdir()
            if entry.is_file() and entry.suffix.lower() in {".pdf", ".docx", ".txt"}
        )
    except OSError:
        return 0


def _documents_subdir(repo: Path) -> Path:
    preferred = repo / "data" / "documents"
    if preferred.is_dir():
        return preferred
    legacy = repo / "_documents"
    if legacy.is_dir():
        return legacy
    return preferred


def resolve_repository_root(fallback: Path) -> Path:
    """Pick a repo root whose documents directory exists on the host."""
    candidates: list[Path] = []
    for raw in (host_repo_root(), os.getenv("REPO_ROOT"), os.getenv("HOST_REPO_ROOT")):
        if not raw:
            continue
        candidates.append(Path(str(raw).replace("\\", "/")))

    candidates.append(fallback)

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate.expanduser())
        if normalized in seen:
            continue
        seen.add(normalized)

        if _has_documents_dir(candidate):
            return candidate.resolve()

        parent = candidate.parent
        if _has_documents_dir(parent):
            logger.warning(
                "Configured repo root %s has no documents directory; using parent %s instead.",
                candidate,
                parent,
            )
            return parent.resolve()

    return fallback.resolve()


def resolve_document_host_path(fallback_documents: Path) -> str:
    if running_inside_container():
        configured = (os.getenv("SCHEDULER_DOCUMENT_HOST_PATH") or "").strip()
        if not configured:
            root = (host_repo_root() or os.getenv("REPO_ROOT") or "").strip()
            if root:
                configured = f"{root.rstrip('/')}/data/documents"
        if configured:
            normalized = normalize_bind_mount_host_path(configured, repo_root=host_repo_root())
            if normalized and (is_windows_host_path(normalized) or not normalized.startswith("/app")):
                logger.info("Using document host bind mount source: %s", normalized)
                return normalized
        raise RuntimeError(
            "Scheduler is running inside Docker but no valid host documents path was configured. "
            "Set SCHEDULER_DOCUMENT_HOST_PATH or HOST_REPO_ROOT to the host path containing ./data/documents."
        )

    candidates: list[Path] = []
    configured = (os.getenv("SCHEDULER_DOCUMENT_HOST_PATH") or "").strip()
    if configured:
        candidates.append(Path(configured.replace("\\", "/")))

    repo_root = resolve_repository_root(fallback_documents.parent)
    candidates.append(_documents_subdir(repo_root))
    candidates.append(fallback_documents)

    seen: set[str] = set()
    ranked: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        ranked.append(candidate)

    best = max(ranked, key=lambda path: (_documents_dir_score(path), path.exists()))
    if _documents_dir_score(best) < 0:
        best = fallback_documents

    normalized = normalize_bind_mount_host_path(str(best), repo_root=host_repo_root())
    if normalized:
        logger.info("Using document host bind mount source: %s", normalized)
        return normalized
    return str(best.resolve())
