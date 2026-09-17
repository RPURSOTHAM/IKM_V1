"""Central path helpers for production layout.

Preferred locations (production):
  data/documents   — uploaded originals (DOCUMENT_ROOT / HOST_DOCUMENTS_DIR)
  data/templates   — document templates
  data/reports     — generated reports
  models/          — embedding / reranker weights
  deploy/          — compose + ops scripts

Legacy aliases (junctions) still resolve for older env files:
  _documents, _templates, _reports, src/models
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root_from_here() -> Path:
    """``src/shared/networking/paths.py`` → repo root."""
    return Path(__file__).resolve().parents[3]


def documents_dir(*, repo_root: Path | None = None) -> Path:
    """Resolve the host documents directory with production-first fallbacks."""
    configured_host = (os.getenv("HOST_DOCUMENTS_DIR") or "").strip()
    if configured_host:
        return Path(configured_host).expanduser().resolve()

    document_root = (os.getenv("DOCUMENT_ROOT") or "").strip()
    if document_root:
        root_path = Path(document_root).expanduser()
        normalized = document_root.replace("\\", "/")
        if not (normalized.startswith("/app") and not root_path.exists()):
            return root_path.resolve()

    root = (repo_root or repo_root_from_here()).resolve()
    for candidate in (
        root / "data" / "documents",
        root / "_documents",  # legacy junction / alias
    ):
        if candidate.is_dir():
            return candidate.resolve()
    # Prefer production path even when creating fresh.
    preferred = root / "data" / "documents"
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred.resolve()


def templates_dir(*, repo_root: Path | None = None) -> Path:
    root = (repo_root or repo_root_from_here()).resolve()
    for candidate in (root / "data" / "templates", root / "_templates"):
        if candidate.is_dir():
            return candidate.resolve()
    preferred = root / "data" / "templates"
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred.resolve()


def reports_dir(*, repo_root: Path | None = None) -> Path:
    root = (repo_root or repo_root_from_here()).resolve()
    for candidate in (root / "data" / "reports", root / "_reports"):
        if candidate.is_dir():
            return candidate.resolve()
    preferred = root / "data" / "reports"
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred.resolve()


def models_dir(*, repo_root: Path | None = None) -> Path:
    configured = (os.getenv("MODELS_ROOT") or os.getenv("SCHEDULER_MODELS_HOST_PATH") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    root = (repo_root or repo_root_from_here()).resolve()
    for candidate in (root / "models", root / "src" / "models"):
        if candidate.is_dir():
            return candidate.resolve()
    preferred = root / "models"
    preferred.mkdir(parents=True, exist_ok=True)
    return preferred.resolve()


def deploy_infrastructure_dir(*, repo_root: Path | None = None) -> Path:
    root = (repo_root or repo_root_from_here()).resolve()
    for candidate in (root / "deploy" / "infrastructure", root / "src" / "dependencies"):
        if candidate.is_dir():
            return candidate.resolve()
    return (root / "deploy" / "infrastructure").resolve()


def deploy_application_dir(*, repo_root: Path | None = None) -> Path:
    root = (repo_root or repo_root_from_here()).resolve()
    for candidate in (root / "deploy" / "application", root / "src" / "deployment"):
        if candidate.is_dir():
            return candidate.resolve()
    return (root / "deploy" / "application").resolve()
