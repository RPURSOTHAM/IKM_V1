"""Reranker model path resolution for offline-first CrossEncoder loading."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from src.features.configuration.platform_settings import REPO_ROOT
from src.features.retrieval.reranking.config import RerankerConfig

_logger = logging.getLogger(__name__)

_WEIGHT_NAMES = (
    "model.safetensors",
    "pytorch_model.bin",
    "pytorch_model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _models_root() -> Path:
    raw = (os.getenv("MODELS_ROOT") or "").strip()
    if raw:
        return Path(raw).expanduser()
    return REPO_ROOT / "src" / "models"


def _is_offline() -> bool:
    return any(
        (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}
        for name in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE")
    )


def _looks_like_path(value: str) -> bool:
    if not value:
        return False
    if value.startswith("/") or value.startswith(".") or ":\\" in value or ":/" in value:
        return True
    if "/" in value and not value.startswith(("http://", "https://")):
        # Hub ids look like "org/name"; local dirs may also contain "/".
        # Treat as path when an absolute/relative filesystem hit exists,
        # or when value starts with known local prefixes.
        lowered = value.replace("\\", "/").lower()
        return (
            lowered.startswith("app/")
            or lowered.startswith("src/")
            or lowered.startswith("models/")
            or lowered.startswith("models--")
            or Path(value).exists()
        )
    if value.startswith("models--") or value.endswith(("-base", "-large", "-v2-m3")):
        # Prefer checking filesystem for short local folder names like bge-reranker-base.
        return True
    return Path(value).exists()


def _expand_candidate(raw: str) -> Path:
    candidate = Path(str(raw).strip()).expanduser()
    if candidate.is_absolute():
        return candidate
    models_root = _models_root()
    under_models = models_root / candidate
    if under_models.exists():
        return under_models
    under_repo = REPO_ROOT / candidate
    if under_repo.exists():
        return under_repo
    return models_root / candidate


def has_required_model_files(path: Path) -> bool:
    """Return True when a directory looks like a loadable Transformers/SentenceTransformer model."""
    if not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = any((path / name).exists() for name in _WEIGHT_NAMES)
    # sentence-transformers CrossEncoder packages may also ship modules.json
    has_st_marker = (path / "modules.json").exists() or (path / "sentence_bert_config.json").exists()
    return has_config and (has_weights or has_st_marker)


def resolve_snapshot_dir(candidate: Path) -> Path | None:
    """Prefer HF hub cache snapshot directories when present."""
    if not candidate.exists():
        return None
    snapshots = candidate / "snapshots"
    if snapshots.is_dir():
        snapshot_dirs = sorted(
            (path for path in snapshots.iterdir() if path.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in snapshot_dirs:
            if has_required_model_files(snap):
                return snap
    if has_required_model_files(candidate):
        return candidate
    return None


def _hub_cache_dir_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def _collect_local_candidates(
    *,
    configured_model: str,
    model_dir: str | None,
) -> list[Path]:
    candidates: list[Path] = []
    seen: set[str] = set()

    def _add(raw: str | Path | None) -> None:
        if raw is None:
            return
        text = str(raw).strip()
        if not text:
            return
        path = _expand_candidate(text) if not isinstance(raw, Path) else raw
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            return
        seen.add(key)
        candidates.append(path)

    if model_dir:
        _add(model_dir)
    if configured_model and _looks_like_path(configured_model):
        _add(configured_model)

    models_root = _models_root()
    # Conventional offline layouts
    for name in (
        "bge-reranker-base",
        "BAAI--bge-reranker-base",
        "models--BAAI--bge-reranker-base",
        _hub_cache_dir_name(configured_model) if "/" in configured_model else None,
        configured_model.replace("/", "--") if "/" in configured_model else configured_model,
    ):
        if name:
            _add(models_root / name)

    return candidates


def resolve_reranker_model_path(
    repo_settings: dict[str, Any] | None = None,
    *,
    config: RerankerConfig | None = None,
) -> str:
    """
    Resolve a local CrossEncoder directory suitable for offline loading.

    Never downloads. When offline and no valid local model exists, returns "" so the
    caller can fall back to RRF without attempting a HuggingFace hub fetch.
    """
    settings = dict(repo_settings or {})
    cfg = config or RerankerConfig.from_env()
    model_dir = (
        settings.get("reranker_model_dir")
        or settings.get("reranking_model_dir")
        or os.getenv("RERANKER_MODEL_DIR")
    )
    configured_model = str(
        settings.get("reranker_model")
        or settings.get("reranking_model")
        or os.getenv("RERANKER_MODEL")
        or os.getenv("RETRIEVAL_RERANKER_MODEL")
        or cfg.model
        or "BAAI/bge-reranker-base"
    ).strip()

    for candidate in _collect_local_candidates(
        configured_model=configured_model,
        model_dir=str(model_dir) if model_dir else None,
    ):
        resolved = resolve_snapshot_dir(candidate)
        if resolved is not None:
            _logger.info(
                "reranker_model_resolved path=%s offline=%s source=%s",
                resolved,
                _is_offline(),
                candidate,
            )
            return str(resolved)

    if _is_offline() or (configured_model and _looks_like_path(configured_model) and not Path(configured_model).exists()):
        _logger.warning(
            "reranker_model_unavailable offline=%s configured=%s models_root=%s "
            "reason=no_valid_local_model_files",
            _is_offline(),
            configured_model,
            _models_root(),
        )
        return ""

    # Online fallback: allow hub id (not used in Docker offline mode).
    _logger.info("reranker_model_resolved hub_id=%s offline=false", configured_model)
    return configured_model
