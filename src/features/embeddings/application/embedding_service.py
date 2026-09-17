from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Sequence

import torch

from src.features.document_processing.core.config import apply_huggingface_cache_env, get_settings
import numpy as np

from src.features.document_processing.core.logger import log
from src.features.chunking.application.chunking_service import Chunk
from src.features.document_processing.utilities.chunk_rewriter import rewrite_chunks_for_embedding
from src.features.document_processing.utilities.text_utils import normalize_text_for_embedding

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingValidationError(ValueError):
    """Raised when an embedding vector fails contract validation."""


def validate_embedding_vector(
    vector: Any,
    *,
    expected_dim: int | None = None,
    context: str = "embedding",
) -> list[float]:
    """Validate a dense embedding for persistence."""
    if vector is None:
        raise EmbeddingValidationError(f"{context}: embedding is None")
    arr = np.asarray(vector, dtype=np.float64)
    if arr.ndim == 0:
        raise EmbeddingValidationError(f"{context}: embedding must be a 1-D vector")
    if arr.ndim > 1:
        arr = np.squeeze(arr)
    if arr.ndim != 1:
        raise EmbeddingValidationError(
            f"{context}: embedding must be 1-D after squeeze, got shape {np.asarray(vector).shape}"
        )
    if arr.size == 0:
        raise EmbeddingValidationError(f"{context}: embedding is empty")
    if not np.isfinite(arr).all():
        raise EmbeddingValidationError(f"{context}: embedding contains NaN or Inf")
    dim = int(arr.shape[0])
    if expected_dim is not None and dim != int(expected_dim):
        raise EmbeddingValidationError(
            f"{context}: embedding dimension {dim} does not match expected {expected_dim}"
        )
    return [float(x) for x in arr.tolist()]


def validate_embedding_batch(
    vectors: Sequence[Any],
    *,
    expected_dim: int | None = None,
    context: str = "embedding_batch",
) -> list[list[float]]:
    """Validate a batch of embeddings; all rows must share the same dimension."""
    if not vectors:
        raise EmbeddingValidationError(f"{context}: no embeddings provided")
    validated: list[list[float]] = []
    batch_dim = expected_dim
    for index, vector in enumerate(vectors):
        item = validate_embedding_vector(
            vector,
            expected_dim=batch_dim,
            context=f"{context}[{index}]",
        )
        if batch_dim is None:
            batch_dim = len(item)
        validated.append(item)
    return validated


MODEL_ID_CACHE_ALIASES: dict[str, tuple[str, ...]] = {
    "bge-base-en": ("models--BAAI--bge-base-en",),
    "bge-small-en": ("models--BAAI--bge-small-en",),
    "bge-large-en": ("models--BAAI--bge-large-en",),
    "bge-m3": ("models--BAAI--bge-m3",),
    "all-MiniLM-L6-v2": ("models--sentence-transformers--all-MiniLM-L6-v2",),
    "all-mpnet-base-v2": ("models--sentence-transformers--all-mpnet-base-v2",),
    "e5-base-v2": ("models--intfloat--e5-base-v2",),
    "e5-large-v2": ("models--intfloat--e5-large-v2",),
}


class EmbeddingModelConfigurationError(RuntimeError):
    """Raised when the configured local embedding model cannot be resolved."""


def _get_device() -> str:
    """Return CUDA if available, otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _snapshot_has_sentence_transformer_files(snapshot_dir: Path) -> bool:
    return (
        (snapshot_dir / "config_sentence_transformers.json").is_file()
        or (snapshot_dir / "modules.json").is_file()
    )


def _pick_complete_local_snapshot(base: Path) -> Path | None:
    """Return a snapshot subfolder under a HF hub-style ``models--*`` dir if it looks loadable."""
    snapshots = base / "snapshots"
    if not snapshots.is_dir():
        return None
    for child in sorted(snapshots.iterdir()):
        if child.is_dir() and _snapshot_has_sentence_transformer_files(child):
            return child
    return None


def _hub_cache_dir_name(model_id: str) -> str | None:
    value = str(model_id or "").strip()
    if "/" not in value:
        return None
    owner, name = value.split("/", 1)
    owner = owner.strip()
    name = name.strip()
    if not owner or not name:
        return None
    return f"models--{owner}--{name}"


def _available_model_dirs(models_root: Path) -> list[str]:
    if not models_root.is_dir():
        return []
    return sorted(
        child.name
        for child in models_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    )


def _model_candidate_names(model_name: str, dir_name: str) -> list[str]:
    candidates: list[str] = []
    for raw in (dir_name, model_name, Path(str(model_name or "")).name):
        value = str(raw or "").strip()
        if value and value not in candidates:
            candidates.append(value)
        for alias in MODEL_ID_CACHE_ALIASES.get(value, ()):
            if alias not in candidates:
                candidates.append(alias)
        hub_dir = _hub_cache_dir_name(value)
        if hub_dir and hub_dir not in candidates:
            candidates.append(hub_dir)
    return candidates


def _resolve_local_model_path(models_root: Path, model_name: str, dir_name: str) -> Path:
    if not models_root.is_dir():
        raise EmbeddingModelConfigurationError(
            "Embedding models root is not mounted or does not exist: "
            f"'{models_root}'. Ensure src/models is mounted to /app/src/models."
        )

    checked: list[str] = []
    for candidate in _model_candidate_names(model_name, dir_name):
        base = (models_root / candidate).resolve()
        checked.append(str(base))
        if not base.is_dir():
            continue
        if _snapshot_has_sentence_transformer_files(base):
            return base
        snapshot = _pick_complete_local_snapshot(base)
        if snapshot is not None:
            return snapshot
        raise EmbeddingModelConfigurationError(
            f"Local embedding model path '{base}' exists but is not a complete "
            "sentence-transformers model (missing config_sentence_transformers.json or modules.json). "
            f"Files discovered: {', '.join(_diagnostic_model_files(base)) or '<none>'}."
        )

    available = ", ".join(_available_model_dirs(models_root)) or "<none>"
    requested = dir_name or model_name
    raise EmbeddingModelConfigurationError(
        f"Local embedding model '{requested}' was not found under '{models_root}'. "
        f"Checked: {', '.join(checked)}. Available model directories: {available}. "
        "Either install the requested model under src/models, update repository embedding_model.local_model_dir, "
        "or configure MODEL_DIR to an installed cache directory."
    )


def _diagnostic_model_files(path: Path, *, limit: int = 20) -> list[str]:
    if not path.exists():
        return []
    names: list[str] = []
    interesting = {
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
        "model.safetensors",
        "pytorch_model.bin",
    }
    for child in sorted(path.rglob("*")):
        if child.is_file() and child.name in interesting:
            names.append(str(child.relative_to(path)))
            if len(names) >= limit:
                break
    return names


def embedding_model_diagnostics(model_name: str | None = None, model_dir: str | None = None) -> dict[str, object]:
    settings = get_settings()
    resolved_model_name = str(model_name or settings.model_name or "").strip()
    resolved_dir_name = str(model_dir or settings.model_dir or "").strip()
    models_root = Path(settings.models_root).resolve()
    diagnostics: dict[str, object] = {
        "model_name": resolved_model_name,
        "model_dir": resolved_dir_name or None,
        "models_root": str(models_root),
        "models_root_exists": models_root.is_dir(),
        "available_model_dirs": _available_model_dirs(models_root),
    }
    try:
        resolved_path = _resolve_local_model_path(models_root, resolved_model_name, resolved_dir_name)
        diagnostics.update(
            {
                "resolved_path": str(resolved_path),
                "exists": resolved_path.is_dir(),
                "files": _diagnostic_model_files(resolved_path),
                "load_status": "not_initialized",
            }
        )
    except EmbeddingModelConfigurationError as exc:
        diagnostics.update({"error": str(exc), "load_status": "configuration_error"})
    return diagnostics


def log_embedding_startup_diagnostics() -> None:
    diagnostics = embedding_model_diagnostics()
    log.info(
        "PIPELINE_TRACE embedding_model_diagnostics "
        "model=%s model_dir=%s models_root=%s exists=%s resolved_path=%s files=%s load_status=%s error=%s",
        diagnostics.get("model_name"),
        diagnostics.get("model_dir"),
        diagnostics.get("models_root"),
        diagnostics.get("exists"),
        diagnostics.get("resolved_path"),
        diagnostics.get("files"),
        diagnostics.get("load_status"),
        diagnostics.get("error"),
    )


def _sentence_transformer_source(model_name: str, model_dir: str | None = None) -> str:
    """Resolve the embedding model source.

    When ``model_dir`` or ``MODEL_DIR`` is configured, prefer the local model under
    MODELS_ROOT. Common short names (for example ``bge-base-en``) are resolved to
    installed HuggingFace cache folder names before failing with a concise config error.
    """
    settings = get_settings()
    models_root = Path(settings.models_root).resolve()
    dir_name = (model_dir or settings.model_dir or "").strip()

    if dir_name:
        resolved = _resolve_local_model_path(models_root, model_name, dir_name)
        log.info("Loading embedding model from local path %s", resolved)
        return str(resolved)

    return model_name


def get_embedding_model(model_name: str, model_dir: str | None = None) -> "SentenceTransformer":
    """Load and cache a sentence-transformer model."""
    from src.features.document_processing.shared_processor.deployment import (
        ProcessorConfigurationError,
        ProcessorConfigurationProvider,
    )

    if not ProcessorConfigurationProvider.is_enabled("embedding"):
        raise ProcessorConfigurationError(
            "Embedding processor is disabled for this IKM deployment; model was not loaded."
        )
    apply_huggingface_cache_env()
    resolved = _sentence_transformer_source(model_name, model_dir=model_dir)
    log.info(
        "PIPELINE_TRACE embedding_model_load_attempt model=%s model_dir=%s resolved=%s",
        model_name,
        model_dir,
        resolved,
    )
    return _load_embedding_model(resolved)


@lru_cache(maxsize=3)
def _load_embedding_model(resolved_model_path_or_id: str) -> "SentenceTransformer":
    from sentence_transformers import SentenceTransformer

    device = _get_device()
    log.info("Loading embedding model '%s' on %s", resolved_model_path_or_id, device)

    try:
        return SentenceTransformer(
            resolved_model_path_or_id,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load embedding model '{resolved_model_path_or_id}': {e}"
        ) from e


def clear_embedding_model_cache() -> None:
    """Drop cached SentenceTransformer instances so disabled deployments free memory."""
    _load_embedding_model.cache_clear()
    log.info("Embedding model LRU cache cleared.")
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def encode_texts(
    texts: list[str],
    model_name: str,
    *,
    model_dir: str | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encode raw texts for semantic chunking helpers."""
    model = get_embedding_model(model_name, model_dir=model_dir)
    if not texts:
        return np.empty((0, 0))
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
    )


def embed_chunks(
    chunks: List[Chunk],
    model_name: str,
    batch_size: int = 64,
    rewrite_mode: str = "rule_based",
    rewrite_workers: int = 4,
    rewrite_max_tokens: int = 700,
    rewrite_timeout_seconds: int = 60,
    min_chunk_words: int = 60,
    model_dir: str | None = None,
    repository_embedding_model: dict[str, Any] | None = None,
) -> None:
    """Generate embeddings for chunk objects in-place."""
    if not chunks:
        return

    log.info("Preparing %d chunk(s) for embedding rewrite mode '%s'.", len(chunks), rewrite_mode)
    rewrite_chunks_for_embedding(
        chunks,
        mode=rewrite_mode,
        max_workers=rewrite_workers,
        max_tokens=rewrite_max_tokens,
        timeout_seconds=rewrite_timeout_seconds,
        min_chunk_words=min_chunk_words,
    )
    log.info("Chunk rewrite/preparation complete for %d chunk(s).", len(chunks))
    model = get_embedding_model(model_name, model_dir=model_dir)

    log.info("Normalizing %d chunk(s) before embedding.", len(chunks))
    texts = [normalize_text_for_embedding(c.text) for c in chunks]
    for chunk, normalized_text in zip(chunks, texts):
        chunk.normalized_text = normalized_text
        chunk.embedding_version = model_name

    valid_indices = [i for i, t in enumerate(texts) if t.strip()]
    skipped_count = len(chunks) - len(valid_indices)

    if not valid_indices:
        log.warning("No valid text to embed across %d chunk(s).", len(chunks))
        return

    if skipped_count:
        log.warning("%d chunk(s) skipped due to empty normalized text.", skipped_count)

    log.info(
        "Computing embeddings for %d/%d chunk(s) (batch_size=%d)",
        len(valid_indices),
        len(chunks),
        batch_size,
    )

    valid_texts = [texts[i] for i in valid_indices]

    try:
        embeddings = model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
    except Exception as e:
        raise RuntimeError(f"Embedding failed for model '{model_name}': {e}") from e

    matrix = np.asarray(embeddings)
    if matrix.ndim == 1 and len(valid_indices) == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] != len(valid_indices):
        raise RuntimeError(
            f"Embedding model '{model_name}' returned unexpected shape "
            f"{getattr(matrix, 'shape', None)}; expected ({len(valid_indices)}, dim)"
        )
    expected_dim = int(matrix.shape[1])
    if expected_dim <= 0:
        raise RuntimeError(f"Embedding model '{model_name}' produced zero-width vectors")

    if repository_embedding_model is not None:
        from src.features.repositories.infrastructure.collection_naming import (
            assert_embedding_dimension_consistency,
        )

        assert_embedding_dimension_consistency(
            repository_embedding_model,
            expected_dim,
            context=f"model={model_name}",
        )

    for row_index, chunk_index in enumerate(valid_indices):
        validated = validate_embedding_vector(
            matrix[row_index],
            expected_dim=expected_dim,
            context=f"chunk[{chunk_index}]",
        )
        chunks[chunk_index].embedding = np.asarray(validated, dtype=np.float32)

    log.info(
        "Embeddings computed: %d assigned, %d skipped, dim=%d model=%s",
        len(valid_indices),
        skipped_count,
        expected_dim,
        model_name,
    )
