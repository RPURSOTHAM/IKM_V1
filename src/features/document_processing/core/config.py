from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[4]
PROCESSOR_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROCESSOR_ROOT / ".env"
DEPENDENCIES_ENV_PATH = REPO_ROOT / "src" / "dependencies" / ".env"
if DEPENDENCIES_ENV_PATH.is_file():
    load_dotenv(DEPENDENCIES_ENV_PATH, override=False)
load_dotenv(ENV_PATH, override=True)


def resolve(key: str, default: Any = None) -> Any:
    """Resolve a configuration value from env or fallback defaults."""
    value = os.environ.get(key)
    if value is not None:
        return value
    return os.getenv(key, default)


def resolve_path(key: str, default: str | None = None) -> str | None:
    raw = resolve(key, default)
    if raw in (None, ""):
        return raw
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return str(path)
    candidate = (REPO_ROOT / path).resolve()
    return str(candidate)


def _optional_str(key: str) -> str | None:
    raw = resolve(key)
    if raw is None or str(raw).strip() == "":
        return None
    return str(raw).strip()


@dataclass
class Settings:
    processor_port: int = int(resolve("PROCESSOR_PORT", "8082"))
    weaviate_url: str = resolve("WEAVIATE_URL", "http://localhost:8080")
    weaviate_api_key: str | None = resolve("WEAVIATE_API_KEY")
    weaviate_grpc_port: int = int(resolve("WEAVIATE_GRPC_PORT", "50051"))
    collection_name: str = resolve("WEAVIATE_COLLECTION", "DocumentChunk")
    chunk_size: int = int(resolve("CHUNK_SIZE", "150"))
    chunk_overlap_sentences: int = int(resolve("CHUNK_OVERLAP_SENTENCES", "2"))
    min_content_words: int = int(resolve("MIN_CONTENT_WORDS", "60"))
    model_name: str = resolve("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    # Hub-style folder under models_root (e.g. models--BAAI--bge-base-en); loads from disk, not ~/.cache/huggingface.
    model_dir: str | None = _optional_str("MODEL_DIR")
    # Absolute root for MODEL_DIR (default: repo src/models). Override in containers if layout differs.
    models_root: str = resolve_path("MODELS_ROOT", "src/models") or str(REPO_ROOT / "src" / "models")
    max_workers: int = int(resolve("MAX_WORKERS", "4"))
    log_level: str = resolve("LOG_LEVEL", "INFO").upper()
    processor_root: str = resolve_path("PROCESSOR_ROOT", "src/processor_service")
    document_root: str = resolve_path("DOCUMENT_ROOT", "_documents")


def get_settings(overrides: Optional[Dict[str, Any]] = None) -> Settings:
    overrides = overrides or {}
    valid_keys = {f.name for f in fields(Settings)}
    filtered = {k: v for k, v in overrides.items() if k in valid_keys}
    return Settings(**filtered)


def apply_huggingface_cache_env() -> None:
    """Pin HuggingFace cache paths to a writable directory.

    Pre-mounted local models are still loaded from ``MODELS_ROOT`` via
    ``_sentence_transformer_source``; downloads must not target a read-only mount.
    """
    existing_hf_home = (os.getenv("HF_HOME") or "").strip()
    if existing_hf_home and (existing_hf_home.startswith("/tmp") or os.access(existing_hf_home, os.W_OK)):
        cache_root = str(Path(existing_hf_home).resolve())
    else:
        models_root_str = resolve_path("MODELS_ROOT", "src/models") or str(REPO_ROOT / "src" / "models")
        models_root = Path(models_root_str).resolve()
        if os.access(models_root, os.W_OK):
            cache_root = str(models_root)
        else:
            cache_root = str(Path(os.getenv("PROCESSOR_HF_CACHE_DIR", "/tmp/huggingface")).resolve())
            Path(cache_root).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = cache_root
    os.environ["TRANSFORMERS_CACHE"] = cache_root
    os.environ["HF_HUB_CACHE"] = cache_root
    os.environ["HUGGINGFACE_HUB_CACHE"] = cache_root
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_root


apply_huggingface_cache_env()
