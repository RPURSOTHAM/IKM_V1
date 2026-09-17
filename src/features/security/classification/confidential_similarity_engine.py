"""Real embedding-based confidential similarity DLP engine.

Uses the platform embedding model with an in-process FAISS/numpy index.
Optional Weaviate backend when SECURITY_SIMILARITY_BACKEND=weaviate.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None


def _store_dir() -> Path:
    configured = (os.getenv("CONFIDENTIAL_CORPUS_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    docs = (os.getenv("HOST_DOCUMENTS_DIR") or os.getenv("DOCUMENT_ROOT") or "").strip()
    if docs and not docs.replace("\\", "/").startswith("/app"):
        base = Path(docs).expanduser().resolve()
    else:
        base = Path.cwd().resolve() / "_documents"
    path = base / "confidential_corpus"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _embed_texts(texts: list[str]) -> np.ndarray:
    """Reuse production embedding stack; fall back to hashing embedder for tests."""
    if not texts:
        return np.zeros((0, 8), dtype=np.float32)
    try:
        from src.features.document_processing.core.config import get_settings
        from src.features.embeddings.application.embedding_service import encode_texts

        settings = get_settings()
        model_name = (
            os.getenv("EMBEDDING_MODEL")
            or getattr(settings, "embedding_model", None)
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        model_dir = os.getenv("MODEL_DIR") or getattr(settings, "model_dir", None)
        vectors = encode_texts(texts, model_name, model_dir=model_dir, batch_size=16)
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # L2 normalize
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms
    except Exception as exc:
        logger.warning("Falling back to hashing embedder for similarity DLP: %s", exc)
        dim = 64
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in text.lower().split():
                out[i, hash(token) % dim] += 1.0
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


def evaluate_similarity_risk(similarity: float) -> dict[str, Any]:
    """Map cosine similarity to DLP decision.

    Thresholds come from dlp_policies.yaml ``similarity``.
    Near-duplicates route to human_review unless ``enforcement.auto_block_enabled``.
    """
    try:
        from src.features.security.dlp.policy_loader import is_auto_block_enabled, load_dlp_policies

        policies = load_dlp_policies() if isinstance(load_dlp_policies(), dict) else {}
        cfg = (policies.get("similarity") or {}) if isinstance(policies, dict) else {}
        auto_block = is_auto_block_enabled()
    except Exception:
        cfg = {}
        auto_block = False
    block_above = float(cfg.get("block_above", 0.95))
    human_min = float(cfg.get("human_review_min", 0.85))
    warn_min = float(cfg.get("warning_min", 0.75))
    score = float(similarity)

    if score > block_above:
        if auto_block:
            return {"status": "block", "severity": "critical", "similarity": score}
        return {"status": "human_review", "severity": "critical", "similarity": score}
    if score >= human_min:
        return {"status": "human_review", "severity": "high", "similarity": score}
    if score >= warn_min:
        return {"status": "warning", "severity": "medium", "similarity": score}
    return {"status": "allow", "severity": "low", "similarity": score}


class ConfidentialSimilarityEngine:
    """Persistent confidential corpus + cosine similarity search."""

    def __init__(self, store_dir: Path | None = None) -> None:
        self.store_dir = store_dir or _store_dir()
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.store_dir / "corpus_meta.jsonl"
        self.vectors_path = self.store_dir / "corpus_vectors.npy"
        self._lock = threading.RLock()
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._meta: list[dict[str, Any]] = []
        self._vectors: np.ndarray | None = None
        self._index = None
        self._load()

    def _load(self) -> None:
        with self._lock:
            self._ids, self._texts, self._meta = [], [], []
            if self.meta_path.is_file():
                for line in self.meta_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._ids.append(str(row.get("id")))
                    self._texts.append(str(row.get("text") or ""))
                    self._meta.append(row)
            if self.vectors_path.is_file() and self._ids:
                try:
                    self._vectors = np.load(self.vectors_path).astype(np.float32)
                except Exception:
                    self._vectors = None
            if self._vectors is None and self._texts:
                self._vectors = _embed_texts(self._texts)
                self._persist_vectors()
            self._rebuild_index()

    def _persist_vectors(self) -> None:
        if self._vectors is None:
            return
        np.save(self.vectors_path, self._vectors)

    def _rebuild_index(self) -> None:
        self._index = None
        if self._vectors is None or len(self._vectors) == 0:
            return
        if faiss is not None:
            dim = int(self._vectors.shape[1])
            index = faiss.IndexFlatIP(dim)
            index.add(self._vectors.astype(np.float32))
            self._index = index

    def add_sensitive_document(
        self,
        text: str,
        *,
        document_id: str | None = None,
        document_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Index a confidential reference document."""
        cleaned = (text or "").strip()
        if not cleaned:
            raise ValueError("Cannot index empty confidential document text.")
        doc_id = document_id or str(uuid.uuid4())
        vector = _embed_texts([cleaned])[0]
        row = {
            "id": doc_id,
            "document_name": document_name or doc_id,
            "text": cleaned[:8000],
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        with self._lock:
            self._ids.append(doc_id)
            self._texts.append(cleaned)
            self._meta.append(row)
            if self._vectors is None or self._vectors.size == 0:
                self._vectors = vector.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])
            with open(self.meta_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=True) + "\n")
            self._persist_vectors()
            self._rebuild_index()
        return doc_id

    def search_similar_content(
        self,
        text: str,
        *,
        top_k: int = 5,
        exclude_ids: set[str] | frozenset[str] | None = None,
        exclude_document_names: set[str] | frozenset[str] | None = None,
    ) -> dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            return {"matches": [], "max_similarity": 0.0}
        excluded = {str(x) for x in (exclude_ids or set()) if x}
        excluded_names = {str(x).lower() for x in (exclude_document_names or set()) if x}
        with self._lock:
            if self._vectors is None or len(self._ids) == 0:
                return {"matches": [], "max_similarity": 0.0}
            query = _embed_texts([cleaned])[0].astype(np.float32)
            # Fetch extra neighbors so exclusions do not empty the result set.
            k = min(max(1, top_k + len(excluded) + 3), len(self._ids))
            matches: list[dict[str, Any]] = []

            def _accept(idx: int, score: float) -> None:
                if idx < 0 or len(matches) >= top_k:
                    return
                ref_id = str(self._ids[idx])
                doc_name = str(self._meta[idx].get("document_name") or "")
                if ref_id in excluded or doc_name.lower() in excluded_names:
                    return
                # Also skip seeded-{id} when bare id is excluded (re-upload self-match).
                if ref_id.startswith("seeded-") and ref_id[len("seeded-") :] in excluded:
                    return
                matches.append(
                    {
                        "reference_id": ref_id,
                        "document_name": doc_name,
                        "similarity": float(score),
                        "severity": evaluate_similarity_risk(float(score))["severity"],
                        "reason": "Embedding similarity to confidential corpus.",
                    }
                )

            if self._index is not None:
                scores, idxs = self._index.search(query.reshape(1, -1), k)
                for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
                    _accept(idx, float(score))
            else:
                sims = self._vectors @ query
                top_idx = np.argsort(-sims)[:k]
                for idx in top_idx.tolist():
                    _accept(idx, float(sims[idx]))
            max_sim = max((m["similarity"] for m in matches), default=0.0)
            return {"matches": matches, "max_similarity": max_sim}

    def check_similarity(self, text: str) -> dict[str, Any]:
        """Backward-compatible API used by UploadSecurityPipeline."""
        return self.search_similar_content(text)


_engine: ConfidentialSimilarityEngine | None = None
_engine_lock = threading.Lock()


def get_similarity_engine() -> ConfidentialSimilarityEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = ConfidentialSimilarityEngine()
        return _engine


def add_sensitive_document(text: str, **kwargs: Any) -> str:
    return get_similarity_engine().add_sensitive_document(text, **kwargs)


def search_similar_content(text: str, **kwargs: Any) -> dict[str, Any]:
    return get_similarity_engine().search_similar_content(text, **kwargs)
