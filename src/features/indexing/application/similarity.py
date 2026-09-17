"""
similarity.py
=============
Vector similarity analysis for document chunks.

Provides three main functions:
1. Duplicate detection (exact/near-exact matches)
2. Match percentage computation (best cross-document similarity)
3. Cross-document clustering (Union-Find grouping based on similarity)
"""

from __future__ import annotations

import hashlib
import numpy as np

from src.features.document_processing.core.logger import log
from src.features.chunking.application.chunking_service import Chunk
from src.features.document_processing.utilities.text_utils import normalize_text_for_embedding


def _normalized_text(chunk: Chunk) -> str:
    return chunk.normalized_text or normalize_text_for_embedding(chunk.text)


def _normalized_word_count(chunk: Chunk) -> int:
    return len(_normalized_text(chunk).split())


# ─────────────────────────────────────────────────────────────────────────────
# UNION-FIND FOR CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────


class UnionFind:
    """Standard disjoint-set data structure for clustering chunks."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> None:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


# ─────────────────────────────────────────────────────────────────────────────
# 1. DUPLICATE DETECTION
# ─────────────────────────────────────────────────────────────────────────────


def detect_duplicates(
    chunks: list[Chunk],
    min_similarity: float = 0.97,
    min_content_words: int = 10,
) -> None:
    """
    Mark chunks as duplicates if they have highly similar normalized text
    and embeddings, OR if they are too short to be meaningful.

    Uses a two-pass approach:
      Pass 1 — Exact duplicates: identical normalized text hash → O(n)
      Pass 2 — Near-duplicates: embedding similarity across surviving chunks
    """
    log.info("Detecting duplicates (sim >= %.2f) ...", min_similarity)

    exact_count = 0
    near_dup_count = 0

    # ✅ FIX 1 (Critical): The original code grouped by hash THEN checked
    # embeddings within each hash group. This is completely wrong:
    #   - Identical text → identical hash → identical embeddings,
    #     so the embedding check inside the hash group was always redundant.
    #   - Near-duplicates with slightly different text land in DIFFERENT hash
    #     groups and were NEVER compared by embedding similarity.
    # Correct approach: hash for exact duplicates (fast path), then embedding
    # similarity across ALL surviving non-duplicate chunks.

    # Pass 1: Exact duplicate detection via normalized-text hash
    hash_seen: dict[str, int] = {}  # hash → index of first seen chunk
    for i, c in enumerate(chunks):
        norm_text = _normalized_text(c)
        if len(norm_text.split()) < min_content_words:
            c.is_duplicate = True
            continue

        if not norm_text:
            c.is_duplicate = True
            continue

        h = hashlib.md5(norm_text.encode("utf-8")).hexdigest()
        if h in hash_seen:
            c.is_duplicate = True
            exact_count += 1
        else:
            hash_seen[h] = i

    # Pass 2: Near-duplicate detection via embedding similarity
    valid_indices = [
        i for i, c in enumerate(chunks)
        if not c.is_duplicate and c.embedding is not None
    ]

    if len(valid_indices) >= 2:
        X = np.stack([chunks[i].embedding for i in valid_indices])
        sim_matrix = X @ X.T

        for local_i in range(len(valid_indices)):
            if chunks[valid_indices[local_i]].is_duplicate:
                continue
            for local_j in range(local_i + 1, len(valid_indices)):
                if chunks[valid_indices[local_j]].is_duplicate:
                    continue
                if float(sim_matrix[local_i, local_j]) >= min_similarity:
                    chunks[valid_indices[local_j]].is_duplicate = True
                    near_dup_count += 1

    # ✅ FIX 2: Removed the dead `kept` set that was built but never used.
    log.info(
        "Found %d duplicate chunk(s): %d exact, %d near-duplicate.",
        exact_count + near_dup_count,
        exact_count,
        near_dup_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. MATCH PERCENTAGE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────


def compute_match_percentages(
    chunks: List[Chunk],
    min_similarity: float,
    exclude_same_doc: bool = True,
    min_content_words: int = 10,
    skip_duplicates: bool = True,
) -> None:
    """
    Compute the BEST cross-document similarity for each chunk (match_pct).
    This value is primarily used for reporting & coloring.
    """
    valid_indices = []
    for i, c in enumerate(chunks):
        if c.embedding is None:
            continue
        if skip_duplicates and c.is_duplicate:
            continue
        if _normalized_word_count(c) < min_content_words:
            continue
        valid_indices.append(i)

    if not valid_indices:
        return

    log.info(
        "Computing match percentages for %d non-duplicate chunk(s) ...",
        len(valid_indices),
    )

    X = np.stack([chunks[i].embedding for i in valid_indices])
    sim_matrix = X @ X.T
    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)

    # ✅ FIX 3: Replace O(n²) pure Python inner loop with vectorized numpy ops.
    # Build a boolean doc-name array for same-doc masking in one shot per row.
    doc_names_arr = np.array([chunks[i].doc_name for i in valid_indices])

    matched_count = 0

    for local_i, chunk_idx in enumerate(valid_indices):
        c_i = chunks[chunk_idx]

        # Copy this chunk's similarity row and zero out positions to exclude.
        row = sim_matrix[local_i].copy()
        row[local_i] = 0.0  # never match self

        if exclude_same_doc:
            # Zero out all same-document columns in one vectorized step.
            row[doc_names_arr == c_i.doc_name] = 0.0

        # Single numpy max call instead of a Python loop over all j.
        best_sim = float(np.max(row)) if row.size > 0 else 0.0

        if best_sim >= min_similarity:
            c_i.match_pct = round(best_sim * 100.0, 2)
            matched_count += 1
        else:
            c_i.match_pct = 0.0

    log.info(
        "%d chunk(s) matched across documents (sim >= %.2f).",
        matched_count,
        min_similarity,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. CROSS-DOCUMENT CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────


def assign_cross_document_clusters(
    chunks: List[Chunk],
    min_similarity: float,
    min_content_words: int = 10,
) -> None:
    """
    Group chunks into cross-document clusters using Union-Find + centroid refinement.

    A valid cluster must have chunks originating from at least 2 different documents.
    """
    log.info("Assigning cross-document clusters (sim >= %.2f) ...", min_similarity)

    valid_indices = []
    for i, c in enumerate(chunks):
        if c.is_duplicate:
            continue
        if c.embedding is None:
            continue
        if _normalized_word_count(c) < min_content_words:
            continue
        valid_indices.append(i)

    if not valid_indices:
        return

    X = np.stack([chunks[i].embedding for i in valid_indices])
    sim_matrix = X @ X.T
    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)

    n = len(valid_indices)
    uf = UnionFind(n)

    # Connect edges above threshold (excluding same-document pairs)
    for i in range(n):
        for j in range(i + 1, n):
            c_i = chunks[valid_indices[i]]
            c_j = chunks[valid_indices[j]]
            if c_i.doc_name == c_j.doc_name:
                continue
            if sim_matrix[i, j] >= min_similarity:
                uf.union(i, j)

    # Extract raw clusters
    clusters_raw: Dict[int, List[int]] = {}
    for i in range(n):
        root = uf.find(i)
        clusters_raw.setdefault(root, []).append(i)

    # Initialize all to 0 (unclustered)
    for c in chunks:
        c.cluster_id = 0

    cluster_id_counter = 1

    # Centroid-filtered cluster refinement
    for root, local_indices in clusters_raw.items():
        docs = {chunks[valid_indices[i]].doc_name for i in local_indices}
        if len(docs) < 2:
            continue

        X_cluster = X[local_indices]
        centroid = np.mean(X_cluster, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        final_members = [
            local_i for local_i in local_indices
            if float(np.dot(X[local_i], centroid)) >= min_similarity
        ]

        final_docs = {chunks[valid_indices[i]].doc_name for i in final_members}
        if len(final_docs) >= 2:
            for local_i in final_members:
                chunks[valid_indices[local_i]].cluster_id = cluster_id_counter
            cluster_id_counter += 1

    log.info("Found %d valid cross-document cluster(s).", cluster_id_counter - 1)
