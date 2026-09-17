"""Production retrieval pipeline stage adapters that reuse existing components."""

from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable

from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
from src.features.retrieval.domain.interfaces import (
    ContextCompressionStage,
    HybridRetrieverStage,
    IntentClassifierStage,
    MetadataFilterGeneratorStage,
    PromptBuilderStage,
    PromptGuardStage,
    QueryRewriterStage,
    RankFusionStage,
    RerankerStage,
)
from src.features.retrieval.domain.models import PipelineState, RetrievalCandidate, StageResult
from src.features.retrieval.strategies.rrf import reciprocal_rank_fusion

_logger = logging.getLogger(__name__)


def _timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - started) * 1000.0


class ExistingPromptGuardAdapter(PromptGuardStage):
    """Canonical PromptGuard — must run before rewrite/intent/retrieval/rerank/prompt build."""

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_prompt_guard:
            return StageResult(
                stage=self.name,
                status="skipped",
                confidence=1.0,
                metadata={"prompt_guard_executed": False, "reason": "disabled"},
            )

        def _execute() -> dict[str, Any]:
            from src.features.security.moderation.hybrid_prompt_guard import HybridPromptGuard
            from src.features.security.moderation.ai_safety_guard import PROMPT_GUARD_VERSION

            result = HybridPromptGuard().check_prompt(state.original_query, pipeline="retrieval")
            result["preferred_model"] = config.preferred_prompt_guard_model
            result["prompt_guard_version"] = result.get("prompt_guard_version") or PROMPT_GUARD_VERSION
            return result

        try:
            result, elapsed = _timed(_execute)
        except Exception as exc:
            # Fail closed for prompt injection when rule guard cannot run.
            state.blocked = True
            state.block_reason = f"prompt guard unavailable: {exc}"
            return StageResult(
                stage=self.name,
                status="blocked",
                confidence=0.0,
                execution_time_ms=0.0,
                error=str(exc),
                warnings=["prompt guard unavailable; pipeline terminated"],
                metadata={
                    "prompt_guard_executed": False,
                    "preferred_model": config.preferred_prompt_guard_model,
                    "pipeline_terminated": True,
                },
            )

        allowed = bool(result.get("allowed", True))
        severity = str(result.get("severity") or "low").lower()
        if not allowed and (
            config.prompt_guard_block_on_high
            or severity in {"medium", "high", "critical"}
            or str(result.get("action") or "").lower() == "block"
        ):
            state.blocked = True
            state.block_reason = str(result.get("reason") or "prompt blocked by guard")
        return StageResult(
            stage=self.name,
            status="ok" if allowed else "blocked",
            confidence=1.0 if allowed else 0.0,
            execution_time_ms=elapsed,
            metadata={
                "prompt_guard_executed": True,
                "allowed": allowed,
                "risk_type": result.get("risk_type"),
                "severity": severity,
                "action": result.get("action") or ("allow" if allowed else "block"),
                "pipeline_terminated": bool(state.blocked),
                "prompt_guard_version": result.get("prompt_guard_version"),
                "hybrid_version": result.get("hybrid_version"),
                "rule_guard": result.get("rule_guard"),
                "model_guard": result.get("model_guard"),
                "model_guard_skipped": result.get("model_guard_skipped"),
                "preferred_model": config.preferred_prompt_guard_model,
                "implementation": "hybrid_rule_then_model",
            },
        )


class HeuristicQueryRewriter(QueryRewriterStage):
    """Lightweight rewriter with optional Qwen3 adapter hook."""

    _FILLER = re.compile(
        r"\b(?:please|kindly|could you|can you|tell me|explain|about|regarding)\b",
        re.IGNORECASE,
    )

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_query_rewrite:
            state.rewritten_query = state.original_query
            state.expanded_queries = []
            state.sub_queries = []
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        def _execute() -> tuple[str, list[str], list[str], bool]:
            cleaned = " ".join(self._FILLER.sub(" ", state.original_query).split()).strip()
            rewritten = cleaned or state.original_query
            expansions: list[str] = []
            provider_available = False
            # Prefer Qwen3 when an injected callable is present; never hard-require it.
            qwen = getattr(self, "_qwen_rewrite", None)
            if callable(qwen):
                try:
                    model_out = qwen(state.original_query, config.preferred_query_rewriter_model)
                    if isinstance(model_out, str) and model_out.strip():
                        rewritten = model_out.strip()
                        expansions = [rewritten]
                        provider_available = True
                    elif isinstance(model_out, (list, tuple)):
                        cleaned_out = [str(item).strip() for item in model_out if str(item).strip()]
                        if cleaned_out:
                            rewritten = cleaned_out[0]
                            expansions = cleaned_out
                            provider_available = True
                except Exception as exc:
                    _logger.warning("Qwen query rewrite unavailable: %s", exc)
            if not provider_available and rewritten.lower() != state.original_query.lower().strip():
                expansions = [rewritten]
            # Multi-clause question decomposition (e.g. "What is X and what are Y?").
            parts = [
                part.strip(" ?.")
                for part in re.split(
                    r"\?\s*(?=(?:what|who|where|when|why|how|which|list|describe)\b)|"
                    r"(?<=\?)\s*|\band\b(?=\s*(?:what|who|where|when|why|how|which)\b)|[;]",
                    rewritten,
                    flags=re.IGNORECASE,
                )
                if part.strip(" ?.")
            ]
            sub_queries = [part if part.endswith("?") else f"{part}?" for part in parts if len(part) > 3]
            if len(sub_queries) <= 1:
                sub_queries = []
            if not sub_queries:
                stem_match = re.match(
                    r"^(what is|what are|who is|who are|list|describe)\s+(.+)$",
                    rewritten.rstrip(" ?."),
                    flags=re.IGNORECASE,
                )
                if stem_match:
                    stem = stem_match.group(1)
                    items = [
                        item.strip(" .")
                        for item in re.split(r",|\band\b", stem_match.group(2), flags=re.IGNORECASE)
                        if item.strip(" .") and len(item.strip(" .")) > 3
                    ]
                    if len(items) >= 2:
                        sub_queries = [f"{stem} {item}?" for item in items]
            if not sub_queries and "," in rewritten:
                items = [
                    item.strip(" .")
                    for item in re.split(r",|\band\b", rewritten.rstrip(" ?."), flags=re.IGNORECASE)
                    if item.strip(" .") and 3 < len(item.strip(" .")) <= 80
                ]
                short_items = [item for item in items if len(item.split()) <= 6]
                if len(short_items) >= 2:
                    sub_queries = [item if item.endswith("?") else f"{item}?" for item in short_items]
            return rewritten, expansions, sub_queries, provider_available

        try:
            (rewritten, expansions, sub_queries, provider_available), elapsed = _timed(_execute)
        except Exception as exc:
            state.rewritten_query = state.original_query
            if config.never_block_on_optional_failure:
                return StageResult(
                    stage=self.name,
                    status="degraded",
                    confidence=0.0,
                    error=str(exc),
                    warnings=["query rewrite failed; using original query"],
                    metadata={"preferred_model": config.preferred_query_rewriter_model},
                )
            raise

        state.rewritten_query = rewritten
        # Never report the original query alone as a meaningful expansion.
        expansions = [
            item
            for item in expansions
            if str(item).strip() and str(item).strip().lower() != state.original_query.strip().lower()
        ]
        state.expanded_queries = expansions
        state.sub_queries = sub_queries
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=0.8,
            execution_time_ms=elapsed,
            metadata={
                "rewritten_query": rewritten,
                "expanded_count": len(expansions),
                "preferred_model": config.preferred_query_rewriter_model,
                "implementation": "heuristic_with_optional_qwen3",
                "rewrite_provider_available": provider_available,
                "expansion_status": (
                    "expanded"
                    if expansions
                    else ("unavailable" if not provider_available else "unchanged")
                ),
                "expansion_fallback": None if expansions else "original_query",
            },
        )


class KeywordIntentClassifier(IntentClassifierStage):
    """Reuse TopicClassifier keyword patterns for query intent routing."""

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_intent_classifier:
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        def _execute() -> dict[str, Any]:
            query = state.rewritten_query or state.original_query
            intent = "factual"
            confidence = 0.55
            topic = "General"
            try:
                from src.features.security.classification.topic_classifier import TopicClassifier

                topics = TopicClassifier().classify_topics(query).get("topics") or []
                if topics:
                    topic = str(topics[0].get("topic") or "General")
                    confidence = float(topics[0].get("confidence") or confidence)
            except Exception:
                pass

            lower = query.lower()
            if any(token in lower for token in ("how to", "procedure", "steps", "process")):
                intent = "procedural"
                confidence = max(confidence, 0.7)
            elif any(token in lower for token in ("where", "which document", "find", "locate")):
                intent = "navigational"
                confidence = max(confidence, 0.65)
            elif any(token in lower for token in ("compare", "difference", "vs", "versus")):
                intent = "comparative"
                confidence = max(confidence, 0.65)
            elif any(token in lower for token in ("why", "reason", "cause")):
                intent = "explanatory"
                confidence = max(confidence, 0.6)
            return {
                "intent": intent,
                "topic": topic,
                "confidence": confidence,
            }

        try:
            intent, elapsed = _timed(_execute)
        except Exception as exc:
            if config.never_block_on_optional_failure:
                state.intent = {"intent": "factual", "topic": "General", "confidence": 0.0}
                return StageResult(
                    stage=self.name,
                    status="degraded",
                    confidence=0.0,
                    error=str(exc),
                    warnings=["intent classification failed"],
                )
            raise

        if float(intent.get("confidence") or 0.0) < config.intent_confidence_threshold:
            intent["intent"] = "factual"
        state.intent = intent
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=float(intent.get("confidence") or 0.0),
            execution_time_ms=elapsed,
            metadata=intent,
        )


class RuleMetadataFilterGenerator(MetadataFilterGeneratorStage):
    """Generate Weaviate property filters from query hints; merge with caller filters."""

    _DOC_PATTERN = re.compile(
        r"\b(?:document|doc|sop|file)\s+(?:named|called|titled)?\s*[\"']?([A-Za-z0-9._\\-]+)[\"']?",
        re.IGNORECASE,
    )
    # Generic filename/token with a known document extension (no document-name hardcoding).
    _FILENAME_PATTERN = re.compile(
        r"(?i)\b([A-Za-z0-9][\w.\-]{0,240}\.(?:pdf|docx?|pptx?|txt|html?|xlsx?|csv))\b"
    )
    _SECTION_PATTERN = re.compile(
        r"\b(?:section|heading)\s+[\"']?([A-Za-z0-9 .\\-_/]+)[\"']?",
        re.IGNORECASE,
    )
    _CATEGORY_PATTERN = re.compile(
        # Require explicit "category" / "document type" — bare "type 2 diabetes" must not match.
        r"\b(?:category|document\s+type)\s+[\"']?([A-Za-z][A-Za-z0-9 .\\-_/]+)[\"']?",
        re.IGNORECASE,
    )

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_metadata_filter_generator:
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        def _execute() -> dict[str, Any]:
            query = state.rewritten_query or state.original_query
            generated: dict[str, Any] = {}
            doc_match = self._DOC_PATTERN.search(query)
            if doc_match:
                generated["doc_name"] = doc_match.group(1).strip()
            else:
                filename_match = self._FILENAME_PATTERN.search(query or "")
                if filename_match:
                    generated["doc_name"] = filename_match.group(1).strip()
            section_match = self._SECTION_PATTERN.search(query)
            if section_match:
                generated["section_name"] = section_match.group(1).strip()
            category_match = self._CATEGORY_PATTERN.search(query)
            if category_match:
                generated["category"] = category_match.group(1).strip()
            # Intent-informed defaults never override caller filters.
            merged = {**generated, **dict(state.filters or {})}
            return {"generated": generated, "merged": merged}

        try:
            payload, elapsed = _timed(_execute)
        except Exception as exc:
            if config.never_block_on_optional_failure:
                return StageResult(
                    stage=self.name,
                    status="degraded",
                    confidence=0.0,
                    error=str(exc),
                    warnings=["metadata filter generation failed"],
                )
            raise

        state.filters = payload["merged"]
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=1.0 if payload["generated"] else 0.5,
            execution_time_ms=elapsed,
            metadata=payload,
        )


class CallableHybridRetriever(HybridRetrieverStage):
    """Adapter that delegates dense/sparse fetches to injected callables."""

    def __init__(
        self,
        dense_fn: Callable[[PipelineState, RetrievalPipelineConfig], list[RetrievalCandidate]],
        sparse_fn: Callable[[PipelineState, RetrievalPipelineConfig], list[RetrievalCandidate]],
    ) -> None:
        self._dense_fn = dense_fn
        self._sparse_fn = sparse_fn

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        def _merge(existing: list[RetrievalCandidate], extra: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
            seen = {str(getattr(item, "chunk_id", "") or id(item)) for item in existing}
            merged = list(existing)
            for item in extra:
                key = str(getattr(item, "chunk_id", "") or id(item))
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
            return merged

        def _execute() -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
            dense = list(self._dense_fn(state, config) or [])
            sparse = list(self._sparse_fn(state, config) or [])
            original_rewritten = state.rewritten_query
            primary = (original_rewritten or state.original_query or "").strip()
            for sub in state.sub_queries or []:
                candidate_query = str(sub or "").strip()
                if not candidate_query or candidate_query.rstrip(" ?.").lower() == primary.rstrip(" ?.").lower():
                    continue
                state.rewritten_query = candidate_query
                dense = _merge(dense, list(self._dense_fn(state, config) or []))
                sparse = _merge(sparse, list(self._sparse_fn(state, config) or []))
            state.rewritten_query = original_rewritten
            return dense, sparse

        try:
            (dense, sparse), elapsed = _timed(_execute)
        except Exception as exc:
            return StageResult(
                stage=self.name,
                status="error",
                confidence=0.0,
                error=str(exc),
                metadata={"preferred_embedding_model": config.preferred_embedding_model},
            )

        state.dense_candidates = dense
        state.sparse_candidates = sparse
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=1.0 if dense or sparse else 0.0,
            execution_time_ms=elapsed,
            metadata={
                "dense_count": len(dense),
                "sparse_count": len(sparse),
                "preferred_embedding_model": config.preferred_embedding_model,
                "implementation": "hybrid_dense_bm25",
            },
        )


class ReciprocalRankFusionStage(RankFusionStage):
    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_hybrid_rrf:
            state.fused_candidates = state.dense_candidates or state.sparse_candidates
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        def _execute() -> list[RetrievalCandidate]:
            return reciprocal_rank_fusion(
                [state.dense_candidates, state.sparse_candidates],
                k=config.rrf_k,
                source_names=["dense", "bm25"],
            )

        try:
            fused, elapsed = _timed(_execute)
        except Exception as exc:
            state.fused_candidates = state.dense_candidates or state.sparse_candidates
            return StageResult(
                stage=self.name,
                status="degraded",
                confidence=0.0,
                error=str(exc),
                warnings=["RRF failed; using dense or sparse list"],
            )

        # Apply min_score on fused RRF scores when configured.
        if config.min_score > 0:
            fused = [item for item in fused if float(item.score) >= config.min_score]
        state.fused_candidates = fused[: config.candidate_k]
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=1.0 if fused else 0.0,
            execution_time_ms=elapsed,
            metadata={"fused_count": len(state.fused_candidates), "rrf_k": config.rrf_k},
        )


class ExistingCrossEncoderReranker(RerankerStage):
    """Only rerank Top-K fused candidates using existing CrossEncoder loader."""

    def __init__(
        self,
        rerank_fn: Callable[[str, list[RetrievalCandidate], RetrievalPipelineConfig], list[RetrievalCandidate]],
    ) -> None:
        self._rerank_fn = rerank_fn

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_rerank:
            state.reranked_candidates = state.fused_candidates[: config.final_top_k]
            state.final_candidates = state.reranked_candidates
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        top_k = state.fused_candidates[: config.rerank_top_k]

        def _execute() -> list[RetrievalCandidate]:
            return self._rerank_fn(
                state.rewritten_query or state.original_query,
                top_k,
                config,
            )

        try:
            reranked, elapsed = _timed(_execute)
        except Exception as exc:
            state.reranked_candidates = top_k
            state.final_candidates = top_k[: config.final_top_k]
            return StageResult(
                stage=self.name,
                status="degraded",
                confidence=0.0,
                error=str(exc),
                warnings=["reranker unavailable; keeping RRF order"],
                metadata={"preferred_model": config.preferred_reranker_model},
            )

        has_rerank_scores = any(
            (getattr(item, "retrieval_signals", None) or {}).get("rerank_score") is not None
            for item in reranked
        )
        if not has_rerank_scores:
            state.reranked_candidates = top_k
            state.final_candidates = top_k[: config.final_top_k]
            return StageResult(
                stage=self.name,
                status="degraded",
                confidence=0.0,
                execution_time_ms=elapsed,
                warnings=["reranker did not apply scores; keeping RRF order"],
                metadata={
                    "preferred_model": config.preferred_reranker_model,
                    "rerank_applied": False,
                },
            )

        if config.rerank_score_threshold > 0:
            reranked = [
                item
                for item in reranked
                if float(item.retrieval_signals.get("rerank_score") or item.score)
                >= config.rerank_score_threshold
            ]
        state.reranked_candidates = reranked
        state.final_candidates = reranked[: config.final_top_k]
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=1.0 if reranked else 0.0,
            execution_time_ms=elapsed,
            metadata={
                "rerank_top_k": config.rerank_top_k,
                "returned": len(state.final_candidates),
                "preferred_model": config.preferred_reranker_model,
                "implementation": "sentence_transformers_cross_encoder",
                "rerank_applied": True,
            },
        )


class LLMLinguaOrExtractiveCompressor(ContextCompressionStage):
    """Prefer LLMLingua-2 when available; otherwise keep top sentences."""

    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_context_compression:
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        def _execute() -> tuple[str, str]:
            texts = [c.text.strip() for c in state.final_candidates if c.text.strip()]
            joined = "\n\n".join(texts)
            implementation = "extractive"
            # Optional LLMLingua-2 adapter (never required).
            try:
                from llmlingua import PromptCompressor  # type: ignore

                compressor = PromptCompressor(config.preferred_compression_model)
                compressed = compressor.compress_prompt(
                    joined,
                    rate=0.5,
                    force_tokens=["\n", "?"],
                )
                if isinstance(compressed, dict):
                    joined = str(compressed.get("compressed_prompt") or joined)
                else:
                    joined = str(compressed)
                implementation = "llmlingua2"
            except Exception:
                sentences = re.split(r"(?<=[.!?])\s+", joined)
                keep = max(1, int(config.compression_keep_sentences))
                joined = " ".join(sentences[:keep]).strip()
                # Soft token budget approximation (~4 chars/token).
                max_chars = max(200, int(config.compression_max_tokens) * 4)
                if len(joined) > max_chars:
                    joined = joined[:max_chars].rsplit(" ", 1)[0]
            return joined, implementation

        try:
            (compressed, implementation), elapsed = _timed(_execute)
        except Exception as exc:
            if config.never_block_on_optional_failure:
                state.compressed_context = "\n\n".join(c.text for c in state.final_candidates[:3])
                return StageResult(
                    stage=self.name,
                    status="degraded",
                    confidence=0.0,
                    error=str(exc),
                    warnings=["compression failed; using raw top chunks"],
                    metadata={"preferred_model": config.preferred_compression_model},
                )
            raise

        state.compressed_context = compressed
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=0.9 if implementation == "llmlingua2" else 0.7,
            execution_time_ms=elapsed,
            metadata={
                "implementation": implementation,
                "preferred_model": config.preferred_compression_model,
                "chars": len(compressed),
            },
        )


class TemplatePromptBuilder(PromptBuilderStage):
    def run(self, state: PipelineState, config: RetrievalPipelineConfig) -> StageResult:
        if not config.enable_prompt_builder:
            return StageResult(stage=self.name, status="skipped", confidence=1.0)

        def _execute() -> str:
            context = state.compressed_context or "\n\n".join(
                f"[{idx}] {c.doc_name} p.{c.page}: {c.text}"
                for idx, c in enumerate(state.final_candidates, start=1)
            )
            citations = []
            for idx, candidate in enumerate(state.final_candidates, start=1):
                citations.append(
                    f"[{idx}] {candidate.doc_name} | section={candidate.section_name} | page={candidate.page}"
                )
            intent = state.intent.get("intent") if state.intent else "factual"
            return (
                "You are an enterprise document assistant.\n"
                f"Intent: {intent}\n"
                f"Question: {state.rewritten_query or state.original_query}\n\n"
                "Use only the provided context. Cite sources by bracket number.\n\n"
                f"Context:\n{context}\n\n"
                f"Sources:\n" + "\n".join(citations)
            )

        try:
            prompt, elapsed = _timed(_execute)
        except Exception as exc:
            if config.never_block_on_optional_failure:
                return StageResult(
                    stage=self.name,
                    status="degraded",
                    confidence=0.0,
                    error=str(exc),
                    warnings=["prompt builder failed"],
                )
            raise

        state.prompt = prompt
        return StageResult(
            stage=self.name,
            status="ok",
            confidence=1.0,
            execution_time_ms=elapsed,
            metadata={"prompt_chars": len(prompt)},
        )
