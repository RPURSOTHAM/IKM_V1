"""Query decomposition regression tests."""

from __future__ import annotations

from src.features.retrieval.application.pipeline.stages import HeuristicQueryRewriter
from src.features.retrieval.configuration.retrieval_config import RetrievalPipelineConfig
from src.features.retrieval.domain.models import PipelineState


def test_multi_part_query_decomposes() -> None:
    rewriter = HeuristicQueryRewriter()
    state = PipelineState(
        original_query="What is the cleaning procedure and what are the acceptance criteria?",
    )
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert len(state.sub_queries) >= 2
    joined = " ".join(state.sub_queries).lower()
    assert "cleaning procedure" in joined
    assert "acceptance criteria" in joined


def test_simple_query_stays_simple() -> None:
    rewriter = HeuristicQueryRewriter()
    state = PipelineState(original_query="cleaning procedure")
    rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert state.sub_queries == []


def test_comma_list_query_decomposes() -> None:
    rewriter = HeuristicQueryRewriter()
    state = PipelineState(
        original_query="What is the purpose, approval process, and retention requirement?",
    )
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert len(state.sub_queries) >= 2
    joined = " ".join(state.sub_queries).lower()
    assert "purpose" in joined
    assert "approval" in joined
    assert "retention" in joined


def test_bare_comma_list_decomposes() -> None:
    rewriter = HeuristicQueryRewriter()
    state = PipelineState(original_query="purpose, approval, and retention requirements")
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert len(state.sub_queries) >= 2
    joined = " ".join(state.sub_queries).lower()
    assert "purpose" in joined
    assert "approval" in joined
    assert "retention" in joined


def test_query_expansion_disabled_returns_empty() -> None:
    rewriter = HeuristicQueryRewriter()
    state = PipelineState(original_query="cleaning procedure")
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=False))
    assert result.status == "skipped"
    assert state.expanded_queries == []
    assert state.sub_queries == []


def test_query_expansion_without_provider_does_not_fabricate() -> None:
    rewriter = HeuristicQueryRewriter()
    state = PipelineState(original_query="cleaning procedure")
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert state.expanded_queries == []
    assert result.metadata.get("expansion_fallback") == "original_query"
    assert result.metadata.get("rewrite_provider_available") is False
    assert state.rewritten_query


def test_query_expansion_with_provider_returns_meaningful_queries() -> None:
    rewriter = HeuristicQueryRewriter()
    rewriter._qwen_rewrite = lambda _query, _model: ["cleaning SOP", "sanitation procedure"]
    state = PipelineState(original_query="cleaning procedure")
    result = rewriter.run(state, RetrievalPipelineConfig(enable_query_rewrite=True))
    assert result.status == "ok"
    assert "cleaning SOP" in state.expanded_queries
    assert "sanitation procedure" in state.expanded_queries
    assert result.metadata.get("rewrite_provider_available") is True


def test_hybrid_retriever_executes_sub_queries() -> None:
    from src.features.retrieval.application.pipeline.stages import CallableHybridRetriever
    from src.features.retrieval.domain.models import RetrievalCandidate

    seen: list[str] = []

    def dense_fn(state, _config):
        q = state.rewritten_query or state.original_query
        seen.append(q)
        return [RetrievalCandidate(chunk_id=q, text=q, doc_name="d", page=1, score=0.9)]

    def sparse_fn(state, _config):
        return []

    state = PipelineState(original_query="What is the purpose?")
    state.rewritten_query = "What is the purpose?"
    state.sub_queries = ["What is the purpose?", "What is the approval process?"]
    retriever = CallableHybridRetriever(dense_fn, sparse_fn)
    retriever.run(state, RetrievalPipelineConfig())
    assert any("approval" in item.lower() for item in seen)
    assert len(state.dense_candidates) >= 2
