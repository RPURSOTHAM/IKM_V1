"""Prometheus export adapter for retrieval metrics."""

from __future__ import annotations

from src.features.retrieval.metrics.models import RetrievalMetrics


def export_retrieval_metrics(metrics: RetrievalMetrics) -> None:
    """Publish retrieval telemetry to the shared metrics registry."""
    try:
        from src.infrastructure.application_support.metrics import get_metrics

        registry = get_metrics(namespace="rag")
        labels = {
            "strategy": metrics.strategy or "unknown",
            "repository_id": metrics.repository_id or "none",
        }
        registry.pipeline_stage_latency.labels(pipeline="retrieval", stage="total").observe(
            metrics.latency.total_ms / 1000.0
        )
        for stage_name, value in metrics.latency.pipeline_trace(debug=True).items():
            if not stage_name.endswith("_ms") or stage_name == "total_ms":
                continue
            stage = stage_name.removesuffix("_ms")
            registry.pipeline_stage_latency.labels(pipeline="retrieval", stage=stage).observe(
                float(value) / 1000.0
            )
    except Exception:
        return
