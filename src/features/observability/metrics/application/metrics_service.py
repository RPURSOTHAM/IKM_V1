"""Metrics recording service — only RAG Benchmark Metrics are persisted."""

from __future__ import annotations

import logging
import sys
import threading
from datetime import datetime, timezone
from typing import Any

from src.features.observability.metrics.application.benchmark_metrics import is_allowed_metric
from src.features.observability.metrics.domain.metrics_models import MetricEventRecord
from src.features.observability.metrics.infrastructure.metrics_repository import get_metrics_repository
from src.features.observability.middleware.request_context import current_context

logger = logging.getLogger(__name__)

_local = threading.local()

# Maps metric_name -> (component, operation) for consistent metadata on every record.
# Only benchmark-allowed metric names belong here.
METRIC_DEFAULTS: dict[str, tuple[str, str]] = {
    "document_processing_completed": ("processor", "process"),
    "document_processing_failed": ("processor", "process"),
    "document_upload_duplicate": ("document_receiver", "upload_duplicate"),
    "processor_cpu_usage": ("processor", "resource"),
    "processor_memory_usage": ("processor", "resource"),
    "chunking": ("chunking_processor", "chunk"),
    "embedding": ("embedding_service", "embed"),
    "blob_api_response_time": ("blob_storage", "api"),
    "blob_storage_used": ("blob_storage", "capacity"),
    "blob_available_capacity": ("blob_storage", "capacity"),
    "blob_largest_file": ("blob_storage", "capacity"),
    "blob_network_usage": ("blob_storage", "network"),
    "postgresql_cpu_usage": ("postgresql", "resource"),
    "postgresql_memory_usage": ("postgresql", "resource"),
    "postgresql_disk_io": ("postgresql", "resource"),
    "postgresql_storage_utilization": ("postgresql", "capacity"),
    "weaviate_query": ("weaviate", "query"),
    "weaviate_cpu_usage": ("weaviate", "resource"),
    "weaviate_memory_usage": ("weaviate", "resource"),
    "weaviate_disk_io": ("weaviate", "resource"),
    "weaviate_database_size": ("weaviate", "capacity"),
    "weaviate_storage_utilization": ("weaviate", "capacity"),
    "neo4j_cpu_usage": ("neo4j", "resource"),
    "neo4j_memory_usage": ("neo4j", "resource"),
    "neo4j_disk_io": ("neo4j", "resource"),
    "neo4j_database_size": ("neo4j", "capacity"),
    "redis_cpu_usage": ("redis", "resource"),
    "redis_memory_usage": ("redis", "resource"),
    "redis_disk_io": ("redis", "resource"),
    "redis_connected_clients": ("redis", "resource"),
    "redis_used_memory": ("redis", "capacity"),
    "redis_max_memory": ("redis", "capacity"),
    "scheduler_job": ("scheduler", "schedule"),
    "queue_wait_time": ("scheduler", "gauge"),
    "queue_depth": ("scheduler", "gauge"),
    "active_jobs": ("scheduler", "gauge"),
    "scheduler_cpu_usage": ("scheduler", "resource"),
    "scheduler_memory_usage": ("scheduler", "resource"),
    "scheduler_active_workers": ("scheduler", "resource"),
    "scheduler_pending_jobs": ("scheduler", "queue"),
    "scheduler_running_jobs": ("scheduler", "queue"),
    "scheduler_completed_jobs": ("scheduler", "queue"),
    "scheduler_failed_jobs": ("scheduler", "queue"),
    "retrieval_duration": ("retrieval_service", "retrieve"),
    "retrieval_precision_at_k": ("retrieval_service", "quality"),
    "retrieval_recall_at_k": ("retrieval_service", "quality"),
    "retrieval_hit_rate": ("retrieval_service", "quality"),
    "retrieval_mrr": ("retrieval_service", "quality"),
}


def is_in_metrics_persist() -> bool:
    return getattr(_local, "in_persist", False)


def enrich_metric_metadata(
    metric_name: str,
    metadata: dict[str, Any] | None,
    *,
    status: str = "success",
    duration_ms: float | None = None,
    ctx_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply standard component/operation defaults and context names to metadata."""
    meta = dict(metadata) if metadata is not None else {}

    defaults = METRIC_DEFAULTS.get(metric_name)
    if defaults:
        meta.setdefault("component", defaults[0])
        meta.setdefault("operation", defaults[1])
    else:
        meta.setdefault("component", "unknown")
        meta.setdefault("operation", "unknown")

    if ctx_metadata:
        meta.setdefault("repository_name", ctx_metadata.get("repository_name"))
        meta.setdefault("document_name", ctx_metadata.get("document_name"))

    meta.setdefault("status", status)
    if duration_ms is not None:
        meta.setdefault("duration_ms", duration_ms)

    if status == "failure" or metric_name.endswith("_error") or metric_name == "document_processing_failed":
        meta.setdefault("duration_before_failure_ms", duration_ms or 0.0)
        exc_type, exc_val, _ = sys.exc_info()
        if exc_val:
            meta.setdefault("exception_type", type(exc_val).__name__)
            meta.setdefault("error", str(exc_val))
            meta.setdefault("error_message", str(exc_val))

    return meta


class MetricsService:
    def _persist(self, event: MetricEventRecord) -> None:
        repo = get_metrics_repository()
        if repo is None:
            return
        _local.in_persist = True
        try:
            repo.insert_event(event)
        except Exception:
            logger.debug("Metrics persist failed for %s", event.metric_name, exc_info=True)
        finally:
            _local.in_persist = False

    def _async_persist(self, event: MetricEventRecord) -> None:
        threading.Thread(target=self._persist, args=(event,), daemon=True).start()

    def record(
        self,
        metric_name: str,
        *,
        duration_ms: float | None = None,
        status: str = "success",
        repository_id: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        request_id: str | None = None,
        correlation_id: str | None = None,
    ) -> None:
        if not is_allowed_metric(metric_name):
            logger.debug("Skipping non-benchmark metric: %s", metric_name)
            return

        ctx = current_context()
        meta = enrich_metric_metadata(
            metric_name,
            metadata,
            status=status,
            duration_ms=duration_ms,
            ctx_metadata=ctx.metadata,
        )

        event = MetricEventRecord(
            timestamp=datetime.now(timezone.utc),
            request_id=request_id or ctx.request_id,
            correlation_id=correlation_id or ctx.correlation_id,
            metric_name=metric_name,
            duration_ms=duration_ms,
            status=status,
            repository_id=repository_id or ctx.repository_id or meta.get("repository_id"),
            document_id=document_id or ctx.document_id or meta.get("document_id"),
            metadata=meta,
        )
        self._async_persist(event)

    def record_http_request(
        self,
        *,
        endpoint: str,
        http_method: str,
        status_code: int,
        duration_ms: float,
        status: str,
        request_id: str | None = None,
        correlation_id: str | None = None,
        user_id: str | None = None,
        repository_id: str | None = None,
        document_id: str | None = None,
        request_size: int = 0,
        response_size: int = 0,
    ) -> None:
        """HTTP request metrics are outside the RAG benchmark list — no-op."""
        return

    def record_error(
        self,
        *,
        component: str,
        operation: str,
        error: str,
        exception_type: str | None = None,
        duration_ms: float | None = None,
        repository_id: str | None = None,
        document_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Structured error events are outside the RAG benchmark list — no-op."""
        return

    def record_chunk_count(self, count: int, *, document_id: str | None = None, repository_id: str | None = None) -> None:
        """Chunk count is outside the RAG benchmark list — no-op."""
        return

    def record_gauge(self, metric_name: str, value: float | int, metadata: dict[str, Any] | None = None) -> None:
        self.record(metric_name, status="success", metadata={"value": value, **(metadata or {})})


_service: MetricsService | None = None


def get_metrics_service() -> MetricsService:
    global _service
    if _service is None:
        _service = MetricsService()
    return _service
