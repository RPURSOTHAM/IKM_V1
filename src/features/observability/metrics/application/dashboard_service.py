"""Dashboard aggregation over metrics_events for the Streamlit observability UI.

Aligned to RAG_Benchmark_Metrics.xlsx via ``benchmark_metrics``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.features.observability.metrics.application.benchmark_metrics import (
    ALLOWED_METRIC_NAMES,
    BENCHMARK_METRICS,
    benchmark_catalog,
    benchmark_summary,
)
from src.features.observability.metrics.infrastructure.metrics_repository import get_metrics_repository

# Dashboard sections keyed by Excel component, listing storage metric names.
DASHBOARD_SECTIONS: dict[str, list[str]] = {
    "document_processing": [
        "document_processing_completed",
        "document_processing_failed",
        "processor_cpu_usage",
        "processor_memory_usage",
    ],
    "chunking": ["chunking"],
    "embeddings": ["embedding"],
    "blob_storage": [
        "blob_api_response_time",
        "blob_storage_used",
        "blob_available_capacity",
        "blob_largest_file",
        "blob_network_usage",
    ],
    "postgresql": [
        "postgresql_cpu_usage",
        "postgresql_memory_usage",
        "postgresql_disk_io",
        "postgresql_storage_utilization",
    ],
    "weaviate": [
        "weaviate_query",
        "weaviate_cpu_usage",
        "weaviate_memory_usage",
        "weaviate_disk_io",
        "weaviate_database_size",
        "weaviate_storage_utilization",
    ],
    "neo4j": [
        "neo4j_cpu_usage",
        "neo4j_memory_usage",
        "neo4j_disk_io",
        "neo4j_database_size",
    ],
    "redis": [
        "redis_cpu_usage",
        "redis_memory_usage",
        "redis_disk_io",
        "redis_connected_clients",
        "redis_used_memory",
        "redis_max_memory",
    ],
    "scheduler": [
        "scheduler_job",
        "queue_wait_time",
        "queue_depth",
        "active_jobs",
        "scheduler_cpu_usage",
        "scheduler_memory_usage",
        "scheduler_active_workers",
        "scheduler_pending_jobs",
        "scheduler_running_jobs",
        "scheduler_completed_jobs",
        "scheduler_failed_jobs",
    ],
    "retrieval": [
        "retrieval_duration",
        "retrieval_precision_at_k",
        "retrieval_recall_at_k",
        "retrieval_hit_rate",
        "retrieval_mrr",
    ],
}

SECTION_LABELS: dict[str, str] = {
    "document_processing": "Document Processing",
    "chunking": "Chunking",
    "embeddings": "Embeddings",
    "blob_storage": "Blob Storage",
    "postgresql": "PostgreSQL",
    "weaviate": "Weaviate",
    "neo4j": "Neo4j",
    "redis": "Redis",
    "scheduler": "Scheduler Service",
    "retrieval": "Retrieval",
}


def _empty_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "success_count": 0,
        "failure_count": 0,
        "avg_duration_ms": None,
        "min_duration_ms": None,
        "max_duration_ms": None,
        "success_rate": None,
        "failure_rate": None,
    }


def _merge_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return _empty_stats()
    count = sum(int(r.get("count") or 0) for r in rows)
    success = sum(int(r.get("success_count") or 0) for r in rows)
    failure = sum(int(r.get("failure_count") or 0) for r in rows)
    durations = [float(r["avg_duration_ms"]) for r in rows if r.get("avg_duration_ms") is not None]
    mins = [float(r["min_duration_ms"]) for r in rows if r.get("min_duration_ms") is not None]
    maxs = [float(r["max_duration_ms"]) for r in rows if r.get("max_duration_ms") is not None]
    total = success + failure
    return {
        "count": count,
        "success_count": success,
        "failure_count": failure,
        "avg_duration_ms": sum(durations) / len(durations) if durations else None,
        "min_duration_ms": min(mins) if mins else None,
        "max_duration_ms": max(maxs) if maxs else None,
        "success_rate": round(success / total * 100.0, 1) if total else None,
        "failure_rate": round(failure / total * 100.0, 1) if total else None,
    }


def _fmt_ms(ms: float | None) -> str | None:
    if ms is None:
        return None
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def _fmt_pct(n: float | None) -> str | None:
    return f"{n:.1f}%" if n is not None else None


def _latest_gauge(repo: Any, metric_name: str, *, date_from, date_to) -> Any:
    events = repo.fetch_recent_events(
        metric_names=[metric_name],
        date_from=date_from,
        date_to=date_to,
        limit=1,
    )
    if not events:
        return None
    meta = events[0].get("metadata") or {}
    return meta.get("value")


def _top_meta(repo: Any, metric_name: str, key: str, *, date_from, date_to, fallback_keys: list[str] | None = None) -> str | None:
    rows = repo.top_metadata_values(
        key,
        metric_names=[metric_name],
        date_from=date_from,
        date_to=date_to,
        fallback_keys=fallback_keys or [],
        limit=1,
    )
    if not rows:
        return None
    return str(rows[0].get("value") or "-")


class DashboardService:
    def build_dashboard(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        repo = get_metrics_repository()
        catalog = benchmark_catalog()
        summary = benchmark_summary()

        if repo is None:
            return {
                "overview": _empty_stats(),
                "sections": {},
                "recent": {},
                "benchmark": {"summary": summary, "metrics": catalog},
                "timeline": [],
            }

        all_metrics = sorted(ALLOWED_METRIC_NAMES)
        by_metric = {
            row["metric_name"]: row
            for row in repo.aggregate_by_metric_name(
                metric_names=all_metrics,
                date_from=date_from,
                date_to=date_to,
            )
        }

        sections: dict[str, Any] = {}
        for section, names in DASHBOARD_SECTIONS.items():
            rows = [by_metric[n] for n in names if n in by_metric]
            stats = _merge_stats(rows)
            stats["label"] = SECTION_LABELS.get(section, section)
            stats["metrics"] = {n: by_metric.get(n, _empty_stats()) for n in names if n in by_metric}
            sections[section] = stats

        # Document processing rates across completed + failed.
        completed = by_metric.get("document_processing_completed", _empty_stats())
        failed = by_metric.get("document_processing_failed", _empty_stats())
        proc_success = int(completed.get("success_count") or completed.get("count") or 0)
        proc_failure = int(failed.get("failure_count") or failed.get("count") or 0)
        proc_total = proc_success + proc_failure
        if "document_processing" in sections:
            sections["document_processing"]["avg_duration_ms"] = completed.get("avg_duration_ms")
            sections["document_processing"]["success_rate"] = (
                round(proc_success / proc_total * 100.0, 1) if proc_total else None
            )
            sections["document_processing"]["failure_rate"] = (
                round(proc_failure / proc_total * 100.0, 1) if proc_total else None
            )
            sections["document_processing"]["processor_names"] = repo.top_metadata_values(
                "processor_type",
                metric_names=["document_processing_completed", "document_processing_failed"],
                date_from=date_from,
                date_to=date_to,
                fallback_keys=["processor_name"],
                limit=10,
            )
            sections["document_processing"]["models"] = repo.top_metadata_values(
                "model_name",
                metric_names=["embedding", "document_processing_completed"],
                date_from=date_from,
                date_to=date_to,
                fallback_keys=["embedding_model"],
                limit=10,
            )

        if "embeddings" in sections:
            sections["embeddings"]["models"] = repo.top_metadata_values(
                "model_name",
                metric_names=["embedding"],
                date_from=date_from,
                date_to=date_to,
                fallback_keys=["embedding_model"],
                limit=5,
            )

        if "scheduler" in sections:
            job = by_metric.get("scheduler_job", _empty_stats())
            sections["scheduler"]["avg_queue_wait_ms"] = repo.avg_metadata_numeric(
                "value",
                metric_names=["queue_wait_time"],
                date_from=date_from,
                date_to=date_to,
                fallback_keys=["queue_wait_time_ms"],
            )
            job_count = int(job.get("count") or 0)
            range_seconds = None
            if date_from and date_to:
                range_seconds = max((date_to - date_from).total_seconds(), 1.0)
            if range_seconds and job_count:
                sections["scheduler"]["jobs_per_sec"] = round(job_count / range_seconds, 4)
                sections["scheduler"]["jobs_per_min"] = round(job_count / (range_seconds / 60.0), 4)
            else:
                sections["scheduler"]["jobs_per_sec"] = None
                sections["scheduler"]["jobs_per_min"] = None
            sections["scheduler"]["latest_queue_depth"] = _latest_gauge(
                repo, "queue_depth", date_from=date_from, date_to=date_to
            )
            sections["scheduler"]["latest_active_jobs"] = _latest_gauge(
                repo, "active_jobs", date_from=date_from, date_to=date_to
            )
            sections["scheduler"]["avg_retry_count"] = repo.avg_metadata_numeric(
                "retry_count",
                metric_names=["scheduler_job"],
                date_from=date_from,
                date_to=date_to,
            )

        # Retrieval latency percentiles from aggregation service when possible.
        if "retrieval" in sections:
            try:
                from src.features.observability.metrics.application.aggregation_service import get_aggregation_service

                latency = get_aggregation_service().latency_summary(
                    metric_name="retrieval_duration",
                    date_from=date_from,
                    date_to=date_to,
                )
                sections["retrieval"]["p95"] = latency.get("p95")
                sections["retrieval"]["p99"] = latency.get("p99")
            except Exception:
                sections["retrieval"]["p95"] = None
                sections["retrieval"]["p99"] = None

        overview_rows = repo.aggregate_by_metric_name(
            metric_names=all_metrics,
            date_from=date_from,
            date_to=date_to,
        )
        overview = _merge_stats(overview_rows)

        recent: dict[str, list[dict[str, Any]]] = {}
        for section, names in DASHBOARD_SECTIONS.items():
            recent[section] = repo.fetch_recent_events(
                metric_names=names,
                date_from=date_from,
                date_to=date_to,
                limit=10,
            )

        # Enrich catalog rows with live values for Streamlit confirmation.
        enriched = []
        for row in catalog:
            source = row["source_metric"]
            stats = by_metric.get(source, _empty_stats())
            value = self._resolve_benchmark_value(
                row,
                stats,
                sections,
                repo=repo,
                date_from=date_from,
                date_to=date_to,
            )
            enriched.append({**row, "current_value": value})

        timeline = repo.fetch_hourly_timeline(date_from=date_from, date_to=date_to)

        return {
            "overview": overview,
            "sections": sections,
            "recent": recent,
            "benchmark": {"summary": summary, "metrics": enriched},
            "timeline": timeline,
            "section_labels": SECTION_LABELS,
        }

    def _resolve_benchmark_value(
        self,
        row: dict[str, Any],
        stats: dict[str, Any],
        sections: dict[str, Any],
        *,
        repo: Any,
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> str | None:
        if row["status"] == "planned":
            return None

        metric = row["metric"]
        source = row["source_metric"]
        path = row.get("value_path") or ""

        if metric == "Processor Name":
            names = (sections.get("document_processing") or {}).get("processor_names") or []
            if names:
                return ", ".join(str(n.get("value")) for n in names[:5] if n.get("value"))
            return _top_meta(
                repo, source, "processor_type", date_from=date_from, date_to=date_to, fallback_keys=["processor_name"]
            )

        if metric in ("Model Used", "Embedding Model"):
            models = (sections.get("embeddings") or {}).get("models") or (
                sections.get("document_processing") or {}
            ).get("models") or []
            if models:
                return ", ".join(str(m.get("value")) for m in models[:5] if m.get("value"))
            return _top_meta(
                repo, source, "model_name", date_from=date_from, date_to=date_to, fallback_keys=["embedding_model"]
            )

        if path == "duration_ms":
            return _fmt_ms(stats.get("avg_duration_ms") or stats.get("max_duration_ms"))

        if path == "avg_duration_ms":
            return _fmt_ms(stats.get("avg_duration_ms"))

        if path == "success_rate":
            if source == "document_processing_completed":
                return _fmt_pct((sections.get("document_processing") or {}).get("success_rate"))
            return _fmt_pct(stats.get("success_rate"))

        if path == "failure_rate":
            if source == "document_processing_failed":
                return _fmt_pct((sections.get("document_processing") or {}).get("failure_rate"))
            rate = stats.get("failure_rate")
            if rate is None and stats.get("count"):
                total = int(stats.get("success_count") or 0) + int(stats.get("failure_count") or 0)
                rate = round(int(stats.get("failure_count") or 0) / total * 100.0, 1) if total else None
            return _fmt_pct(rate)

        if path == "success_count":
            return str(int(stats.get("success_count") or 0))

        if path == "failure_count":
            return str(int(stats.get("failure_count") or 0))

        if path == "jobs_per_sec":
            v = (sections.get("scheduler") or {}).get("jobs_per_sec")
            return f"{v:.4f}" if v is not None else None

        if path == "jobs_per_min":
            v = (sections.get("scheduler") or {}).get("jobs_per_min")
            return f"{v:.4f}" if v is not None else None

        if path == "p95":
            return _fmt_ms((sections.get("retrieval") or {}).get("p95"))

        if path == "p99":
            return _fmt_ms((sections.get("retrieval") or {}).get("p99"))

        if path == "metadata.value":
            gauge = _latest_gauge(repo, source, date_from=date_from, date_to=date_to)
            if gauge is not None:
                return str(gauge)
            # Fallback: queue wait often also on scheduler_job metadata
            if source == "queue_wait_time":
                avg = (sections.get("scheduler") or {}).get("avg_queue_wait_ms")
                return _fmt_ms(avg)
            return None

        if path == "metadata.retry_count":
            avg = (sections.get("scheduler") or {}).get("avg_retry_count")
            return f"{avg:.2f}" if avg is not None else None

        return None


_service: DashboardService | None = None


def get_dashboard_service() -> DashboardService:
    global _service
    if _service is None:
        _service = DashboardService()
    return _service
