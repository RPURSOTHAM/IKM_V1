"""RAG Benchmark Metrics dashboard — aligned to RAG_Benchmark_Metrics.xlsx."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import streamlit as st

from src.features.observability.streamlit.pagination import render_page_size_selector, render_paginated_table


def _fmt_ms(ms: float | None) -> str:
    if ms is None or ms <= 0:
        return "-"
    if ms < 1000:
        return f"{ms:.0f} ms"
    return f"{ms / 1000:.2f} s"


def _fmt_pct(n: float | None) -> str:
    return f"{n:.1f}%" if n is not None else "-"


def _fmt_int(n: float | int | None) -> str:
    if n is None:
        return "-"
    return f"{int(n):,}"


def _status_label(status: str) -> str:
    if status == "collecting":
        return "Collecting"
    if status == "derived":
        return "Derived"
    return "Planned"


def _render_filters() -> dict[str, str]:
    today = datetime.utcnow().date()
    c1, c2 = st.columns(2)
    with c1:
        date_from = st.date_input(
            "From",
            value=today - timedelta(days=7),
            key="metrics_dashboard_date_from",
        )
    with c2:
        date_to = st.date_input(
            "To",
            value=today,
            key="metrics_dashboard_date_to",
        )
    return {
        "date_from": f"{date_from.isoformat()}T00:00:00",
        "date_to": f"{date_to.isoformat()}T23:59:59",
    }


def _render_benchmark_catalog(benchmark: dict[str, Any]) -> None:
    summary = benchmark.get("summary") or {}
    metrics = benchmark.get("metrics") or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Benchmark Metrics", _fmt_int(summary.get("total")))
    c2.metric("Collecting", _fmt_int(summary.get("collecting")))
    c3.metric("Derived", _fmt_int(summary.get("derived")))
    c4.metric("Planned", _fmt_int(summary.get("planned")))

    if not metrics:
        st.info("Benchmark catalog unavailable.")
        return

    rows = []
    current_component = None
    for item in metrics:
        component = item.get("component") or "-"
        if component != current_component:
            current_component = component
        rows.append({
            "Component": component,
            "Category": item.get("category", "-"),
            "Metric": item.get("metric", "-"),
            "Collection": _status_label(item.get("status", "planned")),
            "Storage Key": item.get("source_metric", "-"),
            "Current Value": item.get("current_value") or "-",
            "Notes": item.get("notes", ""),
        })

    page_size = int(st.session_state.get("benchmark_catalog_page_size", st.session_state.get("metrics_page_size", 25)))
    render_paginated_table(rows, key="benchmark_catalog", page_size=page_size)


def _render_section_summary(section_key: str, label: str, section: dict, recent: list[dict]) -> None:
    if int(section.get("count") or 0) <= 0 and not recent:
        return

    with st.expander(label, expanded=(section_key == "document_processing")):
        if section_key == "document_processing":
            completed = (section.get("metrics") or {}).get("document_processing_completed", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Processing Events", _fmt_int(completed.get("count")))
            c2.metric("Avg Processing Time", _fmt_ms(completed.get("avg_duration_ms")))
            c3.metric("Success Rate", _fmt_pct(section.get("success_rate")))
            c4.metric("Failure Rate", _fmt_pct(section.get("failure_rate")))
        elif section_key == "chunking":
            chunking = (section.get("metrics") or {}).get("chunking", {})
            c1, c2 = st.columns(2)
            c1.metric("Chunking Runs", _fmt_int(chunking.get("count")))
            c2.metric("Avg Chunking Time", _fmt_ms(chunking.get("avg_duration_ms")))
        elif section_key == "embeddings":
            emb = (section.get("metrics") or {}).get("embedding", {})
            c1, c2 = st.columns(2)
            c1.metric("Embedding Runs", _fmt_int(emb.get("count")))
            c2.metric("Avg Generation Time", _fmt_ms(emb.get("avg_duration_ms")))
        elif section_key == "blob_storage":
            blob = (section.get("metrics") or {}).get("blob_api_response_time", {})
            c1, c2 = st.columns(2)
            c1.metric("Blob API Calls", _fmt_int(blob.get("count")))
            c2.metric("Avg Response Time", _fmt_ms(blob.get("avg_duration_ms")))
        elif section_key == "weaviate":
            query = (section.get("metrics") or {}).get("weaviate_query", {})
            c1, c2 = st.columns(2)
            c1.metric("Vector Queries", _fmt_int(query.get("count")))
            c2.metric("Avg Retrieval Time", _fmt_ms(query.get("avg_duration_ms")))
        elif section_key == "scheduler":
            job = (section.get("metrics") or {}).get("scheduler_job", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Job Events", _fmt_int(job.get("count")))
            c2.metric("Avg Execution", _fmt_ms(job.get("avg_duration_ms")))
            c3.metric("Queue Depth", _fmt_int(section.get("latest_queue_depth")))
            c4.metric("Active Jobs", _fmt_int(section.get("latest_active_jobs")))
        elif section_key == "retrieval":
            search = (section.get("metrics") or {}).get("retrieval_duration", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Retrievals", _fmt_int(search.get("count")))
            c2.metric("Avg Latency", _fmt_ms(search.get("avg_duration_ms")))
            c3.metric("P95", _fmt_ms(section.get("p95")))
            c4.metric("P99", _fmt_ms(section.get("p99")))
        else:
            c1, c2 = st.columns(2)
            c1.metric("Events", _fmt_int(section.get("count")))
            c2.metric("Avg Duration", _fmt_ms(section.get("avg_duration_ms")))

        if recent:
            st.markdown("**Recent events**")
            event_rows = []
            for ev in recent[:10]:
                meta = ev.get("metadata") or {}
                ts = ev.get("timestamp", "")
                if hasattr(ts, "strftime"):
                    ts = ts.strftime("%Y-%m-%d %H:%M:%S")
                event_rows.append({
                    "Time": str(ts)[:19],
                    "Metric": ev.get("metric_name", "-"),
                    "Status": (ev.get("status") or "-").upper(),
                    "Duration": _fmt_ms(ev.get("duration_ms")),
                    "Detail": meta.get("document_name") or meta.get("processor_type") or "-",
                })
            render_paginated_table(event_rows, key=f"metrics_recent_{section_key}", page_size=5)


def render_metrics_page(request_fn: Any) -> None:
    st.subheader("RAG Benchmark Metrics")
    params = _render_filters()
    render_page_size_selector("metrics")

    try:
        dashboard = request_fn("GET", "/metrics/dashboard", params=params)
    except Exception as exc:
        st.error(f"Failed to load metrics dashboard: {exc}")
        return

    benchmark = dashboard.get("benchmark") or {}
    overview = dashboard.get("overview") or {}
    sections = dashboard.get("sections") or {}
    recent = dashboard.get("recent") or {}
    section_labels = dashboard.get("section_labels") or {}

    _render_benchmark_catalog(benchmark)
    st.divider()

    st.markdown("**Activity summary**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Benchmark Events", _fmt_int(overview.get("count")))
    c2.metric("Success Rate", _fmt_pct(overview.get("success_rate")))
    c3.metric("Failures", _fmt_int(overview.get("failure_count")))
    c4.metric("Avg Latency", _fmt_ms(overview.get("avg_duration_ms")))

    if int(overview.get("count") or 0) == 0:
        st.info(
            "No benchmark metric events in the selected date range yet. "
            "Upload and process a document, run retrieval, or dispatch scheduler jobs to populate values."
        )
        return

    st.divider()
    st.markdown("**By component**")
    for key, label in section_labels.items():
        _render_section_summary(key, label, sections.get(key) or {}, recent.get(key) or [])
