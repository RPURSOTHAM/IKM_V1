"""Application-level usage metrics for integrated RAG clients."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import streamlit as st

from src.features.observability.streamlit.pagination import render_paginated_table


def _number(value: Any) -> int:
    return int(value or 0)


def render_application_usage_page(request_fn: Any) -> None:
    st.subheader("Application Usage")

    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            application_name = st.text_input(
                "Application", key="usage_application", placeholder="All applications"
            ).strip()
        with c2:
            date_from = st.date_input(
                "From",
                value=datetime.utcnow().date() - timedelta(days=7),
                key="usage_date_from",
            )
        with c3:
            date_to = st.date_input(
                "To",
                value=datetime.utcnow().date(),
                key="usage_date_to",
            )

    params: dict[str, str] = {
        "date_from": f"{date_from.isoformat()}T00:00:00",
        "date_to": f"{date_to.isoformat()}T23:59:59",
    }
    if application_name:
        params["application_name"] = application_name

    try:
        payload = request_fn("GET", "/metrics/applications", params=params)
    except Exception as exc:
        st.error(f"Failed to load application usage: {exc}")
        return

    applications = payload.get("applications") or []
    totals = payload.get("totals") or {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Applications", _number(totals.get("application_count")))
    c2.metric("Documents Uploaded", _number(totals.get("documents_uploaded")))
    c3.metric("Search Requests", _number(totals.get("search_requests")))
    c4.metric("AI Queries", _number(totals.get("ai_queries")))
    c5.metric("Failed Requests", _number(totals.get("failed_requests")))

    if not applications:
        st.info("No application usage found for the selected date range.")
        return

    rows = [
        {
            "Application": row.get("application_name") or "unknown",
            "Total Requests": _number(row.get("total_requests")),
            "Documents Uploaded": _number(row.get("documents_uploaded")),
            "Search Requests": _number(row.get("search_requests")),
            "AI Queries": _number(row.get("ai_queries")),
            "Documents Deleted": _number(row.get("documents_deleted")),
            "Failed Requests": _number(row.get("failed_requests")),
            "Active Users": _number(row.get("active_users")),
        }
        for row in applications
    ]
    render_paginated_table(rows, key="application_usage", page_size=25)
