"""Streamlit UI for Observability: Audit Logs and Metrics."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.features.observability.streamlit.application_usage_page import render_application_usage_page
from src.features.observability.streamlit.audit_page import render_audit_page
from src.features.observability.streamlit.metrics_page import render_metrics_page


def render_observability_page(request_fn: Any, api_base_url: str) -> None:
    st.subheader("Observability")
    st.caption("Audit trail and operational metrics for the RAG platform.")

    tab1, tab2, tab3 = st.tabs(["Audit Logs", "Application Usage", "System Metrics"])

    with tab1:
        render_audit_page(request_fn, api_base_url)

    with tab2:
        render_application_usage_page(request_fn)

    with tab3:
        render_metrics_page(request_fn)
