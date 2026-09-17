"""Streamlit audit trail page — business timeline view."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any

import requests
import streamlit as st

from src.features.observability.audit.domain.audit_models import BUSINESS_EVENT_TYPE_PREFIXES
from src.features.observability.streamlit.pagination import (
    get_page,
    render_page_size_selector,
    render_pagination_bar,
    render_text_table,
    reset_page,
)

# Business events shown by default — technical scheduler/processor internals are hidden
_BUSINESS_EVENT_PREFIXES = BUSINESS_EVENT_TYPE_PREFIXES

_CATEGORY_LABELS = {
    "repository": "Repository",
    "document": "Document",
    "processing": "Processing",
    "retrieval": "Search",
    "authentication": "Authentication",
    "access_control": "Access Control",
    "scheduler": "Scheduler",
    "application": "Application Request",
}

_EVENT_LABELS: dict[str, str] = {
    "REPOSITORY_CREATED": "Repository Created",
    "REPOSITORY_SETTINGS_UPDATED": "Repository Settings Updated",
    "REPOSITORY_ACTIVATED": "Repository Activated",
    "REPOSITORY_DEACTIVATED": "Repository Deactivated",
    "REPOSITORY_STATUS_CHANGED": "Repository Status Changed",
    "REPOSITORY_DELETED": "Repository Deleted",
    "DOCUMENT_TYPE_CREATED": "Document Type Created",
    "DOCUMENT_TYPE_UPDATED": "Document Type Updated",
    "DOCUMENT_TYPE_DELETED": "Document Type Deleted",
    "DOCUMENT_KEY_FIELD_CREATED": "Key Field Created",
    "DOCUMENT_KEY_FIELD_UPDATED": "Key Field Updated",
    "DOCUMENT_KEY_FIELD_DELETED": "Key Field Deleted",
    "DOCUMENT_METADATA_FIELD_CREATED": "Metadata Field Created",
    "DOCUMENT_METADATA_FIELD_UPDATED": "Metadata Field Updated",
    "DOCUMENT_METADATA_FIELD_DELETED": "Metadata Field Deleted",
    "DOCUMENT_UPLOADED": "Document Uploaded",
    "DOCUMENT_SUBMITTED": "Document Submitted for Processing",
    "DOCUMENT_REPROCESSED": "Document Reprocessed",
    "DOCUMENT_DELETED": "Document Deleted",
    "DOCUMENT_JOB_DISPATCHED": "Processing Job Started",
    "DOCUMENT_JOB_FAILED": "Processing Job Failed",
    "DOCUMENT_PROCESSING_STARTED": "Processing Started",
    "DOCUMENT_PROCESSING_COMPLETED": "Processing Completed",
    "DOCUMENT_PROCESSING_FAILED": "Processing Failed",
    "SEARCH_EXECUTED": "Search Executed",
    "SEARCH_COMPLETED": "Search Completed",
    "SEARCH_NO_RESULTS": "Search Returned No Results",
    "SEARCH_FAILED": "Search Failed",
    "CITATION_GENERATED": "Citation Generated",
    "USER_LOGIN": "User Logged In",
    "USER_LOGIN_FAILED": "Login Failed",
    "USER_PASSWORD_CHANGED": "Password Changed",
    "PERMISSION_GRANTED": "Permission Granted",
    "PERMISSION_REVOKED": "Permission Revoked",
}


def _is_business_event(event_type: str) -> bool:
    return any(event_type.upper().startswith(prefix) for prefix in _BUSINESS_EVENT_PREFIXES)


def _format_event_label(event_type: str) -> str:
    return _EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())


def _format_category(category: str | None) -> str:
    if not category:
        return "-"
    return _CATEGORY_LABELS.get(category, category.replace("_", " ").title())


def _format_ts(ts: str | None) -> str:
    if not ts:
        return "-"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except ValueError:
        return ts


def _format_status(status: str | None) -> str:
    if not status:
        return "-"
    return status.strip().title()


def _entity_display(item: dict[str, Any]) -> str:
    repo_name = item.get("repository_name")
    doc_name = item.get("document_name")
    entity_id = item.get("entity_id") or ""
    if repo_name and doc_name:
        return f"{repo_name} / {doc_name}"
    if repo_name:
        return repo_name
    if doc_name:
        return doc_name
    meta = item.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("repository_id") and entity_id:
        return entity_id
    return entity_id or "-"


def _user_display(item: dict[str, Any]) -> str:
    meta = item.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    nested = meta.get("user") if isinstance(meta.get("user"), dict) else {}
    display_name = meta.get("display_name") or nested.get("display_name")
    username = meta.get("username") or nested.get("username")
    role = meta.get("role") or nested.get("role")
    user_id = item.get("user_id") or meta.get("user_id") or nested.get("user_id")
    label = display_name or username or user_id or "-"
    if role and label != "-":
        return f"{label} ({role})"
    return str(label)


def _row_from_item(item: dict[str, Any]) -> dict[str, Any]:
    meta = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "Time": _format_ts(item.get("timestamp")),
        "Application": item.get("application_name") or meta.get("application_name") or "unknown",
        "Category": _format_category(item.get("category")),
        "Action": _format_event_label(item.get("event_type", "")),
        "Repository / Document": _entity_display(item),
        "User": _user_display(item),
        "Status": _format_status(item.get("status")),
        "Duration": f"{round(item.get('duration_ms'), 1)} ms" if item.get("duration_ms") is not None else "-",
        "Request ID": item.get("request_id") or meta.get("request_id") or "-",
        "_id": item.get("id"),
    }


def render_audit_page(request_fn: Any, api_base: str) -> None:
    st.subheader("Business Audit Trail")
    st.caption("User-facing actions across repositories, documents, search, and access control.")

    with st.expander("Filters", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            search = st.text_input("Search events", key="audit_search", placeholder="repository name, document, user")
        with col2:
            application_name = st.text_input(
                "Application", key="audit_application", placeholder="All applications"
            ).strip()
        with col3:
            category_options = [
                "",
                "repository",
                "document",
                "processing",
                "retrieval",
                "authentication",
                "access_control",
                "application",
            ]
            category = st.selectbox(
                "Category",
                category_options,
                format_func=lambda c: "All Categories" if not c else _format_category(c),
                key="audit_category",
            )
        with col4:
            status = st.selectbox(
                "Status",
                ["", "success", "failure"],
                format_func=lambda s: "All Statuses" if not s else s.title(),
                key="audit_status",
            )

        col4, col5 = st.columns(2)
        with col4:
            date_from = st.date_input(
                "From", value=datetime.utcnow().date() - timedelta(days=7), key="audit_from"
            )
        with col5:
            date_to = st.date_input("To", value=datetime.utcnow().date(), key="audit_to")

        show_technical = st.checkbox(
            "Show technical / internal events",
            value=False,
            key="audit_show_technical",
            help="When disabled, scheduler internals, processor lifecycle events, and generic API calls are hidden.",
        )

    filter_signature = (
        search,
        application_name,
        category,
        status,
        str(date_from),
        str(date_to),
        show_technical,
    )
    if st.session_state.get("audit_filter_signature") != filter_signature:
        st.session_state["audit_filter_signature"] = filter_signature
        reset_page("audit_events")

    page_size = render_page_size_selector("audit_events")
    page = get_page("audit_events")

    params: dict[str, Any] = {
        "page": page,
        "page_size": page_size,
        "date_from": f"{date_from.isoformat()}T00:00:00",
        "date_to": f"{date_to.isoformat()}T23:59:59",
    }
    if search:
        params["search"] = search
    if application_name:
        params["application_name"] = application_name
    if status:
        params["status"] = status
    if category:
        params["category"] = category
    if not show_technical:
        params["business_only"] = True

    try:
        payload = request_fn("GET", "/audit/events", params=params)
    except requests.HTTPError as exc:
        st.error(f"Failed to load audit events: {exc}")
        return

    all_items: list[dict[str, Any]] = payload.get("items") or []
    api_total = int(payload.get("total") or 0)
    items = all_items
    if not show_technical:
        items = [i for i in all_items if _is_business_event(i.get("event_type", ""))]

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Total Events", api_total)
    col_b.metric("On This Page", len(items))
    col_c.metric("Success", sum(1 for i in items if i.get("status") == "success"))
    col_d.metric("Failures", sum(1 for i in items if i.get("status") == "failure"))

    st.divider()

    if not items and api_total == 0:
        st.info("No events found for the selected filters.")
        return

    if not items:
        st.info("No matching events on this page. Try another page or adjust the filters.")
        render_pagination_bar(
            key="audit_events",
            page=page,
            page_size=page_size,
            total=api_total,
            label="events",
        )
        return

    table_rows = [_row_from_item(item) for item in items]
    render_pagination_bar(
        key="audit_events",
        page=page,
        page_size=page_size,
        total=api_total,
        label="events",
    )
    render_text_table(table_rows, hide_columns={"_id"})

    st.divider()
    st.markdown("**Event details**")
    event_options = [
        f"{r['Time']} | {r['Action']} | {r['Status']} | {r['User']} [ID: {r['_id']}]"
        for r in table_rows
    ]
    selected_label = st.selectbox(
        "Select an audit event to inspect details and metadata:",
        [""] + event_options,
        key="audit_event_inspector_select",
    )

    if not selected_label:
        return

    match = re.search(r"\[ID:\s*(\d+)\]", selected_label)
    if not match:
        return

    selected_id = int(match.group(1))
    selected_item = next((x for x in items if x.get("id") == selected_id), None)
    if not selected_item:
        return

    meta = selected_item.get("metadata") if isinstance(selected_item.get("metadata"), dict) else {}
    nested_user = meta.get("user") if isinstance(meta.get("user"), dict) else {}

    st.markdown(f"**Event ID:** `{selected_id}`")
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown(f"**Event Type:** `{selected_item.get('event_type')}`")
        st.markdown(f"**Application:** `{selected_item.get('application_name') or meta.get('application_name') or 'unknown'}`")
        st.markdown(f"**Category:** `{_format_category(selected_item.get('category'))}`")
        st.markdown(f"**Timestamp:** `{_format_ts(selected_item.get('timestamp'))}`")
        st.markdown(f"**Source:** `{selected_item.get('source') or meta.get('source') or '-'}`")
    with col_i2:
        st.markdown(f"**User:** `{_user_display(selected_item)}`")
        st.markdown(f"**Username:** `{meta.get('username') or nested_user.get('username') or '-'}`")
        st.markdown(f"**Role:** `{meta.get('role') or nested_user.get('role') or '-'}`")
        st.markdown(f"**Auth Provider:** `{meta.get('authentication_provider') or nested_user.get('authentication_provider') or '-'}`")
    with col_i3:
        st.markdown(f"**Entity Type:** `{selected_item.get('entity_type') or '-'}`")
        st.markdown(f"**Entity ID:** `{selected_item.get('entity_id') or '-'}`")
        st.markdown(f"**Request ID:** `{selected_item.get('request_id') or meta.get('request_id') or '-'}`")
        st.markdown(f"**Correlation ID:** `{selected_item.get('correlation_id') or meta.get('correlation_id') or '-'}`")

    col_u1, col_u2, col_u3 = st.columns(3)
    with col_u1:
        st.markdown(f"**Client IP:** `{meta.get('client_ip') or nested_user.get('client_ip') or '-'}`")
    with col_u2:
        st.markdown(f"**Session ID:** `{meta.get('session_id') or nested_user.get('session_id') or '-'}`")
    with col_u3:
        user_agent = meta.get("user_agent") or nested_user.get("user_agent") or "-"
        st.markdown(f"**User Agent:** `{str(user_agent)[:120]}`")

    if selected_item.get("status") == "failure":
        st.markdown("**Error**")
        st.error(selected_item.get("exception") or "Unknown execution failure")
        if selected_item.get("stacktrace"):
            with st.expander("View Stack Trace"):
                st.code(selected_item.get("stacktrace"))

    if meta.get("change_summary"):
        st.markdown("**Change Summary**")
        st.info(str(meta.get("change_summary")))

    changes = selected_item.get("changes") or []
    if changes:
        st.markdown("**Field Changes**")
        change_rows = []
        for change in changes:
            label = change.get("label") or change.get("field")
            before = change.get("old_display", change.get("old"))
            after = change.get("new_display", change.get("new"))
            change_rows.append({"Field": label, "Before": before, "After": after})
        st.dataframe(
            change_rows,
            use_container_width=True,
            hide_index=True,
            column_config={col: st.column_config.TextColumn(col) for col in ("Field", "Before", "After")},
        )

    if meta:
        st.markdown("**Metadata**")
        st.json(meta)
