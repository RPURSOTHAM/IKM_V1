"""Reusable pagination controls for Streamlit observability tables."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

DEFAULT_PAGE_SIZE = 10
PAGE_SIZE_OPTIONS = (10, 25, 50)


def _total_pages(total: int, page_size: int) -> int:
    if total <= 0:
        return 1
    return max(1, (total + page_size - 1) // page_size)


def _page_key(key: str) -> str:
    return f"{key}_page"


def reset_page(key: str) -> None:
    st.session_state[_page_key(key)] = 1


def render_page_size_selector(key: str, *, label: str = "Rows per page") -> int:
    size_key = f"{key}_page_size"
    if size_key not in st.session_state:
        st.session_state[size_key] = DEFAULT_PAGE_SIZE
    selected = st.selectbox(
        label,
        options=list(PAGE_SIZE_OPTIONS),
        index=list(PAGE_SIZE_OPTIONS).index(st.session_state[size_key]),
        key=size_key,
    )
    return int(selected)


def render_pagination_bar(
    *,
    key: str,
    page: int,
    page_size: int,
    total: int,
    label: str = "rows",
) -> int:
    """Render prev/next controls. Returns the active page number."""
    total_pages = _total_pages(total, page_size)
    page = max(1, min(page, total_pages))
    st.session_state[_page_key(key)] = page

    start = (page - 1) * page_size + 1 if total else 0
    end = min(page * page_size, total)

    nav1, nav2, nav3 = st.columns([1, 3, 1])
    with nav1:
        if st.button("Previous", key=f"{key}_prev", disabled=page <= 1, use_container_width=True):
            st.session_state[_page_key(key)] = page - 1
            st.rerun()
    with nav2:
        st.caption(
            f"Page {page} of {total_pages} | "
            f"Showing {start:,}-{end:,} of {total:,} {label}"
        )
    with nav3:
        if st.button("Next", key=f"{key}_next", disabled=page >= total_pages, use_container_width=True):
            st.session_state[_page_key(key)] = page + 1
            st.rerun()

    return page


def get_page(key: str) -> int:
    return max(1, int(st.session_state.get(_page_key(key), 1)))


def render_text_table(rows: list[dict[str, Any]], *, hide_columns: set[str] | None = None) -> None:
    if not rows:
        st.info("No data available yet")
        return
    display_rows = []
    for row in rows:
        item = dict(row)
        for col in hide_columns or set():
            item.pop(col, None)
        display_rows.append(item)
    df = pd.DataFrame(display_rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={col: st.column_config.TextColumn(col, help=None) for col in df.columns},
    )


def paginate_list(rows: list[dict[str, Any]], *, key: str, page_size: int) -> list[dict[str, Any]]:
    if not rows:
        return []
    page = get_page(key)
    total_pages = _total_pages(len(rows), page_size)
    page = max(1, min(page, total_pages))
    st.session_state[_page_key(key)] = page
    start = (page - 1) * page_size
    return rows[start : start + page_size]


def render_paginated_table(
    rows: list[dict[str, Any]],
    *,
    key: str,
    page_size: int | None = None,
    hide_columns: set[str] | None = None,
    row_label: str = "rows",
) -> list[dict[str, Any]]:
    """Render a paginated plain-text table. Returns rows on the current page."""
    if not rows:
        st.info("No data available yet")
        return []

    size = page_size or int(st.session_state.get(f"{key}_page_size", DEFAULT_PAGE_SIZE))
    page = render_pagination_bar(key=key, page=get_page(key), page_size=size, total=len(rows), label=row_label)
    page_rows = paginate_list(rows, key=key, page_size=size)
    render_text_table(page_rows, hide_columns=hide_columns)
    return page_rows
