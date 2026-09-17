"""Unit tests for MeSH / field query builders."""

from __future__ import annotations

import pytest

from src.features.medical_literature.taxonomy.mesh_service import (
    build_clinicaltrials_title_query,
    build_ikm_pubmed_query,
    build_pubmed_search_query,
    format_pubmed_mesh_query,
    format_pubmed_title_query,
    normalize_mesh_terms,
    title_matches_query,
)


def test_normalize_mesh_terms_list_and_string() -> None:
    assert normalize_mesh_terms("Diabetes Mellitus; Hypoglycemia") == [
        "Diabetes Mellitus",
        "Hypoglycemia",
    ]
    assert normalize_mesh_terms(["A", "A", "B"]) == ["A", "B"]


def test_format_mesh_and_title() -> None:
    assert format_pubmed_mesh_query("Diabetes Mellitus") == '"Diabetes Mellitus"[MeSH Terms]'
    assert format_pubmed_mesh_query('"Cancer"[MeSH Terms]') == '"Cancer"[MeSH Terms]'
    assert format_pubmed_title_query("metformin").endswith("[Title]")


def test_build_mesh_only_and_mixed() -> None:
    mesh_only = build_pubmed_search_query(["Diabetes"], mesh_mode=True)
    assert "[MeSH Terms]" in mesh_only
    mixed = build_ikm_pubmed_query(
        query="metformin",
        mesh_terms=["Diabetes Mellitus"],
        title=None,
    )
    assert "metformin" in mixed.lower() or "Title" in mixed
    assert "MeSH" in mixed


def test_ikm_query_requires_input() -> None:
    with pytest.raises(ValueError):
        build_ikm_pubmed_query(query="", mesh_terms=[], title=None)


def test_clinicaltrials_title_helpers() -> None:
    q = build_clinicaltrials_title_query(["diabetes", "metformin"])
    assert "diabetes" in q.lower()
    assert title_matches_query("Metformin for Diabetes Mellitus", ["diabetes"])
    assert not title_matches_query("Asthma study", ["diabetes"])
