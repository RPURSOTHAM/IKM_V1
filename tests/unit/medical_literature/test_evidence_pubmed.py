"""Unit tests for the new PubMed client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.features.medical_literature.providers.pubmed_client import (
    PubMedArticle,
    PubMedClient,
    PubMedSearchPage,
    PubMedSummary,
)
from src.features.medical_literature.taxonomy.mesh_service import (
    build_ikm_pubmed_query,
    build_pubmed_search_query,
)


def test_build_ikm_pubmed_query_mesh_and_title() -> None:
    q = build_ikm_pubmed_query(
        query="",
        mesh_terms=["Diabetes Mellitus", "Hypoglycemia"],
        title="metformin",
    )
    assert '"Diabetes Mellitus"[MeSH Terms]' in q
    assert "metformin" in q


def test_build_pubmed_search_query_mesh_mode() -> None:
    q = build_pubmed_search_query(["Diabetes Mellitus"], mesh_mode=True)
    assert '"Diabetes Mellitus"[MeSH Terms]' in q


def test_build_ikm_requires_input() -> None:
    with pytest.raises(ValueError):
        build_ikm_pubmed_query(query="", mesh_terms=[], title=None)


def test_pubmed_search_page_and_summaries() -> None:
    client = PubMedClient(base_url="https://example.test/eutils", api_key="k", email="a@b.c")
    page = PubMedSearchPage(pmids=["12345678"], total_count=1, retstart=0)
    summary = PubMedSummary(
        pmid="12345678",
        title="Example diabetes article",
        journal="NEJM",
        year="2024",
        published_date="2024 Jan",
        url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
    )
    with (
        patch.object(client, "search_page", return_value=page) as search_page,
        patch.object(client, "fetch_summaries", return_value=[summary]) as fetch_summaries,
    ):
        pmids = client.search(["diabetes"], retmax=5)
        summaries = client.fetch_summaries(pmids)
    search_page.assert_called_once()
    fetch_summaries.assert_called_once_with(["12345678"])
    assert pmids == ["12345678"]
    assert summaries[0].title.startswith("Example")


def test_article_to_document_text() -> None:
    article = PubMedArticle(
        pmid="123",
        title="Title",
        abstract="Abstract body",
        authors="Doe J",
        journal="J",
        year="2024",
        mesh_terms=["Diabetes Mellitus"],
        url="https://pubmed.ncbi.nlm.nih.gov/123/",
    )
    text = article.to_document_text()
    assert "PMID: 123" in text
    assert "Abstract body" in text
    assert "Diabetes Mellitus" in text


def test_search_requires_terms() -> None:
    client = PubMedClient(base_url="https://example.test/eutils")
    page = client.search_page([])
    assert page.pmids == []
    assert page.total_count == 0
