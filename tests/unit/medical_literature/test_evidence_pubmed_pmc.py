"""Unit tests for PMC / Europe PMC helpers (new pubmed_pmc module)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.features.medical_literature.providers.pubmed_pmc import (
    PmcPdfAvailability,
    batch_resolve_pmc_pdf_availability,
    normalize_pmcid,
    pmc_xml_to_plain_text,
)


def test_normalize_pmcid() -> None:
    assert normalize_pmcid("123") == "PMC123"
    assert normalize_pmcid("pmc999") == "PMC999"
    assert normalize_pmcid(None) is None


def test_pmc_xml_to_plain_text() -> None:
    xml = """<?xml version="1.0"?>
    <article>
      <article-title>Hello</article-title>
      <abstract><p>Abstract body</p></abstract>
      <body><p>Full text paragraph</p></body>
    </article>
    """
    text = pmc_xml_to_plain_text(xml)
    assert "Hello" in text
    assert "Abstract body" in text


def test_batch_resolve_uses_oa_urls() -> None:
    with (
        patch(
            "src.features.medical_literature.providers.pubmed_pmc.fetch_ncbi_cloud_pdf_urls",
            return_value=[],
        ),
        patch(
            "src.features.medical_literature.providers.pubmed_pmc.fetch_oa_pdf_urls",
            return_value=[],
        ),
        patch(
            "src.features.medical_literature.providers.pubmed_pmc.discover_pmc_pdf_urls",
            return_value=[],
        ),
    ):
        result = batch_resolve_pmc_pdf_availability({"1": "PMC1"})
    assert isinstance(result["1"], PmcPdfAvailability)
    assert result["1"].has_pdf is False


def test_fetch_oa_pdf_urls_treats_404_as_empty() -> None:
    import requests
    from src.features.medical_literature.providers.pubmed_pmc import fetch_oa_pdf_urls

    response = MagicMock()
    response.status_code = 404
    err = requests.exceptions.HTTPError(response=response)
    err.response = response

    with patch(
        "src.features.medical_literature.providers.pubmed_pmc.request_with_retry",
        side_effect=err,
    ):
        assert fetch_oa_pdf_urls("PMC13524455") == []

