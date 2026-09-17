"""PubMed Central (PMC) open-access PDF resolution and download."""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

from src.features.medical_literature.taxonomy.mesh_service import format_clinicaltrials_query
from src.features.medical_literature.providers.pubmed_http import (
    configure_ncbi_session,
    ncbi_session_headers,
    normalize_entrez_base_url,
    request_with_retry,
)

_logger = logging.getLogger(__name__)

_IDCONV_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
_IDCONV_BATCH_SIZE = 10
_IDCONV_BATCH_PAUSE_SECONDS = 0.34
_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
_PMC_CLOUD_BASE = "https://pmc-oa-opendata.s3.amazonaws.com"
_EUROPEPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


@dataclass(frozen=True)
class EuropePmcSearchResult:
    pmid: str
    pmcid: str
    title: str
    journal: str
    published_date: str
    pdf_urls: tuple[str, ...]
    url: str


def search_europepmc_medline(
    mesh_terms: list[str],
    *,
    page_size: int = 25,
    open_access_only: bool = False,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> list[EuropePmcSearchResult]:
    """Search Europe PMC MEDLINE records (fallback when NCBI E-utilities is unavailable)."""
    terms = [term.strip() for term in mesh_terms if term.strip()]
    if not terms:
        return []

    query_terms = " OR ".join(f"({format_clinicaltrials_query(term)})" for term in terms)
    query = f"({query_terms}) AND SRC:MED"
    if open_access_only:
        query = f"{query} AND OPEN_ACCESS:Y"

    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    response = request_with_retry(
        http,
        "GET",
        _EUROPEPMC_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": str(max(1, min(page_size, 100))),
        },
        timeout=60,
    )

    results: list[EuropePmcSearchResult] = []
    for record in (response.json().get("resultList") or {}).get("result") or []:
        if not isinstance(record, dict):
            continue
        pmid = str(record.get("pmid") or "").strip()
        if not pmid:
            continue
        pmcid = normalize_pmcid(str(record.get("pmcid") or "").strip() or None)
        pdf_urls = tuple(_europepmc_pdf_urls(record))
        title = str(record.get("title") or f"PubMed {pmid}").strip()
        journal = str(record.get("journalTitle") or "").strip()
        published_date = str(record.get("firstPublicationDate") or record.get("pubYear") or "").strip()
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        results.append(
            EuropePmcSearchResult(
                pmid=pmid,
                pmcid=pmcid or "",
                title=title,
                journal=journal,
                published_date=published_date,
                pdf_urls=pdf_urls,
                url=pubmed_url,
            )
        )
    return results


def search_europepmc_open_access(
    mesh_terms: list[str],
    *,
    page_size: int = 25,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> list[EuropePmcSearchResult]:
    """Search Europe PMC for open-access PubMed records with downloadable PDFs."""
    terms = [term.strip() for term in mesh_terms if term.strip()]
    if not terms:
        return []

    query_terms = " OR ".join(f"({format_clinicaltrials_query(term)})" for term in terms)
    query = f"({query_terms}) AND OPEN_ACCESS:Y AND SRC:MED"
    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    response = request_with_retry(
        http,
        "GET",
        _EUROPEPMC_SEARCH_URL,
        params={
            "query": query,
            "format": "json",
            "resultType": "core",
            "pageSize": str(max(1, min(page_size, 100))),
        },
        timeout=60,
    )

    results: list[EuropePmcSearchResult] = []
    for record in (response.json().get("resultList") or {}).get("result") or []:
        if not isinstance(record, dict):
            continue
        pmid = str(record.get("pmid") or "").strip()
        pmcid = normalize_pmcid(str(record.get("pmcid") or "").strip() or None)
        pdf_urls = tuple(_europepmc_pdf_urls(record))
        if not pmid or not pmcid or not pdf_urls:
            continue
        title = str(record.get("title") or f"PubMed {pmid}").strip()
        journal = str(record.get("journalTitle") or "").strip()
        published_date = str(record.get("firstPublicationDate") or record.get("pubYear") or "").strip()
        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        results.append(
            EuropePmcSearchResult(
                pmid=pmid,
                pmcid=pmcid,
                title=title,
                journal=journal,
                published_date=published_date,
                pdf_urls=pdf_urls,
                url=pubmed_url,
            )
        )
    return results


@dataclass(frozen=True)
class PmcResolution:
    pmid: str
    pmcid: str | None
    doi: str | None
    is_open_access: bool


@dataclass(frozen=True)
class PmcPdfAvailability:
    pmid: str
    pmcid: str | None
    has_pdf: bool
    pdf_urls: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PmcPdfResult:
    pmid: str
    pmcid: str
    filename: str
    content: bytes
    source_url: str


def normalize_pmcid(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = str(value).strip().upper()
    if not cleaned:
        return None
    if cleaned.startswith("PMC"):
        return cleaned
    if cleaned.isdigit():
        return f"PMC{cleaned}"
    return cleaned


def _session_headers(email: str = "", tool: str = "rag-builder-pubmed") -> dict[str, str]:
    return ncbi_session_headers(email=email, tool=tool)


def _ftp_to_https(url: str) -> str:
    return re.sub(r"^ftp://", "https://", url.strip(), flags=re.IGNORECASE)


def _s3_url_to_https(url: str) -> str | None:
    cleaned = str(url or "").strip()
    if cleaned.startswith("s3://pmc-oa-opendata/"):
        path = cleaned[len("s3://pmc-oa-opendata/") :].split("?")[0].lstrip("/")
        if path:
            return f"{_PMC_CLOUD_BASE}/{path}"
    return None


def _cloud_pdf_candidates(pmcid: str) -> list[str]:
    normalized = normalize_pmcid(pmcid)
    if not normalized:
        return []
    prefix = f"{normalized}.1"
    return [f"{_PMC_CLOUD_BASE}/{prefix}/{prefix}.pdf"]


def _is_ncbi_pdf_url(url: str) -> bool:
    lowered = str(url or "").strip().lower()
    if not lowered:
        return False
    return (
        "ncbi.nlm.nih.gov" in lowered
        or lowered.startswith(_PMC_CLOUD_BASE.lower())
    )


def _is_pdf_response(response: requests.Response) -> bool:
    content_type = str(response.headers.get("Content-Type") or "").lower()
    return "pdf" in content_type or response.content.startswith(b"%PDF")


def resolve_pmcids(
    pmids: list[str],
    *,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> dict[str, PmcResolution]:
    """Map PMIDs to PMCIDs using the NCBI ID Converter (batched)."""
    cleaned = [str(item).strip() for item in pmids if str(item).strip()]
    if not cleaned:
        return {}

    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    result: dict[str, PmcResolution] = {}
    for start in range(0, len(cleaned), _IDCONV_BATCH_SIZE):
        batch = cleaned[start : start + _IDCONV_BATCH_SIZE]
        params: dict[str, str] = {
            "ids": ",".join(batch),
            "format": "json",
            "tool": tool,
            "idtype": "pmid",
            "versions": "no",
        }
        if email:
            params["email"] = email

        response = request_with_retry(
            http,
            "GET",
            _IDCONV_URL,
            params=params,
            timeout=60,
        )
        payload = response.json()
        records = payload.get("records") or []
        for record in records:
            if not isinstance(record, dict):
                continue
            pmid = str(record.get("pmid") or record.get("requested-id") or "").strip()
            if not pmid:
                continue
            pmcid = normalize_pmcid(str(record.get("pmcid") or "").strip() or None)
            doi = str(record.get("doi") or "").strip() or None
            status = str(record.get("status") or "").strip().lower()
            is_oa = bool(pmcid) and status not in {"error", "failed"}
            result[pmid] = PmcResolution(
                pmid=pmid,
                pmcid=pmcid,
                doi=doi,
                is_open_access=is_oa,
            )
        if start + _IDCONV_BATCH_SIZE < len(cleaned):
            time.sleep(_IDCONV_BATCH_PAUSE_SECONDS)
    return result


def fetch_oa_pdf_urls(
    pmcid: str,
    *,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> list[str]:
    """Return OA PDF URLs from the NCBI PMC OA service when present.

    A 404 from oa.fcgi means the PMCID is not in the OA subset — treat as no PDF.
    """
    normalized = normalize_pmcid(pmcid)
    if not normalized:
        return []

    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    try:
        response = request_with_retry(
            http,
            "GET",
            _OA_URL,
            params={"id": normalized},
            timeout=60,
        )
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        if status == 404:
            _logger.debug("PMC OA lookup 404 for %s (not open-access)", normalized)
            return []
        raise
    except requests.exceptions.RequestException as exc:
        _logger.warning("PMC OA lookup failed for %s: %s", normalized, exc)
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        _logger.debug("PMC OA response was not valid XML for %s", normalized)
        return []

    urls: list[str] = []
    for link in root.findall(".//link"):
        if str(link.attrib.get("format") or "").lower() != "pdf":
            continue
        href = str(link.attrib.get("href") or "").strip()
        if href:
            urls.append(_ftp_to_https(href))
    return urls


def fetch_ncbi_cloud_pdf_urls(
    pmcid: str,
    *,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> list[str]:
    """Return downloadable OA PDF URLs from the NCBI PMC AWS cloud service."""
    normalized = normalize_pmcid(pmcid)
    if not normalized:
        return []

    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    oa_links = fetch_oa_pdf_urls(normalized, session=http, email=email, tool=tool)
    if not oa_links:
        return []

    urls: list[str] = []
    prefix = f"{normalized}.1"
    meta_url = f"{_PMC_CLOUD_BASE}/{prefix}/{prefix}.json"
    try:
        response = request_with_retry(http, "GET", meta_url, timeout=30)
        meta = response.json()
        for key in ("pdf_url", "pdfUrl"):
            https_url = _s3_url_to_https(str(meta.get(key) or ""))
            if https_url:
                urls.append(https_url)
    except Exception:
        _logger.debug("PMC cloud metadata unavailable for %s", normalized)

    urls.extend(_cloud_pdf_candidates(normalized))
    return list(dict.fromkeys(url for url in urls if url))


def _europepmc_pdf_urls(record: dict) -> list[str]:
    urls: list[str] = []
    pmcid = normalize_pmcid(str(record.get("pmcid") or "").strip() or None)

    full_text_urls = (record.get("fullTextUrlList") or {}).get("fullTextUrl") or []
    if isinstance(full_text_urls, dict):
        full_text_urls = [full_text_urls]
    for item in full_text_urls:
        if not isinstance(item, dict):
            continue
        if str(item.get("documentStyle") or "").lower() != "pdf":
            continue
        if str(item.get("availabilityCode") or "").upper() not in {"OA", "OPEN_ACCESS"}:
            continue
        url = str(item.get("url") or "").strip()
        if url and url not in urls:
            urls.append(url)

    if pmcid and urls:
        render_url = f"https://europepmc.org/articles/{pmcid}?pdf=render"
        backend_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        for extra in (render_url, backend_url):
            if extra not in urls:
                urls.append(extra)
    return urls


def discover_pmc_pdf_urls(
    pmcid: str,
    *,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> list[str]:
    """Parse the PMC article page for direct PDF download links (like the PMC UI)."""
    normalized = normalize_pmcid(pmcid)
    if not normalized:
        return []

    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    base = f"https://pmc.ncbi.nlm.nih.gov/articles/{normalized}/"
    try:
        response = http.get(base, timeout=60, headers=_session_headers(email, tool))
        response.raise_for_status()
    except Exception:
        _logger.exception("PMC article page fetch failed for %s", normalized)
        return []

    relative_links = re.findall(r'href="((?:pdf/)?[^"]+\.pdf)"', response.text, flags=re.IGNORECASE)
    urls: list[str] = []
    for rel in dict.fromkeys(relative_links):
        urls.append(f"{base.rstrip('/')}/{rel.lstrip('/')}")
    return urls


def batch_resolve_pmc_pdf_availability(
    pmid_to_pmcid: dict[str, str | None],
    *,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> dict[str, PmcPdfAvailability]:
    """Batch-check which PMIDs have downloadable OA PDFs via NCBI PMC services."""
    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    availability: dict[str, PmcPdfAvailability] = {}

    for pmid, raw_pmcid in pmid_to_pmcid.items():
        pmcid = normalize_pmcid(raw_pmcid)
        pdf_urls: list[str] = []
        if pmcid:
            try:
                pdf_urls.extend(
                    fetch_ncbi_cloud_pdf_urls(
                        pmcid,
                        session=http,
                        email=email,
                        tool=tool,
                    )
                )
            except Exception as exc:  # noqa: BLE001 — never fail search over one PMC miss
                _logger.warning("PMC PDF resolve failed pmid=%s pmcid=%s: %s", pmid, pmcid, exc)
        deduped = tuple(dict.fromkeys(url for url in pdf_urls if _is_ncbi_pdf_url(url)))
        availability[pmid] = PmcPdfAvailability(
            pmid=pmid,
            pmcid=pmcid,
            has_pdf=bool(deduped),
            pdf_urls=deduped,
        )
    return availability


def pmc_pdf_url(pmcid: str) -> str:
    normalized = normalize_pmcid(pmcid) or pmcid
    prefix = f"{normalized}.1"
    return f"{_PMC_CLOUD_BASE}/{prefix}/{prefix}.pdf"


def _download_pdf_from_urls(
    urls: list[str],
    *,
    session: requests.Session,
    email: str,
    tool: str,
) -> tuple[bytes, str] | None:
    headers = _session_headers(email, tool)
    for url in urls:
        try:
            response = session.get(url, timeout=120, allow_redirects=True, headers=headers)
            if response.status_code == 200 and _is_pdf_response(response):
                return bytes(response.content), url
        except Exception:
            _logger.exception("PMC PDF download failed for %s", url)
    return None


def pmc_efetch_id(pmcid: str) -> str | None:
    normalized = normalize_pmcid(pmcid)
    if not normalized:
        return None
    numeric = re.sub(r"^PMC", "", normalized, flags=re.IGNORECASE).strip()
    return numeric or None


def pmc_xml_to_plain_text(xml_text: str) -> str:
    """Extract readable text from a PMC full-text XML response."""
    if not xml_text.strip():
        return ""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        _logger.exception("Failed to parse PMC XML")
        return ""

    parts: list[str] = []
    seen: set[str] = set()
    selectors = (
        ".//article-title",
        ".//abstract-title",
        ".//abstract//p",
        ".//abstract//sec",
        ".//sec-title",
        ".//body//p",
        ".//body//sec",
    )
    for selector in selectors:
        for element in root.findall(selector):
            text = " ".join(chunk.strip() for chunk in element.itertext() if chunk.strip())
            if not text or text in seen:
                continue
            seen.add(text)
            parts.append(text)
    return "\n\n".join(parts)


def fetch_pmc_fulltext_xml(
    pmcid: str,
    *,
    session: requests.Session | None = None,
    email: str = "",
    api_key: str = "",
    tool: str = "rag-builder-pubmed",
    entrez_base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
) -> str:
    """Fetch PMC open-access full text as XML via NCBI E-utilities."""
    efetch_id = pmc_efetch_id(pmcid)
    if not efetch_id:
        return ""

    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)
    params: dict[str, str] = {
        "db": "pmc",
        "id": efetch_id,
        "rettype": "xml",
        "retmode": "xml",
        "tool": tool,
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    response = request_with_retry(
        http,
        "GET",
        f"{normalize_entrez_base_url(entrez_base_url)}/efetch.fcgi",
        params=params,
        timeout=120,
    )
    return response.text


def is_pmc_pdf_downloadable(
    pmcid: str | None,
    *,
    pdf_urls: tuple[str, ...] | list[str] | None = None,
) -> bool:
    """Return True only when open-access PMC PDF URLs are known for this record."""
    if not normalize_pmcid(pmcid):
        return False
    return bool(pdf_urls)


def fetch_pmc_document(
    *,
    pmid: str,
    pmcid: str | None,
    title: str,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
    pdf_urls: list[str] | None = None,
) -> PmcPdfResult | None:
    """Download the original open-access PMC PDF."""
    return download_pmc_pdf(
        pmid=pmid,
        pmcid=pmcid,
        title=title,
        session=session,
        email=email,
        tool=tool,
        pdf_urls=pdf_urls,
    )


def download_pmc_pdf(
    *,
    pmid: str,
    pmcid: str | None,
    title: str,
    session: requests.Session | None = None,
    email: str = "",
    tool: str = "rag-builder-pubmed",
    pdf_urls: list[str] | None = None,
) -> PmcPdfResult | None:
    """Download an open-access PMC PDF from NCBI PMC cloud / OA services."""
    normalized = normalize_pmcid(pmcid)
    http = configure_ncbi_session(session or requests.Session(), email=email, tool=tool)

    candidate_urls = [url for url in (pdf_urls or []) if _is_ncbi_pdf_url(url)]
    if normalized:
        candidate_urls.extend(
            fetch_ncbi_cloud_pdf_urls(
                normalized,
                session=http,
                email=email,
                tool=tool,
            )
        )
        candidate_urls.append(pmc_pdf_url(normalized))
    candidate_urls = list(dict.fromkeys(url for url in candidate_urls if _is_ncbi_pdf_url(url)))

    downloaded = _download_pdf_from_urls(
        candidate_urls,
        session=http,
        email=email,
        tool=tool,
    )
    if not downloaded:
        _logger.info("PMC PDF unavailable for PMID %s (%s)", pmid, normalized)
        return None

    content, source_url = downloaded
    safe_title = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in title[:60])
    pmc_label = normalized or f"PMID{pmid}"
    filename = f"pubmed_{pmid}_{pmc_label}_{safe_title.strip() or 'article'}.pdf"
    return PmcPdfResult(
        pmid=pmid,
        pmcid=pmc_label,
        filename=filename,
        content=content,
        source_url=source_url,
    )
