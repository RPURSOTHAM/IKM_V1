from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any

import requests

from src.features.medical_literature.taxonomy.mesh_service import build_pubmed_search_query, format_pubmed_mesh_query
from src.features.medical_literature.providers.pubmed_http import (
    configure_ncbi_session,
    normalize_entrez_base_url,
    request_with_retry,
)

_logger = logging.getLogger(__name__)

_NCBI_MIN_INTERVAL_SECONDS = 0.11


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    authors: str
    journal: str
    year: str
    mesh_terms: list[str]
    url: str

    def to_document_text(self) -> str:
        mesh_line = ", ".join(self.mesh_terms) if self.mesh_terms else "N/A"
        return (
            f"Source: PubMed\n"
            f"PMID: {self.pmid}\n"
            f"Title: {self.title}\n"
            f"Authors: {self.authors}\n"
            f"Journal: {self.journal}\n"
            f"Year: {self.year}\n"
            f"URL: {self.url}\n"
            f"MeSH Terms: {mesh_line}\n\n"
            f"Abstract:\n{self.abstract or 'No abstract available.'}\n"
        )

    @property
    def filename(self) -> str:
        safe_title = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in self.title[:60])
        return f"pubmed_{self.pmid}_{safe_title.strip() or 'article'}.txt"


@dataclass(frozen=True)
class PubMedSummary:
    pmid: str
    title: str
    journal: str
    year: str
    published_date: str
    url: str


@dataclass(frozen=True)
class PubMedSearchPage:
    pmids: list[str]
    total_count: int
    retstart: int


class PubMedClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        email: str = "",
        tool: str = "rag-builder-pubmed",
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = normalize_entrez_base_url(base_url)
        self.api_key = api_key.strip()
        self.email = email.strip()
        self.tool = tool
        self.session = configure_ncbi_session(session or requests.Session(), email=self.email, tool=self.tool)
        self._last_request_at = 0.0

    def _common_params(self) -> dict[str, str]:
        params: dict[str, str] = {"tool": self.tool}
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return params

    def _throttle(self) -> None:
        cancel = getattr(self, "_cancel_event", None)
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Literature search cancelled.")
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < _NCBI_MIN_INTERVAL_SECONDS:
            remaining = _NCBI_MIN_INTERVAL_SECONDS - elapsed
            # Sleep in short slices so refresh/disconnect can stop quickly.
            end = time.monotonic() + remaining
            while time.monotonic() < end:
                if cancel is not None and cancel.is_set():
                    raise RuntimeError("Literature search cancelled.")
                time.sleep(min(0.05, end - time.monotonic()))

    def _get(self, path: str, *, params: dict[str, Any], timeout: float) -> requests.Response:
        cancel = getattr(self, "_cancel_event", None)
        if cancel is not None and cancel.is_set():
            raise RuntimeError("Literature search cancelled.")
        self._throttle()
        response = request_with_retry(
            self.session,
            "GET",
            f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            timeout=timeout,
            cancel_event=cancel,
        )
        self._last_request_at = time.monotonic()
        return response

    def search(
        self,
        mesh_terms: list[str],
        *,
        retmax: int = 25,
        retstart: int = 0,
        open_access_only: bool = False,
        mesh_mode: bool = False,
    ) -> list[str]:
        return self.search_page(
            mesh_terms,
            retmax=retmax,
            retstart=retstart,
            open_access_only=open_access_only,
            mesh_mode=mesh_mode,
        ).pmids

    def search_page(
        self,
        mesh_terms: list[str],
        *,
        retmax: int = 25,
        retstart: int = 0,
        open_access_only: bool = False,
        mesh_mode: bool = False,
    ) -> PubMedSearchPage:
        if not mesh_terms:
            return PubMedSearchPage(pmids=[], total_count=0, retstart=retstart)
        query = build_pubmed_search_query(mesh_terms, mesh_mode=mesh_mode, title_only=True)
        if not query:
            return PubMedSearchPage(pmids=[], total_count=0, retstart=retstart)
        if open_access_only:
            query = f"({query}) AND free full text[filter]"
        page_size = max(1, min(retmax, 100))
        page_start = max(0, retstart)
        params = {
            **self._common_params(),
            "db": "pubmed",
            "term": query,
            "retmax": str(page_size),
            "retstart": str(page_start),
            "retmode": "json",
        }
        response = self._get("esearch.fcgi", params=params, timeout=60)
        payload = response.json()
        result = payload.get("esearchresult") or {}
        id_list = result.get("idlist") or []
        total_raw = result.get("count") or 0
        try:
            total_count = int(total_raw)
        except (TypeError, ValueError):
            total_count = 0
        pmids = [str(item) for item in id_list if item]
        return PubMedSearchPage(pmids=pmids, total_count=total_count, retstart=page_start)

    def fetch_summaries(self, pmids: list[str]) -> list[PubMedSummary]:
        if not pmids:
            return []
        summaries: list[PubMedSummary] = []
        batch_size = 100
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            params = {
                **self._common_params(),
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "json",
            }
            response = self._get("esummary.fcgi", params=params, timeout=60)
            payload = response.json()
            result = payload.get("result") or {}
            for pmid in batch:
                entry = result.get(pmid) or {}
                if not isinstance(entry, dict):
                    continue
                title = str(entry.get("title") or f"PubMed {pmid}").strip()
                journal = str(entry.get("fulljournalname") or entry.get("source") or "").strip()
                pubdate = str(entry.get("pubdate") or "").strip()
                year = pubdate[:4] if len(pubdate) >= 4 else pubdate
                summaries.append(
                    PubMedSummary(
                        pmid=str(pmid),
                        title=title,
                        journal=journal,
                        year=year,
                        published_date=pubdate,
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    )
                )
        return summaries

    def fetch_articles(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []
        articles: list[PubMedArticle] = []
        batch_size = 50
        for start in range(0, len(pmids), batch_size):
            batch = pmids[start : start + batch_size]
            params = {
                **self._common_params(),
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }
            response = self._get("efetch.fcgi", params=params, timeout=120)
            articles.extend(self._parse_xml(response.text))
        return articles

    def search_and_fetch(self, mesh_terms: list[str], *, retmax: int = 25) -> list[PubMedArticle]:
        pmids = self.search(mesh_terms, retmax=retmax)
        return self.fetch_articles(pmids)

    def _parse_xml(self, xml_text: str) -> list[PubMedArticle]:
        root = ET.fromstring(xml_text)
        articles: list[PubMedArticle] = []
        for article_el in root.findall(".//PubmedArticle"):
            parsed = self._parse_article(article_el)
            if parsed:
                articles.append(parsed)
        return articles

    def _parse_article(self, article_el: ET.Element) -> PubMedArticle | None:
        medline = article_el.find("MedlineCitation")
        if medline is None:
            return None
        pmid_el = medline.find("PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        if not pmid:
            return None

        article = medline.find("Article")
        title = self._element_text(article, "ArticleTitle") if article is not None else ""
        abstract = self._collect_abstract(article)
        journal = ""
        if article is not None:
            journal_el = article.find("Journal/Title")
            journal = (journal_el.text or "").strip() if journal_el is not None else ""
        year = self._extract_year(medline)
        authors = self._collect_authors(article)
        mesh_terms = [
            (el.text or "").strip()
            for el in medline.findall(".//MeshHeading/DescriptorName")
            if (el.text or "").strip()
        ]
        return PubMedArticle(
            pmid=pmid,
            title=title or f"PubMed article {pmid}",
            abstract=abstract,
            authors=authors,
            journal=journal,
            year=year,
            mesh_terms=mesh_terms,
            url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        )

    @staticmethod
    def _element_text(parent: ET.Element | None, path: str) -> str:
        if parent is None:
            return ""
        el = parent.find(path)
        return (el.text or "").strip() if el is not None else ""

    @staticmethod
    def _collect_abstract(article: ET.Element | None) -> str:
        if article is None:
            return ""
        abstract_el = article.find("Abstract")
        if abstract_el is None:
            return ""
        parts: list[str] = []
        for text_el in abstract_el.findall("AbstractText"):
            label = text_el.attrib.get("Label", "").strip()
            content = "".join(text_el.itertext()).strip()
            if not content:
                continue
            parts.append(f"{label}: {content}" if label else content)
        return "\n".join(parts)

    @staticmethod
    def _collect_authors(article: ET.Element | None) -> str:
        if article is None:
            return ""
        names: list[str] = []
        for author in article.findall("AuthorList/Author"):
            collective = author.find("CollectiveName")
            if collective is not None and (collective.text or "").strip():
                names.append(collective.text.strip())
                continue
            last = (author.findtext("LastName") or "").strip()
            fore = (author.findtext("ForeName") or author.findtext("Initials") or "").strip()
            if last:
                names.append(f"{fore} {last}".strip())
        return "; ".join(names)

    @staticmethod
    def _extract_year(medline: ET.Element) -> str:
        for path in ("Article/Journal/JournalIssue/PubDate/Year", "DateCompleted/Year", "DateRevised/Year"):
            el = medline.find(path)
            if el is not None and (el.text or "").strip():
                return el.text.strip()
        pub_date = medline.find("Article/Journal/JournalIssue/PubDate")
        if pub_date is not None:
            medline_date = pub_date.findtext("MedlineDate") or ""
            if medline_date:
                return medline_date[:4] if len(medline_date) >= 4 else medline_date
        return ""
