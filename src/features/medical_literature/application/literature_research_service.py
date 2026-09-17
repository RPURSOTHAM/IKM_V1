"""Orchestrate external evidence search/import through the standard DMS pipeline."""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.features.documents.schemas.document_schemas import DocumentRegisterRequest
from src.features.documents.application.document_service import DocumentReceiverService
from src.features.medical_literature.providers.clinical_trials_client import (
    ClinicalTrialRecord,
    ClinicalTrialsClient,
)
from src.features.medical_literature.configuration.literature_config import (
    MedicalLiteracyConfig,
    load_config,
)
from src.features.medical_literature.application.converters import (
    evidence_provenance_metadata,
    write_clinicaltrial_docx,
    write_pubmed_structured_text,
)
from src.features.medical_literature.domain.literature_exceptions import (
    EvidenceDuplicateError,
    EvidenceError,
    EvidenceExternalError,
    EvidenceValidationError,
)
from src.features.medical_literature.providers.pubmed_client import PubMedClient, PubMedSummary
from src.features.medical_literature.providers.pubmed_pmc import (
    batch_resolve_pmc_pdf_availability,
    fetch_pmc_document,
    resolve_pmcids,
)
from src.features.medical_literature.schemas.literature_schemas import (
    ClinicalTrialsSearchRequest,
    EvidenceImportRequest,
    EvidenceImportResponse,
    EvidenceImportResultItem,
    EvidencePreviewItem,
    EvidenceRegistryItem,
    EvidenceRegistryListResponse,
    EvidenceSearchResponse,
    PubMedSearchRequest,
)
from src.features.medical_literature.infrastructure.evidence_repository import EvidenceImportStore
from src.features.medical_literature.taxonomy.mesh_service import (
    normalize_mesh_terms,
    title_matches_query,
)

logger = logging.getLogger(__name__)

_SAFE_ID = re.compile(r"[^A-Za-z0-9._-]+")
_PUBMED_SEARCH_PAGE_SIZE = 50
_PUBMED_MAX_CANDIDATES_SCANNED = 2000

# Back-compat alias
EvidenceConfig = MedicalLiteracyConfig


def _utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _safe_filename(prefix: str, record_id: str, ext: str) -> str:
    cleaned = _SAFE_ID.sub("_", record_id).strip("._") or "record"
    return f"{prefix}_{cleaned}_{uuid.uuid4().hex[:8]}{ext}"


def _collect_search_terms(
    *,
    query: str = "",
    mesh_terms: list[str] | None = None,
    title: str | None = None,
) -> list[str]:
    terms: list[str] = []
    for part in (query, title or ""):
        for term in normalize_mesh_terms(part):
            if term not in terms:
                terms.append(term)
    for term in normalize_mesh_terms(mesh_terms):
        if term not in terms:
            terms.append(term)
    if not terms:
        raise EvidenceValidationError("PubMed search requires query, title, or MeSH terms")
    return terms


class EvidenceService:
    def __init__(
        self,
        *,
        store: EvidenceImportStore | None = None,
        config: MedicalLiteracyConfig | None = None,
        pubmed: PubMedClient | None = None,
        clinicaltrials: ClinicalTrialsClient | None = None,
        document_receiver: DocumentReceiverService | None = None,
    ) -> None:
        self.config = config or load_config()
        self.store = store or get_evidence_import_store()
        if self.store is None:
            raise EvidenceError(
                "Evidence import registry is unavailable (Postgres not configured)",
                details={"reason": "store_unavailable"},
            )
        self.pubmed = pubmed or self.config.build_pubmed_client()
        self.clinicaltrials = clinicaltrials or self.config.build_clinicaltrials_client()
        self._receiver = document_receiver

    def _receiver_svc(self) -> DocumentReceiverService:
        if self._receiver is None:
            self._receiver = DocumentReceiverService()
        return self._receiver

    def search_pubmed(
        self,
        *,
        repository_id: str,
        body: PubMedSearchRequest,
    ) -> EvidenceSearchResponse:
        try:
            max_results = self.config.clamp_max_results(body.max_results)
        except ValueError as exc:
            raise EvidenceValidationError(str(exc)) from exc

        terms = _collect_search_terms(
            query=body.query,
            mesh_terms=body.mesh_terms,
            title=body.title,
        )
        mesh_mode = bool(body.mesh_terms) and not (body.query or "").strip() and not (body.title or "").strip()
        try:
            summaries = self._search_pubmed_summaries(
                terms=terms,
                max_results=max_results,
                mesh_mode=mesh_mode,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("pubmed_search_failed")
            raise EvidenceExternalError(
                "PubMed search failed",
                details={"reason": type(exc).__name__, "error": str(exc)[:300]},
            ) from exc

        # PMC enrichment is optional — missing OA / 404 must not fail the search.
        pmc_map: dict[str, Any] = {}
        availability: dict[str, Any] = {}
        try:
            pmc_map = resolve_pmcids(
                [item.pmid for item in summaries],
                session=self.pubmed.session,
                email=self.pubmed.email,
                tool=self.pubmed.tool,
            )
            availability = batch_resolve_pmc_pdf_availability(
                {
                    pmid: (pmc_map.get(pmid).pmcid if pmc_map.get(pmid) else None)
                    for pmid in [item.pmid for item in summaries]
                },
                session=self.pubmed.session,
                email=self.pubmed.email,
                tool=self.pubmed.tool,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("pubmed_pmc_enrichment_skipped: %s", exc)
        imported = self.store.imported_ids(
            repository_id=repository_id,
            source="pubmed",
            record_ids=[item.pmid for item in summaries],
        )
        results: list[EvidencePreviewItem] = []
        for item in summaries:
            pmc = pmc_map.get(item.pmid)
            avail = availability.get(item.pmid)
            pmc_id = (pmc.pmcid if pmc else None) or (avail.pmcid if avail else None)
            pdf_available = bool(avail and avail.has_pdf)
            results.append(
                EvidencePreviewItem(
                    source="pubmed",
                    record_id=item.pmid,
                    title=item.title,
                    summary=f"PMID {item.pmid}" + (f" | {item.journal}" if item.journal else ""),
                    published_date=item.published_date or item.year or None,
                    pdf_available=pdf_available,
                    full_text_available=bool(pmc_id) or pdf_available,
                    already_imported=item.pmid in imported,
                    external_url=item.url,
                    metadata={
                        "pmc_id": pmc_id,
                        "journal": item.journal,
                        "doi": (pmc.doi if pmc else None),
                        "pdf_urls": list(avail.pdf_urls) if avail else [],
                    },
                )
            )
        return EvidenceSearchResponse(
            source="pubmed",
            query=body.query,
            max_results=max_results,
            count=len(results),
            results=results,
        )

    def _search_pubmed_summaries(
        self,
        *,
        terms: list[str],
        max_results: int,
        mesh_mode: bool,
    ) -> list[PubMedSummary]:
        target = max(1, min(max_results, 100))
        collected: list[PubMedSummary] = []
        collected_pmids: set[str] = set()
        scanned = 0
        for use_oa in (True, False):
            retstart = 0
            total_count: int | None = None
            while len(collected) < target and scanned < _PUBMED_MAX_CANDIDATES_SCANNED:
                if total_count is not None and retstart >= total_count:
                    break
                page = self.pubmed.search_page(
                    terms,
                    retmax=_PUBMED_SEARCH_PAGE_SIZE,
                    retstart=retstart,
                    open_access_only=use_oa,
                    mesh_mode=mesh_mode,
                )
                if total_count is None:
                    total_count = page.total_count
                page_pmids = [pmid for pmid in page.pmids if pmid not in collected_pmids]
                scanned += len(page.pmids)
                retstart += len(page.pmids)
                if not page_pmids:
                    if not page.pmids:
                        break
                    continue
                for summary in self.pubmed.fetch_summaries(page_pmids):
                    if summary.pmid in collected_pmids:
                        continue
                    if not mesh_mode and not title_matches_query(summary.title, terms):
                        continue
                    collected_pmids.add(summary.pmid)
                    collected.append(summary)
                    if len(collected) >= target:
                        return collected[:target]
            if len(collected) >= target:
                break
        return collected[:target]

    def search_clinical_trials(
        self,
        *,
        repository_id: str,
        body: ClinicalTrialsSearchRequest,
    ) -> EvidenceSearchResponse:
        try:
            max_results = self.config.clamp_max_results(body.max_results)
        except ValueError as exc:
            raise EvidenceValidationError(str(exc)) from exc

        cond = (body.condition or "").strip()
        nct = (body.nct_id or "").strip()
        title_q = (body.title or "").strip()
        if not cond and not nct and not title_q:
            raise EvidenceValidationError(
                "ClinicalTrials search requires condition, nct_id, or title"
            )

        records: list[ClinicalTrialRecord] = []
        if nct:
            study = self.clinicaltrials.fetch_study(nct)
            parsed = self.clinicaltrials.parse_study(study)
            if parsed is None:
                raise EvidenceValidationError(
                    "ClinicalTrials record not found", details={"nct_id": nct}
                )
            records = [parsed]
        else:
            terms = normalize_mesh_terms([part for part in (cond, title_q) if part])
            records = self.clinicaltrials.search(terms, page_size=max_results)

        imported = self.store.imported_ids(
            repository_id=repository_id,
            source="clinicaltrials.gov",
            record_ids=[r.nct_id for r in records],
        )
        query = nct or cond or title_q
        results = [
            EvidencePreviewItem(
                source="clinicaltrials.gov",
                record_id=r.nct_id,
                title=r.title,
                summary=(r.summary or "")[:280] or None,
                published_date=r.published_date or None,
                pdf_available=False,
                full_text_available=True,
                already_imported=r.nct_id in imported,
                external_url=r.url,
                metadata={
                    "status": r.status,
                    "condition": ", ".join(r.conditions) if r.conditions else None,
                    "phase": ", ".join(r.phases) if r.phases else None,
                    "conditions": list(r.conditions or []),
                    "phases": list(r.phases or []),
                },
            )
            for r in records
        ]
        return EvidenceSearchResponse(
            source="clinicaltrials.gov",
            query=query,
            max_results=max_results,
            count=len(results),
            results=results,
        )

    def list_imports(
        self,
        *,
        repository_id: str,
        limit: int = 200,
        offset: int = 0,
    ) -> EvidenceRegistryListResponse:
        rows = self.store.list_for_repository(
            repository_id=repository_id,
            limit=limit,
            offset=offset,
        )
        items = [
            EvidenceRegistryItem(
                repository_id=r.repository_id,
                source=r.source,  # type: ignore[arg-type]
                source_record_id=r.source_record_id,
                published_date=r.published_date,
                document_id=r.document_id,
                imported_at=r.imported_at.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
                if r.imported_at.tzinfo is None
                else r.imported_at.isoformat().replace("+00:00", "Z"),
                status=r.status,
                external_url=r.external_url,
                metadata=r.metadata,
            )
            for r in rows
        ]
        return EvidenceRegistryListResponse(
            repository_id=repository_id,
            count=len(items),
            items=items,
        )

    async def import_selected(
        self,
        *,
        repository_id: str,
        body: EvidenceImportRequest,
    ) -> EvidenceImportResponse:
        if not body.items:
            raise EvidenceValidationError("At least one evidence item is required for import")

        results: list[EvidenceImportResultItem] = []
        for item in body.items:
            source = item.source
            record_id = str(item.record_id).strip()
            try:
                existing = self.store.get(
                    repository_id=repository_id,
                    source=source,
                    source_record_id=record_id,
                )
                if existing is not None:
                    results.append(
                        EvidenceImportResultItem(
                            source=source,
                            record_id=record_id,
                            status="already_imported",
                            document_id=existing.document_id,
                            message="Record already imported into this repository",
                        )
                    )
                    continue

                document_id = await self._import_one(
                    repository_id=repository_id,
                    source=source,
                    record_id=record_id,
                    document_type_id=body.document_type_id,
                    submit_for_processing=body.submit_for_processing,
                )
                results.append(
                    EvidenceImportResultItem(
                        source=source,
                        record_id=record_id,
                        status="imported",
                        document_id=document_id,
                        message="Imported through standard document pipeline",
                    )
                )
            except EvidenceDuplicateError as exc:
                results.append(
                    EvidenceImportResultItem(
                        source=source,
                        record_id=record_id,
                        status="already_imported",
                        document_id=(exc.details or {}).get("document_id"),
                        message=exc.message,
                    )
                )
            except EvidenceError as exc:
                logger.warning(
                    "evidence_import_failed repo=%s source=%s record=%s code=%s",
                    repository_id,
                    source,
                    record_id,
                    exc.code,
                )
                results.append(
                    EvidenceImportResultItem(
                        source=source,
                        record_id=record_id,
                        status="failed",
                        message=exc.message,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "evidence_import_unexpected repo=%s source=%s record=%s",
                    repository_id,
                    source,
                    record_id,
                )
                results.append(
                    EvidenceImportResultItem(
                        source=source,
                        record_id=record_id,
                        status="failed",
                        message=(
                            "Document creation or processing queue submission failed "
                            f"({type(exc).__name__}: {str(exc)[:240]})"
                        ),
                    )
                )

        return EvidenceImportResponse(repository_id=repository_id, results=results)

    async def _import_one(
        self,
        *,
        repository_id: str,
        source: str,
        record_id: str,
        document_type_id: str | None,
        submit_for_processing: bool,
    ) -> str:
        imported_at = _utc_iso()
        upload_dir = Path(self._receiver_svc()._live_settings().upload_dir)
        evidence_dir = upload_dir / "external_evidence" / repository_id
        evidence_dir.mkdir(parents=True, exist_ok=True)

        if source == "pubmed":
            pmid = str(record_id).strip()
            if not pmid.isdigit():
                raise EvidenceValidationError("Invalid PubMed ID", details={"pmid": pmid})
            articles = self.pubmed.fetch_articles([pmid])
            if not articles:
                raise EvidenceValidationError("PubMed record not found", details={"pmid": pmid})
            article = articles[0]
            pmc_map = resolve_pmcids(
                [pmid],
                session=self.pubmed.session,
                email=self.pubmed.email,
                tool=self.pubmed.tool,
            )
            pmc = pmc_map.get(pmid)
            pmc_id = pmc.pmcid if pmc else None
            avail = batch_resolve_pmc_pdf_availability(
                {pmid: pmc_id},
                session=self.pubmed.session,
                email=self.pubmed.email,
                tool=self.pubmed.tool,
            ).get(pmid)
            pdf_urls = list(avail.pdf_urls) if avail else []
            pdf_result = fetch_pmc_document(
                pmid=pmid,
                pmcid=pmc_id or (avail.pmcid if avail else None),
                title=article.title,
                session=self.pubmed.session,
                email=self.pubmed.email,
                tool=self.pubmed.tool,
                pdf_urls=pdf_urls,
            )
            if pdf_result is not None:
                filename = pdf_result.filename or _safe_filename("pubmed", pmid, ".pdf")
                dest = evidence_dir / filename
                dest.write_bytes(pdf_result.content)
            else:
                filename = article.filename if hasattr(article, "filename") else _safe_filename(
                    "pubmed", pmid, ".txt"
                )
                dest = evidence_dir / filename
                write_pubmed_structured_text(dest, article)
            published_date = article.year or None
            external_url = article.url
            title = article.title
            relative_path = str(dest.relative_to(upload_dir)).replace("\\", "/")
            provenance = evidence_provenance_metadata(
                source="pubmed",
                source_record_id=article.pmid,
                repository_id=repository_id,
                published_date=published_date,
                external_url=external_url,
                imported_at=imported_at,
                extra={
                    "pmc_id": pmc_id,
                    "doi": pmc.doi if pmc else None,
                    "journal": article.journal,
                    "title": title,
                    "pdf_available": pdf_result is not None,
                    "full_text_available": True,
                    "import_format": "pdf" if pdf_result is not None else "text",
                },
            )
            source_record_id = article.pmid
        elif source == "clinicaltrials.gov":
            study = self.clinicaltrials.fetch_study(record_id)
            trial = self.clinicaltrials.parse_study(study)
            if trial is None:
                raise EvidenceValidationError(
                    "ClinicalTrials record not found", details={"nct_id": record_id}
                )
            try:
                docx_name, docx_bytes = self.clinicaltrials.build_study_docx(study)
                filename = docx_name or _safe_filename("clinicaltrials", trial.nct_id, ".docx")
                dest = evidence_dir / filename
                dest.write_bytes(docx_bytes)
            except Exception:
                filename = _safe_filename("clinicaltrials", trial.nct_id, ".docx")
                dest = evidence_dir / filename
                write_clinicaltrial_docx(dest, trial, raw_study=study)
            published_date = trial.published_date or None
            external_url = trial.url
            title = trial.title
            relative_path = str(dest.relative_to(upload_dir)).replace("\\", "/")
            provenance = evidence_provenance_metadata(
                source="clinicaltrials.gov",
                source_record_id=trial.nct_id,
                repository_id=repository_id,
                published_date=published_date,
                external_url=external_url,
                imported_at=imported_at,
                extra={
                    "status": trial.status,
                    "condition": ", ".join(trial.conditions) if trial.conditions else None,
                    "phase": ", ".join(trial.phases) if trial.phases else None,
                    "title": title,
                },
            )
            source_record_id = trial.nct_id
        else:
            raise EvidenceValidationError("Unsupported evidence source", details={"source": source})

        register = DocumentRegisterRequest(
            document_name=filename,
            original_file_name=filename,
            repository_type="local",
            repository_path=relative_path,
            repository_id=repository_id,
            document_type_id=document_type_id,
            submit_for_processing=submit_for_processing,
            metadata=provenance,
        )
        response = await self._receiver_svc().register_document(register)
        document_id = response.document_id

        try:
            self.store.insert(
                repository_id=repository_id,
                source=source,
                source_record_id=source_record_id,
                document_id=document_id,
                published_date=published_date,
                external_url=external_url,
                status="imported",
                metadata={"title": title, "filename": filename},
            )
        except Exception as exc:  # noqa: BLE001
            existing = self.store.get(
                repository_id=repository_id,
                source=source,
                source_record_id=source_record_id,
            )
            if existing is not None:
                raise EvidenceDuplicateError(
                    "Record already imported into this repository",
                    details={
                        "document_id": existing.document_id,
                        "source": source,
                        "source_record_id": source_record_id,
                    },
                ) from exc
            raise EvidenceError(
                "Failed to persist evidence import registry entry",
                details={"reason": type(exc).__name__},
            ) from exc

        return document_id


_store: EvidenceImportStore | None = None
_service: EvidenceService | None = None


def get_evidence_import_store() -> EvidenceImportStore | None:
    global _store
    if _store is not None:
        return _store
    from src.features.repositories.configuration.repository_config import postgres_params_for_repositories

    params = postgres_params_for_repositories()
    if params is None:
        return None
    candidate = EvidenceImportStore(params)
    candidate.ensure_schema()
    _store = candidate
    return _store


def get_evidence_service() -> EvidenceService:
    global _service
    if _service is not None:
        return _service
    _service = EvidenceService()
    return _service
