"""Unit tests for EvidenceService search/import orchestration (mocked providers)."""

from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.features.medical_literature.configuration.literature_config import MedicalLiteracyConfig
from src.features.medical_literature.providers.clinical_trials_client import ClinicalTrialRecord
from src.features.medical_literature.providers.pubmed_client import PubMedArticle, PubMedSummary
from src.features.medical_literature.schemas.literature_schemas import (
    ClinicalTrialsSearchRequest,
    EvidenceImportItem,
    EvidenceImportRequest,
    PubMedSearchRequest,
)
from src.features.medical_literature.application.literature_research_service import EvidenceService
from src.features.medical_literature.infrastructure.evidence_repository import EvidenceImportRecord


@pytest.fixture
def repo_id() -> str:
    return str(uuid.uuid4())


def _store_mock() -> MagicMock:
    store = MagicMock()
    store.imported_ids.return_value = set()
    store.get.return_value = None
    store.list_for_repository.return_value = []
    return store


def test_search_pubmed_marks_already_imported(repo_id: str) -> None:
    store = _store_mock()
    store.imported_ids.return_value = {"111"}
    pubmed = MagicMock()
    pubmed.session = MagicMock()
    pubmed.email = ""
    pubmed.tool = "test"
    summaries = [
        PubMedSummary(
            pmid="111",
            title="Imported diabetes study",
            journal="J",
            year="2024",
            published_date="2024",
            url="https://pubmed.ncbi.nlm.nih.gov/111/",
        ),
        PubMedSummary(
            pmid="222",
            title="New diabetes study",
            journal="J",
            year="2024",
            published_date="2024",
            url="https://pubmed.ncbi.nlm.nih.gov/222/",
        ),
    ]
    svc = EvidenceService(
        store=store,
        config=MedicalLiteracyConfig(pubmed_max_results=25, absolute_max_results=100),
        pubmed=pubmed,
        clinicaltrials=MagicMock(),
        document_receiver=MagicMock(),
    )
    with (
        patch.object(svc, "_search_pubmed_summaries", return_value=summaries),
        patch(
            "src.features.medical_literature.application.literature_research_service.resolve_pmcids",
            return_value={},
        ),
        patch(
            "src.features.medical_literature.application.literature_research_service.batch_resolve_pmc_pdf_availability",
            return_value={},
        ),
    ):
        resp = svc.search_pubmed(
            repository_id=repo_id,
            body=PubMedSearchRequest(query="diabetes", max_results=10),
        )
    assert resp.count == 2
    assert resp.results[0].already_imported is True
    assert resp.results[1].already_imported is False
    assert resp.max_results == 10


def test_search_rejects_unbounded_max_results(repo_id: str) -> None:
    svc = EvidenceService(
        store=_store_mock(),
        config=MedicalLiteracyConfig(absolute_max_results=25),
        pubmed=MagicMock(),
        clinicaltrials=MagicMock(),
        document_receiver=MagicMock(),
    )
    from src.features.medical_literature.domain.literature_exceptions import EvidenceValidationError

    with pytest.raises(EvidenceValidationError):
        svc.search_pubmed(
            repository_id=repo_id,
            body=PubMedSearchRequest.model_construct(query="x", max_results=1000),
        )


@pytest.mark.asyncio
async def test_import_duplicate_short_circuits(repo_id: str) -> None:
    store = _store_mock()
    store.get.return_value = EvidenceImportRecord(
        id=str(uuid.uuid4()),
        repository_id=repo_id,
        source="pubmed",
        source_record_id="123",
        published_date="2024",
        document_id=str(uuid.uuid4()),
        imported_at=datetime.utcnow(),
        status="imported",
        external_url=None,
        metadata={},
    )
    svc = EvidenceService(
        store=store,
        pubmed=MagicMock(),
        clinicaltrials=MagicMock(),
        document_receiver=MagicMock(),
    )
    resp = await svc.import_selected(
        repository_id=repo_id,
        body=EvidenceImportRequest(
            items=[EvidenceImportItem(source="pubmed", record_id="123")],
        ),
    )
    assert resp.results[0].status == "already_imported"
    store.insert.assert_not_called()


@pytest.mark.asyncio
async def test_import_pubmed_registers_document(repo_id: str, tmp_path: Path) -> None:
    store = _store_mock()
    pubmed = MagicMock()
    pubmed.session = MagicMock()
    pubmed.email = ""
    pubmed.tool = "test"
    pubmed.fetch_articles.return_value = [
        PubMedArticle(
            pmid="999",
            title="Article",
            abstract="Abstract",
            authors="Doe",
            journal="J",
            year="2024",
            mesh_terms=[],
            url="https://pubmed.ncbi.nlm.nih.gov/999/",
        )
    ]
    receiver = MagicMock()
    live = MagicMock()
    live.upload_dir = str(tmp_path)
    receiver._live_settings.return_value = live
    doc_id = str(uuid.uuid4())
    registered = MagicMock()
    registered.document_id = doc_id
    receiver.register_document = AsyncMock(return_value=registered)

    svc = EvidenceService(
        store=store,
        pubmed=pubmed,
        clinicaltrials=MagicMock(),
        document_receiver=receiver,
    )
    with (
        patch(
            "src.features.medical_literature.application.literature_research_service.resolve_pmcids",
            return_value={},
        ),
        patch(
            "src.features.medical_literature.application.literature_research_service.batch_resolve_pmc_pdf_availability",
            return_value={},
        ),
        patch(
            "src.features.medical_literature.application.literature_research_service.fetch_pmc_document",
            return_value=None,
        ),
    ):
        resp = await svc.import_selected(
            repository_id=repo_id,
            body=EvidenceImportRequest(
                items=[EvidenceImportItem(source="pubmed", record_id="999")],
            ),
        )
    assert resp.results[0].status == "imported"
    assert resp.results[0].document_id == doc_id
    store.insert.assert_called_once()
    receiver.register_document.assert_awaited()


@pytest.mark.asyncio
async def test_import_clinicaltrials(repo_id: str, tmp_path: Path) -> None:
    store = _store_mock()
    trials = MagicMock()
    study = {"protocolSection": {"identificationModule": {"nctId": "NCT01234567"}}}
    trials.fetch_study.return_value = study
    trials.parse_study.return_value = ClinicalTrialRecord(
        nct_id="NCT01234567",
        title="Trial",
        status="COMPLETED",
        conditions=["Diabetes"],
        phases=["PHASE2"],
        summary="Summary",
        url="https://clinicaltrials.gov/study/NCT01234567",
        published_date="2024-01-01",
    )
    trials.build_study_docx.return_value = ("clinicaltrials_NCT01234567.docx", b"PK\x03\x04fake")

    receiver = MagicMock()
    live = MagicMock()
    live.upload_dir = str(tmp_path)
    receiver._live_settings.return_value = live
    registered = MagicMock()
    registered.document_id = str(uuid.uuid4())
    receiver.register_document = AsyncMock(return_value=registered)

    svc = EvidenceService(
        store=store,
        pubmed=MagicMock(),
        clinicaltrials=trials,
        document_receiver=receiver,
    )
    resp = await svc.import_selected(
        repository_id=repo_id,
        body=EvidenceImportRequest(
            items=[EvidenceImportItem(source="clinicaltrials.gov", record_id="NCT01234567")],
        ),
    )
    assert resp.results[0].status == "imported"


def test_search_clinical_trials(repo_id: str) -> None:
    store = _store_mock()
    trials = MagicMock()
    trials.search.return_value = [
        ClinicalTrialRecord(
            nct_id="NCT01234567",
            title="Diabetes trial",
            status="COMPLETED",
            conditions=["Diabetes"],
            phases=[],
            summary="S",
            url="https://clinicaltrials.gov/study/NCT01234567",
            published_date="2024",
        )
    ]
    svc = EvidenceService(
        store=store,
        pubmed=MagicMock(),
        clinicaltrials=trials,
        document_receiver=MagicMock(),
    )
    resp = svc.search_clinical_trials(
        repository_id=repo_id,
        body=ClinicalTrialsSearchRequest(condition="diabetes", max_results=5),
    )
    assert resp.count == 1
    assert resp.results[0].record_id == "NCT01234567"
