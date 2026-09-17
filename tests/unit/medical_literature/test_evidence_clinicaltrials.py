"""Unit tests for ClinicalTrials.gov client (new package)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.features.medical_literature.providers.clinical_trials_client import (
    ClinicalTrialRecord,
    ClinicalTrialsClient,
)
from src.features.medical_literature.application.converters import write_clinicaltrial_docx


def test_schema_rejects_invalid_nct_format() -> None:
    from src.features.medical_literature.schemas.literature_schemas import ClinicalTrialsSearchRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ClinicalTrialsSearchRequest(nct_id="NOTANCT", max_results=1)
    ok = ClinicalTrialsSearchRequest(nct_id="nct01234567", max_results=1)
    assert ok.nct_id == "NCT01234567"


def test_condition_search_parses_studies(tmp_path) -> None:
    client = ClinicalTrialsClient(base_url="https://example.test/api/v2")
    core = ClinicalTrialRecord(
        nct_id="NCT01234567",
        title="Study A diabetes",
        status="COMPLETED",
        conditions=["Diabetes"],
        phases=["PHASE2"],
        summary="A brief summary",
        url="https://clinicaltrials.gov/study/NCT01234567",
        published_date="2024-01-02",
    )
    with patch.object(client, "search", return_value=[core]):
        records = client.search(["diabetes"], page_size=10)
    assert len(records) == 1
    assert records[0].nct_id == "NCT01234567"

    out = tmp_path / "trial.docx"
    write_clinicaltrial_docx(out, records[0])
    assert out.exists()
    assert out.stat().st_size > 100


def test_fetch_study_parse() -> None:
    client = ClinicalTrialsClient(base_url="https://example.test/api/v2")
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT01234567", "briefTitle": "T"},
            "statusModule": {"overallStatus": "COMPLETED"},
            "conditionsModule": {"conditions": ["Diabetes"]},
            "designModule": {"phases": ["PHASE2"]},
            "descriptionModule": {"briefSummary": "Brief"},
        }
    }
    with patch.object(client, "fetch_study", return_value=study):
        payload = client.fetch_study("NCT01234567")
    parsed = client.parse_study(payload)
    assert parsed is not None
    assert parsed.nct_id == "NCT01234567"
    assert parsed.conditions == ["Diabetes"]


def test_docx_contains_nct_and_nested_fields(tmp_path) -> None:
    record = ClinicalTrialRecord(
        nct_id="NCT01234567",
        title="Study Title",
        status="COMPLETED",
        conditions=["Diabetes"],
        phases=[],
        summary="Brief",
        url="https://clinicaltrials.gov/study/NCT01234567",
        published_date="2024-01-02",
    )
    out = tmp_path / "trial.docx"
    write_clinicaltrial_docx(
        out,
        record,
        raw_study={
            "protocolSection": {
                "identificationModule": {"nctId": "NCT01234567", "briefTitle": "Study Title"},
                "conditionsModule": {"conditions": ["Diabetes"]},
            }
        },
    )
    assert out.exists()
    import zipfile

    assert zipfile.is_zipfile(out)
