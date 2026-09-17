from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from src.features.medical_literature.export.json_to_docx import json_to_docx_bytes
from src.features.medical_literature.taxonomy.mesh_service import (
    build_clinicaltrials_title_query,
    title_matches_query,
)


@dataclass(frozen=True)
class ClinicalTrialRecord:
    nct_id: str
    title: str
    status: str
    conditions: list[str]
    phases: list[str]
    summary: str
    url: str
    published_date: str = ""

    @property
    def filename(self) -> str:
        return self.docx_filename

    @property
    def docx_filename(self) -> str:
        safe_title = "".join(ch if ch.isalnum() or ch in " -_" else "_" for ch in self.title[:60])
        return f"clinicaltrials_{self.nct_id}_{safe_title.strip() or 'study'}.docx"


def _date_from_struct(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("date") or "").strip()
    return str(value or "").strip()


def _trial_published_date_from_status(status_mod: dict[str, Any]) -> str:
    """Prefer first public post date, then start / completion / last update."""
    for key in (
        "studyFirstPostDateStruct",
        "startDateStruct",
        "primaryCompletionDateStruct",
        "completionDateStruct",
        "lastUpdatePostDateStruct",
    ):
        date_value = _date_from_struct(status_mod.get(key))
        if date_value:
            return date_value
    for key in ("studyFirstSubmitDate", "studyFirstSubmitQcDate", "lastUpdateSubmitDate", "statusVerifiedDate"):
        date_value = str(status_mod.get(key) or "").strip()
        if date_value:
            return date_value
    return ""


class ClinicalTrialsClient:
    def __init__(self, *, base_url: str, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def search(
        self,
        mesh_terms: list[str],
        *,
        page_size: int = 25,
        exclude_nct_ids: set[str] | frozenset[str] | None = None,
    ) -> list[ClinicalTrialRecord]:
        if not mesh_terms:
            return []
        query = build_clinicaltrials_title_query(mesh_terms)
        if not query:
            return []

        # Title API hits can include related drugs (e.g. acetaminophen) without the
        # literal query in the brief title. Page through until we fill the requested
        # count of true title matches (or exhaust the result set).
        # Already-in-repo NCTs never consume a slot — keep paging to replace them.
        target = max(1, min(int(page_size), 100))
        excluded = {str(nct).strip().upper() for nct in (exclude_nct_ids or set()) if str(nct).strip()}
        page_fetch = 100
        max_pages = 20
        records: list[ClinicalTrialRecord] = []
        seen_nct: set[str] = set()
        page_token: str | None = None

        for _ in range(max_pages):
            cancel = getattr(self, "_cancel_event", None)
            if cancel is not None and cancel.is_set():
                raise RuntimeError("Literature search cancelled.")
            if len(records) >= target:
                break
            params: dict[str, str] = {
                "query.titles": query,
                "pageSize": str(page_fetch),
                "format": "json",
            }
            if page_token:
                params["pageToken"] = page_token

            response = self.session.get(f"{self.base_url}/studies", params=params, timeout=60)
            response.raise_for_status()
            payload = response.json()
            studies = payload.get("studies") or []
            if not studies:
                break

            for study in studies:
                parsed = self.parse_study(study)
                if not parsed:
                    continue
                if parsed.nct_id in seen_nct:
                    continue
                seen_nct.add(parsed.nct_id)
                if parsed.nct_id.upper() in excluded:
                    continue
                if not title_matches_query(parsed.title, mesh_terms):
                    continue
                records.append(parsed)
                if len(records) >= target:
                    break

            page_token = str(payload.get("nextPageToken") or "").strip() or None
            if not page_token:
                break

        return records[:target]

    def fetch_study(self, nct_id: str) -> dict[str, Any]:
        """Fetch the complete study JSON from ClinicalTrials.gov."""
        normalized = str(nct_id or "").strip()
        if not normalized:
            raise ValueError("NCT ID is required.")
        response = self.session.get(
            f"{self.base_url}/studies/{normalized}",
            params={"format": "json"},
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(f"Unexpected ClinicalTrials.gov response for {normalized}.")
        return payload

    def build_study_docx(self, study: dict[str, Any]) -> tuple[str, bytes]:
        """Build a DOCX payload from the full study JSON response."""
        record = self.parse_study(study)
        if not record:
            raise ValueError("ClinicalTrials.gov study JSON is missing an NCT ID.")
        return record.docx_filename, json_to_docx_bytes(study)

    def parse_study(self, study: dict[str, Any]) -> ClinicalTrialRecord | None:
        return self._parse_study(study)

    def _parse_study(self, study: dict[str, Any]) -> ClinicalTrialRecord | None:
        protocol = study.get("protocolSection") or {}
        identification = protocol.get("identificationModule") or {}
        nct_id = str(identification.get("nctId") or "").strip()
        if not nct_id:
            return None
        title = str(identification.get("briefTitle") or identification.get("officialTitle") or nct_id).strip()
        status_mod = protocol.get("statusModule") or {}
        status = str(status_mod.get("overallStatus") or "Unknown").strip()
        conditions_mod = protocol.get("conditionsModule") or {}
        conditions = [str(item).strip() for item in (conditions_mod.get("conditions") or []) if str(item).strip()]
        design_mod = protocol.get("designModule") or {}
        phases = [str(item).strip() for item in (design_mod.get("phases") or []) if str(item).strip()]
        description_mod = protocol.get("descriptionModule") or {}
        summary = str(
            description_mod.get("briefSummary")
            or description_mod.get("detailedDescription")
            or ""
        ).strip()
        return ClinicalTrialRecord(
            nct_id=nct_id,
            title=title,
            status=status,
            conditions=conditions,
            phases=phases,
            summary=summary,
            url=f"https://clinicaltrials.gov/study/{nct_id}",
            published_date=_trial_published_date_from_status(status_mod if isinstance(status_mod, dict) else {}),
        )
