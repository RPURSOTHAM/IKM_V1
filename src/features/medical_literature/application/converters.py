"""Convert external evidence records into ingestible document files."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from src.features.medical_literature.providers.clinical_trials_client import ClinicalTrialRecord
from src.features.medical_literature.providers.pubmed_client import PubMedArticle


def write_pubmed_structured_text(path: Path, article: PubMedArticle) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(article.to_document_text(), encoding="utf-8")
    return path


def write_clinicaltrial_docx(
    path: Path,
    record: ClinicalTrialRecord,
    *,
    raw_study: dict[str, Any] | None = None,
) -> Path:
    """Write a DOCX package for a ClinicalTrials.gov record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    study = raw_study
    title = f"ClinicalTrials.gov — {record.nct_id}"
    try:
        from src.features.medical_literature.export.json_to_docx import json_to_docx_bytes

        payload = study if isinstance(study, dict) else _record_as_dict(record)
        if isinstance(payload, dict):
            enriched = {
                "provenance": {
                    "source_type": "external_evidence",
                    "source": "clinicaltrials.gov",
                    "source_record_id": record.nct_id,
                    "external_url": record.url,
                    "published_date": record.published_date,
                },
                "study": payload,
            }
        else:
            enriched = payload
        path.write_bytes(json_to_docx_bytes(enriched, title=title))
        return path
    except Exception:
        paragraphs = _clinicaltrial_paragraphs(record)
        _write_minimal_docx(path, paragraphs)
        return path


def evidence_provenance_metadata(
    *,
    source: str,
    source_record_id: str,
    repository_id: str,
    published_date: str | None,
    external_url: str | None,
    imported_at: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "source_type": "external_evidence",
        "source": source,
        "source_record_id": source_record_id,
        "published_date": published_date,
        "external_url": external_url,
        "imported_at": imported_at,
        "repository_id": repository_id,
    }
    if extra:
        meta.update(extra)
    return meta


def _record_as_dict(record: ClinicalTrialRecord) -> dict[str, Any]:
    return {
        "nct_id": record.nct_id,
        "title": record.title,
        "status": record.status,
        "published_date": record.published_date,
        "conditions": list(record.conditions or []),
        "phases": list(record.phases or []),
        "summary": record.summary,
        "url": record.url,
    }


def _clinicaltrial_paragraphs(record: ClinicalTrialRecord) -> list[str]:
    paragraphs = [
        "EXTERNAL EVIDENCE — ClinicalTrials.gov",
        f"NCT ID: {record.nct_id}",
        f"Title: {record.title}",
    ]
    if record.status:
        paragraphs.append(f"Status: {record.status}")
    if record.published_date:
        paragraphs.append(f"Published / Updated: {record.published_date}")
    if record.conditions:
        paragraphs.append(f"Conditions: {', '.join(record.conditions)}")
    if record.phases:
        paragraphs.append(f"Phases: {', '.join(record.phases)}")
    if record.url:
        paragraphs.append(f"URL: {record.url}")
    paragraphs.append("")
    paragraphs.append("Summary")
    paragraphs.append(record.summary or "(No summary available.)")
    paragraphs.append("")
    paragraphs.append(
        f"Provenance: source_type=external_evidence; source=clinicaltrials.gov; "
        f"source_record_id={record.nct_id}"
    )
    return paragraphs


def _write_minimal_docx(path: Path, paragraphs: list[str]) -> None:
    body_parts = []
    for text in paragraphs:
        safe = escape(text).replace("\n", "</w:t><w:br/><w:t>")
        body_parts.append(f'<w:p><w:r><w:t xml:space="preserve">{safe}</w:t></w:r></w:p>')
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(body_parts)}<w:sectPr/></w:body></w:document>'
    )
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
"""
    rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", document_xml)
