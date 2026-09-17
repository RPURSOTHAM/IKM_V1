"""Unit tests for page-1 colon metadata (shared DOCX/PDF)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXTRACTORS = _HERE.parent.parent
if str(_EXTRACTORS) not in sys.path:
    sys.path.insert(0, str(_EXTRACTORS))

from template_compliance.page_meta import (
    colon_fields_from_page_lines,
    column_key,
    metadata_from_colon_fields,
)


class PageMetaTests(unittest.TestCase):
    def test_same_line_key_value(self) -> None:
        fields = colon_fields_from_page_lines(
            [
                "Title: Microbiological Monitoring of Classified Environment",
                "Document No.: TMP-001",
                "Version No: 1.0",
                "Effective Date: 30/04/2026",
                "Review Date: 29/04/2028",
            ]
        )
        meta = metadata_from_colon_fields(fields)
        self.assertEqual(meta["title"], "Microbiological Monitoring of Classified Environment")
        self.assertEqual(meta["document_no"], "TMP-001")
        self.assertEqual(meta["version_no"], "1.0")
        self.assertEqual(meta["effective_date"], "30/04/2026")
        self.assertEqual(meta["review_date"], "29/04/2028")

    def test_empty_key_uses_next_line(self) -> None:
        fields = colon_fields_from_page_lines(
            [
                "Document No.:",
                "BMS1-QCM-SOP-0069",
                "Version No.:",
                "4.0",
            ]
        )
        meta = metadata_from_colon_fields(fields)
        self.assertEqual(meta["document_no"], "BMS1-QCM-SOP-0069")
        self.assertEqual(meta["version_no"], "4.0")

    def test_ignores_non_colon_text(self) -> None:
        fields = colon_fields_from_page_lines(
            [
                "STANDARD OPERATING PROCEDURE",
                "Title: Real Title Here Exactly",
                "Purpose text without a colon is ignored",
            ]
        )
        meta = metadata_from_colon_fields(fields)
        self.assertEqual(meta["title"], "Real Title Here Exactly")
        self.assertNotIn("Title", meta)
        self.assertEqual(len(fields), 1)

    def test_does_not_dump_non_control_keys_into_metadata(self) -> None:
        fields = [
            {"name": "Title", "value": "DEVIATION RECURRENCE CHECK INSTRUCTIONS"},
            {"name": "Document No", "value": "GL-CQA-ANN-0383"},
            {"name": "Version No", "value": "2.0"},
            {"name": "Department", "value": "Corporate Quality Assurance"},
            {"name": "Facility", "value": "Global"},
            {"name": "Definitions", "value": "Deviation Departure from an approved instruction."},
            {
                "name": "This procedure applies to",
                "value": "Management of Deviations.",
            },
        ]
        meta = metadata_from_colon_fields(fields)
        self.assertEqual(meta["title"], "DEVIATION RECURRENCE CHECK INSTRUCTIONS")
        self.assertEqual(meta["document_no"], "GL-CQA-ANN-0383")
        self.assertEqual(meta["version_no"], "2.0")
        self.assertNotIn("Department", meta)
        self.assertNotIn("Facility", meta)
        self.assertNotIn("Definitions", meta)
        self.assertNotIn("This procedure applies to", meta)

    def test_rejects_sentence_as_document_no(self) -> None:
        fields = [
            {"name": "Document No", "value": "Repeated Deviation"},
            {
                "name": "Effective Date",
                "value": "Define the amendment team, including the Amendment Owner and Section QA.",
            },
            {"name": "Version No", "value": "1.0"},
            {"name": "Review Date", "value": "15/10/2027"},
            {"name": "Document No.", "value": "BMS1-QA-SOP-0008"},
        ]
        meta = metadata_from_colon_fields(fields)
        self.assertEqual(meta["document_no"], "BMS1-QA-SOP-0008")
        self.assertEqual(meta["version_no"], "1.0")
        self.assertEqual(meta["review_date"], "15/10/2027")
        self.assertEqual(meta["effective_date"], "")

    def test_column_key_dynamic(self) -> None:
        self.assertEqual(column_key("Signature and Date"), "signature_and_date")
        self.assertEqual(column_key("Role"), "role")

    def test_dual_empty_keys_skip_prose(self) -> None:
        fields = colon_fields_from_page_lines(
            [
                "Document No.: Effective Date:",
                "Signatures",
                "Signature and Date",
                "BMS1-QCM-SOP-0069 30/04/2026",
                "Version No: Review Date:",
                "4.0",
                "29/04/2028",
            ]
        )
        meta = metadata_from_colon_fields(fields)
        self.assertEqual(meta["document_no"], "BMS1-QCM-SOP-0069")
        self.assertEqual(meta["effective_date"], "30/04/2026")
        self.assertEqual(meta["version_no"], "4.0")
        self.assertEqual(meta["review_date"], "29/04/2028")


if __name__ == "__main__":
    unittest.main()
