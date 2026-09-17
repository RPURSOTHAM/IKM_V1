"""Verify citation metadata for a DOCX against an exported CSV."""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

from src.features.document_processing.loaders.docxloader import DocxLoader
from src.features.chunking.strategies.chunking_strategies import chunk_document_with_strategy


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text[:120]


def main() -> int:
    docx_path = Path(r"c:\Users\Neevan\Downloads\SOP-CL-TM-001_Study_Startup.docx")
    csv_path = Path(r"c:\Users\Neevan\Downloads\2026-06-25T07-21_export.csv")

    export_rows: list[dict[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            export_rows.append(row)

    print("=== CSV Export Summary ===")
    print(f"Total chunks: {len(export_rows)}")
    for i, row in enumerate(export_rows):
        text_preview = (row.get("text") or "")[:80].replace("\n", " ")
        print(
            f"{i + 1}. page={row.get('page')} section={row.get('section_name')} "
            f"category={row.get('category')} | {text_preview}..."
        )

    loader = DocxLoader()
    blocks = loader.load(docx_path, citation_retainment=True)
    pages = sorted(b.page for b in blocks if b.page)
    print(f"\n=== Loader produced {len(blocks)} blocks ===")
    print(f"Page range in blocks: {min(pages)}-{max(pages)}")

    chunks = chunk_document_with_strategy(
        blocks,
        docx_path.name,
        strategy="section-based",
        chunk_size=80,
        overlap_sentences=1,
        min_content_words=6,
        citation_retainment=True,
    )
    print(f"\n=== Chunking produced {len(chunks)} chunks ===")
    for i, c in enumerate(chunks):
        text_preview = (c.text or "")[:80].replace("\n", " ")
        chunk_type = getattr(c, "chunk_type", "?")
        print(
            f"{i + 1}. page={c.page} end_page={c.end_page} section={c.section_name} "
            f"type={chunk_type} | {text_preview}..."
        )

    print("\n=== Citation Accuracy Comparison ===")
    matched = 0
    page_mismatches: list[str] = []
    section_mismatches: list[str] = []
    not_found: list[str] = []

    for row in export_rows:
        export_text = _normalize_text(row.get("text", ""))
        export_page = int(row.get("page") or 0)
        export_section = (row.get("section_name") or "").strip()

        best = None
        best_score = 0.0
        for chunk in chunks:
            chunk_text = _normalize_text(chunk.text)
            if export_text in chunk_text or chunk_text in export_text:
                score = min(len(export_text), len(chunk_text)) / max(len(export_text), len(chunk_text), 1)
                if score > best_score:
                    best_score = score
                    best = chunk
            elif export_text[:40] and export_text[:40] in chunk_text:
                score = 0.5
                if score > best_score:
                    best_score = score
                    best = chunk

        if best is None:
            not_found.append(f"CSV page={export_page} section={export_section}: {export_text[:60]}...")
            continue

        matched += 1
        chunk_page = best.page
        chunk_section = (best.section_name or "").strip()

        page_ok = chunk_page == export_page
        section_ok = export_section in chunk_section or chunk_section in export_section or export_section.split()[0] in chunk_section

        status_parts = []
        if not page_ok:
            page_mismatches.append(
                f"  page CSV={export_page} vs chunk={chunk_page} | section={export_section} | {export_text[:50]}..."
            )
            status_parts.append(f"PAGE MISMATCH (csv={export_page}, chunk={chunk_page})")
        if not section_ok:
            section_mismatches.append(
                f"  section CSV={export_section!r} vs chunk={chunk_section!r} | page csv={export_page} chunk={chunk_page}"
            )
            status_parts.append(f"SECTION MISMATCH (csv={export_section!r}, chunk={chunk_section!r})")

        if page_ok and section_ok:
            print(f"OK  page={export_page} section={export_section}")
        else:
            print(f"FAIL {', '.join(status_parts)} | {export_text[:50]}...")

    print(f"\nMatched by text: {matched}/{len(export_rows)}")
    print(f"Page mismatches: {len(page_mismatches)}")
    print(f"Section mismatches: {len(section_mismatches)}")
    print(f"Not found in chunks: {len(not_found)}")

    if page_mismatches:
        print("\n--- Page mismatches ---")
        for m in page_mismatches:
            print(m)
    if section_mismatches:
        print("\n--- Section mismatches ---")
        for m in section_mismatches:
            print(m)
    if not_found:
        print("\n--- Not found ---")
        for m in not_found:
            print(m)

    # Block-level page audit for key sections
    print("\n=== Block-level page audit (headings + first paragraph) ===")
    seen_sections: set[str] = set()
    for block in blocks:
        text = (block.text or "").strip()
        if re.match(r"^\d+\.\d*", text) and block.page:
            key = text[:40]
            if key not in seen_sections:
                seen_sections.add(key)
                print(f"  page {block.page:2d} line {block.line_number:3d} | {text[:70]}")

    return 0 if not page_mismatches and not section_mismatches and not not_found else 1


if __name__ == "__main__":
    sys.exit(main())
