"""Deep compare CSV chunks against DOCX table/paragraph content."""
from __future__ import annotations

import csv
import re
from pathlib import Path

from docx import Document


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip())


def table_as_pipe_text(table) -> str:
    rows = []
    for row in table.rows:
        cells = [c.text.strip() for c in row.cells if c.text.strip()]
        if cells:
            rows.append(" | ".join(cells))
    return " | ".join(rows)


def main() -> None:
    docx_path = Path(r"c:\Users\Neevan\Downloads\SOP-CL-TM-001_Study_Startup.docx")
    csv_path = Path(r"c:\Users\Neevan\Downloads\2026-06-25T07-21_export.csv")

    doc = Document(str(docx_path))
    all_para_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    all_tables = [table_as_pipe_text(t) for t in doc.tables]

    print("=== DOCX Tables ===")
    for i, ttext in enumerate(all_tables):
        print(f"\nTable {i} ({len(ttext)} chars):")
        print(ttext[:300], "...")

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print("\n\n=== Per-chunk deep match ===")
    for row in rows:
        csv_text = row["text"]
        csv_page = row["page"]
        csv_section = row["section_name"]
        csv_cat = row["category"]

        # Extract key phrases (3+ word sequences)
        phrases = []
        for part in re.split(r"[|,\n]", csv_text):
            part = part.strip()
            if len(part) > 15:
                phrases.append(part[:50])

        para_hits = sum(1 for p in phrases if norm(p[:30]) in norm(all_para_text))
        table_hits = sum(1 for p in phrases if any(norm(p[:30]) in norm(t) for t in all_tables))

        # Best table match
        best_table = -1
        best_overlap = 0
        csv_norm = norm(csv_text)
        for i, ttext in enumerate(all_tables):
            overlap = 0
            for p in phrases:
                if norm(p[:30]) in norm(ttext):
                    overlap += 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_table = i

        status = "OK" if (para_hits + table_hits) >= min(2, len(phrases)) or best_overlap >= 2 else "WEAK"
        print(f"\n{status} page={csv_page} section={csv_section} category={csv_cat}")
        print(f"  text: {csv_text[:100]}...")
        print(f"  phrase hits: para={para_hits} table={table_hits} best_table={best_table} overlap={best_overlap}")
        if best_table >= 0 and best_overlap > 0:
            print(f"  maps to Table {best_table}")

    # Section-to-heading mapping from docx
    print("\n\n=== Expected section order in DOCX ===")
    current_h1 = ""
    for p in doc.paragraphs:
        t = p.text.strip()
        if not t:
            continue
        style = p.style.name if p.style else ""
        if "Heading 1" in style:
            current_h1 = t
            print(f"H1: {t}")
        elif "Heading 2" in style:
            print(f"  H2 under {current_h1}: {t}")


if __name__ == "__main__":
    main()
