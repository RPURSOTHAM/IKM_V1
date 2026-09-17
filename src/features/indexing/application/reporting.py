"""
reporting.py
============
Excel report generation and data structuring for Neo4j exports.
"""

from __future__ import annotations

import os
import numpy as np
import xlsxwriter

from src.features.document_processing.core.logger import log
from src.features.chunking.application.chunking_service import Chunk
from src.features.document_processing.utilities.text_utils import clean_text_for_display, normalize_text_for_embedding


def _normalized_text(chunk: Chunk) -> str:
    return chunk.normalized_text or normalize_text_for_embedding(chunk.text)


def _normalized_word_count(chunk: Chunk) -> int:
    return len(_normalized_text(chunk).split())


# ─────────────────────────────────────────────────────────────────────────────
# PAIRWISE SIMILARITY EXPORT (for Neo4j)
# ─────────────────────────────────────────────────────────────────────────────


def build_similarity_pairs(
    chunks: list[Chunk],
    min_similarity: float,
    min_content_words: int = 10,
) -> list[dict]:
    """
    Build document-level similarity pairs for Neo4j edge export (`SIMILAR_TO`).
    Filters out duplicates and same-document pairs, then aggregates chunk-level
    matches into the payload expected by `Neo4jGraph.push_similarity()`.
    """
    valid_indices = []
    for i, c in enumerate(chunks):
        if c.is_duplicate or c.embedding is None:
            continue
        if _normalized_word_count(c) < min_content_words:
            continue
        valid_indices.append(i)

    if not valid_indices:
        return []

    X = np.stack([chunks[i].embedding for i in valid_indices])
    sim_matrix = X @ X.T
    np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)

    aggregated: dict[tuple[str, str], dict[str, object]] = {}
    for i in range(len(valid_indices)):
        c_i = chunks[valid_indices[i]]
        for j in range(i + 1, len(valid_indices)):
            c_j = chunks[valid_indices[j]]
            if c_i.doc_name == c_j.doc_name:
                continue
            sim = float(sim_matrix[i, j])
            if sim >= min_similarity:
                if c_i.doc_name <= c_j.doc_name:
                    src_doc, tgt_doc = c_i.doc_name, c_j.doc_name
                    src_section, tgt_section = c_i.section_name, c_j.section_name
                else:
                    src_doc, tgt_doc = c_j.doc_name, c_i.doc_name
                    src_section, tgt_section = c_j.section_name, c_i.section_name

                pair_key = (src_doc, tgt_doc)
                bucket = aggregated.setdefault(
                    pair_key,
                    {
                        "src_doc": src_doc,
                        "tgt_doc": tgt_doc,
                        "scores": [],
                        "section_scores": {},
                    },
                )
                bucket["scores"].append(sim)

                if src_section or tgt_section:
                    if src_section and tgt_section and src_section != tgt_section:
                        section_name = f"{src_section} <> {tgt_section}"
                    else:
                        section_name = src_section or tgt_section
                    section_scores = bucket["section_scores"]
                    section_scores.setdefault(section_name, []).append(sim)

    pair_results: list[dict] = []
    for bucket in aggregated.values():
        section_results = []
        for section_name, scores in sorted(bucket["section_scores"].items()):
            if not scores:
                continue
            section_results.append(
                {
                    "section_name": section_name,
                    "weighted_sim": round(max(scores), 4),
                }
            )

        pair_results.append(
            {
                "src_doc": bucket["src_doc"],
                "tgt_doc": bucket["tgt_doc"],
                "overall_sim": round(max(bucket["scores"]), 4) if bucket["scores"] else 0.0,
                "section_results": section_results,
            }
        )

    return pair_results


# ─────────────────────────────────────────────────────────────────────────────
# EXCEL REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────


def build_pairwise_similarity_report(
    chunks: list[Chunk],
    output_path: str,
    min_similarity: float,
    min_content_words: int = 10,
) -> None:
    """Generate similarity_report.xlsx."""
    if not chunks:
        log.warning("No chunks to report.")
        return

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    detail_path = output_path
    if not detail_path.endswith(".xlsx"):
        detail_path = os.path.join(output_path, "similarity_report.xlsx")

    report_min_similarity = max(min_similarity, 0.75)

    # Filter out invalid chunks early
    valid_chunks = []
    embeddings = []
    for c in chunks:
        if c.is_duplicate or c.embedding is None:
            continue
        if _normalized_word_count(c) < min_content_words:
            continue
        valid_chunks.append(c)
        embeddings.append(c.embedding)

    if not valid_chunks:
        log.warning("No valid chunks for reporting.")
        return

    X = np.stack(embeddings)

    doc_to_indices: dict[str, list[int]] = {}
    for idx, c in enumerate(valid_chunks):
        doc_to_indices.setdefault(c.doc_name, []).append(idx)

    doc_names_sorted = sorted(doc_to_indices.keys())

    if len(doc_names_sorted) < 2:
        log.info("Only 1 document present. Pairwise report skipped.")
        return

    # ── Detail Report setup ──────────────────────────────────────────
    wb1 = xlsxwriter.Workbook(detail_path)

    # ✅ FIX 4: Wrap ALL workbook operations in try/finally so both workbooks
    # are explicitly closed even if an exception occurs mid-report.
    # xlsxwriter's own docs warn that missing close() corrupts the output
    # file since data is only flushed to disk inside close().
    try:
        ws1 = wb1.add_worksheet("Matches")
        hdr1 = wb1.add_format({
            "bold": True, "bg_color": "#1F4E79", "font_color": "white",
            "border": 1, "font_name": "Arial", "font_size": 10,
        })
        plain1 = wb1.add_format({
            "border": 1, "text_wrap": True, "valign": "top",
            "font_name": "Arial", "font_size": 10,
        })
        center1 = wb1.add_format({
            "border": 1, "align": "center", "valign": "vcenter",
            "font_name": "Arial", "font_size": 10,
        })

        det_headers = [
            "Source Document", "Section Context", "Page", "Source Content",
            "Target Document", "Section Context", "Page", "Target Content",
            "Similarity %",
        ]
        det_col_widths = [30, 25, 6, 60, 30, 25, 6, 60, 14]

        for col, (h, w) in enumerate(zip(det_headers, det_col_widths)):
            ws1.write(0, col, h, hdr1)
            ws1.set_column(col, col, w)
        ws1.set_row(0, 22)
        ws1.freeze_panes(1, 0)

        # ── Summary Report setup ─────────────────────────────────────────
        # ── Execute comparisons ──────────────────────────────────────────
        detail_row = 1

        total_pairs = len(doc_names_sorted) * (len(doc_names_sorted) - 1) // 2
        log.info(
            "Generating similarity report for %d document pair(s) (threshold >= %.0f%%) ...",
            total_pairs,
            report_min_similarity * 100.0,
        )

        for i, src_doc in enumerate(doc_names_sorted):
            src_indices = doc_to_indices.get(src_doc, [])
            if not src_indices:
                continue

            log.info("  [%d/%d] %s", i + 1, len(doc_names_sorted), src_doc)
            X_src = X[src_indices]

            for tgt_doc in doc_names_sorted[i + 1:]:
                tgt_indices = doc_to_indices.get(tgt_doc, [])
                if not tgt_indices:
                    continue

                X_tgt = X[tgt_indices]
                sim_matrix = X_src @ X_tgt.T
                np.clip(sim_matrix, 0.0, 1.0, out=sim_matrix)

                for a, src_idx in enumerate(src_indices):
                    for b, tgt_idx in enumerate(tgt_indices):
                        sim = float(sim_matrix[a, b])
                        if sim < report_min_similarity:
                            continue

                        src_chunk = valid_chunks[src_idx]
                        tgt_chunk = valid_chunks[tgt_idx]
                        sim_pct = round(sim * 100.0, 2)

                        src_display = clean_text_for_display(_normalized_text(src_chunk))
                        tgt_display = clean_text_for_display(_normalized_text(tgt_chunk))

                        if not src_display or not tgt_display:
                            continue

                        ws1.write(detail_row, 0, src_chunk.doc_name, plain1)
                        ws1.write(detail_row, 1, src_chunk.section_path or src_chunk.section_name, plain1)
                        ws1.write(detail_row, 2, src_chunk.page, center1)
                        ws1.write(detail_row, 3, src_display, plain1)
                        ws1.write(detail_row, 4, tgt_chunk.doc_name, plain1)
                        ws1.write(detail_row, 5, tgt_chunk.section_path or tgt_chunk.section_name, plain1)
                        ws1.write(detail_row, 6, tgt_chunk.page, center1)
                        ws1.write(detail_row, 7, tgt_display, plain1)
                        ws1.write(detail_row, 8, sim_pct, center1)
                        detail_row += 1

    finally:
        wb1.close()

    log.info("Detail  -> %s  (%d rows)", detail_path, detail_row - 1)


def build_chunk_quality_report(
    chunks: list[Chunk],
    output_path: str,
    min_content_words: int = 10,
    chunk_size_target: int | None = None,
) -> None:
    """Generate an Excel workbook for manual chunk-quality review."""
    if not chunks:
        log.warning("No chunks available for chunk-quality reporting.")
        return

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    quality_path = output_path
    if not quality_path.endswith(".xlsx"):
        quality_path = os.path.join(output_path, "chunk_quality_report.xlsx")

    wb = xlsxwriter.Workbook(quality_path)
    try:
        ws_chunks = wb.add_worksheet("Chunk Audit")
        ws_summary = wb.add_worksheet("Doc Summary")

        hdr = wb.add_format({
            "bold": True, "bg_color": "#1F4E79", "font_color": "white",
            "border": 1, "font_name": "Arial", "font_size": 10,
        })
        plain = wb.add_format({
            "border": 1, "text_wrap": True, "valign": "top",
            "font_name": "Arial", "font_size": 10,
        })
        center = wb.add_format({
            "border": 1, "align": "center", "valign": "vcenter",
            "font_name": "Arial", "font_size": 10,
        })
        good_fmt = wb.add_format({
            "bg_color": "#C6EFCE", "border": 1, "align": "center",
            "valign": "vcenter", "bold": True, "font_name": "Arial", "font_size": 10,
        })
        review_fmt = wb.add_format({
            "bg_color": "#FFEB9C", "border": 1, "align": "center",
            "valign": "vcenter", "bold": True, "font_name": "Arial", "font_size": 10,
        })
        poor_fmt = wb.add_format({
            "bg_color": "#FFC7CE", "border": 1, "align": "center",
            "valign": "vcenter", "bold": True, "font_name": "Arial", "font_size": 10,
        })

        headers = [
            "Doc Name",
            "Section Name",
            "Section Path",
            "Page",
            "Chunk ID",
            "Word Count",
            "Sentence Count",
            "Character Count",
            "Duplicate",
            "Cluster ID",
            "Match %",
            "Quality",
            "Quality Notes",
            "Chunk Text",
        ]
        widths = [38, 24, 34, 8, 38, 12, 14, 14, 10, 12, 12, 12, 38, 90]

        for col, (title, width) in enumerate(zip(headers, widths)):
            ws_chunks.write(0, col, title, hdr)
            ws_chunks.set_column(col, col, width)
        ws_chunks.set_row(0, 22)
        ws_chunks.freeze_panes(1, 0)
        ws_chunks.autofilter(0, 0, len(chunks), len(headers) - 1)

        doc_summary: dict[str, dict[str, float]] = {}

        for row, chunk in enumerate(chunks, start=1):
            raw_text = (chunk.raw_text or chunk.text or "").strip()
            clean_text = clean_text_for_display(raw_text) or raw_text
            word_count = len(raw_text.split())
            sentence_count = sum(raw_text.count(mark) for mark in ".!?")
            sentence_count = max(sentence_count, 1 if raw_text else 0)
            char_count = len(raw_text)

            notes: list[str] = []
            quality = "Good"
            quality_format = good_fmt

            if word_count < min_content_words:
                notes.append(f"below_min_words<{min_content_words}")
                quality = "Poor"
                quality_format = poor_fmt
            elif chunk_size_target and word_count < max(min_content_words, int(chunk_size_target * 0.35)):
                notes.append("short_vs_target")
                quality = "Review"
                quality_format = review_fmt

            if chunk_size_target and word_count > int(chunk_size_target * 1.15):
                notes.append("long_vs_target")
                if quality != "Poor":
                    quality = "Review"
                    quality_format = review_fmt

            if sentence_count <= 1:
                notes.append("single_sentence")
                if quality == "Good":
                    quality = "Review"
                    quality_format = review_fmt

            if chunk.is_duplicate:
                notes.append("duplicate")
                if quality == "Good":
                    quality = "Review"
                    quality_format = review_fmt

            if not clean_text.strip():
                notes.append("empty_display_text")
                quality = "Poor"
                quality_format = poor_fmt

            ws_chunks.write(row, 0, chunk.doc_name, plain)
            ws_chunks.write(row, 1, chunk.section_name, plain)
            ws_chunks.write(row, 2, chunk.section_path or chunk.section_name, plain)
            ws_chunks.write(row, 3, chunk.page, center)
            ws_chunks.write(row, 4, chunk.id, plain)
            ws_chunks.write(row, 5, word_count, center)
            ws_chunks.write(row, 6, sentence_count, center)
            ws_chunks.write(row, 7, char_count, center)
            ws_chunks.write(row, 8, "Yes" if chunk.is_duplicate else "No", center)
            ws_chunks.write(row, 9, "" if chunk.cluster_id is None else chunk.cluster_id, center)
            ws_chunks.write(row, 10, "" if chunk.match_pct is None else round(float(chunk.match_pct), 2), center)
            ws_chunks.write(row, 11, quality, quality_format)
            ws_chunks.write(row, 12, ", ".join(notes) if notes else "ok", plain)
            ws_chunks.write(row, 13, raw_text, plain)

            stats = doc_summary.setdefault(
                chunk.doc_name,
                {
                    "chunks": 0,
                    "words_total": 0,
                    "min_words": float("inf"),
                    "max_words": 0,
                    "review_count": 0,
                    "poor_count": 0,
                    "duplicate_count": 0,
                },
            )
            stats["chunks"] += 1
            stats["words_total"] += word_count
            stats["min_words"] = min(stats["min_words"], word_count)
            stats["max_words"] = max(stats["max_words"], word_count)
            stats["review_count"] += int(quality == "Review")
            stats["poor_count"] += int(quality == "Poor")
            stats["duplicate_count"] += int(chunk.is_duplicate)

        summary_headers = [
            "Doc Name",
            "Chunk Count",
            "Avg Words",
            "Min Words",
            "Max Words",
            "Review Chunks",
            "Poor Chunks",
            "Duplicate Chunks",
        ]
        summary_widths = [40, 14, 12, 12, 12, 14, 12, 16]

        for col, (title, width) in enumerate(zip(summary_headers, summary_widths)):
            ws_summary.write(0, col, title, hdr)
            ws_summary.set_column(col, col, width)
        ws_summary.set_row(0, 22)
        ws_summary.freeze_panes(1, 0)

        for row, doc_name in enumerate(sorted(doc_summary), start=1):
            stats = doc_summary[doc_name]
            avg_words = round(stats["words_total"] / stats["chunks"], 2) if stats["chunks"] else 0
            ws_summary.write(row, 0, doc_name, plain)
            ws_summary.write(row, 1, int(stats["chunks"]), center)
            ws_summary.write(row, 2, avg_words, center)
            ws_summary.write(row, 3, 0 if stats["min_words"] == float("inf") else int(stats["min_words"]), center)
            ws_summary.write(row, 4, int(stats["max_words"]), center)
            ws_summary.write(row, 5, int(stats["review_count"]), center)
            ws_summary.write(row, 6, int(stats["poor_count"]), center)
            ws_summary.write(row, 7, int(stats["duplicate_count"]), center)
    finally:
        wb.close()

    log.info("Chunk quality report -> %s  (%d chunks)", quality_path, len(chunks))


def build_post_embedding_chunk_report(
    chunks: list[Chunk],
    output_path: str,
    min_content_words: int = 10,
) -> None:
    """
    Generate an Excel workbook for chunks after embedding/similarity eligibility.
    This focuses on chunks as they exist after processing rather than raw text only.
    """
    if not chunks:
        log.warning("No chunks available for post-embedding reporting.")
        return

    processed_chunks = []
    for chunk in chunks:
        rewritten_text = chunk.rewritten_text or chunk.text
        normalized_text = chunk.normalized_text or normalize_text_for_embedding(rewritten_text)
        embedding_ready = chunk.embedding is not None
        eligible = (
            embedding_ready
            and not chunk.is_duplicate
            and len(rewritten_text.split()) >= min_content_words
            and bool(normalized_text.strip())
        )
        processed_chunks.append((chunk, rewritten_text, normalized_text, eligible))

    out_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(out_dir, exist_ok=True)

    report_path = output_path
    if not report_path.endswith(".xlsx"):
        report_path = os.path.join(output_path, "post_embedding_chunk_report.xlsx")

    wb = xlsxwriter.Workbook(report_path)
    try:
        ws_chunks = wb.add_worksheet("Processed Chunks")
        ws_summary = wb.add_worksheet("Usage Summary")

        hdr = wb.add_format({
            "bold": True, "bg_color": "#1F4E79", "font_color": "white",
            "border": 1, "font_name": "Arial", "font_size": 10,
        })
        plain = wb.add_format({
            "border": 1, "text_wrap": True, "valign": "top",
            "font_name": "Arial", "font_size": 10,
        })
        center = wb.add_format({
            "border": 1, "align": "center", "valign": "vcenter",
            "font_name": "Arial", "font_size": 10,
        })
        used_fmt = wb.add_format({
            "bg_color": "#C6EFCE", "border": 1, "align": "center",
            "valign": "vcenter", "bold": True, "font_name": "Arial", "font_size": 10,
        })
        filtered_fmt = wb.add_format({
            "bg_color": "#FFEB9C", "border": 1, "align": "center",
            "valign": "vcenter", "bold": True, "font_name": "Arial", "font_size": 10,
        })
        excluded_fmt = wb.add_format({
            "bg_color": "#FFC7CE", "border": 1, "align": "center",
            "valign": "vcenter", "bold": True, "font_name": "Arial", "font_size": 10,
        })

        headers = [
            "Doc Name",
            "Section Name",
            "Section Path",
            "Page",
            "Chunk ID",
            "Raw Chunk Text",
            "Rewritten Chunk Text",
            "Normalized Text Used For Embedding",
            "Embedding Present",
            "Embedding Dim",
            "Duplicate",
            "Cluster ID",
            "Match %",
            "Used In Similarity",
            "Processing Status",
            "Processing Notes",
        ]
        widths = [38, 24, 34, 8, 38, 65, 65, 65, 14, 14, 10, 12, 12, 14, 18, 38]

        for col, (title, width) in enumerate(zip(headers, widths)):
            ws_chunks.write(0, col, title, hdr)
            ws_chunks.set_column(col, col, width)
        ws_chunks.set_row(0, 22)
        ws_chunks.freeze_panes(1, 0)
        ws_chunks.autofilter(0, 0, len(processed_chunks), len(headers) - 1)

        doc_summary: dict[str, dict[str, int]] = {}

        for row, (chunk, rewritten_text, normalized_text, eligible) in enumerate(processed_chunks, start=1):
            notes: list[str] = []
            status = "Used"
            status_fmt = used_fmt

            if chunk.embedding is None:
                notes.append("missing_embedding")
                status = "Excluded"
                status_fmt = excluded_fmt
            if len(rewritten_text.split()) < min_content_words:
                notes.append(f"below_min_words<{min_content_words}")
                status = "Excluded"
                status_fmt = excluded_fmt
            if not normalized_text.strip():
                notes.append("empty_normalized_text")
                status = "Excluded"
                status_fmt = excluded_fmt
            if rewritten_text == "[SKIP: Non-narrative content]":
                notes.append("non_narrative_content")
                status = "Excluded"
                status_fmt = excluded_fmt
            if chunk.is_duplicate:
                notes.append("duplicate_filtered")
                if status != "Excluded":
                    status = "Filtered"
                    status_fmt = filtered_fmt

            embedding_dim = 0
            if chunk.embedding is not None:
                embedding_dim = int(getattr(chunk.embedding, "shape", [len(chunk.embedding)])[0])

            ws_chunks.write(row, 0, chunk.doc_name, plain)
            ws_chunks.write(row, 1, chunk.section_name, plain)
            ws_chunks.write(row, 2, chunk.section_path or chunk.section_name, plain)
            ws_chunks.write(row, 3, chunk.page, center)
            ws_chunks.write(row, 4, chunk.id, plain)
            ws_chunks.write(row, 5, chunk.raw_text or "", plain)
            ws_chunks.write(row, 6, rewritten_text, plain)
            ws_chunks.write(row, 7, normalized_text, plain)
            ws_chunks.write(row, 8, "Yes" if chunk.embedding is not None else "No", center)
            ws_chunks.write(row, 9, embedding_dim, center)
            ws_chunks.write(row, 10, "Yes" if chunk.is_duplicate else "No", center)
            ws_chunks.write(row, 11, "" if chunk.cluster_id is None else chunk.cluster_id, center)
            ws_chunks.write(row, 12, "" if chunk.match_pct is None else round(float(chunk.match_pct), 2), center)
            ws_chunks.write(row, 13, "Yes" if eligible else "No", center)
            ws_chunks.write(row, 14, status, status_fmt)
            ws_chunks.write(row, 15, ", ".join(notes) if notes else "eligible", plain)

            stats = doc_summary.setdefault(
                chunk.doc_name,
                {"total": 0, "used": 0, "filtered": 0, "excluded": 0},
            )
            stats["total"] += 1
            stats["used"] += int(status == "Used")
            stats["filtered"] += int(status == "Filtered")
            stats["excluded"] += int(status == "Excluded")

        summary_headers = ["Doc Name", "Total Chunks", "Used", "Filtered", "Excluded"]
        summary_widths = [40, 14, 10, 10, 10]
        for col, (title, width) in enumerate(zip(summary_headers, summary_widths)):
            ws_summary.write(0, col, title, hdr)
            ws_summary.set_column(col, col, width)
        ws_summary.set_row(0, 22)
        ws_summary.freeze_panes(1, 0)

        for row, doc_name in enumerate(sorted(doc_summary), start=1):
            stats = doc_summary[doc_name]
            ws_summary.write(row, 0, doc_name, plain)
            ws_summary.write(row, 1, stats["total"], center)
            ws_summary.write(row, 2, stats["used"], center)
            ws_summary.write(row, 3, stats["filtered"], center)
            ws_summary.write(row, 4, stats["excluded"], center)
    finally:
        wb.close()

    log.info("Post-embedding chunk report -> %s  (%d chunks)", report_path, len(processed_chunks))
