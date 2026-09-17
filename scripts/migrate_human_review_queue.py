from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.security.review.human_review_queue import HumanReviewQueue
from src.features.security.review.review_decisions import (
    STATE_APPROVED,
    STATE_APPROVED_MASKED,
    STATE_COMPLETED,
    STATE_PENDING,
    STATE_REJECTED,
    STATE_REPROCESSED,
    effective_available_actions,
    effective_risk_level,
)

TERMINAL_STATUSES = {
    STATE_APPROVED,
    STATE_APPROVED_MASKED,
    STATE_REJECTED,
    STATE_REPROCESSED,
    STATE_COMPLETED,
}


@dataclass
class MigrationStats:
    total_rows: int = 0
    kept_rows: int = 0
    archived_completed: int = 0
    removed_orphans: int = 0
    removed_malformed: int = 0
    normalized_status: int = 0
    backfilled_risk_level: int = 0
    backfilled_dlp_decision: int = 0


def _load_json(path: Path) -> Any:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status_from_decision(decision: str) -> str | None:
    dec = str(decision or "").upper()
    if dec == "ALLOW":
        return STATE_APPROVED
    if dec == "MASK_AND_ALLOW":
        return STATE_APPROVED_MASKED
    if dec == "BLOCK":
        return STATE_REJECTED
    return None


def _derive_dlp_decision(item: dict[str, Any], normalized_status: str) -> str:
    existing = str(item.get("dlp_decision") or "").strip().upper()
    if existing:
        return existing
    if normalized_status == STATE_APPROVED:
        return "ALLOW"
    if normalized_status == STATE_APPROVED_MASKED:
        return "MASK_AND_ALLOW"
    if normalized_status == STATE_REJECTED:
        return "BLOCK"
    return "HUMAN_REVIEW"


def _has_backing_document(item: dict[str, Any], docs_dir: Path, document_ids: set[str]) -> bool:
    doc_id = str(item.get("document_id") or "").strip()
    doc_name = str(item.get("document_name") or "").strip()
    if doc_id in document_ids:
        return True
    candidates: list[Path] = []
    if doc_name:
        candidates.append(docs_dir / doc_name)
    if doc_id:
        candidates.append(docs_dir / doc_id)
        candidates.extend(Path(p) for p in glob.glob(str(docs_dir / f"{doc_id}.*")))
    return any(path.exists() for path in candidates)


def _latest_timestamp(item: dict[str, Any]) -> float:
    for key in ("reviewed_at", "updated_at", "created_at"):
        value = item.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def migrate_queue(
    *,
    queue_path: Path,
    rag_api_path: Path,
    archive_path: Path,
    backup_path: Path,
    apply: bool,
) -> MigrationStats:
    stats = MigrationStats()
    raw_queue = _load_json(queue_path)
    if not isinstance(raw_queue, list):
        raise RuntimeError(f"Queue file must be a JSON list: {queue_path}")
    stats.total_rows = len(raw_queue)

    rag_db = _load_json(rag_api_path)
    documents = (rag_db or {}).get("documents") if isinstance(rag_db, dict) else {}
    document_ids = set(documents.keys()) if isinstance(documents, dict) else set()
    docs_dir = queue_path.parent

    by_review_id: dict[str, dict[str, Any]] = {}
    archive_rows: list[dict[str, Any]] = []

    for row in raw_queue:
        if not isinstance(row, dict):
            stats.removed_malformed += 1
            archive_rows.append(
                {
                    "archived_at": _now_iso(),
                    "archive_reason": "malformed_row",
                    "row": row,
                }
            )
            continue
        review_id = str(row.get("review_id") or "").strip()
        doc_id = str(row.get("document_id") or "").strip()
        if not review_id or not doc_id:
            stats.removed_malformed += 1
            archive_rows.append(
                {
                    "archived_at": _now_iso(),
                    "archive_reason": "missing_review_or_document_id",
                    "row": row,
                }
            )
            continue
        existing = by_review_id.get(review_id)
        if existing is None or _latest_timestamp(row) >= _latest_timestamp(existing):
            by_review_id[review_id] = row

    cleaned_rows: list[dict[str, Any]] = []
    for row in by_review_id.values():
        normalized_status = HumanReviewQueue._normalize_status(str(row.get("status") or ""))
        mapped_from_decision = _status_from_decision(str(row.get("reviewer_decision") or ""))
        if normalized_status == STATE_PENDING and mapped_from_decision is not None:
            normalized_status = mapped_from_decision
        if normalized_status != str(row.get("status") or ""):
            stats.normalized_status += 1

        dlp_decision = _derive_dlp_decision(row, normalized_status)
        if not row.get("dlp_decision"):
            stats.backfilled_dlp_decision += 1

        normalized = {
            **row,
            "status": normalized_status,
            "review_status": normalized_status,
            "dlp_decision": dlp_decision,
        }
        if not normalized.get("risk_level"):
            stats.backfilled_risk_level += 1
        normalized["risk_level"] = effective_risk_level(normalized)
        normalized["available_actions"] = effective_available_actions(normalized)
        normalized.setdefault("decision_history", list(normalized.get("audit_trail") or []))

        if not _has_backing_document(normalized, docs_dir, document_ids):
            stats.removed_orphans += 1
            archive_rows.append(
                {
                    "archived_at": _now_iso(),
                    "archive_reason": "orphaned_review_row",
                    "row": normalized,
                }
            )
            continue

        if normalized_status in TERMINAL_STATUSES:
            stats.archived_completed += 1
            archive_rows.append(
                {
                    "archived_at": _now_iso(),
                    "archive_reason": "completed_review",
                    "row": normalized,
                }
            )
            continue

        cleaned_rows.append(normalized)

    cleaned_rows.sort(key=_latest_timestamp, reverse=True)
    stats.kept_rows = len(cleaned_rows)

    if apply:
        backup_path.write_text(json.dumps(raw_queue, indent=2), encoding="utf-8")
        queue_path.write_text(json.dumps(cleaned_rows, indent=2), encoding="utf-8")
        with archive_path.open("a", encoding="utf-8") as handle:
            for row in archive_rows:
                handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate and clean human review queue.")
    parser.add_argument(
        "--queue-path",
        default="C:/Users/DELL/Downloads/rag-builder/_documents/human_review_queue.json",
        help="Path to human_review_queue.json",
    )
    parser.add_argument(
        "--rag-api-path",
        default="C:/Users/DELL/Downloads/rag-builder/src/dms_service/consumer_api_service/data/rag_api.json",
        help="Path to rag_api.json containing document metadata",
    )
    parser.add_argument(
        "--archive-path",
        default="C:/Users/DELL/Downloads/rag-builder/_documents/human_review_queue.archive.jsonl",
        help="Path to archive output JSONL",
    )
    parser.add_argument(
        "--backup-path",
        default="C:/Users/DELL/Downloads/rag-builder/_documents/human_review_queue.backup.pre_migration.json",
        help="Path to write full backup before migration",
    )
    parser.add_argument("--apply", action="store_true", help="Persist changes to disk")
    args = parser.parse_args()

    stats = migrate_queue(
        queue_path=Path(args.queue_path),
        rag_api_path=Path(args.rag_api_path),
        archive_path=Path(args.archive_path),
        backup_path=Path(args.backup_path),
        apply=args.apply,
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
