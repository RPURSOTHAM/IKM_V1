#!/usr/bin/env python3
"""Phase 3 migration: ensure Basic document types and migrate repository key_fields.

Idempotent. For every repository:
1. Create Basic document type (system root) when missing.
2. Set settings.document_type_id to Basic when unset.
3. Migrate settings.key_fields into Basic (skip names already present).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.features.document_types.infrastructure.document_type_migration import migrate_repository_document_types
from src.features.repositories.application.repository_service import get_repository_service


@dataclass
class MigrationStats:
    repositories_seen: int = 0
    basic_created: int = 0
    basic_existing: int = 0
    document_type_id_set: int = 0
    fields_migrated: int = 0
    fields_skipped: int = 0
    errors: list[str] = field(default_factory=list)


def migrate_repository(repository_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    repo_service = get_repository_service()
    settings_payload = repo_service.get_settings(repository_id)
    settings = dict(settings_payload.get("settings") or {})
    return migrate_repository_document_types(
        repository_id,
        dry_run=dry_run,
        settings=settings,
        update_settings=lambda rid, patch: repo_service.update_settings(rid, patch),
    )


def run_migration(*, dry_run: bool = False, repository_id: str | None = None) -> MigrationStats:
    stats = MigrationStats()
    repo_service = get_repository_service()
    if repository_id:
        repositories = [{"repository_id": repository_id}]
    else:
        listing = repo_service.list_repositories()
        repositories = listing.get("repositories") or []

    for repo in repositories:
        rid = str(repo.get("repository_id") or "").strip()
        if not rid:
            continue
        stats.repositories_seen += 1
        try:
            result = migrate_repository(rid, dry_run=dry_run)
            if result.get("basic_created"):
                stats.basic_created += 1
            else:
                stats.basic_existing += 1
            if result.get("document_type_id_set"):
                stats.document_type_id_set += 1
            stats.fields_migrated += int(result.get("fields_migrated") or 0)
            stats.fields_skipped += int(result.get("fields_skipped") or 0)
            print(
                f"[ok] repository={rid} basic_created={result.get('basic_created')} "
                f"migrated={result.get('fields_migrated')} skipped={result.get('fields_skipped')}"
            )
        except Exception as exc:
            message = f"repository={rid} error={exc}"
            stats.errors.append(message)
            print(f"[error] {message}", file=sys.stderr)
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report actions without writing.")
    parser.add_argument("--repository-id", default=None, help="Limit migration to one repository.")
    args = parser.parse_args(argv)

    stats = run_migration(dry_run=bool(args.dry_run), repository_id=args.repository_id)
    print(
        "summary: "
        f"seen={stats.repositories_seen} "
        f"basic_created={stats.basic_created} "
        f"basic_existing={stats.basic_existing} "
        f"document_type_id_set={stats.document_type_id_set} "
        f"fields_migrated={stats.fields_migrated} "
        f"fields_skipped={stats.fields_skipped} "
        f"errors={len(stats.errors)}"
    )
    return 1 if stats.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
