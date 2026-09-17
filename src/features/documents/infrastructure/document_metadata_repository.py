from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any


class MetadataStore:
    """Small JSON-backed metadata store for document lifecycle state."""

    def __init__(self, db_path: str):
        path = Path(db_path)
        if path.suffix.lower() in {".db", ".sqlite", ".sqlite3"}:
            path = path.with_suffix(".json")
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.init()

    def init(self) -> None:
        with self._lock:
            if not self.path.exists():
                self._write_unlocked({"documents": {}, "repository_config": None})

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"documents": {}, "repository_config": None}
        raw = self.path.read_text(encoding="utf-8") or "{}"
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError:
            # Recover truncated/concatenated writes: keep the first complete JSON value.
            loaded, _idx = json.JSONDecoder().raw_decode(raw)
        if not isinstance(loaded, dict):
            return {"documents": {}, "repository_config": None}
        return loaded or {"documents": {}, "repository_config": None}

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, indent=2, sort_keys=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(self.path)

    def upsert_repository_config(self, repository_type: str, config: dict[str, Any]) -> dict[str, Any]:
        now = time.time()
        record = {"repository_type": repository_type, "config": config, "updated_at": now}
        with self._lock:
            data = self._read_unlocked()
            data["repository_config"] = record
            self._write_unlocked(data)
        return record

    def get_repository_config(self, default_type: str, default_path: str) -> dict[str, Any]:
        with self._lock:
            row = self._read_unlocked().get("repository_config")
        if not row:
            return {"repository_type": default_type, "config": {"document_root": default_path}, "updated_at": None}
        return row

    @staticmethod
    def _record_matches_hashes(
        record: dict[str, Any],
        *,
        content_hash: str | None,
        content_text_hash: str | None,
    ) -> bool:
        metadata = record.get("metadata") or {}
        existing_bytes = str(record.get("content_hash") or metadata.get("content_hash") or "")
        existing_text = str(record.get("content_text_hash") or metadata.get("content_text_hash") or "")
        if content_hash and existing_bytes and existing_bytes == content_hash:
            return True
        if content_text_hash and existing_text and existing_text == content_text_hash:
            return True
        return False

    def find_duplicate_by_content_hash(
        self,
        content_hash: str,
        *,
        content_text_hash: str | None = None,
        repository_id: str | None = None,
        collection_name: str | None = None,
        tenant_id: str | None = None,
        exclude_document_id: str | None = None,
        non_blocking_statuses: frozenset[str] | None = None,
    ) -> dict[str, Any] | None:
        """Locate an existing document with the same byte hash or normalized-text hash."""
        if not content_hash and not content_text_hash:
            return None
        skip_statuses = non_blocking_statuses or frozenset()
        with self._lock:
            rows = list(self._read_unlocked().get("documents", {}).values())
        if repository_id:
            rows = [
                row
                for row in rows
                if row.get("repository_id") == repository_id
                or (row.get("metadata") or {}).get("repository_id") == repository_id
            ]
        elif collection_name:
            rows = [row for row in rows if row.get("collection_name") == collection_name]
        if tenant_id:
            rows = [row for row in rows if row.get("tenant_id") == tenant_id]
        for record in rows:
            doc_id = str(record.get("document_id") or "")
            if exclude_document_id and doc_id == str(exclude_document_id):
                continue
            status = str(record.get("status") or "").strip().lower()
            if status in skip_statuses:
                continue
            if self._record_matches_hashes(
                record,
                content_hash=content_hash,
                content_text_hash=content_text_hash,
            ):
                return dict(record)
        return None

    def insert_document_if_not_duplicate(
        self,
        record: dict[str, Any],
        *,
        content_hash: str,
        content_text_hash: str | None = None,
        repository_id: str | None = None,
        collection_name: str | None = None,
        tenant_id: str | None = None,
        non_blocking_statuses: frozenset[str] | None = None,
    ) -> dict[str, Any] | None:
        """Insert document record unless an in-scope duplicate already exists.

        Returns the conflicting record when a duplicate is detected, otherwise None.
        """
        skip_statuses = non_blocking_statuses or frozenset()
        incoming_text = content_text_hash or str(
            record.get("content_text_hash") or (record.get("metadata") or {}).get("content_text_hash") or ""
        ) or None
        with self._lock:
            rows = list(self._read_unlocked().get("documents", {}).values())
            scoped = rows
            if repository_id:
                scoped = [
                    row
                    for row in scoped
                    if row.get("repository_id") == repository_id
                    or (row.get("metadata") or {}).get("repository_id") == repository_id
                ]
            elif collection_name:
                scoped = [row for row in scoped if row.get("collection_name") == collection_name]
            if tenant_id:
                scoped = [row for row in scoped if row.get("tenant_id") == tenant_id]
            exclude_id = str(record.get("document_id") or "") or None
            for existing in scoped:
                doc_id = str(existing.get("document_id") or "")
                if exclude_id and doc_id == exclude_id:
                    continue
                status = str(existing.get("status") or "").strip().lower()
                if status in skip_statuses:
                    continue
                if self._record_matches_hashes(
                    existing,
                    content_hash=content_hash,
                    content_text_hash=incoming_text,
                ):
                    return dict(existing)
            data = self._read_unlocked()
            documents = data.setdefault("documents", {})
            doc_id = str(record["document_id"])
            if doc_id in documents:
                raise ValueError(f"Document already exists: {doc_id}")
            documents[doc_id] = dict(record)
            self._write_unlocked(data)
        return None

    def insert_document(self, record: dict[str, Any]) -> None:
        with self._lock:
            data = self._read_unlocked()
            documents = data.setdefault("documents", {})
            if record["document_id"] in documents:
                raise ValueError(f"Document already exists: {record['document_id']}")
            documents[record["document_id"]] = dict(record)
            self._write_unlocked(data)

    def upsert_document(self, record: dict[str, Any]) -> dict[str, Any]:
        """Insert or merge a document record (used when rehydrating from MySQL jobs)."""
        with self._lock:
            data = self._read_unlocked()
            documents = data.setdefault("documents", {})
            doc_id = str(record["document_id"])
            existing = documents.get(doc_id)
            if existing is None:
                documents[doc_id] = dict(record)
                self._write_unlocked(data)
                return dict(record)
            merged = dict(existing)
            for key, value in record.items():
                if value in (None, "", {}):
                    continue
                if key not in merged or merged.get(key) in (None, "", {}):
                    merged[key] = value
            documents[doc_id] = merged
            self._write_unlocked(data)
            return dict(merged)

    def patch_document(self, document_id: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        """Force-update selected top-level document fields (e.g. processing, document_type_*)."""
        with self._lock:
            data = self._read_unlocked()
            record = data.setdefault("documents", {}).get(document_id)
            if not record:
                return None
            for key, value in fields.items():
                if value is None:
                    continue
                record[key] = value
            self._write_unlocked(data)
            return dict(record)

    def update_document_status(
        self,
        document_id: str,
        status: str,
        *,
        queued_at: float | None = None,
        completed_at: float | None = None,
        error_details: str | None = None,
    ) -> None:
        with self._lock:
            data = self._read_unlocked()
            record = data.setdefault("documents", {}).get(document_id)
            if not record:
                return
            record["status"] = status
            if queued_at is not None:
                record["queue_submission_timestamp"] = queued_at
            if completed_at is not None:
                record["processing_completion_timestamp"] = completed_at
            if error_details is not None:
                record["error_details"] = error_details
            self._write_unlocked(data)

    SYSTEM_METADATA_KEYS = frozenset({"job_id", "batch_id", "repository_id"})

    def update_document_metadata(
        self,
        document_id: str,
        metadata: dict[str, Any],
        *,
        merge: bool = True,
    ) -> dict[str, Any] | None:
        with self._lock:
            data = self._read_unlocked()
            record = data.setdefault("documents", {}).get(document_id)
            if not record:
                return None
            old_meta = dict(record.get("metadata") or {})
            system_values = {key: old_meta[key] for key in self.SYSTEM_METADATA_KEYS if key in old_meta}
            if merge:
                updated_meta = {**old_meta, **metadata}
            else:
                updated_meta = {**system_values, **metadata}
            for key, value in system_values.items():
                updated_meta[key] = value
            record["metadata"] = updated_meta
            record["metadata_updated_at"] = time.time()
            self._write_unlocked(data)
            return dict(record)

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._read_unlocked().get("documents", {}).get(document_id)
        return dict(row) if row else None

    def list_documents(
        self,
        *,
        status: str | None = None,
        tenant_id: str | None = None,
        collection_name: str | None = None,
        repository_id: str | None = None,
        document_type: str | None = None,
        original_file_name: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._read_unlocked().get("documents", {}).values())
        if status:
            rows = [row for row in rows if row.get("status") == status]
        if tenant_id:
            rows = [row for row in rows if row.get("tenant_id") == tenant_id]
        if collection_name:
            rows = [row for row in rows if row.get("collection_name") == collection_name]
        if repository_id:
            rows = [
                row
                for row in rows
                if row.get("repository_id") == repository_id
                or (row.get("metadata") or {}).get("repository_id") == repository_id
            ]
        if document_type:
            rows = [row for row in rows if row.get("document_type") == document_type]
        if original_file_name:
            needle = original_file_name.lower()
            rows = [
                row
                for row in rows
                if needle in str(row.get("original_file_name") or "").lower()
                or needle in str(row.get("document_name") or "").lower()
            ]
        if metadata_filters:
            rows = [row for row in rows if self._matches_metadata(row, metadata_filters)]
        rows.sort(key=lambda row: row.get("upload_timestamp", 0), reverse=True)
        return [dict(row) for row in rows[offset : offset + limit]]

    @staticmethod
    def _matches_metadata(record: dict[str, Any], metadata_filters: dict[str, Any]) -> bool:
        for key, expected in metadata_filters.items():
            if expected in (None, ""):
                continue
            if key.startswith("metadata."):
                actual = (record.get("metadata") or {}).get(key.split(".", 1)[1])
            else:
                actual = record.get(key)
            if str(actual) != str(expected):
                return False
        return True

    def count_documents(
        self,
        *,
        status: str | None = None,
        tenant_id: str | None = None,
        collection_name: str | None = None,
        repository_id: str | None = None,
        document_type: str | None = None,
        original_file_name: str | None = None,
        metadata_filters: dict[str, Any] | None = None,
    ) -> int:
        return len(
            self.list_documents(
                status=status,
                tenant_id=tenant_id,
                collection_name=collection_name,
                repository_id=repository_id,
                document_type=document_type,
                original_file_name=original_file_name,
                metadata_filters=metadata_filters,
                limit=1_000_000,
                offset=0,
            )
        )

    def delete_document(self, document_id: str) -> bool:
        with self._lock:
            data = self._read_unlocked()
            documents = data.setdefault("documents", {})
            if document_id not in documents:
                return False
            del documents[document_id]
            self._write_unlocked(data)
        return True
