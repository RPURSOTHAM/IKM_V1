"""Unit tests for Phase 3.2 RepositoryStore persistence (mocked MySQL)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from src.features.repositories.domain.constants import STATUS_ARCHIVED, STATUS_CONFIGURING
from src.features.repositories.domain.repository_exceptions import (
    DuplicateRepository,
    RepositoryNotFound,
)
from src.features.repositories.domain.repository import RepositorySettings
from src.features.repositories.infrastructure.repository_repository import RepositoryStore
from src.infrastructure.database.document_jobs import MySQLConnectionParams


def _params() -> MySQLConnectionParams:
    return MySQLConnectionParams(
        host="localhost",
        port=3306,
        user="rag",
        password="rag",
        database="rag_builder",
    )


def _utc() -> datetime:
    return datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc).replace(tzinfo=None)


def _row(
    *,
    repo_id: str = "11111111-1111-4111-8111-111111111111",
    name: str = "PlantDocs",
    status: str = STATUS_CONFIGURING,
) -> tuple[Any, ...]:
    now = _utc()
    return (
        repo_id,
        name,
        "owner-1",
        "Owner One",
        "PlantDocs",
        None,
        status,
        now,
        now,
        None,
    )


class _Cursor:
    def __init__(self, *, fetchone=None, fetchall=None, rowcount: int = 1):
        self._fetchone = fetchone
        self._fetchall = fetchall or []
        self.rowcount = rowcount
        self.statements: list[tuple[str, Any]] = []

    def execute(self, sql: str, params=None) -> None:
        self.statements.append((sql, params))

    def fetchone(self):
        if callable(self._fetchone):
            return self._fetchone(self)
        return self._fetchone

    def fetchall(self):
        return list(self._fetchall)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _Conn:
    def __init__(self, cursor: _Cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


def test_row_mapping_to_domain_model() -> None:
    record = RepositoryStore._row_to_record(_row())
    assert record.repository_id == "11111111-1111-4111-8111-111111111111"
    assert record.name == "PlantDocs"
    assert record.status == STATUS_CONFIGURING
    assert record.owner_user_name == "Owner One"
    assert record.settings_locked_at is None


def test_create_repository_with_settings_commits_transaction() -> None:
    cursor = _Cursor(fetchone=None, rowcount=1)
    conn = _Conn(cursor)
    store = RepositoryStore(_params())

    with patch.object(store, "_connect", return_value=conn):
        record, settings = store.create_repository_with_settings(
            name="PlantDocs",
            owner_user_id="owner-1",
            weaviate_collection="PlantDocs",
            settings={"chunk_size": 512},
            repository_id="11111111-1111-4111-8111-111111111111",
        )

    assert conn.committed is True
    assert conn.rolled_back is False
    assert record.name == "PlantDocs"
    assert isinstance(settings, RepositorySettings)
    assert settings.settings["chunk_size"] == 512
    assert len(cursor.statements) == 2
    assert "INSERT INTO repository" in cursor.statements[0][0]
    assert "INSERT INTO repository_settings" in cursor.statements[1][0]


def test_create_repository_with_settings_rolls_back_on_duplicate() -> None:
    class _DupCursor(_Cursor):
        def execute(self, sql: str, params=None) -> None:
            self.statements.append((sql, params))
            err = Exception("Duplicate entry")
            err.args = (1062, "Duplicate entry")
            raise err

    cursor = _DupCursor()
    conn = _Conn(cursor)
    store = RepositoryStore(_params())

    with patch.object(store, "_connect", return_value=conn):
        with pytest.raises(DuplicateRepository):
            store.create_repository_with_settings(
                name="PlantDocs",
                owner_user_id="owner-1",
                weaviate_collection="PlantDocs",
                settings={},
            )

    assert conn.rolled_back is True
    assert conn.committed is False


def test_archive_is_soft_delete() -> None:
    repo_id = "11111111-1111-4111-8111-111111111111"
    calls = {"n": 0}

    def _fetchone(cursor: _Cursor):
        calls["n"] += 1
        # after update, require_by_id loads row
        return _row(status=STATUS_ARCHIVED)

    cursor = _Cursor(fetchone=_fetchone, rowcount=1)
    conn = _Conn(cursor)
    store = RepositoryStore(_params())

    with patch.object(store, "_connect", return_value=conn):
        record = store.soft_delete_repository(repo_id)

    assert record.status == STATUS_ARCHIVED
    assert any("UPDATE repository SET status" in sql for sql, _ in cursor.statements)


def test_require_by_id_raises_persistence_not_found() -> None:
    cursor = _Cursor(fetchone=None)
    conn = _Conn(cursor)
    store = RepositoryStore(_params())
    with patch.object(store, "_connect", return_value=conn):
        with pytest.raises(RepositoryNotFound):
            store.require_by_id("missing")


def test_load_and_save_settings_domain_object() -> None:
    now = _utc()
    cursor = _Cursor(fetchone=({"chunk_size": 400}, now))
    # first call for load uses fetchone tuple; upsert uses execute only
    store = RepositoryStore(_params())

    load_cursor = _Cursor(fetchone=('{"chunk_size": 400}', now))
    load_conn = _Conn(load_cursor)
    with patch.object(store, "_connect", return_value=load_conn):
        loaded = store.load_settings("11111111-1111-4111-8111-111111111111")
    assert loaded.settings["chunk_size"] == 400

    save_cursor = _Cursor(rowcount=1)
    save_conn = _Conn(save_cursor)
    with patch.object(store, "_connect", return_value=save_conn):
        saved = store.save_settings(
            RepositorySettings(
                repository_id="11111111-1111-4111-8111-111111111111",
                settings={"chunk_size": 512},
            )
        )
    assert saved.settings["chunk_size"] == 512
    assert any("INSERT INTO repository_settings" in sql for sql, _ in save_cursor.statements)


def test_hard_delete_uses_transaction() -> None:
    cursor = _Cursor(rowcount=1)
    conn = _Conn(cursor)
    store = RepositoryStore(_params())
    with patch.object(store, "_connect", return_value=conn):
        deleted = store.delete_repository("11111111-1111-4111-8111-111111111111")
    assert deleted is True
    assert conn.committed is True
    assert len(cursor.statements) == 3
