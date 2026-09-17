from __future__ import annotations

from types import SimpleNamespace

from src.simulators.streamlit_document_uploader import (
    _CachedUploadFile,
    _coerce_upload_file_list,
    _upload_files_for_submission,
)


class _FakeUploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_coerce_upload_file_list_handles_single_and_multiple() -> None:
    one = _FakeUploadedFile("a.pdf", b"pdf")
    assert _coerce_upload_file_list(None) == []
    assert _coerce_upload_file_list(one) == [one]
    assert _coerce_upload_file_list([one]) == [one]


def test_cached_upload_file_exposes_name_and_bytes() -> None:
    cached = _CachedUploadFile("report.pdf", b"hello")
    assert cached.name == "report.pdf"
    assert cached.getvalue() == b"hello"


def test_upload_files_for_submission_uses_cache_when_widget_empty(monkeypatch) -> None:
    fake_state = {
        "upload_file_cache": [{"name": "cached.txt", "bytes": b"cached-bytes"}],
    }
    monkeypatch.setattr(
        "src.simulators.streamlit_document_uploader.st.session_state",
        fake_state,
        raising=False,
    )

    resolved = _upload_files_for_submission(None)
    assert len(resolved) == 1
    assert resolved[0].name == "cached.txt"
    assert resolved[0].getvalue() == b"cached-bytes"


def test_upload_files_for_submission_refreshes_cache_from_live_widget(monkeypatch) -> None:
    fake_state: dict[str, object] = {}
    monkeypatch.setattr(
        "src.simulators.streamlit_document_uploader.st.session_state",
        fake_state,
        raising=False,
    )
    live = _FakeUploadedFile("live.pdf", b"live-bytes")

    resolved = _upload_files_for_submission(live)
    assert resolved == [live]
    assert fake_state["upload_file_cache"] == [{"name": "live.pdf", "bytes": b"live-bytes"}]
