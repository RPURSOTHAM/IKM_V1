"""Shared HTTP helpers for NCBI / Europe PMC connector requests."""

from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests import Response
from urllib3.exceptions import ProtocolError

_logger = logging.getLogger(__name__)

_RETRYABLE_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    ProtocolError,
)


def normalize_entrez_base_url(base_url: str) -> str:
    url = str(base_url or "https://eutils.ncbi.nlm.nih.gov/entrez/eutils").strip().rstrip("/")
    if url.startswith("http://"):
        url = "https://" + url[len("http://") :]
    return url


def ncbi_session_headers(*, email: str = "", tool: str = "rag-builder-pubmed") -> dict[str, str]:
    contact = str(email or "").strip() or "rag-builder@local"
    return {
        "User-Agent": f"{tool}/1.0 ({contact})",
        "Accept": "application/json, text/xml, */*",
    }


def configure_ncbi_session(
    session: requests.Session,
    *,
    email: str = "",
    tool: str = "rag-builder-pubmed",
) -> requests.Session:
    session.headers.update(ncbi_session_headers(email=email, tool=tool))
    return session


_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


def request_with_retry(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 60,
    retries: int = 3,
    backoff_seconds: float = 0.75,
    cancel_event: Any | None = None,
) -> Response:
    last_error: Exception | None = None
    attempts = max(1, retries)
    for attempt in range(attempts):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Literature search cancelled.")
        try:
            response = session.request(method, url, params=params, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in _RETRYABLE_HTTP_STATUS:
                raise
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = backoff_seconds * (attempt + 1)
            if status == 429:
                delay = max(delay, 2.0)
            _logger.warning(
                "Retrying %s %s after HTTP %s (attempt %s/%s)",
                method,
                url,
                status,
                attempt + 2,
                attempts,
            )
            _sleep_interruptible(delay, cancel_event)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            delay = backoff_seconds * (attempt + 1)
            _logger.warning("Retrying %s %s after %s (attempt %s/%s)", method, url, exc, attempt + 2, attempts)
            _sleep_interruptible(delay, cancel_event)
    assert last_error is not None
    raise last_error


def _sleep_interruptible(delay: float, cancel_event: Any | None) -> None:
    end = time.monotonic() + max(0.0, float(delay or 0.0))
    while time.monotonic() < end:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("Literature search cancelled.")
        time.sleep(min(0.1, end - time.monotonic()))
