from __future__ import annotations

from typing import Any

import httpx
import pytest


def assert_status(
    response: httpx.Response,
    expected: int | set[int],
    *,
    endpoint: str,
) -> None:
    allowed = {expected} if isinstance(expected, int) else expected
    if response.status_code not in allowed:
        detail = response.text[:500]
        pytest.fail(
            f"{endpoint} -> expected {sorted(allowed)}, got {response.status_code}: {detail}"
        )


def assert_error(
    response: httpx.Response,
    expected_status: int,
    *,
    endpoint: str,
    code: str | None = None,
    message_contains: str | None = None,
    details_key: str | None = None,
) -> dict[str, Any]:
    if response.status_code != expected_status:
        pytest.fail(
            f"{endpoint} -> expected {expected_status}, got {response.status_code}: {response.text[:500]}"
        )
    body = response.json()
    error = body.get("error")
    if not isinstance(error, dict):
        pytest.fail(f"{endpoint} -> response missing error envelope: {body}")

    if code is not None and error.get("code") != code:
        pytest.fail(f"{endpoint} -> expected error.code={code!r}, got {error.get('code')!r}")

    message = str(error.get("message") or "")
    if message_contains and message_contains.lower() not in message.lower():
        pytest.fail(
            f"{endpoint} -> expected message containing {message_contains!r}, got {message!r}"
        )

    if not error.get("request_id"):
        pytest.fail(f"{endpoint} -> error.response missing request_id")

    if details_key is not None:
        details = error.get("details")
        if not isinstance(details, dict) or details_key not in details:
            pytest.fail(f"{endpoint} -> expected details[{details_key!r}], got {details!r}")

    return error
