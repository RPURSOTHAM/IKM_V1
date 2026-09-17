"""Extract SOP metadata from unique filtered-out text via OpenAI.

Help
----
Letterhead fields (title, document number, dates, facility) are removed
by filtering. This pass runs **after** filtering. It sends the **unique**
TEXT blocks from the removed set to ``SIMPLE_TASK_MODEL`` and asks for
one best value per key in ``config.json`` → ``metadata``.

Response handling:

1. Require a JSON object.
2. If the first reply is not JSON, retry once with a stricter prompt.
3. If that still fails, fill keys by matching aliases in the text
   (``Title:``, ``Document No.:``, ``Facility`` / ``Department`` rows).
4. LLM or parse errors are logged and returned as empty/partial values.
   They do not raise, so extract → filter → group still finishes.

What this does not change
-------------------------
Extraction, filter rules, grouping, or inline-icon splits.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI, OpenAIError

from ..document import ContentKind, DocumentContent, DocumentItem, TextBlock
from ..errors import RECOVERABLE
from ..failures import log_failure

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.json"
ENV_PATH = ROOT / ".env"
MAX_PROMPT_CHARS = 80_000

_ENV_STATE = {"loaded": False}


@dataclass
class MetadataResult:
    """Metadata values plus how they were obtained."""

    values: dict[str, str | None] = field(default_factory=dict)
    source: str = "empty"
    error: str | None = None
    unique_block_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict of values and provenance."""

        return {
            "values": self.values,
            "source": self.source,
            "error": self.error,
            "unique_block_count": self.unique_block_count,
        }


def extract_metadata(
    removed: DocumentContent,
    on_progress: Callable[[str, float], None] | None = None,
) -> MetadataResult:
    """Fill config metadata keys from unique removed TEXT blocks.

    Never raises. On LLM or parse failure, logs and returns whatever
    values could be recovered (possibly all ``None``).
    """

    def report(message: str, fraction: float) -> None:
        """Forward a clamped metadata fraction when a callback was supplied."""

        if on_progress is not None:
            on_progress(message, min(1.0, max(0.0, fraction)))

    report("Collecting letterhead text…", 0.05)
    schema = _load_metadata_schema()
    keys = list(schema.keys())
    empty = {key: None for key in keys}
    unique_texts = unique_filtered_text_blocks(removed.items)
    blob = "\n".join(unique_texts)
    if not blob.strip():
        logger.warning("Metadata extraction skipped: no unique filtered text blocks.")
        report("No letterhead text for metadata", 1.0)
        return MetadataResult(
            values=empty,
            source="empty",
            error="No unique filtered text blocks.",
            unique_block_count=0,
        )

    prompt_text = blob if len(blob) <= MAX_PROMPT_CHARS else blob[:MAX_PROMPT_CHARS]
    try:
        report("Asking the metadata model…", 0.35)
        raw = _ask_llm_for_json(prompt_text, keys)
        if raw is not None:
            values = _coerce_values(raw, schema)
            report("Metadata extracted", 1.0)
            return MetadataResult(
                values=values,
                source="llm",
                unique_block_count=len(unique_texts),
            )
        logger.warning("Metadata LLM reply was not JSON; retrying once.")
        report("Retrying metadata as JSON…", 0.65)
        raw = _ask_llm_for_json(prompt_text, keys, retry=True)
        if raw is not None:
            values = _coerce_values(raw, schema)
            report("Metadata extracted after retry", 1.0)
            return MetadataResult(
                values=values,
                source="llm_retry",
                unique_block_count=len(unique_texts),
            )
        logger.warning("Metadata LLM retry was not JSON; using manual parse.")
        report("Parsing metadata from letterhead text…", 0.85)
        values = _manual_parse(blob, schema)
        report("Metadata filled from letterhead text", 1.0)
        return MetadataResult(
            values=values,
            source="manual",
            error="LLM did not return JSON after retry; used manual parse.",
            unique_block_count=len(unique_texts),
        )
    except (*RECOVERABLE, OpenAIError) as exc:
        log_failure("analysis", "metadata LLM extraction", exc)
        try:
            report("Parsing metadata from letterhead text…", 0.85)
            values = _manual_parse(blob, schema)
            report("Metadata filled from letterhead text", 1.0)
            return MetadataResult(
                values=values,
                source="manual",
                error=str(exc),
                unique_block_count=len(unique_texts),
            )
        except RECOVERABLE as parse_exc:
            log_failure("analysis", "metadata manual parse", parse_exc)
            report("Metadata extraction failed", 1.0)
            return MetadataResult(
                values=empty,
                source="error",
                error=f"{exc}; manual parse: {parse_exc}",
                unique_block_count=len(unique_texts),
            )


def unique_filtered_text_blocks(items: list[DocumentItem]) -> list[str]:
    """Unique TEXT wording from filtered-out items, first occurrence kept."""

    seen: set[str] = set()
    texts: list[str] = []
    for item in items:
        if item.kind is not ContentKind.TEXT or not isinstance(item.content, TextBlock):
            continue
        text = (item.content.text or "").strip()
        key = " ".join(text.split()).lower()
        if not key or key in seen:
            continue
        seen.add(key)
        texts.append(text)
    return texts


def _load_metadata_schema() -> dict[str, list[str]]:
    """Read ``metadata`` keys and aliases from ``config.json``."""

    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        raw = data.get("metadata") or {}
        schema: dict[str, list[str]] = {}
        for key, aliases in raw.items():
            if isinstance(aliases, list):
                schema[str(key)] = [str(alias) for alias in aliases]
            else:
                schema[str(key)] = [str(aliases)]
        if schema:
            return schema
    except RECOVERABLE as exc:
        log_failure("analysis", f"read metadata keys from {CONFIG_PATH}", exc)
    return {
        "title": ["title", "Title:"],
        "document no.": ["document no.", "document no", "Document No."],
        "version no": ["version no", "version", "Version No."],
        "effective date": ["effective date", "effective dt", "Effective Date:"],
        "review date": ["review date", "review dt", "Review Date:"],
        "facility": ["facility", "Facility:"],
        "department": ["department", "Department:"],
        "sub department": ["sub department", "Sub Department:"],
    }


def _ask_llm_for_json(
    text: str,
    keys: list[str],
    retry: bool = False,
) -> dict[str, Any] | None:
    """Call OpenAI and parse a JSON object, or return ``None``."""

    _ensure_env_loaded()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    model = (os.environ.get("SIMPLE_TASK_MODEL") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")
    if not model:
        raise RuntimeError("SIMPLE_TASK_MODEL is not set.")

    key_list = ", ".join(f'"{key}"' for key in keys)
    system = (
        "You extract metadata from SOP / quality-document text. "
        "For every key, pick the single best value found in the text. "
        "If a key is missing, use null. "
        f"Keys: {key_list}. "
        "Respond with a JSON object only. No markdown, no commentary."
    )
    user = text
    if retry:
        user = (
            "Your previous reply was not valid JSON. "
            "Reply again with a JSON object only, using exactly these keys: "
            f"{key_list}.\n\nDocument text:\n{text}"
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = ""
    if response.choices:
        content = response.choices[0].message.content or ""
    parsed = _parse_json_object(content)
    if parsed is None:
        logger.warning("Metadata LLM response was not JSON (retry=%s).", retry)
    return parsed


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from model text, including fenced replies."""

    stripped = (text or "").strip()
    if not stripped:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    nested = data.get("metadata")
    if isinstance(nested, dict):
        return nested
    return data


def _coerce_values(
    raw: dict[str, Any],
    schema: dict[str, list[str]],
) -> dict[str, str | None]:
    """Map a model dict onto config keys; unknown keys are ignored."""

    alias_to_key: dict[str, str] = {}
    for key, aliases in schema.items():
        alias_to_key[_norm_label(key)] = key
        for alias in aliases:
            alias_to_key[_norm_label(alias)] = key

    values = {key: None for key in schema}
    for raw_key, raw_value in raw.items():
        canon = alias_to_key.get(_norm_label(str(raw_key)))
        if canon is None:
            continue
        values[canon] = _clean_value(raw_value)
    return values


def _manual_parse(text: str, schema: dict[str, list[str]]) -> dict[str, str | None]:
    """Fill keys from labeled lines and Facility/Department footer rows."""

    values = {key: None for key in schema}
    alias_to_key: dict[str, str] = {}
    for key, aliases in schema.items():
        alias_to_key[_norm_label(key)] = key
        for alias in aliases:
            alias_to_key[_norm_label(alias)] = key

    labeled = re.compile(
        r"(?im)^[ \t]*(?P<label>[^:\n]{2,40})\s*:\s*(?P<value>.+?)\s*$"
    )
    for match in labeled.finditer(text):
        canon = alias_to_key.get(_norm_label(match.group("label")))
        if canon is None:
            continue
        value = _clean_value(match.group("value"))
        if value and not values[canon]:
            values[canon] = value

    footer = re.search(
        r"Facility\s*:?\s*(?P<facility>.+?)\s+"
        r"Department\s*:?\s*(?P<department>.+?)\s+"
        r"Sub\s*Department\s*:?\s*(?P<sub>.+?)(?:\s*$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if footer:
        mapping = {
            "facility": footer.group("facility"),
            "department": footer.group("department"),
            "sub department": footer.group("sub"),
        }
        for key, raw in mapping.items():
            if key in values and not values[key]:
                values[key] = _clean_value(raw)
    return values


def _norm_label(label: str) -> str:
    """Lowercase label without a trailing colon, for alias matching."""

    return re.sub(r"\s+", " ", (label or "").strip().rstrip(":").lower())


def _clean_value(value: Any) -> str | None:
    """Turn a model/manual value into a stripped string, or ``None``."""

    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text or text.lower() in {"null", "none", "n/a", "-"}:
        return None
    return text


def _ensure_env_loaded() -> None:
    """Load ``.env`` into ``os.environ`` once, without overriding existing keys."""

    if _ENV_STATE["loaded"]:
        return
    _ENV_STATE["loaded"] = True
    if not ENV_PATH.exists():
        return
    try:
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, value = stripped.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except RECOVERABLE as exc:
        log_failure("analysis", f"read {ENV_PATH}", exc)
