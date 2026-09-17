"""Submit one document to a running src.processor_service HTTP service (POST /process).

Assumes the processor is already up (e.g. port 8082). Picks a single file from the repo ``_documents`` folder
unless you pass an explicit path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[1]
PROCESSOR_ENV = REPO_ROOT / "src" / "src.processor_service" / ".env"

for candidate in (REPO_ROOT / ".env", PROCESSOR_ENV):
    load_dotenv(candidate, override=False)

DEFAULT_PROCESSOR_URL = "http://localhost:8082"
_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


def pick_one_document_from_documents(documents_dir: Path | str | None = None) -> Path:
    """Return one file from ``_documents`` (newest by mtime among supported types)."""
    root = Path(documents_dir or (REPO_ROOT / "_documents")).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Documents directory does not exist: {root}")
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS]
    if not files:
        raise FileNotFoundError(
            f"No {_SUPPORTED_EXTENSIONS} files in {root}",
        )
    return max(files, key=lambda p: p.stat().st_mtime)


def _document_id_for_file(path: Path, explicit_id: str | None) -> str:
    if explicit_id:
        return explicit_id
    try:
        uuid.UUID(path.stem)
    except ValueError:
        return str(uuid.uuid4())
    return path.stem


def _http_json(method: str, url: str, *, body: dict[str, Any] | None = None, timeout: float) -> tuple[int, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode()
            payload = json.loads(raw) if raw.strip() else None
            return resp.status, payload
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        try:
            detail = json.loads(err_body) if err_body.strip() else err_body
        except json.JSONDecodeError:
            detail = err_body or str(exc)
        raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {detail}") from exc


def submit_document_for_processing(
    *,
    base_url: str = DEFAULT_PROCESSOR_URL,
    documents_dir: Path | str | None = None,
    document_path: Path | str | None = None,
    document_id: str | None = None,
    collection_name: str | None = None,
    tenant_id: str | None = None,
    weaviate_url: str | None = None,
    weaviate_api_key: str | None = None,
    model_name: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap_sentences: int | None = None,
    min_content_words: int | None = None,
    timeout: float = 120.0,
) -> dict[str, Any]:
    """POST ``/process`` for one document.

    If ``document_path`` is omitted, picks the newest supported file under ``documents_dir``
    (default: ``<repo>/_documents``). The processor must see the same file under its ``DOCUMENT_ROOT``
    using ``document_name`` (basename only).
    """
    base = (base_url or DEFAULT_PROCESSOR_URL).rstrip("/")
    path = Path(document_path).expanduser().resolve() if document_path else pick_one_document_from_documents(documents_dir)

    payload: dict[str, Any] = {
        "document_id": _document_id_for_file(path, document_id),
        "document_name": path.name,
        "collection_name": collection_name or os.getenv("WEAVIATE_COLLECTION", "DocumentChunk"),
    }
    tid = tenant_id if tenant_id is not None else os.getenv("DEFAULT_TENANT_ID")
    if tid:
        payload["tenant_id"] = tid
    wu = weaviate_url if weaviate_url is not None else os.getenv("WEAVIATE_URL")
    if wu:
        payload["weaviate_url"] = wu
    wk = weaviate_api_key if weaviate_api_key is not None else os.getenv("WEAVIATE_API_KEY")
    if wk:
        payload["weaviate_api_key"] = wk
    mn = model_name if model_name is not None else (os.getenv("MODEL_NAME") or os.getenv("EMBEDDING_MODEL"))
    if mn:
        payload["model_name"] = mn
    if chunk_size is not None:
        payload["chunk_size"] = chunk_size
    if chunk_overlap_sentences is not None:
        payload["chunk_overlap_sentences"] = chunk_overlap_sentences
    if min_content_words is not None:
        payload["min_content_words"] = min_content_words

    _, body = _http_json("POST", f"{base}/process", body=payload, timeout=timeout)
    return body if isinstance(body, dict) else {"response": body}


def _get_processor_health(base: str, timeout: float) -> dict[str, Any]:
    _, st = _http_json("GET", f"{base.rstrip('/')}/health", body=None, timeout=timeout)
    return st if isinstance(st, dict) else {}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Submit one file from _documents to src.processor_service on port 8082.")
    p.add_argument("--base-url", default=os.getenv("PROCESSOR_BASE_URL", DEFAULT_PROCESSOR_URL))
    p.add_argument("--documents-dir", type=Path, default=None, help="Override repo _documents path.")
    p.add_argument("--document-path", type=Path, default=None, help="Process this file instead of auto-pick.")
    p.add_argument("--document-id", default=None)
    p.add_argument("--dry-run", action="store_true", help="Print payload only.")
    p.add_argument("--wait", action="store_true", help="Poll /health until current_status is terminal.")
    p.add_argument("--wait-timeout", type=float, default=3600.0)
    p.add_argument("--poll-interval", type=float, default=2.0)
    return p


def main() -> int:
    args = _parser().parse_args()
    base = args.base_url.rstrip("/")

    if args.dry_run:
        path = Path(args.document_path).expanduser().resolve() if args.document_path else pick_one_document_from_documents(
            args.documents_dir
        )
        payload_preview = {
            "document_id": _document_id_for_file(path, args.document_id),
            "document_name": path.name,
            "collection_name": os.getenv("WEAVIATE_COLLECTION", "DocumentChunk"),
        }
        print(json.dumps({"picked_file": str(path), "payload": payload_preview}, indent=2))
        return 0

    try:
        body = submit_document_for_processing(
            base_url=base,
            documents_dir=args.documents_dir,
            document_path=args.document_path,
            document_id=args.document_id,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(json.dumps({"submitted": body, "base_url": base}, indent=2))

    if not args.wait:
        return 0

    deadline = time.monotonic() + args.wait_timeout
    terminal = {"completed", "failed", "stopped"}
    doc_id = body.get("document_id") if isinstance(body, dict) else None
    while time.monotonic() < deadline:
        st = _get_processor_health(base, timeout=30.0)
        current = st.get("current_status")
        last_term = st.get("last_job_terminal_status")
        last_doc = st.get("last_job_document_id")
        if current in terminal:
            print(json.dumps({"final_status": st}, indent=2))
            return 0 if current == "completed" else 1
        if (
            doc_id
            and last_term in terminal
            and str(last_doc or "") == str(doc_id)
            and current == "idle"
        ):
            print(json.dumps({"final_status": st, "resolved_via": "last_job_terminal_status"}, indent=2))
            return 0 if last_term == "completed" else 1
        time.sleep(max(0.5, args.poll_interval))

    print("Wait timeout.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
