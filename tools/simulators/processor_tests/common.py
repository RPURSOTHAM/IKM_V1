"""Shared helpers for individual processor integration tests."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.features.document_processing.core.contract import ProcessRequest, ProcessorResult
from src.features.document_processing.processors import get_processor
from src.features.document_processing.shared_processor.types import ProcessorType

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parents[2]
FIXTURE_PATH = TESTS_DIR / "fixtures" / "sample_processor_test.txt"
DEFAULT_COLLECTION = "ProcessorTestCollection"
DEFAULT_PROCESSOR_URL = "http://localhost:3100"
_SUPPORTED_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})


@dataclass
class TestContext:
    processor_type: str
    document_id: str
    document_name: str
    document_path: Path
    collection_name: str = DEFAULT_COLLECTION
    repository_id: str | None = "processor-test-repo"
    mode: str = "direct"
    base_url: str = DEFAULT_PROCESSOR_URL
    model_name: str = "BAAI/bge-base-en"
    model_dir: str = "bge-base-en"
    metadata_fields: list[dict[str, Any]] = field(default_factory=list)


def load_test_env() -> None:
    """Load processor and dependency env; map Docker service names to localhost for host runs."""
    for candidate in (
        REPO_ROOT / "src" / "processor_service" / ".env",
        REPO_ROOT / "src" / "dependencies" / ".env",
        REPO_ROOT / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)

    if os.getenv("RUNNING_IN_DOCKER") != "1":
        os.environ["WEAVIATE_URL"] = "http://localhost:8086"
        os.environ["WEAVIATE_GRPC_PORT"] = "50051"
    elif not os.getenv("WEAVIATE_URL"):
        os.environ["WEAVIATE_URL"] = "http://localhost:8086"
    if not os.getenv("WEAVIATE_API_KEY"):
        os.environ["WEAVIATE_API_KEY"] = "weaviate_secret_key"
    if not os.getenv("NEO4J_URI"):
        os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    if not os.getenv("NEO4J_USER"):
        os.environ["NEO4J_USER"] = "neo4j"
    if not os.getenv("NEO4J_PASSWORD"):
        os.environ["NEO4J_PASSWORD"] = "password"
    if not os.getenv("REDIS_HOST"):
        os.environ["REDIS_HOST"] = "localhost"
    if not os.getenv("REDIS_PORT"):
        os.environ["REDIS_PORT"] = "6379"
    if not os.getenv("REDIS_PASSWORD"):
        os.environ["REDIS_PASSWORD"] = "redis_password"

    models_root = os.getenv("MODELS_ROOT")
    if not models_root or models_root.startswith("/app/"):
        os.environ["MODELS_ROOT"] = str((REPO_ROOT / "src" / "models").resolve())
    if not os.getenv("MODEL_DIR"):
        os.environ["MODEL_DIR"] = "bge-base-en"
    if not os.getenv("MODEL_NAME"):
        os.environ["MODEL_NAME"] = "BAAI/bge-base-en"

    doc_root = os.getenv("DOCUMENT_ROOT")
    if not doc_root or doc_root.startswith("/app/"):
        os.environ["DOCUMENT_ROOT"] = str((REPO_ROOT / "_documents").resolve())


def ensure_repo_root_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def new_document_id(prefix: str = "proc-test") -> str:
    return f"{prefix}-{uuid.uuid4()}"


def prepare_document(
    *,
    document_id: str | None = None,
    copy_to_documents_root: bool = True,
    source: Path | None = None,
) -> tuple[str, str, Path]:
    """Return (document_id, document_name, absolute_path) and optionally copy into DOCUMENT_ROOT."""
    doc_id = document_id or new_document_id()
    source_path = (source or FIXTURE_PATH).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Fixture not found: {source_path}")

    document_name = f"{doc_id}{source_path.suffix.lower()}"
    documents_root = Path(os.environ["DOCUMENT_ROOT"]).expanduser().resolve()
    documents_root.mkdir(parents=True, exist_ok=True)
    target = documents_root / document_name

    if copy_to_documents_root:
        shutil.copy2(source_path, target)
        return doc_id, document_name, target

    return doc_id, document_name, source_path


def pick_existing_document(documents_dir: Path | None = None) -> Path:
    root = Path(documents_dir or os.environ.get("DOCUMENT_ROOT", REPO_ROOT / "_documents")).resolve()
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No supported documents in {root}")
    return max(files, key=lambda p: p.stat().st_mtime)


def build_request(ctx: TestContext) -> ProcessRequest:
    payload: dict[str, Any] = {
        "processor_type": ctx.processor_type,
        "document_id": ctx.document_id,
        "document_name": ctx.document_name,
        "document_path": str(ctx.document_path),
        "collection_name": ctx.collection_name,
        "repository_id": ctx.repository_id,
        "model_name": ctx.model_name,
        "model_dir": ctx.model_dir,
        "chunk_size": 256,
        "chunk_overlap_sentences": 1,
        "chunking_strategy": "fixed-overlap-based",
        "citation_retainment": True,
    }
    if ctx.metadata_fields:
        payload["metadata_fields"] = ctx.metadata_fields
        payload["document_type_id"] = "processor-test-doc-type"
    return ProcessRequest(**payload)


def run_direct(ctx: TestContext) -> ProcessorResult:
    request = build_request(ctx)
    processor = get_processor(ctx.processor_type)
    statuses: list[tuple[str, float | None]] = []

    def set_status(name: str, pct: float | None = None) -> None:
        statuses.append((name, pct))

    result = processor.run(
        request,
        ctx.document_path,
        set_status=set_status,
        check_stop=lambda: False,
    )
    result.document_metadata = {
        **(result.document_metadata or {}),
        "_test_statuses": statuses,
    }
    return result


def _http_json(method: str, url: str, *, body: dict[str, Any] | None = None, timeout: float = 60.0) -> tuple[int, Any]:
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
            return resp.status, json.loads(raw) if raw.strip() else None
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode(errors="replace")
        try:
            detail = json.loads(err_body) if err_body.strip() else err_body
        except json.JSONDecodeError:
            detail = err_body or str(exc)
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc


def get_processor_health(base_url: str, timeout: float = 30.0) -> dict[str, Any]:
    _, payload = _http_json("GET", f"{base_url.rstrip('/')}/health", timeout=timeout)
    return payload if isinstance(payload, dict) else {}


def wait_for_processor_job(
    base_url: str,
    document_id: str,
    *,
    timeout: float = 900.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    terminal = {"completed", "failed", "stopped"}
    while time.monotonic() < deadline:
        health = get_processor_health(base_url)
        current = health.get("current_status")
        last_term = health.get("last_job_terminal_status")
        last_doc = health.get("last_job_document_id")
        if current in terminal:
            return health
        if last_term in terminal and str(last_doc or "") == document_id and current == "idle":
            return {**health, "resolved_via": "last_job_terminal_status"}
        time.sleep(max(0.5, poll_interval))
    raise TimeoutError(f"Processor did not finish within {timeout}s")


def run_http(ctx: TestContext, *, wait: bool = True, wait_timeout: float = 900.0) -> dict[str, Any]:
    request = build_request(ctx)
    payload = request.model_dump(exclude_none=True)
    payload["document_path"] = f"/app/documents/{ctx.document_name}"

    base = ctx.base_url.rstrip("/")
    health = get_processor_health(base, timeout=10.0)
    if health.get("current_status") not in {None, "idle", "unknown"}:
        raise RuntimeError(f"Processor busy: {json.dumps(health)}")

    _, body = _http_json("POST", f"{base}/process", body=payload, timeout=120.0)
    out: dict[str, Any] = {"submitted": body, "health_before": health}
    if wait:
        out["final_health"] = wait_for_processor_job(base, ctx.document_id, timeout=wait_timeout)
    return out


def _terminal_status_from_health(final: dict[str, Any]) -> str | None:
    current = final.get("current_status")
    last_term = final.get("last_job_terminal_status")
    if current in {"completed", "failed", "stopped"}:
        return str(current)
    if last_term in {"completed", "failed", "stopped"}:
        return str(last_term)
    return None


def validate_weaviate(*, collection_name: str, document_name: str, min_objects: int = 1) -> list[str]:
    errors: list[str] = []
    try:
        from weaviate.classes.query import Filter

        from src.infrastructure.document_databases.weaviate_store import close_weaviate_client, get_weaviate_client

        close_weaviate_client()
        weaviate_url = os.environ.get("WEAVIATE_URL", "http://localhost:8086")
        api_key = os.environ.get("WEAVIATE_API_KEY") or "weaviate_secret_key"
        client = get_weaviate_client(weaviate_url, api_key)
        if not client.collections.exists(collection_name):
            return [f"Weaviate collection '{collection_name}' does not exist"]
        col = client.collections.get(collection_name)
        result = col.query.fetch_objects(
            filters=Filter.by_property("doc_name").equal(document_name),
            limit=max(min_objects, 5),
        )
        count = len(result.objects)
        if count < min_objects:
            errors.append(f"Expected >= {min_objects} Weaviate objects for doc_name={document_name!r}, got {count}")
    except Exception as exc:
        errors.append(f"Weaviate validation failed: {exc}")
    return errors


def validate_neo4j(
    *,
    document_id: str,
    processor_type: str,
    payload_contains: str | None = None,
) -> list[str]:
    errors: list[str] = []
    try:
        import json

        from neo4j import GraphDatabase

        from src.infrastructure.document_databases.neo4j_store import _neo4j_credentials

        uri, user, password = _neo4j_credentials()
        driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            with driver.session() as session:
                row = session.run(
                    """
                    MATCH (a:DocumentArtifact {document_id: $document_id, processor_type: $processor_type})
                    RETURN a.payload_json AS payload_json
                    """,
                    document_id=document_id,
                    processor_type=processor_type,
                ).single()
                if not row or not row.get("payload_json"):
                    errors.append(
                        f"Neo4j artifact missing for document_id={document_id} processor_type={processor_type}"
                    )
                elif payload_contains:
                    payload_raw = row["payload_json"]
                    payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
                    if payload_contains not in json.dumps(payload):
                        errors.append(f"Neo4j payload missing expected content: {payload_contains!r}")
        finally:
            driver.close()
    except Exception as exc:
        errors.append(f"Neo4j validation failed: {exc}")
    return errors


def validate_redis(*, document_id: str) -> list[str]:
    errors: list[str] = []
    try:
        from src.infrastructure.document_databases.redis_store import get_render_payload

        payload = get_render_payload(document_id=document_id)
        if not payload:
            errors.append(f"Redis render payload missing for document_id={document_id}")
        elif not payload.get("content"):
            errors.append("Redis render payload has empty content")
    except Exception as exc:
        errors.append(f"Redis validation failed: {exc}")
    return errors


def cleanup_artifacts(ctx: TestContext) -> None:
    if ctx.processor_type == ProcessorType.CHUNKING_VECTORIZING.value:
        try:
            from weaviate.classes.query import Filter

            from src.infrastructure.document_databases.weaviate_store import get_weaviate_client
            from src.features.document_processing.core.config import get_settings

            settings = get_settings()
            client = get_weaviate_client(settings.weaviate_url, settings.weaviate_api_key or None)
            if client.collections.exists(ctx.collection_name):
                col = client.collections.get(ctx.collection_name)
                col.data.delete_many(
                    where=Filter.by_property("doc_name").equal(ctx.document_name),
                    verbose=False,
                    dry_run=False,
                )
        except Exception:
            pass
    elif ctx.processor_type in {
        ProcessorType.METADATA_EXTRACTION.value,
        ProcessorType.TEMPLATE_EXTRACTION.value,
        ProcessorType.REFERENCE_DOCUMENT_EXTRACTION.value,
    }:
        try:
            from src.infrastructure.document_databases.neo4j_store import delete_document_graph

            delete_document_graph(ctx.document_id)
        except Exception:
            pass
    elif ctx.processor_type == ProcessorType.CONVERSION_FOR_RENDERING.value:
        try:
            from src.infrastructure.document_databases.redis_store import _redis_client, render_cache_key

            _redis_client().delete(render_cache_key(ctx.document_id))
        except Exception:
            pass

    try:
        staged = Path(os.environ["DOCUMENT_ROOT"]) / ctx.document_name
        if staged.is_file() and ctx.document_name.startswith("proc-test-"):
            staged.unlink()
    except Exception:
        pass


def print_result(title: str, payload: dict[str, Any] | ProcessorResult, *, errors: list[str]) -> int:
    if isinstance(payload, ProcessorResult):
        body = payload.model_dump()
    else:
        body = payload
    print(json.dumps({title: body, "errors": errors}, indent=2, default=str))
    if errors:
        print(f"FAIL: {title}", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"PASS: {title}")
    return 0


def run_processor_test(
    processor_type: str,
    *,
    mode: str = "direct",
    base_url: str = DEFAULT_PROCESSOR_URL,
    validate_fn: Callable[[TestContext, ProcessorResult | dict[str, Any]], list[str]],
    metadata_fields: list[dict[str, Any]] | None = None,
    expect_failure: bool = False,
    keep_artifacts: bool = False,
    wait_timeout: float = 900.0,
) -> int:
    ensure_repo_root_on_path()
    load_test_env()

    doc_id, doc_name, doc_path = prepare_document(copy_to_documents_root=True)
    ctx = TestContext(
        processor_type=processor_type,
        document_id=doc_id,
        document_name=doc_name,
        document_path=doc_path,
        mode=mode,
        base_url=base_url,
        metadata_fields=metadata_fields or [],
    )

    errors: list[str] = []
    try:
        if mode == "http":
            http_out = run_http(ctx, wait=True, wait_timeout=wait_timeout)
            final = http_out.get("final_health") or {}
            status = _terminal_status_from_health(final)
            if expect_failure:
                if status not in {"failed", "stopped"}:
                    errors.append(f"Expected failure but got terminal status={status!r}")
            elif status != "completed":
                errors.append(f"HTTP job did not complete successfully: {json.dumps(final)}")
            else:
                errors.extend(validate_fn(ctx, http_out))
            return print_result(processor_type, http_out, errors=errors)

        result = run_direct(ctx)
        if expect_failure:
            errors.append("Expected processor to fail but it succeeded")
        else:
            if result.storage_backend == "weaviate" and result.artifacts.get("embedded_chunks", 0) <= 0:
                errors.append("chunking produced zero embedded chunks")
            errors.extend(validate_fn(ctx, result))
        return print_result(processor_type, result, errors=errors)
    except Exception as exc:
        if expect_failure:
            print(f"PASS: {processor_type} (expected failure: {exc})")
            return 0
        print(f"FAIL: {processor_type}: {exc}", file=sys.stderr)
        return 1
    finally:
        if not keep_artifacts:
            cleanup_artifacts(ctx)
