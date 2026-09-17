"""RAG System - Test Harness: Streamlit UI for uploads, Weaviate repository, RabbitMQ queues, and scheduler."""

from __future__ import annotations

import json
import importlib
import os
import sys
import socket
import struct
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests
import streamlit as st

# Streamlit wraps stdout/stderr on Windows; reconfigure can raise OSError EINVAL.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

CURRENT_FILE = Path(__file__).resolve()


def _find_repository_root(current_file: Path) -> Path:
    """Find the nearest ancestor that exposes this project as an importable package."""
    required_layout = [
        Path("src"),
        Path("src/shared"),
        Path("src/features"),
        Path("src/processor_service"),
    ]
    # Prefer tools/simulators (current layout); accept src/simulators for older checkouts.
    simulator_markers = (Path("tools/simulators"), Path("src/simulators"))
    required_files = [
        Path("src/shared/networking/document_paths.py"),
        # Prefer feature-based chunking; keep legacy shared path as fallback.
        Path("src/features/chunking"),
    ]
    for candidate in (current_file.parent, *current_file.parents):
        if not all((candidate / path).exists() for path in required_layout):
            continue
        if not any((candidate / path).exists() for path in required_files):
            continue
        if any((candidate / marker).exists() for marker in simulator_markers):
            return candidate
    raise RuntimeError(
        "Could not resolve repository root from "
        f"{current_file}. Expected a parent containing src/shared, src/features, "
        "and tools/simulators (or src/simulators)."
    )


REPO_ROOT = _find_repository_root(CURRENT_FILE)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _is_running_in_docker() -> bool:
    if os.getenv("RUNNING_IN_DOCKER") or os.getenv("DOCKER_CONTAINER"):
        return True
    return Path("/.dockerenv").exists()


def _configure_document_environment() -> None:
    """Set practical defaults so local Streamlit runs do not need shell setup."""
    running_in_docker = _is_running_in_docker()
    if running_in_docker:
        os.environ.setdefault("DEPLOYMENT_MODE", "docker")
        os.environ.setdefault("DOCUMENT_ROOT", "/app/documents")
        os.environ.setdefault("HOST_DOCUMENTS_DIR", "/app/documents")
        os.environ.setdefault("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", "/app/documents")
        return

    if os.name == "nt":
        documents_dir = str((REPO_ROOT / "_documents").resolve())
        os.environ.setdefault("DEPLOYMENT_MODE", "local")
        os.environ.setdefault("DOCUMENT_ROOT", documents_dir)
        os.environ.setdefault("HOST_DOCUMENTS_DIR", documents_dir)
        os.environ.setdefault("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", documents_dir)


_configure_document_environment()

if Path.cwd().resolve() != REPO_ROOT:
    print("Detected execution outside repository root.")
    print("Using repository root:")
    print(REPO_ROOT)


def _print_startup_diagnostics() -> None:
    checks = {
        "Whether src package exists": REPO_ROOT / "src" / "__init__.py",
        "Whether src/shared exists": REPO_ROOT / "src" / "shared",
        "Whether document_paths.py exists": REPO_ROOT / "src" / "shared" / "networking" / "document_paths.py",
        "Whether chunking/strategies.py exists": REPO_ROOT / "src" / "shared" / "chunking" / "strategies.py",
    }
    print("Current working directory")
    print(Path.cwd().resolve())
    print("Resolved repository root")
    print(REPO_ROOT)
    print("Deployment mode")
    print(os.getenv("DEPLOYMENT_MODE", ""))
    print("Document root")
    print(os.getenv("DOCUMENT_ROOT", ""))
    print("Processor path")
    print(os.getenv("PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER", ""))
    print("Host uploads path")
    print(os.getenv("HOST_DOCUMENTS_DIR", ""))
    print("API URL")
    print(os.getenv("RAG_API_BASE_URL", os.getenv("CONSUMER_API_BASE_URL", "http://localhost:8088")))
    print("Processor URL")
    print(os.getenv("CHUNKING_PROCESSOR_URL", os.getenv("PROCESSOR_API_URL", "http://localhost:3100")))
    print("Reference Processor URL")
    print(os.getenv("REFERENCE_EXTRACTION_URL", "http://localhost:3110"))
    print("Python executable")
    print(sys.executable)
    print("Python version")
    print(sys.version)
    print("sys.path")
    print(sys.path)
    for label, path in checks.items():
        print(f"{label}: {path.exists()} ({path})")


def _run_import_self_test() -> None:
    modules = [
        "src.shared.networking.document_paths",
        "src.features.chunking.domain.chunking_strategy",
    ]
    for module in modules:
        try:
            importlib.import_module(module)
            print(f"✓ Imported {module}")
        except Exception as exc:
            print(f"✗ Failed to import {module}: {exc}")


_print_startup_diagnostics()
_run_import_self_test()

from src.shared.networking.document_paths import (
    build_processor_document_path,
    host_documents_dir,
    processor_documents_container_dir,
)
from src.features.chunking.domain.chunking_strategy import CHUNKING_STRATEGIES, normalize_chunking_strategy

OLD_DEFAULT_API_BASE_URL = "http://localhost:8088"
DEFAULT_API_BASE_URL = os.getenv("RAG_API_BASE_URL", os.getenv("CONSUMER_API_BASE_URL", "http://localhost:8088"))
print("DEFAULT_API_BASE_URL =", DEFAULT_API_BASE_URL)
DEFAULT_API_PREFIX = os.getenv("CONSUMER_API_PREFIX", "/api/v1")
DEFAULT_API_KEY = os.getenv("CONSUMER_API_KEY", os.getenv("RAG_API_KEY", ""))
DEFAULT_CHUNKING_PROCESSOR_URL = os.getenv(
    "CHUNKING_PROCESSOR_URL",
    os.getenv("PROCESSOR_API_URL", "http://localhost:3100"),
).rstrip("/")
DEFAULT_REFERENCE_EXTRACTION_URL = os.getenv("REFERENCE_EXTRACTION_URL", "http://localhost:3110").rstrip("/")
DOCUMENTS_DIR = host_documents_dir(repo_root=REPO_ROOT)
PROCESSOR_DOCUMENTS_DIR = processor_documents_container_dir()
DEFAULT_RABBITMQ_MANAGEMENT_URL = os.getenv("RABBITMQ_MANAGEMENT_URL", "http://localhost:15672")
DEFAULT_RABBITMQ_USER = os.getenv("RABBITMQ_USER", os.getenv("RABBITMQ_MANAGEMENT_USER", "rabbitmq_user"))
DEFAULT_RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", os.getenv("RABBITMQ_MANAGEMENT_PASS", "rabbitmq_password"))
DEFAULT_SCHEDULER_HOST = os.getenv("SCHEDULER_HOST", "localhost")
DEFAULT_SCHEDULER_PORT = int(os.getenv("SCHEDULER_TCP_PORT", "3200"))
DEFAULT_PROCESSOR_PORTS = os.getenv("PROCESSOR_PORTS", "3100,3101,3102,3103,3104")
SUPPORTED_TYPES = ["pdf", "docx", "txt"]
APP_TITLE = "RAG System - Test Harness"
CHUNKING_STRATEGY_OPTIONS = sorted(CHUNKING_STRATEGIES)
TEST_EMBEDDING_MODELS = [
    "bge-base-en",
    "bge-small-en",
    "bge-large-en",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "e5-base-v2",
    "e5-large-v2",
]

DEFAULT_METADATA_FIELDS_TEXT = """document_number | SOP No | string
title | SOP Title | string
document_version | Version No | string
effective_date | Effective Date | date
review_date | Review Date | date"""

DEFAULT_KEY_FIELDS_TEXT = """document_number | string | true | SOP No
title | string | false | SOP Title
document_version | string | false | Version No
effective_date | date | false | Effective Date
review_date | date | false | Review Date"""


class ProcessRequestError(RuntimeError):
    def __init__(self, message: str, logs: dict[str, Any]):
        super().__init__(message)
        self.logs = logs


class ComplianceBlockError(ProcessRequestError):
    def __init__(self, details: dict[str, Any], logs: dict[str, Any]):
        super().__init__("Upload blocked due to compliance violation", logs)
        self.details = details


class BackendUnavailableError(RuntimeError):
    def __init__(self, message: str, diagnostics: list[dict[str, Any]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _chunking_processor_url() -> str:
    return st.session_state.get("chunking_processor_url", DEFAULT_CHUNKING_PROCESSOR_URL).rstrip("/")


def _reference_processor_url() -> str:
    return st.session_state.get("reference_extraction_url", DEFAULT_REFERENCE_EXTRACTION_URL).rstrip("/")


def _api_base_url() -> str:
    return st.session_state.get("api_base_url", DEFAULT_API_BASE_URL).rstrip("/")


_PROCESSOR_DOC_ROOT_CACHE: dict[str, str | None] = {}


def _processor_container_document_root(processor_url: str) -> str | None:
    """Ask the processor for its document root (e.g. /app/documents in Docker).

    Returns None when the processor is unreachable or reports a host-style path.
    """
    if processor_url in _PROCESSOR_DOC_ROOT_CACHE:
        return _PROCESSOR_DOC_ROOT_CACHE[processor_url]
    root: str | None = None
    try:
        response = requests.get(f"{processor_url}/debug/documents", timeout=5)
        response.raise_for_status()
        payload = response.json()
        candidate = str(payload.get("document_root") or payload.get("DOCUMENT_ROOT") or "").strip()
        if candidate.startswith("/"):
            root = candidate.rstrip("/")
    except Exception:
        root = None
    _PROCESSOR_DOC_ROOT_CACHE[processor_url] = root
    return root


def _build_path_mapping(saved_path: Path) -> dict[str, str]:
    """Map a saved host file to the path the processor can read.

    Windows Streamlit + Docker processors: the host `_documents` folder is
    bind-mounted at the processor's container document root, so we must send
    the POSIX container path even though DEPLOYMENT_MODE defaults to 'local'.
    """
    container_root = _processor_container_document_root(_chunking_processor_url())
    if container_root and os.name == "nt":
        resolved = Path(saved_path).expanduser().resolve()
        relative = resolved.relative_to(Path(DOCUMENTS_DIR).expanduser().resolve())
        return {
            "saved_host_path": str(resolved),
            "container_document_path": f"{container_root}/{relative.as_posix()}",
            "host_documents_dir": str(Path(DOCUMENTS_DIR).expanduser().resolve()),
            "processor_documents_container_dir": container_root,
        }
    return build_processor_document_path(saved_path, host_root=DOCUMENTS_DIR)


def _request_with_retries(method: str, url: str, *, timeout: float, **kwargs: Any) -> requests.Response:
    delay = 0.5
    # ConnectionAbortedError (WinError 10053) surfaces as ConnectionError when a
    # processor restarts mid-request; retry with backoff instead of deferring once.
    retryable = (
        requests.ConnectionError,
        requests.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )
    for attempt in range(1, 4):
        try:
            response = requests.request(method, url, timeout=timeout, **kwargs)
            if response.status_code not in {502, 503, 504} or attempt == 3:
                return response
        except retryable:
            if attempt == 3:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"Request failed without response: {method} {url}")


def _service_catalog() -> dict[str, dict[str, Any]]:
    return {
        "dms": {
            "label": "DMS Service",
            "url": _api_base_url(),
            "health_paths": ["/health"],
        },
        "chunking": {
            "label": "Chunking Processor",
            "url": _chunking_processor_url(),
            "health_paths": ["/health", "/status"],
        },
        "reference": {
            "label": "Reference Extraction Processor",
            "url": _reference_processor_url(),
            "health_paths": ["/health", "/status"],
        },
        "neo4j": {
            "label": "Neo4j",
            "url": os.getenv("NEO4J_HTTP_URL", "http://localhost:7474").rstrip("/"),
            "health_paths": ["/"],
        },
        "weaviate": {
            "label": "Weaviate",
            "url": os.getenv("WEAVIATE_URL", "http://localhost:8086").rstrip("/"),
            "health_paths": ["/v1/.well-known/ready"],
        },
    }


def _probe_http_service(name: str, spec: dict[str, Any], *, timeout: float = 3.0) -> dict[str, Any]:
    base_url = str(spec["url"]).rstrip("/")
    last_error = ""
    last_status_code: int | None = None
    for path in spec["health_paths"]:
        url = f"{base_url}{path}"
        try:
            response = requests.get(url, timeout=timeout)
            last_status_code = response.status_code
            if response.status_code == 200:
                try:
                    body: Any = response.json()
                except ValueError:
                    body = response.text[:500]
                return {
                    "name": name,
                    "label": spec["label"],
                    "url": base_url,
                    "health_url": url,
                    "ok": True,
                    "state": "healthy",
                    "status_code": response.status_code,
                    "response": body,
                    "error": None,
                    "suggested_fix": None,
                }
            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
        except requests.Timeout as exc:
            last_error = f"Timed out while checking {url}: {exc}"
            return {
                "name": name,
                "label": spec["label"],
                "url": base_url,
                "health_url": url,
                "ok": False,
                "state": "starting",
                "status_code": None,
                "response": None,
                "error": last_error,
                "suggested_fix": _suggest_service_fix(name, "starting"),
            }
        except requests.ConnectionError as exc:
            last_error = str(exc)
            # Try next health path before giving up (e.g. /health vs /status).
            continue
        except requests.RequestException as exc:
            last_error = str(exc)
    if last_error and (
        "Connection" in last_error or "Remote" in last_error or "refused" in last_error.lower()
    ):
        state = "starting"
    else:
        state = "starting" if last_status_code in {202, 429, 500, 502, 503, 504} else "unreachable"
    return {
        "name": name,
        "label": spec["label"],
        "url": base_url,
        "health_url": f"{base_url}{spec['health_paths'][0]}",
        "ok": False,
        "state": state,
        "status_code": last_status_code,
        "response": None,
        "error": last_error or "Health check failed",
        "suggested_fix": _suggest_service_fix(name, state),
    }


def _probe_http_service_with_retry(
    name: str,
    spec: dict[str, Any],
    *,
    timeout: float = 5.0,
    attempts: int = 3,
) -> dict[str, Any]:
    """Probe with short retries — processors often restart and drop the first connection."""
    last: dict[str, Any] | None = None
    for attempt in range(max(1, attempts)):
        last = _probe_http_service(name, spec, timeout=timeout)
        if last.get("ok"):
            return last
        if attempt + 1 < attempts:
            time.sleep(0.8)
    return last or _probe_http_service(name, spec, timeout=timeout)


def _suggest_service_fix(name: str, state: str) -> str:
    mode = os.getenv("DEPLOYMENT_MODE", "local")
    if state == "starting":
        return "Wait a few seconds and refresh. If it stays yellow, check the service logs for startup errors."
    if name == "dms":
        return "Start the API service on port 8088, then confirm /health responds. Check RAG_API_BASE_URL if you changed the port."
    if name == "chunking":
        return "Start the chunking processor on port 3100, or update CHUNKING_PROCESSOR_URL / PROCESSOR_API_URL."
    if name == "reference":
        return "Start the reference processor on port 3110, or update REFERENCE_EXTRACTION_URL."
    return f"Start the {name} service for {mode} deployment and verify the configured URL."


def _service_indicator(item: dict[str, Any]) -> str:
    state = str(item.get("state") or ("healthy" if item.get("ok") else "unreachable")).lower()
    if state == "healthy":
        return "🟢 Healthy"
    if state == "starting":
        return "🟡 Starting"
    return "🔴 Unreachable"


def _required_upload_services() -> list[dict[str, Any]]:
    catalog = _service_catalog()
    return [
        _probe_http_service_with_retry("dms", catalog["dms"], timeout=5.0),
        _probe_http_service_with_retry("chunking", catalog["chunking"], timeout=5.0),
        _probe_http_service_with_retry("reference", catalog["reference"], timeout=5.0),
    ]


def _processors_ready(health: list[dict[str, Any]] | None = None) -> bool:
    items = health or _required_upload_services()
    return all(item.get("ok") for item in items if item.get("name") in {"chunking", "reference"})


def _scheduler_tcp_health() -> dict[str, Any]:
    host = st.session_state.get("scheduler_host", DEFAULT_SCHEDULER_HOST)
    port = int(st.session_state.get("scheduler_port", DEFAULT_SCHEDULER_PORT))
    url = f"tcp://{host}:{port}"
    try:
        payload = _scheduler_tcp_request("healthcheck")
        ok = str(payload.get("status", "")).lower() == "healthy"
        return {
            "name": "scheduler",
            "label": "Scheduler",
            "url": url,
            "health_url": url,
            "ok": ok,
            "status_code": None,
            "response": payload,
            "error": None if ok else str(payload),
        }
    except Exception as exc:
        return {
            "name": "scheduler",
            "label": "Scheduler",
            "url": url,
            "health_url": url,
            "ok": False,
            "status_code": None,
            "response": None,
            "error": str(exc),
        }


def _discover_backends() -> list[dict[str, Any]]:
    catalog = _service_catalog()
    results = [
        _probe_http_service("reference", catalog["reference"]),
        _probe_http_service("chunking", catalog["chunking"]),
        _scheduler_tcp_health(),
        _probe_http_service("dms", catalog["dms"]),
        _probe_http_service("neo4j", catalog["neo4j"]),
        _probe_http_service("weaviate", catalog["weaviate"]),
    ]
    st.session_state["backend_discovery"] = results
    return results


def _required_backend_names(processing_mode: str) -> list[str]:
    return ["dms", "chunking", "reference"]


def _check_required_backends(processing_mode: str) -> list[dict[str, Any]]:
    """Non-blocking preflight — returns diagnostics; never raises."""
    catalog = _service_catalog()
    return [
        _probe_http_service_with_retry(name, catalog[name], timeout=5.0)
        for name in _required_backend_names(processing_mode)
    ]

def _api_path(resource: str) -> str:
    prefix = st.session_state.get("api_prefix", DEFAULT_API_PREFIX).rstrip("/")
    if resource in {"/health", "/status", "/process", "/stop"}:
        return resource
    if not resource.startswith("/"):
        resource = f"/{resource}"
    return f"{prefix}{resource}"


def _api_auth_headers() -> dict[str, str]:
    """Prefer a JWT; retain API-key support for legacy harness deployments."""
    token = str(
        st.session_state.get("api_access_token") or st.session_state.get("api_access_token_input") or ""
    ).strip()
    if token:
        return {"Authorization": f"Bearer {token}"}
    api_key = str(st.session_state.get("api_key", DEFAULT_API_KEY)).strip()
    return {"X-API-Key": api_key} if api_key else {}


def _request(method: str, path: str, **kwargs: Any) -> Any:
    url = f"{_api_base_url()}{_api_path(path)}"
    timeout = kwargs.pop("timeout", 120)
    headers = dict(kwargs.pop("headers", {}) or {})
    if path != "/health":
        for key, value in _api_auth_headers().items():
            headers.setdefault(key, value)
    response = _request_with_retries(method, url, timeout=timeout, headers=headers or None, **kwargs)
    response.raise_for_status()
    if response.status_code == 204 or not (response.text or "").strip():
        return None
    return response.json()


def _ensure_temporary_repository() -> dict[str, Any]:
    """Create one API-managed temporary repository for a test batch."""
    payload = _request("POST", "/repositories/temporary", timeout=30)
    return payload if isinstance(payload, dict) else {}


def _load_temporary_repository_documents() -> dict[str, Any]:
    """Return temporary repository documents for the Streamlit test panel."""
    payload = _request("GET", "/repositories/temporary/documents", timeout=30)
    return payload if isinstance(payload, dict) else {}


def _request_with_logs_at(
    base_url: str,
    method: str,
    path: str,
    **kwargs: Any,
) -> tuple[Any, dict[str, Any]]:
    url = f"{base_url.rstrip('/')}{path}"
    timeout = kwargs.pop("timeout", 120)
    headers = dict(kwargs.pop("headers", {}) or {})
    request_json = kwargs.get("json")
    logs: dict[str, Any] = {
        "backend_url": base_url.rstrip("/"),
        "http_method": method,
        "endpoint": path,
        "api_url_called": url,
        "request_json_sent": request_json,
        "timeout": timeout,
        "attempts": 3,
    }
    try:
        response = _request_with_retries(method, url, timeout=timeout, headers=headers or None, **kwargs)
    except (requests.ConnectionError, requests.Timeout) as exc:
        logs["connection_exception"] = str(exc)
        raise RuntimeError(f"Backend request failed after 3 attempts: {method} {url}\n{exc}") from exc
    logs["response_status_code"] = response.status_code
    logs["response_body"] = response.text
    if response.status_code == 400:
        try:
            body = response.json()
            if body.get("status") == "upload_blocked":
                raise ComplianceBlockError(body, logs)
        except Exception as json_exc:
            if isinstance(json_exc, ComplianceBlockError):
                raise
            pass
    if response.status_code == 422:
        if path.rstrip("/").endswith("/process"):
            raise ProcessRequestError(
                f"/process returned 422. Check the request JSON and document_path. Response: {response.text}",
                logs,
            )
        response.raise_for_status()
    response.raise_for_status()
    if response.status_code == 204 or not (response.text or "").strip():
        return None, logs
    return response.json(), logs


def _request_with_logs(method: str, path: str, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    return _request_with_logs_at(_api_base_url(), method, path, **kwargs)


def _format_http_error(exc: requests.HTTPError) -> str:
    if exc.response is None:
        return str(exc)
    try:
        payload = exc.response.json()
    except ValueError:
        return exc.response.text or str(exc)
    if isinstance(payload, dict) and "error" in payload:
        err = payload["error"]
        if isinstance(err, dict):
            return str(err.get("message") or err)
    detail = payload.get("detail", payload)
    if isinstance(detail, dict):
        return detail.get("message") or detail.get("error") or str(detail)
    return str(detail)


def _load_documents() -> list[dict[str, Any]]:
    params: dict[str, Any] = {"limit": 500, "offset": 0}
    tenant_id = st.session_state.get("filter_tenant_id", "").strip()
    collection_name = st.session_state.get("filter_collection_name", "").strip()
    status = st.session_state.get("filter_status", "").strip()
    if tenant_id:
        params["tenant_id"] = tenant_id
    if collection_name:
        params["collection_name"] = collection_name
    if status:
        params["status"] = status
    response = _request("GET", "/documents", params=params)
    if isinstance(response, dict):
        documents = response.get("documents", [])
    else:
        documents = response
    return [document for document in documents if isinstance(document, dict)]


def _refresh_document_status(document_id: str) -> dict[str, Any]:
    return _request("GET", f"/documents/{document_id}/status")


def _document_chunks(
    document_id: str,
    *,
    limit: int,
    offset: int,
    include_text: bool,
    include_vector: bool,
) -> dict[str, Any]:
    return _request(
        "GET",
        f"/documents/{quote(document_id, safe='')}/chunks",
        params={
            "limit": limit,
            "offset": offset,
            "include_text": str(include_text).lower(),
            "include_vector": str(include_vector).lower(),
        },
        timeout=120,
    )


def _submit_document_for_processing(document_id: str) -> dict[str, Any]:
    return _request("POST", f"/documents/{document_id}/submit", timeout=120)


def _parse_metadata_fields(raw: str) -> list[dict[str, Any]]:
    """Parse JSON or line-based metadata fields entered by a tester."""
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Metadata fields JSON must be an array.")
        return [item for item in payload if isinstance(item, dict)]

    fields: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        parts = [part.strip() for part in value.split("|")]
        if len(parts) < 2:
            raise ValueError(
                f"Line {line_no}: use 'field_name | Display Label | data_type'."
            )
        field_name = parts[0]
        display_label = parts[1]
        data_type = parts[2] if len(parts) >= 3 and parts[2] else "string"
        if not field_name or not display_label:
            raise ValueError(f"Line {line_no}: field_name and display label are required.")
        fields.append(
            {
                "field_name": field_name,
                "display_label": display_label,
                "data_type": data_type,
            }
        )
    return fields


def _normalize_key_field_row(item: dict[str, Any]) -> dict[str, Any]:
    name = str(item.get("name") or item.get("field_name") or "").strip()
    if not name:
        raise ValueError("Each key field must include a name.")
    field_type = str(item.get("type") or item.get("data_type") or "string").strip().lower()
    required_raw = item.get("required")
    if isinstance(required_raw, str):
        required = required_raw.strip().lower() in {"1", "true", "yes", "required"}
    else:
        required = bool(required_raw)
    normalized: dict[str, Any] = {
        "name": name,
        "type": field_type or "string",
        "required": required,
    }
    description = str(item.get("description") or item.get("display_label") or "").strip()
    if description:
        normalized["description"] = description
    return normalized


def _parse_key_fields(raw: str) -> list[dict[str, Any]]:
    """Parse JSON or line-based repository key fields for extraction/validation."""
    text = (raw or "").strip()
    if not text:
        return []
    if text.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError("Key fields JSON must be an array.")
        return [_normalize_key_field_row(item) for item in payload if isinstance(item, dict)]

    fields: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        parts = [part.strip() for part in value.split("|")]
        if len(parts) < 1 or not parts[0]:
            raise ValueError(
                f"Line {line_no}: use 'name | type | required | description'."
            )
        try:
            fields.append(
                _normalize_key_field_row(
                    {
                        "name": parts[0],
                        "type": parts[1] if len(parts) >= 2 and parts[1] else "string",
                        "required": parts[2] if len(parts) >= 3 else False,
                        "description": parts[3] if len(parts) >= 4 else None,
                    }
                )
            )
        except ValueError as exc:
            raise ValueError(f"Line {line_no}: {exc}") from exc
    return fields


def _format_key_fields_text(fields: list[dict[str, Any]] | None) -> str:
    if not fields:
        return DEFAULT_KEY_FIELDS_TEXT
    lines: list[str] = []
    for item in fields:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        field_type = str(item.get("type") or "string")
        required = "true" if item.get("required") else "false"
        description = str(item.get("description") or "").strip()
        lines.append(f"{name} | {field_type} | {required} | {description}".rstrip(" |"))
    return "\n".join(lines) if lines else DEFAULT_KEY_FIELDS_TEXT


def _processor_base_urls() -> list[str]:
    configured = os.getenv("PROCESSOR_BASE_URLS", "").strip()
    if configured:
        return [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]
    ports = [port.strip() for port in DEFAULT_PROCESSOR_PORTS.split(",") if port.strip()]
    return [f"http://localhost:{port}" for port in ports]


def _find_idle_processor() -> str:
    errors: list[str] = []
    for base_url in _processor_base_urls():
        try:
            response = requests.get(f"{base_url}/status", timeout=5)
            response.raise_for_status()
            status = response.json()
            if status.get("health") == "healthy" and status.get("state") == "idle":
                return base_url
            errors.append(f"{base_url}: {status.get('state') or status.get('current_status')}")
        except Exception as exc:
            errors.append(f"{base_url}: {exc}")
    raise RuntimeError("No idle processor found. " + "; ".join(errors))


def _run_metadata_extraction_test(
    document: dict[str, Any],
    metadata_fields: list[dict[str, Any]],
) -> dict[str, Any]:
    if not metadata_fields:
        raise ValueError("Enter at least one metadata field.")

    processor_url = _find_idle_processor()
    document_id = str(document.get("document_id") or "")
    repository_path = document.get("repository_path") or document.get("container_document_path")
    container_document_path = document.get("container_document_path")
    if repository_path and not container_document_path:
        container_document_path = _build_path_mapping(Path(repository_path))["container_document_path"]
    payload = {
        "processor_type": "metadata_extraction",
        "document_id": f"metadata-test-{uuid.uuid4()}",
        "document_name": document.get("document_name") or document_id,
        "original_file_name": document.get("original_file_name") or document.get("document_name"),
        "document_path": container_document_path,
        "collection_name": document.get("collection_name"),
        "tenant_id": document.get("tenant_id"),
        "repository_id": document.get("repository_id") or (document.get("metadata") or {}).get("repository_id"),
        "document_type_id": "streamlit-custom-fields",
        "metadata_fields": metadata_fields,
        "extraction_model": {"provider": "rule_based"},
    }
    response = requests.post(f"{processor_url}/process", json=payload, timeout=15)
    response.raise_for_status()

    last_status: dict[str, Any] = {}
    for _ in range(120):
        time.sleep(1)
        status_response = requests.get(f"{processor_url}/status", timeout=10)
        status_response.raise_for_status()
        last_status = status_response.json()
        current = str(last_status.get("current_status") or "").lower()
        last_terminal = str(last_status.get("last_job_terminal_status") or "").lower()
        if current in {"completed", "failed", "stopped"} or last_terminal in {"completed", "failed", "stopped"}:
            break
    else:
        raise TimeoutError("Metadata extraction test did not finish within 120 seconds.")

    metadata = last_status.get("document_metadata") or last_status.get("last_job_document_metadata") or {}
    return {
        "processor_url": processor_url,
        "status": last_status,
        "extracted_fields": metadata.get("extracted_fields") or [],
        "document_metadata": metadata,
    }


def _delete_document_tracking(document_id: str, delete_file: bool = True) -> None:
    # POST alias: same deployment may run an older process without DELETE; POST avoids some proxy DELETE blocks.
    _request(
        "POST",
        f"/documents/{document_id}/delete",
        params={"delete_file": str(delete_file).lower()},
        timeout=60,
    )


def _stable_document_id(filename: str) -> str:
    stem = Path(filename).stem.strip().lower()
    cleaned = "".join(char if char.isalnum() else "-" for char in stem)
    cleaned = "-".join(part for part in cleaned.split("-") if part)
    return cleaned or f"document-{uuid.uuid4()}"


def _save_uploaded_file(uploaded: Any) -> Path:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    destination = (DOCUMENTS_DIR / Path(uploaded.name).name).resolve()
    destination.write_bytes(uploaded.getvalue())
    return destination


def _build_process_payload(
    *,
    document_id: str,
    filename: str,
    container_document_path: str,
    collection_name: str,
    tenant_id: str,
    repository_id: str,
    document_type_id: str | None = None,
) -> dict[str, Any]:
    if not container_document_path:
        raise ValueError("container document_path is missing.")
    is_windows_path = container_document_path.startswith(("C:\\", "C:/", "D:\\", "D:/")) or (
        len(container_document_path) >= 2 and container_document_path[1] == ":"
    )
    if is_windows_path and os.getenv("DEPLOYMENT_MODE", "").lower() != "local":
        raise ValueError(
            f"Refusing to send a Windows host path to a non-local processor: {container_document_path}"
        )
    payload: dict[str, Any] = {
        "processor_type": "chunking_vectorizing",
        "document_id": document_id,
        "document_name": filename,
        "original_file_name": filename,
        "document_path": container_document_path,
        "collection_name": collection_name or "DocumentChunk",
        "tenant_id": tenant_id or "default",
        "repository_id": repository_id or "default",
        "repository_settings": {},
        "chunk_size": 500,
        "chunk_overlap_sentences": 2,
        "min_content_words": 20,
        "chunking_strategy": normalize_chunking_strategy(
            os.getenv("STREAMLIT_CHUNKING_STRATEGY", "sentence-based")
        ),
        "chunking_config": {},
        "citation_retainment": True,
        "document_type_id": (document_type_id or "").strip(),
        "metadata_fields": [],
        "extraction_model": {},
        # Host-side UploadSecurityPipeline already ran before this request.
        "security_prevalidated": True,
    }
    model_name = (os.getenv("MODEL_NAME") or os.getenv("STREAMLIT_MODEL_NAME") or "").strip()
    model_dir = (os.getenv("MODEL_DIR") or os.getenv("STREAMLIT_MODEL_DIR") or "").strip()
    if model_name:
        payload["model_name"] = model_name
    if model_dir:
        payload["model_dir"] = model_dir
    return payload


def _build_reference_payload(
    *,
    document_id: str,
    filename: str,
    container_document_path: str,
    tenant_id: str,
    repository_id: str,
    document_type_id: str | None = None,
) -> dict[str, Any]:
    return {
        "processor_type": "reference_extraction",
        "document_id": document_id,
        "document_name": filename,
        "original_file_name": filename,
        "document_path": container_document_path,
        "tenant_id": tenant_id or "default",
        "repository_id": repository_id or "default",
        "document_type_id": (document_type_id or "").strip() or None,
    }


def _upload_documents(
    uploaded_files: list[Any],
    collection_name: str,
    tenant_id: str,
    submit_for_processing: bool,
    repository_id: str | None = None,
    document_type_id: str | None = None,
    document_id_override: str | None = None,
    processing_mode: str = "Chunking + Embedding Only",
) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []
    logs: list[dict[str, Any]] = []
    if submit_for_processing:
        logs.append({"backend_preflight": _check_required_backends(processing_mode)})

    for index, uploaded in enumerate(uploaded_files, start=1):
        saved_path = _save_uploaded_file(uploaded)
        path_mapping = _build_path_mapping(saved_path)
        container_document_path = path_mapping["container_document_path"]
        document_id = (
            document_id_override.strip()
            if document_id_override and len(uploaded_files) == 1
            else _stable_document_id(uploaded.name)
        )
        payload = _build_process_payload(
            document_id=document_id,
            filename=Path(uploaded.name).name,
            container_document_path=container_document_path,
            collection_name=collection_name.strip() or "DocumentChunk",
            tenant_id=tenant_id.strip() or "default",
            repository_id=(repository_id or "").strip() or "default",
            document_type_id=document_type_id,
        )
        reference_payload = _build_reference_payload(
            document_id=document_id,
            filename=Path(uploaded.name).name,
            container_document_path=container_document_path,
            tenant_id=tenant_id.strip() or "default",
            repository_id=(repository_id or "").strip() or "default",
            document_type_id=document_type_id,
        )
        log_entry: dict[str, Any] = {
            "saved_host_path": path_mapping["saved_host_path"],
            "container_document_path": container_document_path,
            "host_documents_dir": path_mapping["host_documents_dir"],
            "processor_documents_container_dir": path_mapping["processor_documents_container_dir"],
            "api_url_called": None,
            "request_json_sent": payload,
            "response_status_code": None,
            "response_body": None,
        }
        result: Any = {"status": "saved_only", "document_id": document_id}
        reference_result: Any = None

        # Hard gate security validation check on the saved host file
        from src.features.security.application.upload_security_pipeline import run_security_pipeline
        security_result = run_security_pipeline(Path(path_mapping["saved_host_path"]), document_id=document_id)
        # Keep status aligned with DLP decision (override logic can leave mixed fields).
        if str(security_result.get("dlp_decision") or "").upper() == "HUMAN_REVIEW":
            security_result = {
                **security_result,
                "status": "human_review",
                "requires_human_review": True,
            }
        log_entry["security_pipeline"] = {
            "status": security_result.get("status"),
            "severity": security_result.get("severity"),
            "risk_level": security_result.get("risk_level"),
            "dlp_decision": security_result.get("dlp_decision"),
            "available_reviewer_actions": security_result.get("available_reviewer_actions"),
            "reason": security_result.get("reason"),
            "detected_categories": security_result.get("detected_categories"),
            "pipeline_stages": security_result.get("pipeline_stages"),
            "document_type": security_result.get("document_type"),
            "topics": security_result.get("topics"),
            "moderation_result": security_result.get("moderation_result"),
            "actions": security_result.get("actions"),
            "detection_count": len(security_result.get("detections") or []),
        }

        if security_result.get("status") == "block":
            # Hard-block path: structural errors OR reviewer explicitly chose Block.
            # Reviewer-blocked documents must not be re-routed to human_review —
            # the decision was already made. All other content blocks (non-structural,
            # auto_block disabled) are coerced to human_review for human decision.
            try:
                from src.features.security.dlp.policy_loader import is_auto_block_enabled

                auto_block = is_auto_block_enabled()
            except Exception:
                auto_block = False
            structural = any(
                phrase in str(security_result.get("reason") or "").lower()
                for phrase in (
                    "unsupported file extension",
                    "file does not exist",
                    "file size exceeds",
                    "could not be parsed",
                )
            )
            reviewer_blocked = "blocked by reviewer" in str(security_result.get("reason") or "").lower()
            if auto_block or structural or reviewer_blocked:
                raise ComplianceBlockError(security_result, log_entry)
            from src.features.security.review.review_decisions import enrich_policy_result

            security_result = enrich_policy_result(
                {
                    **security_result,
                    "status": "human_review",
                    "requires_human_review": True,
                    "reason": security_result.get("reason")
                    or "Elevated risk signals. Choose Allow / Mask and Allow / Block.",
                    "actions": list(security_result.get("actions") or []) + ["route_to_human_review"],
                }
            )
            try:
                from src.features.security.review.human_review_queue import HumanReviewQueue
                from src.shared.networking.document_paths import resolve_security_documents_dir

                HumanReviewQueue(
                    queue_file=str(resolve_security_documents_dir() / "human_review_queue.json")
                ).add_to_queue(
                    document_id,
                    Path(path_mapping["saved_host_path"]).name,
                    str(security_result.get("severity") or "high"),
                    str(security_result.get("reason") or "human_review"),
                    list(security_result.get("detections") or []),
                    risk_level=str(security_result.get("risk_level") or "HIGH"),
                    dlp_decision="human_review",
                    available_actions=list(security_result.get("available_reviewer_actions") or []),
                    dlp_reason=str(security_result.get("reason") or ""),
                )
            except Exception:
                pass
            log_entry["security_pipeline"] = {
                **(log_entry.get("security_pipeline") or {}),
                "status": "human_review",
                "dlp_decision": security_result.get("dlp_decision"),
                "available_reviewer_actions": security_result.get("available_reviewer_actions"),
                "reason": security_result.get("reason"),
            }

        if _live_scan_needs_review(security_result) or security_result.get("status") == "human_review":
            st.session_state.pop("last_compliance_block", None)
            result = security_result
            reference_result = security_result
        elif submit_for_processing and _should_auto_process_on_upload(security_result):
            backend_diag = _check_required_backends(processing_mode)
            log_entry["backend_preflight"] = backend_diag
            if not _processors_ready(backend_diag):
                result = {
                    "status": "processor_deferred",
                    "document_id": document_id,
                    "reason": "Processors are starting — file saved; processing runs after Allow or when backends recover.",
                }
            else:
                result, reference_result, proc_logs, proc_err = _invoke_processors_safely(
                    processing_mode=processing_mode,
                    payload=payload,
                    reference_payload=reference_payload,
                )
                log_entry.update(proc_logs)
                if proc_err:
                    log_entry["processor_error"] = proc_err
        else:
            result = {
                "status": "saved_pending_review" if _live_scan_needs_review(security_result) else "saved_only",
                "document_id": document_id,
            }
        logs.append(log_entry)
        documents.append(
            {
                "document_id": document_id,
                "document_name": Path(uploaded.name).name,
                "original_file_name": Path(uploaded.name).name,
                "repository_path": path_mapping["saved_host_path"],
                "container_document_path": container_document_path,
                "collection_name": payload["collection_name"],
                "tenant_id": payload["tenant_id"],
                "security_result": security_result,
                "process_response": result,
                "reference_extraction_response": reference_result,
            }
        )
        try:
            print(f"saved host path: {path_mapping['saved_host_path']}")
            print(f"container document path: {container_document_path}")
            print(f"API URL called: {log_entry.get('api_url_called')}")
            print(f"request JSON sent: {json.dumps(payload, indent=2)}")
            print(f"response status code: {log_entry.get('response_status_code')}")
            print(f"response body: {log_entry.get('response_body')}")
        except OSError:
            pass

    return {
        "uploaded_count": len(documents),
        "batch_id": str(uuid.uuid4()),
        "documents": documents,
        "logs": logs,
    }


def _load_document_references(document_id: str) -> dict[str, Any]:
    return _request("GET", f"/documents/{quote(document_id, safe='')}/references", timeout=60)


def _load_document_referenced_by(document_id: str) -> dict[str, Any]:
    return _request("GET", f"/documents/{quote(document_id, safe='')}/referenced-by", timeout=60)


def _load_document_reference_graph(document_id: str) -> dict[str, Any]:
    return _request("GET", f"/documents/{quote(document_id, safe='')}/reference-graph", timeout=60)


def _show_empty_reference_hint(payload: dict[str, Any], key: str) -> None:
    items = payload.get(key) or []
    if not items:
        st.warning(
            "Neo4j reference graph is empty for this document_id. "
            "The document may not contain extractable references, processing may still be running, "
            "or Neo4j may be unavailable."
        )


def render_reference_lookup(document_id: str, *, key_prefix: str) -> None:
    if not document_id:
        return
    st.subheader("Reference Graph Lookup")
    st.caption(f"Using document_id: `{document_id}`")
    c_refs, c_by, c_graph = st.columns(3)
    with c_refs:
        if st.button("Get references", key=f"{key_prefix}::references", use_container_width=True):
            try:
                payload = _load_document_references(document_id)
                st.session_state[f"{key_prefix}::references_payload"] = payload
                _show_empty_reference_hint(payload, "references")
            except requests.ConnectionError:
                st.error(f"Backend is not running or is unreachable at {_api_base_url()}.")
            except Exception as exc:
                st.error(f"Unable to load references: {exc}")
    with c_by:
        if st.button("Get referenced-by", key=f"{key_prefix}::referenced_by", use_container_width=True):
            try:
                payload = _load_document_referenced_by(document_id)
                st.session_state[f"{key_prefix}::referenced_by_payload"] = payload
                _show_empty_reference_hint(payload, "referenced_by")
            except requests.ConnectionError:
                st.error(f"Backend is not running or is unreachable at {_api_base_url()}.")
            except Exception as exc:
                st.error(f"Unable to load referenced-by: {exc}")
    with c_graph:
        if st.button("Get reference graph", key=f"{key_prefix}::reference_graph", use_container_width=True):
            try:
                payload = _load_document_reference_graph(document_id)
                st.session_state[f"{key_prefix}::reference_graph_payload"] = payload
                if not (payload.get("nodes") or []) and not (payload.get("edges") or []):
                    st.warning(
                        "Neo4j reference graph is empty for this document_id. "
                        "The document may not contain extractable references, processing may still be running, "
                        "or Neo4j may be unavailable."
                    )
            except requests.ConnectionError:
                st.error(f"Backend is not running or is unreachable at {_api_base_url()}.")
            except Exception as exc:
                st.error(f"Unable to load reference graph: {exc}")

    for label, state_key in [
        ("References response", f"{key_prefix}::references_payload"),
        ("Referenced-by response", f"{key_prefix}::referenced_by_payload"),
        ("Reference graph response", f"{key_prefix}::reference_graph_payload"),
    ]:
        if st.session_state.get(state_key):
            with st.expander(label, expanded=True):
                st.json(st.session_state[state_key])


def _load_repositories(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    cache_key = "upload_repositories_cache"
    if force_refresh:
        st.session_state.pop(cache_key, None)
        st.session_state.pop(f"{cache_key}_error", None)
    if cache_key not in st.session_state:
        try:
            payload = _request("GET", "/repositories", timeout=30)
            repositories = payload.get("repositories", []) if isinstance(payload, dict) else payload
            st.session_state[cache_key] = [
                repo for repo in (repositories or []) if isinstance(repo, dict) and repo.get("repository_id")
            ]
            st.session_state.pop(f"{cache_key}_error", None)
        except Exception as exc:
            st.session_state[cache_key] = []
            st.session_state[f"{cache_key}_error"] = str(exc)
    return list(st.session_state.get(cache_key) or [])


def _load_repository_upload_options(*, force_refresh: bool = False) -> list[dict[str, Any]]:
    """Active repositories formatted for upload dropdowns."""
    cache_key = "upload_repository_options_cache"
    if force_refresh:
        st.session_state.pop(cache_key, None)
        st.session_state.pop(f"{cache_key}_error", None)
    if cache_key not in st.session_state:
        try:
            payload = _request("GET", "/repositories/options", params={"status": "active"}, timeout=30)
            rows = payload if isinstance(payload, list) else []
            st.session_state[cache_key] = [
                row for row in rows if isinstance(row, dict) and str(row.get("id") or "").strip()
            ]
            st.session_state.pop(f"{cache_key}_error", None)
        except Exception as exc:
            st.session_state[cache_key] = []
            st.session_state[f"{cache_key}_error"] = str(exc)
    return list(st.session_state.get(cache_key) or [])


def _repository_record_for_upload(
    repository_id: str,
    *,
    repositories: list[dict[str, Any]],
    repository_options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    for repo in repositories:
        if str(repo.get("repository_id") or "") == repository_id:
            return repo
    for option in repository_options:
        if str(option.get("id") or "") == repository_id:
            return {
                "repository_id": option.get("id"),
                "name": option.get("name"),
                "status": "active",
                "settings": {"document_type_id": option.get("default_document_type_id")},
            }
    return None


def _load_document_types(*, repository_id: str | None = None) -> list[dict[str, Any]]:
    params: dict[str, Any] = {}
    if repository_id:
        params["repository_id"] = str(repository_id).strip()
    payload = _request("GET", "/document-types", params=params or None, timeout=60)
    rows = payload.get("document_types", []) if isinstance(payload, dict) else payload
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return [row for row in (payload or []) if isinstance(row, dict)]


def _dedupe_document_types_by_id(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for row in rows:
        document_type_id = str(row.get("document_type_id") or "").strip()
        if not document_type_id or document_type_id in seen:
            continue
        seen.add(document_type_id)
        unique.append(row)
    return unique


def _document_type_option_label(
    row: dict[str, Any],
    *,
    repository_name: str | None = None,
) -> str:
    name = str(row.get("name") or row.get("document_type_id") or "Unknown")
    depth = row.get("depth_level")
    suffix_parts: list[str] = []
    if repository_name:
        suffix_parts.append(repository_name)
    elif str(row.get("repository_id") or "").strip():
        suffix_parts.append(str(row.get("repository_id"))[:8])
    if depth is not None and int(depth or 0) > 0:
        suffix_parts.append(f"L{int(depth)}")
    if suffix_parts:
        return f"{name} ({', '.join(suffix_parts)})"
    return name


def _load_document_types_for_repository(repository_id: str) -> list[dict[str, Any]]:
    if not repository_id:
        return []
    return _dedupe_document_types_by_id(_load_document_types(repository_id=repository_id))


def _repository_default_document_type_id(repository: dict[str, Any] | None) -> str | None:
    if not repository:
        return None
    settings = repository.get("settings") if isinstance(repository.get("settings"), dict) else {}
    value = str((settings or {}).get("document_type_id") or "").strip()
    return value or None


def _repository_default_document_type_name(
    repository: dict[str, Any] | None,
    document_types: list[dict[str, Any]],
) -> str | None:
    default_id = _repository_default_document_type_id(repository)
    if not default_id:
        return None
    for row in document_types:
        if str(row.get("document_type_id") or "") == default_id:
            return str(row.get("name") or default_id)
    return default_id


def _document_type_name_by_id(document_types: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(row.get("document_type_id")): str(row.get("name") or row.get("document_type_id"))
        for row in document_types
        if row.get("document_type_id")
    }


DOCUMENT_TYPE_QUICK_TEMPLATES: tuple[dict[str, str], ...] = (
    {
        "key": "sop",
        "name": "SOP",
        "parent_name": "Basic",
        "description": "Standard Operating Procedure",
    },
    {
        "key": "manufacturing_sop",
        "name": "Manufacturing SOP",
        "parent_name": "SOP",
        "description": "Manufacturing standard operating procedure",
    },
)


def _clear_document_type_caches() -> None:
    for key in ("upload_repositories_cache", "upload_repository_options_cache"):
        st.session_state.pop(key, None)
        st.session_state.pop(f"{key}_error", None)


def _find_document_type_by_name(
    document_types: list[dict[str, Any]],
    name: str,
) -> dict[str, Any] | None:
    target = str(name or "").strip().casefold()
    if not target:
        return None
    for row in document_types:
        if str(row.get("name") or "").strip().casefold() == target:
            return row
    return None


def _basic_document_type(document_types: list[dict[str, Any]]) -> dict[str, Any] | None:
    for row in document_types:
        if row.get("is_system") and str(row.get("name") or "").strip().casefold() == "basic":
            return row
    return _find_document_type_by_name(document_types, "Basic")


def _document_type_has_children(document_types: list[dict[str, Any]], document_type_id: str) -> bool:
    parent_id = str(document_type_id or "").strip()
    return any(str(row.get("parent_document_type_id") or "") == parent_id for row in document_types)


def _deletable_document_types(document_types: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in document_types
        if not row.get("is_system")
        and str(row.get("document_type_id") or "").strip()
        and not _document_type_has_children(document_types, str(row.get("document_type_id")))
    ]


def _flatten_document_type_tree(
    nodes: list[dict[str, Any]],
    *,
    parent_name: str | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: str(item.get("name") or "")):
        display_name = str(node.get("name") or "")
        row = {
            "name": f"{'  ' * depth}{display_name}" if depth else display_name,
            "display_name": display_name,
            "depth_level": node.get("depth_level", depth),
            "parent": parent_name or "",
            "document_type_id": node.get("document_type_id"),
            "is_system": bool(node.get("is_system")),
            "is_active": bool(node.get("is_active", True)),
            "description": str(node.get("description") or ""),
        }
        rows.append(row)
        children = node.get("children") or []
        if isinstance(children, list) and children:
            rows.extend(
                _flatten_document_type_tree(
                    children,
                    parent_name=display_name,
                    depth=depth + 1,
                )
            )
    return rows


def _resolve_template_parent_id(
    document_types: list[dict[str, Any]],
    parent_name: str,
) -> str | None:
    parent = _find_document_type_by_name(document_types, parent_name)
    if parent is None and str(parent_name or "").strip().casefold() == "basic":
        basic = _basic_document_type(document_types)
        return str(basic.get("document_type_id") or "").strip() or None
    if parent is None:
        return None
    return str(parent.get("document_type_id") or "").strip() or None


def _document_types_missing_for_templates(
    document_types: list[dict[str, Any]],
    *,
    template_keys: tuple[str, ...] | None = None,
) -> list[dict[str, str]]:
    missing: list[dict[str, str]] = []
    for template in DOCUMENT_TYPE_QUICK_TEMPLATES:
        if template_keys is not None and template["key"] not in template_keys:
            continue
        if _find_document_type_by_name(document_types, template["name"]) is None:
            missing.append(template)
    return missing


def _create_repository_document_type(repository_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request(
        "POST",
        f"/repositories/{quote(repository_id, safe='')}/document-types",
        json=payload,
        timeout=60,
    )


def _delete_repository_document_type(repository_id: str, document_type_id: str) -> None:
    _request(
        "DELETE",
        f"/repositories/{quote(repository_id, safe='')}/document-types/{quote(document_type_id, safe='')}",
        timeout=60,
    )


def _provision_document_type_templates(
    repository_id: str,
    document_types: list[dict[str, Any]],
    *,
    template_keys: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    created: list[dict[str, Any]] = []
    errors: list[str] = []
    working = list(document_types)
    for template in DOCUMENT_TYPE_QUICK_TEMPLATES:
        if template_keys is not None and template["key"] not in template_keys:
            continue
        if _find_document_type_by_name(working, template["name"]) is not None:
            continue
        parent_id = _resolve_template_parent_id(working, template["parent_name"])
        if not parent_id:
            errors.append(
                f"Cannot add {template['name']}: parent type '{template['parent_name']}' was not found."
            )
            continue
        payload = {
            "name": template["name"],
            "parent_document_type_id": parent_id,
            "description": template["description"],
        }
        record = _create_repository_document_type(repository_id, payload)
        created.append(record)
        working.append(record)
    return created, errors


def _render_repository_document_types_section(
    repository_id: str,
    *,
    repository_name: str | None = None,
) -> None:
    st.subheader("Document types")
    st.caption(
        "Document types are scoped to each repository. New repositories start with Basic only; "
        "add SOP and other types here so they appear on the Upload tab."
    )

    try:
        repository_document_types = _load_document_types_for_repository(repository_id)
    except requests.HTTPError as exc:
        st.error(f"Failed to load document types: {_format_http_error(exc)}")
        return
    except Exception as exc:
        st.error(f"Failed to load document types: {exc}")
        return

    try:
        tree_payload = _request(
            "GET",
            f"/repositories/{quote(repository_id, safe='')}/document-types/tree",
            timeout=60,
        )
        tree_nodes = tree_payload.get("tree") if isinstance(tree_payload, dict) else []
        tree_rows = _flatten_document_type_tree(tree_nodes if isinstance(tree_nodes, list) else [])
    except Exception:
        tree_rows = [
            {
                "name": str(row.get("name") or ""),
                "display_name": str(row.get("name") or ""),
                "depth_level": row.get("depth_level"),
                "parent": "",
                "document_type_id": row.get("document_type_id"),
                "is_system": bool(row.get("is_system")),
                "is_active": bool(row.get("is_active", True)),
                "description": str(row.get("description") or ""),
            }
            for row in sorted(repository_document_types, key=lambda item: str(item.get("name") or ""))
        ]

    if tree_rows:
        st.dataframe(
            [
                {
                    "name": row.get("name"),
                    "parent": row.get("parent") or "—",
                    "depth": row.get("depth_level"),
                    "system": row.get("is_system"),
                    "active": row.get("is_active"),
                    "document_type_id": row.get("document_type_id"),
                    "description": row.get("description") or "",
                }
                for row in tree_rows
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No document types found for this repository.")

    default_type_id = None
    try:
        settings_payload = _request("GET", f"/repositories/{quote(repository_id, safe='')}/settings", timeout=60)
        settings = settings_payload.get("settings") if isinstance(settings_payload.get("settings"), dict) else {}
        default_type_id = str((settings or {}).get("document_type_id") or "").strip() or None
    except Exception:
        pass

    type_options = [""] + [
        str(row.get("document_type_id"))
        for row in repository_document_types
        if str(row.get("document_type_id") or "").strip()
    ]
    type_labels = {
        str(row.get("document_type_id")): _document_type_option_label(row, repository_name=repository_name)
        for row in repository_document_types
        if str(row.get("document_type_id") or "").strip()
    }
    default_index = type_options.index(default_type_id) if default_type_id in type_options else 0

    with st.form(f"repo_default_document_type::{repository_id}"):
        selected_default = st.selectbox(
            "Default document type for uploads",
            options=type_options,
            index=default_index,
            format_func=lambda value: "Not configured" if not value else type_labels.get(value, value),
            help=(
                "Used when Upload > Document Type is left as Repository default. "
                "Can be changed even after the repository is activated."
            ),
        )
        save_default = st.form_submit_button("Save default document type")
    if save_default:
        if not selected_default:
            st.warning("Choose a document type to set as the repository default.")
        else:
            try:
                _request(
                    "PATCH",
                    f"/repositories/{quote(repository_id, safe='')}/settings",
                    json={"document_type_id": selected_default},
                    timeout=60,
                )
                st.success("Default document type updated.")
                _clear_document_type_caches()
                _rerun_fragment_only()
            except requests.HTTPError as exc:
                st.error(f"Save default document type failed: {_format_http_error(exc)}")
            except Exception as exc:
                st.error(f"Save default document type failed: {exc}")

    missing_sop = _document_types_missing_for_templates(repository_document_types, template_keys=("sop",))
    missing_msop = _document_types_missing_for_templates(
        repository_document_types,
        template_keys=("manufacturing_sop",),
    )
    quick_col1, quick_col2 = st.columns(2)
    with quick_col1:
        add_sop = st.button(
            "Quick add: SOP",
            key=f"repo_doc_type_add_sop::{repository_id}",
            disabled=not missing_sop,
            help="Creates SOP under Basic for this repository.",
        )
    with quick_col2:
        add_msop = st.button(
            "Quick add: Manufacturing SOP",
            key=f"repo_doc_type_add_msop::{repository_id}",
            disabled=not missing_msop,
            help="Creates Manufacturing SOP under SOP (adds SOP first if missing).",
        )

    if add_sop:
        try:
            created, errors = _provision_document_type_templates(
                repository_id,
                repository_document_types,
                template_keys=("sop",),
            )
            if created:
                st.success(f"Added document type: {created[-1].get('name')}")
                _clear_document_type_caches()
                _rerun_fragment_only()
            for message in errors:
                st.error(message)
        except requests.HTTPError as exc:
            st.error(f"Quick add SOP failed: {_format_http_error(exc)}")
        except Exception as exc:
            st.error(f"Quick add SOP failed: {exc}")

    if add_msop:
        try:
            created, errors = _provision_document_type_templates(
                repository_id,
                repository_document_types,
                template_keys=("sop", "manufacturing_sop"),
            )
            if created:
                names = ", ".join(str(item.get("name") or "type") for item in created)
                st.success(f"Added document type(s): {names}")
                _clear_document_type_caches()
                _rerun_fragment_only()
            for message in errors:
                st.error(message)
        except requests.HTTPError as exc:
            st.error(f"Quick add Manufacturing SOP failed: {_format_http_error(exc)}")
        except Exception as exc:
            st.error(f"Quick add Manufacturing SOP failed: {exc}")

    parent_options = [""] + [
        str(row.get("document_type_id"))
        for row in repository_document_types
        if str(row.get("document_type_id") or "").strip()
    ]
    parent_labels = {
        "": "Root (no parent)",
        **type_labels,
    }
    with st.form(f"repo_create_document_type::{repository_id}"):
        new_type_name = st.text_input("Name", placeholder="e.g. SOP, Policy, Work Instruction")
        new_type_parent = st.selectbox(
            "Parent type",
            options=parent_options,
            format_func=lambda value: parent_labels.get(value, value),
            help="Child types inherit key fields from ancestors (Basic → SOP → Manufacturing SOP).",
        )
        new_type_description = st.text_area("Description", height=80)
        create_type = st.form_submit_button("Create document type", type="primary")

    if create_type:
        trimmed_name = str(new_type_name or "").strip()
        if not trimmed_name:
            st.error("Document type name is required.")
        elif _find_document_type_by_name(repository_document_types, trimmed_name) is not None:
            st.error(f"A document type named '{trimmed_name}' already exists in this repository.")
        else:
            payload: dict[str, Any] = {"name": trimmed_name}
            if new_type_parent:
                payload["parent_document_type_id"] = new_type_parent
            if str(new_type_description or "").strip():
                payload["description"] = str(new_type_description).strip()
            try:
                created = _create_repository_document_type(repository_id, payload)
                st.success(f"Created document type: {created.get('name') or trimmed_name}")
                _clear_document_type_caches()
                _rerun_fragment_only()
            except requests.HTTPError as exc:
                st.error(f"Create document type failed: {_format_http_error(exc)}")
            except Exception as exc:
                st.error(f"Create document type failed: {exc}")

    deletable_types = _deletable_document_types(repository_document_types)
    if deletable_types:
        delete_options = [str(row.get("document_type_id")) for row in deletable_types]
        delete_labels = {
            str(row.get("document_type_id")): str(row.get("name") or row.get("document_type_id"))
            for row in deletable_types
        }
        with st.form(f"repo_delete_document_type::{repository_id}"):
            delete_type_id = st.selectbox(
                "Delete document type",
                options=delete_options,
                format_func=lambda value: delete_labels.get(value, value),
                help="Only leaf types (no children) can be deleted. Basic is protected.",
            )
            confirm_delete = st.checkbox("Confirm delete")
            delete_type = st.form_submit_button("Delete selected type")
        if delete_type:
            if not confirm_delete:
                st.error("Check Confirm delete before removing a document type.")
            else:
                try:
                    _delete_repository_document_type(repository_id, delete_type_id)
                    st.success(f"Deleted document type: {delete_labels.get(delete_type_id, delete_type_id)}")
                    _clear_document_type_caches()
                    _rerun_fragment_only()
                except requests.HTTPError as exc:
                    st.error(f"Delete document type failed: {_format_http_error(exc)}")
                except Exception as exc:
                    st.error(f"Delete document type failed: {exc}")


_UPLOAD_FILE_CACHE_KEY = "upload_file_cache"


class _CachedUploadFile:
    """Session-state snapshot of a Streamlit UploadedFile."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def _coerce_upload_file_list(uploaded_files: Any) -> list[Any]:
    if not uploaded_files:
        return []
    if isinstance(uploaded_files, list):
        return uploaded_files
    return [uploaded_files]


def _refresh_upload_file_cache(uploaded_files: Any) -> None:
    files = _coerce_upload_file_list(uploaded_files)
    if not files:
        return
    st.session_state[_UPLOAD_FILE_CACHE_KEY] = [
        {"name": Path(str(item.name)).name, "bytes": item.getvalue()}
        for item in files
    ]


def _cached_upload_files() -> list[_CachedUploadFile]:
    rows = st.session_state.get(_UPLOAD_FILE_CACHE_KEY) or []
    cached: list[_CachedUploadFile] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        data = row.get("bytes")
        if name and isinstance(data, (bytes, bytearray)) and data:
            cached.append(_CachedUploadFile(name, bytes(data)))
    return cached


def _upload_files_for_submission(current_uploaded: Any) -> list[Any]:
    """Return live uploader files, or cached bytes when Streamlit clears the widget on rerun."""
    current = _coerce_upload_file_list(current_uploaded)
    if current:
        _refresh_upload_file_cache(current)
        return current
    return _cached_upload_files()


def _upload_documents_via_api(
    uploaded_files: list[Any],
    *,
    collection_name: str,
    tenant_id: str,
    submit_for_processing: bool,
    repository_id: str | None = None,
    document_type_id: str | None = None,
) -> dict[str, Any]:
    if not uploaded_files:
        raise ValueError("No files to upload. Select a file and click Upload again.")

    url = f"{_api_base_url()}{_api_path('/documents/upload')}"
    headers = _api_auth_headers()

    data: dict[str, str] = {
        "submit_for_processing": str(bool(submit_for_processing)).lower(),
    }
    if collection_name.strip():
        data["collection_name"] = collection_name.strip()
    if tenant_id.strip():
        data["tenant_id"] = tenant_id.strip()
    if repository_id:
        data["repository_id"] = repository_id
    if document_type_id:
        data["document_type_id"] = document_type_id

    multipart_files: list[tuple[str, tuple[str, bytes, str]]] = []
    for uploaded in uploaded_files:
        filename = Path(str(uploaded.name)).name
        content = uploaded.getvalue()
        suffix = Path(filename).suffix.lower()
        media_type = "application/octet-stream"
        if suffix == ".pdf":
            media_type = "application/pdf"
        elif suffix == ".docx":
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif suffix == ".txt":
            media_type = "text/plain"
        multipart_files.append(("files", (filename, content, media_type)))

    if not multipart_files:
        raise ValueError(
            "Upload file data was empty. Re-select your file in the picker and click Upload again."
        )

    response = _request_with_retries(
        "POST",
        url,
        timeout=180,
        headers=headers or None,
        data=data,
        files=multipart_files,
    )
    if response.status_code == 202:
        payload = response.json() if (response.text or "").strip() else {}
        err = payload.get("error") if isinstance(payload, dict) else {}
        details = err.get("details") if isinstance(err, dict) else {}
        severity = str((details or {}).get("severity_level") or "medium").lower()
        risk_map = {"critical": "CRITICAL", "high": "HIGH", "medium": "MEDIUM", "low": "LOW"}
        risk_level = risk_map.get(severity, "MEDIUM")
        filename = Path(str(uploaded_files[0].name)).name if uploaded_files else "document"
        # Compliance human-review responses may not include document_id.
        # Derive it from the stored filename when possible (UUID-style stem).
        detected_doc_id = str((details or {}).get("document_id") or "").strip()
        if not detected_doc_id:
            suggested = Path(filename).stem
            if suggested and "-" in suggested:
                detected_doc_id = suggested
        mapped_documents = [
            {
                "document_id": detected_doc_id or None,
                "document_name": filename,
                "original_file_name": filename,
                "collection_name": collection_name.strip() or "DocumentChunk",
                "tenant_id": tenant_id.strip() or "default",
                "status": "human_review",
                "document_type_id": document_type_id,
                "document_type_name": None,
                "security_result": {
                    "status": "human_review",
                    "severity": severity,
                    "risk_level": risk_level,
                    "dlp_decision": "HUMAN_REVIEW",
                    "requires_human_review": True,
                    "reason": str((details or {}).get("blocking_reason") or (err or {}).get("message") or "Pending human review"),
                    "detected_categories": list((details or {}).get("detected_categories") or []),
                    "detections": [],
                    "available_reviewer_actions": ["ALLOW", "MASK_AND_ALLOW", "BLOCK"],
                },
                "process_response": {
                    "status": "human_review",
                    "source": "dms_upload_api_202",
                },
                "reference_extraction_response": None,
            }
        ]
        return {
            "uploaded_count": 1,
            "batch_id": None,
            "documents": mapped_documents,
            "logs": [
                {
                    "api_url_called": url,
                    "request_form_sent": data,
                    "response_status_code": response.status_code,
                    "response_body": payload,
                }
            ],
        }
    response.raise_for_status()
    payload = response.json() if (response.text or "").strip() else {}

    rows = payload.get("documents", []) if isinstance(payload, dict) else []
    mapped_documents: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        security_scan = metadata.get("security_scan") if isinstance(metadata.get("security_scan"), dict) else {}
        doc_status = str(row.get("status") or "")
        security_result = dict(security_scan) if security_scan else {}
        if security_result and not security_result.get("requires_human_review"):
            security_result["requires_human_review"] = bool(metadata.get("requires_human_review"))
        needs_review = _live_scan_needs_review(security_result) or doc_status == "human_review"
        process_status = "human_review" if needs_review else doc_status
        mapped_documents.append(
            {
                "document_id": row.get("document_id"),
                "document_name": row.get("document_name"),
                "original_file_name": row.get("original_file_name"),
                "collection_name": row.get("collection_name"),
                "tenant_id": row.get("tenant_id"),
                "status": doc_status,
                "document_type_id": row.get("document_type_id"),
                "document_type_name": row.get("document_type_name"),
                "security_result": security_result,
                "process_response": {
                    "status": process_status,
                    "source": "dms_upload_api",
                },
                "reference_extraction_response": None,
            }
        )

    return {
        "uploaded_count": payload.get("uploaded_count", len(mapped_documents)) if isinstance(payload, dict) else len(mapped_documents),
        "batch_id": payload.get("batch_id") if isinstance(payload, dict) else None,
        "documents": mapped_documents,
        "logs": [
            {
                "api_url_called": url,
                "request_form_sent": data,
                "response_status_code": response.status_code,
                "response_body": payload,
            }
        ],
    }


def _status_badge(status: str) -> str:
    normalized = (status or "unknown").lower()
    if normalized in {"completed"}:
        return "completed"
    if normalized in {"failed", "queue_failed", "not_in_scheduler", "stopped", "stopping"}:
        return "failed"
    if normalized in {"queued", "starting", "collecting_chunks", "embedding", "storing", "processing"}:
        return "processing"
    if normalized in {"uploaded", "registered", "received"}:
        return "waiting"
    return normalized


def _is_completed_document(document: dict[str, Any]) -> bool:
    return str(document.get("status", "")).lower() == "completed"


def _document_label(document: dict[str, Any]) -> str:
    name = document.get("original_file_name") or document.get("document_name") or document.get("document_id")
    status = document.get("status") or "unknown"
    return f"{name} ({status})"


def _rerun_fragment_only() -> None:
    """Full page rerun (fragment decorator removed; always rerun the whole page)."""
    st.rerun()


def _review_local_fallback_enabled() -> bool:
    return os.getenv("STREAMLIT_REVIEW_LOCAL_FALLBACK", "").strip().lower() in {"1", "true", "yes", "on"}


def _review_decide_url(review_id: str) -> str:
    encoded = quote(review_id, safe="")
    return f"{_api_base_url()}{_api_path(f'/security/reviews/{encoded}/decide')}"


# --- Human review queue (frontend wiring) ---------------------------------

_DECISION_LABELS: dict[str, str] = {
    "ALLOW": "Allow",
    "MASK_AND_ALLOW": "Mask and Allow",
    "BLOCK": "Block",
}

# HUMAN_REVIEW is a DLP workflow state; it must never appear as a reviewer action.
_DISALLOWED_REVIEWER_ACTIONS = {"HUMAN_REVIEW"}
_PENDING_REVIEW_STATUSES = frozenset({"PENDING", "PENDING_REVIEW", ""})


def _review_queue_status(item: dict[str, Any] | None) -> str:
    return str((item or {}).get("status") or "").upper()


def _is_pending_review(item: dict[str, Any] | None) -> bool:
    return _review_queue_status(item) in _PENDING_REVIEW_STATUSES


def _local_review_item(review_id: str) -> dict[str, Any] | None:
    try:
        from src.features.security.review.human_review_service import HumanReviewService

        return HumanReviewService().get(review_id)
    except Exception:
        return None


def _decision_label(decision: str) -> str:
    key = str(decision or "").upper()
    return _DECISION_LABELS.get(key, key.replace("_", " ").title())


def _live_scan_needs_review(report: dict[str, Any] | None) -> bool:
    """True when the latest DLP scan requires a reviewer dropdown."""
    if not report:
        return False
    if report.get("review_completed"):
        return False
    outcome = report.get("review_outcome") or {}
    outcome_status = str(outcome.get("status") or outcome.get("review_status") or "").upper()
    if outcome_status in {"APPROVED", "APPROVED_MASKED", "REJECTED", "REPROCESSED", "COMPLETED"}:
        return False
    status = str(report.get("status") or "").lower()
    dlp = str(report.get("dlp_decision") or "").upper()
    if status in {"allow", "allowed", "pass", "block", "mask_and_allow"} and dlp in {
        "ALLOW",
        "BLOCK",
        "MASK_AND_ALLOW",
    }:
        return False
    if status == "human_review" or dlp == "HUMAN_REVIEW":
        return True
    if report.get("requires_human_review") and status not in {"allow", "allowed", "pass", "block", "mask_and_allow"}:
        return True
    return False


def _should_auto_process_on_upload(security_result: dict[str, Any]) -> bool:
    """Only LOW-risk DLP ALLOW may be processed automatically during upload.
    MEDIUM / HIGH / CRITICAL always wait for reviewer dropdown (config-driven via
    auto_allow_only_low in dlp_policies.yaml).
    """
    if _live_scan_needs_review(security_result):
        return False
    if security_result.get("requires_human_review"):
        return False
    dlp = str(security_result.get("dlp_decision") or "").upper()
    status = str(security_result.get("status") or "").lower()
    if dlp and dlp not in {"ALLOW", ""}:
        return False
    if status not in {"allow", "allowed", "pass"}:
        return False
    # Enforce low-risk-only auto-process from config.
    risk = str(security_result.get("risk_level") or "").upper()
    if risk and risk not in {"LOW", ""}:
        return False
    return True


def _invoke_processors_safely(
    *,
    processing_mode: str,
    payload: dict[str, Any],
    reference_payload: dict[str, Any],
) -> tuple[Any, Any, dict[str, Any], str | None]:
    """POST /process to chunking/reference backends; never raises."""
    result: Any = {"status": "saved_only", "document_id": payload.get("document_id")}
    reference_result: Any = None
    log_entry: dict[str, Any] = {}
    error_note: str | None = None
    try:
        if processing_mode in {"Chunking + Embedding Only", "Run Both"}:
            result, request_logs = _request_with_logs_at(
                _chunking_processor_url(),
                "POST",
                "/process",
                json=payload,
                timeout=120,
            )
            log_entry.update(request_logs)
        if processing_mode in {"Reference Extraction Only", "Run Both"}:
            reference_result, reference_logs = _request_with_logs_at(
                _reference_processor_url(),
                "POST",
                "/process",
                json=reference_payload,
                timeout=120,
            )
            log_entry["reference_extraction"] = reference_logs
        if processing_mode == "Reference Extraction Only":
            result = reference_result
    except (ProcessRequestError, RuntimeError, requests.RequestException) as exc:
        error_note = str(exc)
        result = {
            "status": "processor_deferred",
            "document_id": payload.get("document_id"),
            "reason": "Processing deferred — file is saved; retry from review Allow or when processor is healthy.",
            "detail": error_note[:500],
        }
    return result, reference_result, log_entry, error_note


def _effective_security_status(document_id: str, original_status: str) -> tuple[str, dict[str, Any] | None]:
    """Map terminal reviewer decisions onto pipeline status for UI display.

    A fresh DLP ``human_review`` outcome always wins over a stale queue row so
    re-uploads show Allow / Mask / Block instead of a prior Block.
    """
    original = str(original_status or "").lower()
    if not document_id:
        return original_status, None
    review_item = _fetch_review_item(f"rev_{document_id}")
    if original == "human_review":
        return "human_review", review_item
    if not review_item:
        return original_status, None
    queue_status = str(review_item.get("status") or "").upper()
    if queue_status == "REJECTED":
        return "block", review_item
    if queue_status == "APPROVED":
        return "allow", review_item
    if queue_status == "APPROVED_MASKED":
        return "mask_and_allow", review_item
    if queue_status == "PENDING":
        # Do not let stale pending rows override a fresh clean decision.
        # Pending queue should drive UI only when scan status is transient
        # (queued/in-progress/deferred) or explicitly human_review.
        stable_clean_statuses = {"allow", "accepted", "pass", "ok"}
        if original in stable_clean_statuses:
            return original_status, review_item
        return "human_review", review_item
    return original_status, review_item


def _sync_session_after_review(review_id: str, updated: dict[str, Any]) -> None:
    """Refresh cached upload/harness results after a reviewer action."""
    document_id = str(updated.get("document_id") or review_id.removeprefix("rev_"))
    queue_status = str(updated.get("status") or "").upper()
    decision = str(updated.get("reviewer_decision") or "").upper()
    status_map = {
        "REJECTED": "block",
        "APPROVED": "allow",
        "APPROVED_MASKED": "mask_and_allow",
    }
    dlp_map = {
        "block": "BLOCK",
        "allow": "ALLOW",
        "mask_and_allow": "MASK_AND_ALLOW",
    }
    pipeline_status = status_map.get(queue_status)
    if not pipeline_status:
        return

    reason = (
        f"Reviewer decision: {decision} by {updated.get('reviewer') or 'reviewer'}"
        + (f" — {updated.get('comments')}" if updated.get("comments") else "")
    )

    harness_doc_id = st.session_state.get("harness_scan_doc_id")
    if harness_doc_id and str(harness_doc_id) == document_id:
        result = dict(st.session_state.get("harness_scan_result") or {})
        result["status"] = pipeline_status
        result["dlp_decision"] = dlp_map[pipeline_status]
        result["requires_human_review"] = False
        result["review_completed"] = True
        result["reason"] = reason
        result["review_outcome"] = updated
        if pipeline_status == "mask_and_allow":
            result["masking_applied"] = True
        st.session_state["harness_scan_result"] = result

    last_upload = st.session_state.get("last_upload_result")
    if isinstance(last_upload, dict):
        for doc in last_upload.get("documents") or []:
            if str(doc.get("document_id")) != document_id:
                continue
            security_result = dict(doc.get("security_result") or {})
            security_result["status"] = pipeline_status
            security_result["dlp_decision"] = dlp_map[pipeline_status]
            security_result["requires_human_review"] = False
            security_result["review_completed"] = True
            security_result["reason"] = reason
            security_result["review_outcome"] = updated
            if pipeline_status == "mask_and_allow":
                security_result["masking_applied"] = True
            doc["security_result"] = security_result
            doc["review_outcome"] = updated
            if pipeline_status == "block":
                doc["process_response"] = {
                    "status": "blocked_by_reviewer",
                    "review_id": review_id,
                    "reason": reason,
                }
            elif pipeline_status in {"allow", "mask_and_allow"}:
                doc["process_response"] = updated.get("processing") or doc.get("process_response")
        st.session_state["last_upload_result"] = last_upload


def _fetch_review_item(review_id: str) -> dict[str, Any] | None:
    """Load review from Consumer API, falling back to local queue file.

    When Streamlit runs scans on the host, the local queue may be PENDING while
    the API container still holds a stale REJECTED row — prefer local PENDING.
    """
    encoded = quote(review_id, safe="")
    api_item: dict[str, Any] | None = None
    try:
        api_item = _request("GET", f"/security/reviews/{encoded}", timeout=30)
    except Exception:
        pass
    local_item = _local_review_item(review_id)
    if api_item and local_item:
        if not _is_pending_review(api_item) and _is_pending_review(local_item):
            return local_item
        return api_item
    return api_item or local_item


def _fetch_review_actions(review_id: str) -> list[str]:
    encoded = quote(review_id, safe="")
    raw: list[str] = []
    try:
        payload = _request("GET", f"/security/reviews/{encoded}/actions", timeout=30)
        raw = [str(a).upper() for a in ((payload or {}).get("available_actions") or [])]
    except Exception:
        pass
    item = _fetch_review_item(review_id)
    if item:
        from src.features.security.review.review_decisions import effective_available_actions

        # Always prefer risk-correct actions (fixes stale LOW→Allow-only rows).
        raw = [str(a).upper() for a in effective_available_actions(item)]
    # HUMAN_REVIEW is a DLP state, never a reviewer-selectable action.
    return [a for a in raw if a not in _DISALLOWED_REVIEWER_ACTIONS]


def _list_pending_reviews() -> list[dict[str, Any]]:
    try:
        payload = _request("GET", "/security/reviews", params={"review_status": "PENDING"}, timeout=30)
        return list((payload or {}).get("reviews") or [])
    except Exception:
        try:
            from src.features.security.review.human_review_service import HumanReviewService

            return HumanReviewService().list_pending()
        except Exception:
            return []


def _review_api_bases() -> list[str]:
    """Consumer API first; chunking processor as fallback when DMS lacks review routes."""
    bases = [_api_base_url().rstrip("/")]
    processor = _chunking_processor_url().rstrip("/")
    if processor not in bases:
        bases.append(processor)
    return bases


def _reopen_review_queue_local(review_id: str, *, reason: str) -> None:
    """Reopen on the host queue file (Streamlit scan source of truth)."""
    try:
        from src.features.security.review.human_review_queue import HumanReviewQueue
        from src.shared.networking.document_paths import resolve_security_documents_dir

        queue = HumanReviewQueue(
            queue_file=str(resolve_security_documents_dir() / "human_review_queue.json")
        )
        queue._load_queue()
        for item in queue.queue:
            if item.get("review_id") != review_id:
                continue
            if not _is_pending_review(item):
                queue.reopen_for_review(review_id, reason=reason)
            return
    except Exception:
        pass


def _apply_review_decision_local(
    review_id: str,
    *,
    decision: str,
    reviewer: str,
    comments: str,
) -> dict[str, Any] | None:
    """Apply via host queue — used when API containers lag behind Streamlit scans."""
    from src.features.security.review.human_review_queue import HumanReviewQueue
    from src.features.security.review.human_review_service import HumanReviewService
    from src.shared.networking.document_paths import resolve_security_documents_dir

    queue_file = str(resolve_security_documents_dir() / "human_review_queue.json")
    queue = HumanReviewQueue(queue_file=queue_file)
    queue._load_queue()
    if not any(item.get("review_id") == review_id for item in queue.queue):
        return None
    _reopen_review_queue_local(
        review_id,
        reason="Local apply — prior terminal status cleared.",
    )
    return HumanReviewService(queue=queue).apply_decision(
        review_id,
        decision=decision,
        reviewer=reviewer,
        comments=comments,
    )


def _apply_review_decision(
    review_id: str,
    *,
    decision: str,
    reviewer: str,
    comments: str,
) -> dict[str, Any]:
    """POST /api/v1/security/reviews/{id}/decide — local queue first, then Consumer API."""
    encoded = quote(review_id, safe="")
    rel_path = f"/security/reviews/{encoded}/decide"
    body = {"reviewer": reviewer, "decision": str(decision).upper(), "comments": comments}
    errors: list[str] = []

    # Host queue is authoritative for Streamlit harness uploads.
    local_result = _apply_review_decision_local(
        review_id,
        decision=body["decision"],
        reviewer=reviewer,
        comments=comments,
    )
    if local_result is not None:
        decision_applied = str(local_result.get("reviewer_decision") or body.get("decision") or "").upper()
        queue_status_applied = str(local_result.get("status") or local_result.get("review_status") or "ACTIONED").upper()
        st.session_state["last_review_api_call"] = {
            "method": "POST",
            "url": _review_decide_url(review_id),
            "request_body": body,
            "response_status_code": 200,
            "response_body": f"Decision={decision_applied} → queue={queue_status_applied} (applied locally — no HTTP round-trip needed; host queue is shared with DMS volume)",
            "via": "local_host_queue",
        }
        return local_result

    # Sync API container queue before remote decide.
    if not _is_pending_review(_fetch_review_item(review_id)):
        _reopen_review_on_all_backends(
            review_id,
            reason="Apply reviewer decision — prior terminal status cleared.",
        )

    for base in _review_api_bases():
        api_url = f"{base}{_api_path(rel_path)}"
        try:
            result, logs = _request_with_logs_at(
                base,
                "POST",
                _api_path(rel_path),
                json=body,
                timeout=60,
            )
            st.session_state["last_review_api_call"] = {
                "method": "POST",
                "url": api_url,
                "request_body": body,
                "response_status_code": logs.get("response_status_code"),
                "response_body": logs.get("response_body"),
                "via": "consumer_api" if base == _api_base_url().rstrip("/") else "processor_api",
            }
            return result if isinstance(result, dict) else {}
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            detail = exc.response.text[:300] if exc.response is not None else str(exc)
            try:
                if exc.response is not None:
                    payload = exc.response.json()
                    if isinstance(payload, dict) and payload.get("detail"):
                        detail = str(payload["detail"])
            except Exception:
                pass
            errors.append(f"{api_url} -> HTTP {status}: {detail}")
            st.session_state["last_review_api_call"] = {
                "method": "POST",
                "url": api_url,
                "request_body": body,
                "response_status_code": status,
                "response_body": detail,
                "via": "failed",
            }
            if status == 422 and "already actioned" in str(detail).lower():
                _reopen_review_on_all_backends(
                    review_id,
                    reason="Retry after stale terminal review status on API.",
                )
                try:
                    result, logs = _request_with_logs_at(
                        base,
                        "POST",
                        _api_path(rel_path),
                        json=body,
                        timeout=60,
                    )
                    st.session_state["last_review_api_call"] = {
                        "method": "POST",
                        "url": api_url,
                        "request_body": body,
                        "response_status_code": logs.get("response_status_code"),
                        "response_body": logs.get("response_body"),
                        "via": "consumer_api_retry" if base == _api_base_url().rstrip("/") else "processor_api_retry",
                    }
                    return result if isinstance(result, dict) else {}
                except requests.HTTPError as retry_exc:
                    retry_status = retry_exc.response.status_code if retry_exc.response is not None else None
                    retry_detail = retry_exc.response.text[:300] if retry_exc.response is not None else str(retry_exc)
                    errors.append(f"{api_url} (retry) -> HTTP {retry_status}: {retry_detail}")
                continue
            if status == 422:
                # Fall through to local queue — host file is authoritative for Streamlit scans.
                break
            if status in {404, 405}:
                continue
            raise
        except (requests.ConnectionError, requests.Timeout) as exc:
            errors.append(f"{api_url} -> {exc}")
            continue

    if _review_local_fallback_enabled():
        from src.features.security.review.human_review_service import HumanReviewService

        result = HumanReviewService().apply_decision(
            review_id,
            decision=decision,
            reviewer=reviewer,
            comments=comments,
        )
        st.session_state["last_review_api_call"] = {
            "method": "POST",
            "url": _review_decide_url(review_id),
            "request_body": body,
            "via": "local_fallback",
            "errors": errors,
        }
        return result

    st.session_state["last_review_api_call"] = {
        "method": "POST",
        "url": _review_decide_url(review_id),
        "request_body": body,
        "via": "failed",
        "errors": errors,
    }

    raise RuntimeError(
        "Review API unreachable on all backends:\n- "
        + "\n- ".join(errors)
        + "\n\nRestart DMS (docker compose recreate dms-service) so security mounts load, "
        "or set STREAMLIT_REVIEW_LOCAL_FALLBACK=1."
    )


def _render_last_review_api_call() -> None:
    call = st.session_state.get("last_review_api_call")
    if not call:
        return
    with st.expander("Last review API call (server-side HTTP)", expanded=True):
        st.code(
            f"{call.get('method', 'POST')} {call.get('url')}\n"
            f"Body: {json.dumps(call.get('request_body') or {}, indent=2)}\n"
            f"Status: {call.get('response_status_code')}\n"
            f"Via: {call.get('via')}",
            language="text",
        )
        if call.get("response_body"):
            st.text("Response body")
            st.code(str(call.get("response_body"))[:4000], language="json")
        if call.get("error"):
            st.error(str(call.get("error")))


def _reopen_review_on_all_backends(review_id: str, *, reason: str) -> None:
    """Reset terminal queue rows to PENDING on API and host queue."""
    _reopen_review_queue_local(review_id, reason=reason)
    encoded = quote(review_id, safe="")
    try:
        _request(
            "POST",
            f"/security/reviews/{encoded}/reopen",
            json={},
            timeout=30,
        )
    except Exception:
        pass
    try:
        reopen_fn = getattr(
            __import__(
                "src.features.security.review.human_review_service",
                fromlist=["HumanReviewService"],
            ).HumanReviewService,
            "reopen",
            None,
        )
        if callable(reopen_fn):
            from src.features.security.review.human_review_service import HumanReviewService

            HumanReviewService().reopen(review_id, reason=reason)
    except Exception:
        pass


def _ensure_review_pending(review_id: str, *, reason: str) -> dict[str, Any] | None:
    """Ensure queue is PENDING when live DLP asks for human review."""
    item = _fetch_review_item(review_id)
    if _is_pending_review(item):
        return item
    _reopen_review_on_all_backends(review_id, reason=reason)
    return _fetch_review_item(review_id)


def _reopen_review_if_needed(review_id: str, *, reason: str) -> dict[str, Any] | None:
    """Backward-compatible alias — always syncs API + host queue."""
    return _ensure_review_pending(review_id, reason=reason)


def _render_review_queue_section(
    *,
    review_id: str,
    report: dict[str, Any],
    document_name: str,
    key_prefix: str,
) -> None:
    """Inline reviewer actions — shown on Upload when DLP requires review."""

    st.markdown("---")
    st.subheader("Review Queue")
    st.caption(f"Document `{document_name}` · queue id `{review_id}`")
    st.caption(f"Decide endpoint: `{_review_decide_url(review_id)}`")

    live_needs_review = _live_scan_needs_review(report)
    review_item = _fetch_review_item(review_id)

    # Never show a stale "approved" screen when this upload still needs human review.
    if live_needs_review and review_item and not _is_pending_review(review_item):
        st.warning(
            "A previous reviewer decision was on file, but this upload requires a **new** "
            "explicit choice. The queue will reopen when you apply a decision below."
        )
        _ensure_review_pending(
            review_id,
            reason="Live DLP human_review after upload — prior decision cleared.",
        )
        review_item = _fetch_review_item(review_id)

    from src.features.security.review.review_decisions import effective_risk_level

    risk_source = review_item or report
    risk_level = (
        effective_risk_level(
            {
                "severity": (risk_source or {}).get("severity") or report.get("severity"),
                "risk_level": (risk_source or {}).get("risk_level") or report.get("risk_level"),
                "dlp_decision": (risk_source or {}).get("dlp_decision")
                or report.get("dlp_decision")
                or report.get("status"),
                "available_actions": (risk_source or {}).get("available_actions")
                or report.get("available_reviewer_actions")
                or [],
            }
        )
        if (review_item or report)
        else "—"
    )
    # Prefer live pipeline severity when available; queue severity can be stale
    # for legacy rows that were written before risk normalization fixes.
    severity = report.get("severity") or (review_item or {}).get("severity") or "—"
    dlp_decision = (
        (review_item or report).get("dlp_decision")
        or report.get("dlp_decision")
        or report.get("status")
        or "—"
    )
    review_status = (review_item or {}).get("status") or (
        "PENDING" if live_needs_review else "N/A"
    )

    if str(report.get("status") or "").lower() == "block" and str(dlp_decision).upper() == "BLOCK":
        st.error("DLP hard-blocked this document — reviewer Allow/Mask actions do not apply.")
        st.write(f"**Reason:** {report.get('reason') or 'Blocked by security policy'}")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk", str(risk_level).upper())
    c2.metric("Severity", severity)
    c3.metric("DLP Decision", str(dlp_decision).upper())
    c4.metric("Review Status", review_status)

    queue_status = str((review_item or {}).get("status") or "").upper()
    is_pending = _is_pending_review(review_item) or (
        live_needs_review and not (review_item or {}).get("reviewer_decision")
    )

    _raw_available = _fetch_review_actions(review_id) or [
        str(a).upper() for a in (report.get("available_reviewer_actions") or [])
    ]
    available = [a for a in _raw_available if a not in _DISALLOWED_REVIEWER_ACTIONS]
    if not available:
        available = ["ALLOW", "MASK_AND_ALLOW", "BLOCK"]

    if review_item and not is_pending and not live_needs_review:
        decision = str(review_item.get("reviewer_decision") or "").upper()
        st.info(
            f"Review already completed: **{_decision_label(decision)}** "
            f"by `{review_item.get('reviewer')}` (status `{queue_status}`)."
        )
        history = review_item.get("decision_history") or review_item.get("audit_trail") or []
        if history:
            st.write("**Decision history**")
            st.dataframe(history, use_container_width=True)
        return

    st.write(
        "Choose **one** action in the form, then click **Apply reviewer decision**. "
        "Nothing is approved until you click Apply."
    )
    labeled_options = {_decision_label(a): a for a in available}
    with st.form(key=f"{key_prefix}_review_form", clear_on_submit=False):
        label = st.selectbox("Reviewer action", options=list(labeled_options.keys()), index=0)
        reviewer = st.text_input("Reviewer identity", value="harness-reviewer")
        comments = st.text_area("Comments", value="")
        submitted = st.form_submit_button("Apply reviewer decision", type="primary")

    if not submitted:
        return

    selected_decision = labeled_options[label]
    try:
        updated = _apply_review_decision(
            review_id,
            decision=selected_decision,
            reviewer=(reviewer or "").strip() or "harness-reviewer",
            comments=comments or "",
        )
        st.session_state["review_reviewer"] = (reviewer or "").strip() or "harness-reviewer"
        _sync_session_after_review(review_id, updated)

        processing = dict(updated.get("processing") or {})
        if selected_decision in {"ALLOW", "MASK_AND_ALLOW"} and not processing.get("triggered"):
            processing = _process_document_after_review(review_id, updated) or processing
            updated["processing"] = processing
            _sync_session_after_review(review_id, updated)

        st.session_state["review_flash"] = {
            "decision": selected_decision,
            "document_name": document_name,
            "review_id": review_id,
            "queue_status": updated.get("status") or updated.get("review_status"),
            "processing": processing,
        }
        # Full rerun so pending review banners/forms clear after the decision.
        st.rerun()
    except Exception as exc:
        st.error(f"Review action failed: {exc}")
        _render_last_review_api_call()


def _render_review_flash() -> None:
    flash = st.session_state.pop("review_flash", None)
    if not isinstance(flash, dict):
        return
    decision = str(flash.get("decision") or "").upper()
    name = flash.get("document_name") or "document"
    processing = flash.get("processing") or {}
    if decision == "BLOCK":
        st.error(f"**Blocked** — `{name}` will not be processed or indexed.")
    elif decision == "MASK_AND_ALLOW":
        st.warning(f"**Masked and allowed** — sensitive values redacted for `{name}`.")
        if processing.get("triggered"):
            mode = str(processing.get("mode") or "")
            if mode.startswith("dms"):
                st.success("Queued for processing via DMS (scheduler will dispatch).")
            else:
                st.success("Processing started with masked content.")
        else:
            st.info(
                f"Masking applied. Processing deferred: "
                f"`{processing.get('reason') or 'processor unavailable'}`"
            )
    elif decision == "ALLOW":
        st.success(f"**Allowed** — `{name}` may continue to processing.")
        if processing.get("triggered"):
            mode = str(processing.get("mode") or "")
            if mode.startswith("dms"):
                st.success("Queued for processing via DMS (scheduler will dispatch).")
            else:
                st.success("Processing started.")
        else:
            st.info(
                f"Allowed. Processing deferred: "
                f"`{processing.get('reason') or 'processor unavailable'}`"
            )
    else:
        st.success(
            f"Applied **{_decision_label(decision)}** — status `{flash.get('queue_status')}`"
        )
    if processing:
        st.caption(f"Processing detail: `{processing}`")
    _render_last_review_api_call()


def _process_document_after_review(
    review_id: str,
    updated: dict[str, Any],
) -> dict[str, Any] | None:
    """Resume processing after Allow / Mask — prefer DMS queue, then direct /process."""
    document_id = str(updated.get("document_id") or review_id.removeprefix("rev_"))
    document_name = str(updated.get("document_name") or "")
    masking = str(updated.get("reviewer_decision") or "").upper() == "MASK_AND_ALLOW"

    # Prefer the real pipeline: DMS submit → RabbitMQ → scheduler → processors.
    # Direct POST to localhost:3100 is fragile (processor restarts abort the socket).
    submit_err: str | None = "missing_document_id" if not document_id else None
    if document_id:
        try:
            queued = _submit_document_for_processing(document_id)
            status = str((queued or {}).get("status") or "queued")
            return {
                "triggered": True,
                "continues": True,
                "mode": "dms_submit",
                "status": status,
                "body": queued,
                "reason": f"queued_via_dms:{status}",
                "masking_applied": masking,
            }
        except Exception as submit_exc:
            submit_err = str(submit_exc)

    host_path: Path | None = None
    container_path = ""
    collection_name = "DocumentChunk"
    tenant_id = "default"
    repository_id = "default"
    processing_mode = st.session_state.get("processing_mode") or "Chunking + Embedding Only"

    last_upload = st.session_state.get("last_upload_result")
    if isinstance(last_upload, dict):
        for doc in last_upload.get("documents") or []:
            if str(doc.get("document_id")) != document_id:
                continue
            if doc.get("repository_path"):
                host_path = Path(str(doc["repository_path"]))
            container_path = str(doc.get("container_document_path") or "")
            collection_name = str(doc.get("collection_name") or collection_name)
            tenant_id = str(doc.get("tenant_id") or tenant_id)
            repository_id = str(doc.get("repository_id") or repository_id)
            break

    if host_path is None or not host_path.is_file():
        try:
            from src.shared.networking.document_paths import resolve_security_documents_dir

            docs = resolve_security_documents_dir()
            if document_name and (docs / document_name).is_file():
                host_path = docs / document_name
            else:
                host_path = next(
                    (p for p in docs.iterdir() if p.is_file() and document_id in p.name),
                    None,
                )
        except Exception:
            host_path = None

    if host_path is None or not host_path.is_file():
        return {
            "triggered": False,
            "reason": f"document_file_not_found; dms_submit_failed={submit_err}",
            "continues": False,
            "masking_applied": masking,
        }

    if not container_path:
        try:
            container_path = _build_path_mapping(host_path)["container_document_path"]
        except Exception:
            container_path = str(host_path)

    payload = _build_process_payload(
        document_id=document_id,
        filename=host_path.name,
        container_document_path=container_path,
        collection_name=collection_name,
        tenant_id=tenant_id,
        repository_id=repository_id,
    )
    reference_payload = _build_reference_payload(
        document_id=document_id,
        filename=host_path.name,
        container_document_path=container_path,
        tenant_id=tenant_id,
        repository_id=repository_id,
    )
    result, _ref, _logs, err = _invoke_processors_safely(
        processing_mode=str(processing_mode),
        payload=payload,
        reference_payload=reference_payload,
    )
    status = str((result or {}).get("status") or "")
    triggered = status not in {
        "",
        "processor_deferred",
        "saved_only",
        "human_review",
        "upload_blocked",
        "blocked_by_reviewer",
    }
    reason = err or status
    if submit_err:
        reason = f"{reason}; dms_submit_failed={submit_err}"
    return {
        "triggered": triggered,
        "continues": triggered,
        "mode": "streamlit_processor",
        "status": status,
        "body": result,
        "reason": reason,
        "masking_applied": masking,
    }


def _render_persistent_upload_reviews() -> None:
    """Show review form(s) for documents that need a human decision.

    Driven entirely from the **live queue** — never from cached session state
    flags like review_completed or dlp_decision. This ensures the dropdown
    always appears after upload for MEDIUM/HIGH/CRITICAL documents, and
    disappears once the reviewer acts.
    """
    last_upload = st.session_state.get("last_upload_result")
    if not isinstance(last_upload, dict):
        return

    pending_docs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for doc in last_upload.get("documents") or []:
        doc_id = str(doc.get("document_id") or "")
        if not doc_id:
            continue
        review_id = f"rev_{doc_id}"
        # Always fetch live queue state — do not rely on session security_result.
        review_item = _fetch_review_item(review_id)
        if review_item and not _is_pending_review(review_item):
            # Reviewer already acted — show outcome, no form.
            continue
        # Show form if DLP queued it OR the original scan said human_review.
        security_res = doc.get("security_result") or {}
        originally_needed_review = (
            _live_scan_needs_review(security_res)
            or str(security_res.get("status") or "").lower() == "human_review"
            or str(security_res.get("dlp_decision") or "").upper() == "HUMAN_REVIEW"
            or bool(security_res.get("requires_human_review"))
        )
        has_live_scan = bool(security_res)
        if not originally_needed_review:
            if review_item is None:
                # Never queued — truly clean document.
                continue
            if has_live_scan:
                # Live scan is present and clean for this upload; ignore stale queue rows.
                continue
        if review_item is not None or originally_needed_review:
            pending_docs.append((doc, review_item or {}))

    if not pending_docs:
        return

    st.markdown("---")
    st.subheader("Review decision required")
    st.caption("Choose Allow / Mask and Allow / Block below — processing waits until you click Apply.")
    for doc, _review_item in pending_docs:
        doc_id = str(doc.get("document_id"))
        security_res = doc.get("security_result") or {}
        # Pass the live scan report but force status=human_review so the form renders.
        report = {
            **security_res,
            "status": "human_review",
            "requires_human_review": True,
            "dlp_decision": security_res.get("dlp_decision") or "HUMAN_REVIEW",
            "repository_path": doc.get("repository_path"),
        }
        _render_review_queue_section(
            review_id=f"rev_{doc_id}",
            report=report,
            document_name=str(doc.get("document_name") or "document"),
            key_prefix=f"persistent_upload_review_{doc_id}",
        )


def render_upload_page() -> None:
    st.header("Upload Documents")
    _render_review_flash()
    uploaded_files = st.file_uploader(
        "Select PDF, DOCX, or TXT files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="upload_file_picker",
    )
    if uploaded_files:
        _refresh_upload_file_cache(uploaded_files)
    document_id_override = ""
    if uploaded_files and len(uploaded_files) == 1:
        document_id_override = st.text_input(
            "Document ID",
            value=_stable_document_id(uploaded_files[0].name),
            help="Used for POST /process and later Neo4j reference lookups.",
        )
    elif uploaded_files:
        st.caption("Multiple files use stable document IDs generated from each filename.")

    selected_repo: dict[str, Any] | None = None
    refresh_col, temp_list_col, _ = st.columns([1, 1, 4])
    with refresh_col:
        if st.button("Refresh repositories", key="upload_refresh_repositories"):
            _load_repositories(force_refresh=True)
            _load_repository_upload_options(force_refresh=True)
            _clear_document_type_caches()
            st.rerun()
    with temp_list_col:
        if st.button("View temporary batches", key="upload_view_temporary_documents"):
            try:
                st.session_state["temporary_repository_documents"] = _load_temporary_repository_documents()
            except Exception as exc:
                st.error(f"Unable to load temporary documents: {exc}")

    temporary_docs = st.session_state.get("temporary_repository_documents")
    if isinstance(temporary_docs, dict):
        with st.expander("Temporary upload-batch documents", expanded=True):
            st.caption(
                "Documents are retained for the configured period after successful processing, then deleted with their "
                "batch repository and Weaviate data."
            )
            st.json(temporary_docs)

    repositories = _load_repositories()
    repository_options = _load_repository_upload_options()
    repo_load_error = st.session_state.get("upload_repositories_cache_error") or st.session_state.get(
        "upload_repository_options_cache_error"
    )
    if repo_load_error:
        st.warning(f"Unable to load repositories: {repo_load_error}")

    upload_repo_rows = repository_options or [
        {
            "id": str(repo.get("repository_id") or ""),
            "name": str(repo.get("name") or repo.get("repository_id") or ""),
            "default_document_type": "",
            "default_document_type_id": _repository_default_document_type_id(repo),
        }
        for repo in repositories
        if str(repo.get("status") or "").lower() == "active" and repo.get("repository_id")
    ]

    repo_options = [""] + [str(row.get("id")) for row in upload_repo_rows if row.get("id")]
    repo_labels = {
        str(row.get("id")): (
            f"{row.get('name') or row.get('id')} "
            f"(default type: {row.get('default_document_type') or 'not set'})"
        )
        for row in upload_repo_rows
        if row.get("id")
    }
    if len(repo_options) <= 1:
        st.error(
            "No active repositories are available. Create or activate a repository on the Repository tab, "
            "then click Refresh repositories."
        )
    temporary_upload = st.checkbox(
        "Create a temporary repository for this upload batch",
        value=False,
        key="upload_use_temporary_repository",
        help=(
            "Uploads without a repository receive one new temporary batch repository. IKM runs the normal "
            "processing pipeline and removes that batch after the configured successful-processing retention period."
        ),
    )
    selected_repository_id = st.selectbox(
        "Repository",
        options=repo_options,
        key="upload_repository_id",
        format_func=lambda value: "Manual collection upload" if not value else repo_labels.get(value, value),
        disabled=temporary_upload,
        help="Select an active repository to bind uploaded documents to that repository.",
    )
    if temporary_upload:
        selected_repository_id = ""
        st.info("One API-managed temporary repository will be created for this upload request.")
    if selected_repository_id:
        selected_repo = _repository_record_for_upload(
            selected_repository_id,
            repositories=repositories,
            repository_options=repository_options,
        )
        if selected_repo and str(selected_repo.get("status", "")).lower() != "active":
            st.warning("This repository is not active yet. Activate it before uploading documents.")

    selected_document_type_id: str | None = None

    if selected_repository_id:
        try:
            repository_document_types = _load_document_types_for_repository(selected_repository_id)
        except Exception as exc:
            repository_document_types = []
            st.warning(f"Unable to load document types for repository: {exc}")
    else:
        repository_document_types = []

    default_document_type_id = _repository_default_document_type_id(selected_repo)
    default_document_type_name = _repository_default_document_type_name(selected_repo, repository_document_types)
    if selected_repository_id:
        option_match = next(
            (row for row in upload_repo_rows if str(row.get("id") or "") == selected_repository_id),
            None,
        )
        if option_match and not default_document_type_name:
            default_document_type_name = str(option_match.get("default_document_type") or "").strip() or None
        if default_document_type_id:
            st.caption(
                "Repository default document type: "
                f"`{default_document_type_name or default_document_type_id}`"
            )
        else:
            st.caption("Repository default document type: not configured.")
    elif len(repo_options) > 1:
        st.caption(
            "Choose a repository to enable document type selection (Basic, SOP, etc.). "
            "Add types on the Repository tab if only Basic is listed."
            "Manual collection upload skips repository binding."
        )

    selected_repo_name = str(selected_repo.get("name") or "") if selected_repo else None
    document_type_options = [""] + [
        str(item.get("document_type_id"))
        for item in repository_document_types
        if item.get("document_type_id")
    ]
    document_type_labels = {
        str(item.get("document_type_id")): _document_type_option_label(
            item,
            repository_name=selected_repo_name,
        )
        for item in repository_document_types
        if item.get("document_type_id")
    }
    default_index = 0
    if default_document_type_id and default_document_type_id in document_type_options:
        default_index = document_type_options.index(default_document_type_id)
    selected_document_type_id = st.selectbox(
        "Document Type",
        options=document_type_options,
        index=default_index,
        key="upload_document_type_id",
        format_func=lambda value: (
            "Repository default"
            if not value
            else document_type_labels.get(value, value)
        ),
        disabled=not selected_repository_id,
        help=(
            "Available after you select a repository. "
            "Leave as Repository default to use the repository's configured default type."
        ),
    )

    default_collection = (
        str(selected_repo.get("weaviate_collection") or "")
        if selected_repo
        else os.getenv("WEAVIATE_COLLECTION", "DocumentChunk")
    )
    default_tenant = (
        str(selected_repo.get("default_tenant_id") or "")
        if selected_repo
        else os.getenv("DEFAULT_TENANT_ID", "default")
    )
    collection_name = st.text_input("Collection name", value=default_collection)
    tenant_id = st.text_input("Tenant ID", value=default_tenant)
    submit_for_processing = st.checkbox("Submit for processing immediately", value=True)
    processing_mode = st.selectbox(
        "Processing mode",
        options=[
            "Reference Extraction Only",
            "Chunking + Embedding Only",
            "Run Both",
        ],
        index=1,
        help="Reference extraction runs on the dedicated service (port 3110). Chunking runs on port 3100.",
    )
    processor_joiner = "\\" if os.getenv("DEPLOYMENT_MODE", "").lower() == "local" else "/"
    st.caption(
        f"Host uploads: `{DOCUMENTS_DIR}` -> processor path: `{PROCESSOR_DOCUMENTS_DIR}{processor_joiner}<filename>`. "
        f"Deployment mode: `{os.getenv('DEPLOYMENT_MODE', 'local')}`. "
        f"Chunking: `{_chunking_processor_url()}/process`. "
        f"Reference extraction: `{_reference_processor_url()}/process`."
    )

    if st.button("Refresh service status", use_container_width=False):
        st.session_state.pop("upload_service_health", None)
        st.session_state.pop("upload_service_health_at", None)
    health_checked_at = float(st.session_state.get("upload_service_health_at") or 0.0)
    cached_health = st.session_state.get("upload_service_health")
    health_is_stale = (time.time() - health_checked_at) > 15.0
    # Re-probe automatically when the cache is stale or a service previously
    # looked down, so a transient failure never leaves the button stuck disabled.
    if cached_health is None or health_is_stale or not all(item.get("ok") for item in cached_health):
        st.session_state["upload_service_health"] = _required_upload_services()
        st.session_state["upload_service_health_at"] = time.time()
    upload_service_health = st.session_state["upload_service_health"]
    processors_ok = _processors_ready(upload_service_health)
    uploads_enabled = bool(_coerce_upload_file_list(uploaded_files) or st.session_state.get(_UPLOAD_FILE_CACHE_KEY))

    st.subheader("Backend status")
    service_cols = st.columns(3)
    for col, item in zip(service_cols, upload_service_health):
        with col:
            st.write(f"**{item.get('label')}**")
            st.write(_service_indicator(item))
            st.caption(f"`{item.get('url')}`")
            if not item.get("ok"):
                st.caption(f"_{item.get('state', 'unreachable')}_ — {str(item.get('error') or '')[:120]}")

    if uploaded_files and submit_for_processing and not processors_ok:
        st.info(
            "Chunking/reference processors are not ready yet. You can still **upload and review** — "
            "processing runs after you choose Allow / Mask and Allow, or when backends recover."
        )
    elif uploaded_files and not submit_for_processing:
        st.caption("Security scan and human review work without processor backends.")

    if st.button("Upload", type="primary", disabled=not uploads_enabled):
        try:
            files_for_upload = _upload_files_for_submission(uploaded_files)
            result = _upload_documents_via_api(
                files_for_upload,
                collection_name=collection_name.strip(),
                tenant_id=tenant_id.strip(),
                submit_for_processing=submit_for_processing,
                repository_id=None if temporary_upload else (selected_repository_id or None),
                document_type_id=None if temporary_upload else ((selected_document_type_id or "").strip() or None),
            )
            documents = result.get("documents") or []
            submitted_docs = []
            pending_docs = []
            for doc in documents:
                doc_res = doc.get("process_response") or {}
                security_res = doc.get("security_result") or {}
                if doc_res.get("status") == "human_review" or _live_scan_needs_review(security_res):
                    pending_docs.append(doc)
                else:
                    submitted_docs.append(doc)

            if submitted_docs and not pending_docs:
                st.success(
                    f"Submitted {len(submitted_docs)} document(s) "
                    f"to the current processor backend."
                )
            elif pending_docs and not submitted_docs:
                st.info(
                    f"{len(pending_docs)} document(s) require human review "
                    "(Allow / Mask and Allow / Block) before processing continues."
                )
            elif submitted_docs and pending_docs:
                st.success(f"Submitted {len(submitted_docs)} document(s).")
                st.info(f"{len(pending_docs)} document(s) pending human review.")

            for doc in documents:
                doc_res = doc.get("process_response") or {}
                security_res = doc.get("security_result") or {}
                # Prefer dedicated security report; fall back to process payload for review/block cases.
                report = security_res or doc_res
                doc_id = str(doc.get("document_id") or "")
                scan_status = str(report.get("status") or doc_res.get("status") or "")
                if _live_scan_needs_review(report):
                    scan_status = "human_review"
                # Display live DLP outcome — not stale queue APPROVED mapped to allow.
                display_status = scan_status
                display_reason = report.get("reason") or doc_res.get("reason")
                status, review_item = _effective_security_status(doc_id, scan_status)
                severity = (
                    report.get("severity")
                    or doc_res.get("severity")
                    or (review_item or {}).get("severity")
                    or "unknown"
                )
                if str(severity).lower() in {"", "none", "null", "unknown", "n/a"}:
                    mapped = {
                        "BLOCK": "critical",
                        "HUMAN_REVIEW": "high",
                        "WARN": "medium",
                        "WARNING": "medium",
                        "ALLOW": "low",
                    }
                    sev_from_dlp = mapped.get(str((report.get("dlp_decision") or "").upper()))
                    sev_from_status = mapped.get(str((report.get("status") or "").upper()))
                    severity = sev_from_dlp or sev_from_status or "low"

                st.subheader(f"Security pipeline — {doc.get('document_name')}")
                stages = report.get("pipeline_stages") or [
                    "file_validation",
                    "regex_sensitive_scan",
                    "ner",
                    "document_classification",
                    "topic_classification",
                    "similarity",
                    "llm_moderation",
                    "dlp_policy",
                ]
                st.caption("Stages: " + " → ".join(stages))
                st.write(
                    f"**Decision:** `{display_status}` | **Severity:** `{severity}`"
                )
                risk_level = report.get("risk_level") or (review_item or {}).get("risk_level")
                dlp_decision_display = report.get("dlp_decision") or (review_item or {}).get("dlp_decision")
                if risk_level:
                    st.write(
                        f"**Risk:** `{risk_level}` | **DLP Decision:** `{dlp_decision_display}`"
                    )
                if display_reason and display_reason != report.get("reason"):
                    st.caption(f"Scan reason: {display_reason}")
                elif display_reason:
                    st.write(f"**Reason:** {display_reason}")

                review_id = f"rev_{doc.get('document_id')}" if doc.get("document_id") else None
                live_status = str(report.get("status") or doc_res.get("status") or "").lower()
                dlp_decision = str(report.get("dlp_decision") or "").upper()
                needs_review = _live_scan_needs_review(report) or live_status == "human_review"
                if review_id and needs_review:
                    st.info(
                        f"Human review required — use the **Review decision required** section below "
                        f"for `{doc.get('document_name')}` (queue `{review_id}`)."
                    )
                elif review_item and str(review_item.get("status") or "").upper() != "PENDING":
                    st.caption(
                        f"Reviewer outcome: **{review_item.get('reviewer_decision')}** "
                        f"by `{review_item.get('reviewer')}`"
                    )
                    if review_id:
                        st.caption(f"Review queue id: `{review_id}`")
                elif review_id and status != "allow":
                    st.caption(f"Review queue id: `{review_id}`")

                if status == "block" and review_item and not needs_review:
                    st.error("Upload blocked by reviewer — file will not be processed.")
                    st.write(f"**Reason:** {report.get('reason') or review_item.get('comments') or 'Blocked by reviewer'}")
                    continue
                st.write(f"**Document Type:** {report.get('document_type', 'Unknown')}")
                topics_str = ", ".join(
                    f"{t.get('topic')} ({int(t.get('confidence', 0) * 100)}%)"
                    for t in (report.get("topics") or [])
                )
                if topics_str:
                    st.write(f"**Topics:** {topics_str}")
                categories = report.get("detected_categories") or []
                if categories:
                    st.write("**Detected categories:** " + ", ".join(str(c) for c in categories))
                detections = report.get("detections") or []
                if detections:
                    st.write("**Detections (masked):**")
                    for det in detections:
                        loc_info = f" ({det.get('location')})" if det.get("location") else ""
                        st.write(
                            f"- **Type:** {det.get('type') or det.get('category')}{loc_info} "
                            f"| **Masked:** `{det.get('masked_value')}`"
                        )
                moderation = report.get("moderation_result") or {}
                if moderation:
                    st.write(
                        f"**LLM moderation:** action=`{moderation.get('action')}` "
                        f"flagged=`{moderation.get('flagged')}`"
                    )
                actions = report.get("actions") or []
                if actions:
                    st.write("**Policy actions:** " + ", ".join(str(a) for a in actions))

                if needs_review or live_status == "human_review" or status == "human_review":
                    st.warning(f"Pending human review — {doc.get('document_name')}")
                elif status == "block":
                    st.error(f"Upload blocked by DLP — {doc.get('document_name')}")
                elif status == "mask_and_allow":
                    st.warning(f"Sensitive data masked — {doc.get('document_name')}")
                    st.write("**Action:** Values masked before chunking/embedding/indexing.")
                    masked_text = report.get("masked_text")
                    if not masked_text and doc.get("repository_path"):
                        try:
                            from src.features.security.dlp.sensitive_data_detector import mask_text_content

                            masked_text = mask_text_content(
                                Path(str(doc["repository_path"])).read_text(encoding="utf-8", errors="replace")
                            )
                        except Exception:
                            masked_text = None
                    if masked_text:
                        with st.expander("Masked document preview"):
                            st.text(masked_text)
                elif status in ("allowed_with_warning", "allow_with_warning") or severity == "medium":
                    st.warning(f"Medium-risk sensitive data handled — {doc.get('document_name')}")
                    st.write("**Action:** Allowed with warning.")
                elif status in ("allow", "allowed", "pass"):
                    st.success(f"Security checks passed — {doc.get('document_name')}")
            st.session_state["last_upload_result"] = result
            if documents:
                st.session_state["last_processed_document_id"] = documents[-1].get("document_id")
            deferred = [
                d
                for d in documents
                if str((d.get("process_response") or {}).get("status") or "") in {
                    "processor_deferred",
                    "processor_unavailable",
                    "processor_error",
                }
            ]
            if deferred and pending_docs:
                st.caption(
                    "File saved and queued for review — chunking processor was unavailable; "
                    "choose Allow / Mask and Allow after backends recover."
                )
        except ComplianceBlockError as exc:
            block_reason = exc.details.get("reason") or "Blocked by security policy"
            reviewer_blocked = "blocked by reviewer" in block_reason.lower()
            if reviewer_blocked:
                st.error("Upload blocked — a reviewer previously chose **Block** for this document.")
                st.write(f"**Reason:** {block_reason}")
                st.info(
                    "To re-process this file, re-upload it with a **different Document ID** "
                    "(or rename the file) to start a fresh security review."
                )
            else:
                st.error(f"Upload blocked ({exc.details.get('severity') or 'policy'})")
                st.write(f"**Reason:** {block_reason}")
                st.write(f"**Severity:** {exc.details.get('severity')}")
                st.write(f"**DLP Decision:** `{exc.details.get('dlp_decision') or 'BLOCK'}`")
                if exc.details.get("stage_results"):
                    with st.expander("Security stage debug", expanded=True):
                        st.json(exc.details.get("stage_results"))
                if exc.details.get("decision_trace"):
                    with st.expander("DLP decision trace"):
                        st.json(exc.details.get("decision_trace"))
                st.info(
                    "Content risk files are not auto-blocked — they show a review dropdown. "
                    "This hard block is for structural upload failures (or when "
                    "`enforcement.auto_block_enabled` is true in dlp_policies.yaml)."
                )
            st.session_state["last_compliance_block"] = {
                "reason": block_reason,
                "security": {
                    "status": exc.details.get("status"),
                    "severity": exc.details.get("severity"),
                    "dlp_decision": exc.details.get("dlp_decision"),
                    "risk_level": exc.details.get("risk_level"),
                },
            }
            st.session_state.pop("last_upload_result", None)
            categories = exc.details.get("detected_categories") or []
            if categories:
                st.write("**Detected Categories:**")
                for cat in categories:
                    st.write(f"- {cat}")
            detections = exc.details.get("detections") or []
            if detections:
                st.write("**Detections (sample):**")
                for det in detections[:25]:
                    st.write(
                        f"- **Category:** {det.get('category')} | **Type:** {det.get('type')} "
                        f"| **Location:** {det.get('location')}"
                    )
                    st.write(f"  * **Safe Masked Example:** `{det.get('masked_value')}`")
                if len(detections) > 25:
                    st.caption(f"Showing 25 of {len(detections)} detections.")
            st.write(f"**Recommended Action:** {exc.details.get('recommended_action') or 'Do not process this file.'}")
            # Critical hard-block: do not render Allow/Mask queue UI on this page load.
            return
        except ProcessRequestError as exc:
            if st.session_state.get("last_upload_result"):
                st.warning(f"Processor call failed: {exc}")
                st.subheader("Last Upload Logs")
                st.json(st.session_state["last_upload_result"])
            else:
                st.error(str(exc))
        except BackendUnavailableError as exc:
            st.warning(str(exc))
            st.subheader("Backend diagnostics")
            st.json(exc.diagnostics)
        except ValueError as exc:
            st.error(str(exc))
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            if exc.response is not None and exc.response.status_code == 422:
                st.error(
                    "Document upload API returned 422. "
                    "Check repository, document type, and that a file is attached. "
                    f"Response: {detail}"
                )
            else:
                st.error(f"Upload failed: {detail}")
        except requests.ConnectionError:
            st.error(f"Backend is not running or is unreachable at {_api_base_url()}.")
        except RuntimeError as exc:
            if st.session_state.get("last_upload_result"):
                st.warning(
                    "File saved. Processor backend was temporarily unavailable — "
                    "use the review dropdown below or retry when port 3100 is healthy."
                )
            elif "Backend request failed" in str(exc):
                st.warning(
                    "Processor backend unavailable (port 3100). Upload security scan still works — "
                    "processing runs after you Apply Allow / Mask and Allow."
                )
            else:
                st.error(str(exc))
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

    if st.session_state.get("last_upload_result"):
        st.subheader("Last Upload")
        st.json(st.session_state["last_upload_result"])
        last_docs = st.session_state["last_upload_result"].get("documents") or []
        if last_docs:
            ids = [str(doc.get("document_id")) for doc in last_docs if doc.get("document_id")]
            selected_lookup_id = st.selectbox(
                "Reference lookup document",
                options=ids,
                index=max(len(ids) - 1, 0),
                key="last_upload_reference_doc",
            )
            render_reference_lookup(selected_lookup_id, key_prefix=f"last_upload_refs::{selected_lookup_id}")

    # Show only the current upload's review section for a clean/fresh UI.
    _render_persistent_upload_reviews()


@st.fragment()
def _documents_tab_fragment() -> None:
    with st.expander("Filters", expanded=False):
        st.text_input("Status", key="filter_status")
        st.text_input("Collection name", key="filter_collection_name")
        st.text_input("Tenant ID", key="filter_tenant_id")

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Refresh list", use_container_width=True, key="docs_btn_refresh_list"):
            _rerun_fragment_only()
    with col2:
        refresh_all = st.button("Refresh all from database", use_container_width=True, key="docs_btn_refresh_all")

    try:
        documents = _load_documents()
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"Unable to load documents: {detail}")
        return
    except Exception as exc:
        st.error(f"Unable to load documents: {exc}")
        return

    refreshed: dict[str, dict[str, Any]] = {}
    if refresh_all:
        progress = st.progress(0, text="Refreshing document statuses")
        for index, document in enumerate(documents, start=1):
            document_id = document["document_id"]
            try:
                refreshed[document_id] = _refresh_document_status(document_id)
            except Exception as exc:
                refreshed[document_id] = {"error": str(exc)}
            progress.progress(index / max(len(documents), 1), text=f"Refreshed {index} of {len(documents)}")
        progress.empty()
        st.toast("Document job statuses refreshed from API (MySQL-backed).")
        documents = _load_documents()
        st.session_state["docs_last_manual_refresh"] = refreshed

    if not documents:
        st.info("No uploaded documents found.")
        return

    rows = []
    for document in documents:
        rows.append(
            {
                "document_id": document.get("document_id"),
                "document_name": document.get("document_name"),
                "original_file_name": document.get("original_file_name"),
                "status": document.get("status"),
                "status_group": _status_badge(document.get("status", "")),
                "collection_name": document.get("collection_name"),
                "tenant_id": document.get("tenant_id"),
                "queued_at": document.get("queue_submission_timestamp"),
                "completed_at": document.get("processing_completion_timestamp"),
                "error_details": document.get("error_details"),
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    manual = st.session_state.get("docs_last_manual_refresh")
    if manual:
        with st.expander("Last manual “refresh all” responses", expanded=False):
            st.json(manual)

    st.subheader("Metadata Extraction Test")
    st.caption(
        "Enter custom fields and run the metadata extractor directly against a selected uploaded document."
    )
    metadata_doc_id = st.selectbox(
        "Document for metadata extraction",
        options=[str(doc["document_id"]) for doc in documents if doc.get("document_id")],
        format_func=lambda value: _document_label(
            next((doc for doc in documents if str(doc.get("document_id")) == value), {"document_id": value})
        ),
        key="metadata_test_doc",
    )
    metadata_fields_text = st.text_area(
        "Metadata fields",
        value=st.session_state.get("metadata_fields_text", DEFAULT_METADATA_FIELDS_TEXT),
        key="metadata_fields_text",
        height=140,
        help="One field per line: field_name | Display Label | data_type. You can also paste a JSON array.",
    )
    with st.expander("Parsed metadata field payload", expanded=False):
        try:
            st.json(_parse_metadata_fields(metadata_fields_text))
        except Exception as exc:
            st.warning(str(exc))
    if st.button("Run metadata extraction test", type="primary", key="run_metadata_extraction_test"):
        try:
            selected_doc = next(doc for doc in documents if str(doc.get("document_id")) == metadata_doc_id)
            fields = _parse_metadata_fields(metadata_fields_text)
            with st.spinner("Running metadata extraction on an idle processor..."):
                result = _run_metadata_extraction_test(selected_doc, fields)
            extracted = result.get("extracted_fields") or []
            st.success(f"Metadata extraction completed on {result.get('processor_url')}.")
            if extracted:
                st.dataframe(extracted, use_container_width=True, hide_index=True)
            else:
                st.info("No fields were extracted.")
            st.session_state["last_metadata_extraction_result"] = result
        except requests.HTTPError as exc:
            st.error(f"Metadata extraction failed: {_format_http_error(exc)}")
        except Exception as exc:
            st.error(f"Metadata extraction failed: {exc}")
    if st.session_state.get("last_metadata_extraction_result"):
        with st.expander("Last metadata extraction raw result", expanded=False):
            st.json(st.session_state["last_metadata_extraction_result"])

    st.subheader("Document Chunks")
    completed_docs = [doc for doc in documents if doc.get("document_id") and _is_completed_document(doc)]
    if not completed_docs:
        st.info("No completed documents available for chunk inspection. Failed documents do not have indexed chunks.")
    else:
        chunk_doc_id = st.selectbox(
            "Completed document",
            options=[str(doc["document_id"]) for doc in completed_docs],
            format_func=lambda value: next(
                (
                    _document_label(doc)
                    for doc in completed_docs
                    if str(doc.get("document_id")) == value
                ),
                value,
            ),
            key="documents_chunks_doc",
        )
        render_document_chunks_viewer(chunk_doc_id, key_prefix=f"documents_chunks::{chunk_doc_id}")

    selected = st.selectbox(
        "View document details",
        options=[""] + [doc["document_id"] for doc in documents],
        format_func=lambda value: "Select a document" if not value else _document_label(
            next((doc for doc in documents if str(doc.get("document_id")) == value), {"document_id": value})
        ),
        key="documents_tab_select_doc",
    )
    if selected:
        document = next(doc for doc in documents if doc["document_id"] == selected)
        st.json(document)
        st.subheader("Key fields & validation")
        doc_col1, doc_col2 = st.columns(2)
        with doc_col1:
            if st.button("Load extracted key fields", key=f"doc_load_kf::{selected}"):
                try:
                    st.session_state[f"doc_key_fields::{selected}"] = _request(
                        "GET",
                        f"/documents/{quote(selected, safe='')}/key-fields",
                        timeout=60,
                    )
                except requests.HTTPError as exc:
                    st.error(f"Key fields request failed: {_format_http_error(exc)}")
                except Exception as exc:
                    st.error(f"Key fields request failed: {exc}")
        with doc_col2:
            if st.button("Load validation result", key=f"doc_load_val::{selected}"):
                try:
                    st.session_state[f"doc_validation::{selected}"] = _request(
                        "GET",
                        f"/documents/{quote(selected, safe='')}/validation",
                        timeout=60,
                    )
                except requests.HTTPError as exc:
                    st.error(f"Validation request failed: {_format_http_error(exc)}")
                except Exception as exc:
                    st.error(f"Validation request failed: {exc}")
        if st.session_state.get(f"doc_key_fields::{selected}"):
            with st.expander("Extracted key fields", expanded=True):
                st.json(st.session_state[f"doc_key_fields::{selected}"])
        if st.session_state.get(f"doc_validation::{selected}"):
            with st.expander("Validation result", expanded=True):
                st.json(st.session_state[f"doc_validation::{selected}"])
        if not _is_completed_document(document):
            st.info("This document is not completed, so it is not counted as indexed and has no chunks.")
        c_resubmit, c_delete = st.columns(2)
        with c_resubmit:
            if st.button("Resubmit to processing queue", key=f"doc_resubmit::{selected}", type="primary"):
                try:
                    result = _submit_document_for_processing(selected)
                    st.success(f"Queued again. Status: {result.get('status', '')}")
                    st.session_state[f"doc_action_result::{selected}"] = result
                    _rerun_fragment_only()
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Resubmit failed: {detail}")
                except Exception as exc:
                    st.error(f"Resubmit failed: {exc}")
        with c_delete:
            delete_file = st.checkbox(
                "Also delete stored file (local repo)", value=True, key=f"doc_del_file::{selected}"
            )
            if st.button("Delete from API tracking", key=f"doc_delete::{selected}"):
                try:
                    _delete_document_tracking(selected, delete_file=delete_file)
                    st.success(
                        "Document removed from API metadata" + (" and file deleted." if delete_file else ".")
                    )
                    _rerun_fragment_only()
                except requests.HTTPError as exc:
                    detail = exc.response.text if exc.response is not None else str(exc)
                    st.error(f"Delete failed: {detail}")
                except Exception as exc:
                    st.error(f"Delete failed: {exc}")


def render_documents_page() -> None:
    st.header("Uploaded Documents")
    st.caption(
        "Use **Refresh list** to reload rows from the API, or **Refresh all from database** to re-fetch each document’s "
        "`GET /api/v1/documents/{id}/status` (MySQL `document_job`, fragment-only update). "
        "Use **Resubmit** if a job is stuck in a non-terminal state after infra issues."
    )
    _documents_tab_fragment()


def _repo_collections() -> list[str]:
    data = _request("GET", "/repository/collections")
    return list(data.get("collections") or [])


def _repo_status() -> dict[str, Any]:
    return _request("GET", "/repository/status", timeout=60)


def _repo_config() -> dict[str, Any]:
    return _request("GET", "/repository/config", timeout=60)


def _repo_tenants(collection: str) -> tuple[list[dict[str, Any]], str | None, bool | None]:
    path = f"/repository/collections/{quote(collection, safe='')}/tenants"
    try:
        data = _request("GET", path, timeout=60)
        return list(data.get("tenants") or []), None, data.get("multi_tenancy_enabled")
    except requests.HTTPError as exc:
        return [], _format_http_error(exc), None
    except Exception as exc:
        return [], str(exc), None


def _repo_documents(collection: str, tenant: str | None) -> tuple[list[str], str | None]:
    path = f"/repository/collections/{quote(collection, safe='')}/documents"
    params: dict[str, str] | None = {"tenant": tenant} if tenant else None
    try:
        data = _request("GET", path, params=params, timeout=120)
        return list(data.get("documents") or []), None
    except requests.HTTPError as exc:
        return [], _format_http_error(exc)
    except Exception as exc:
        return [], str(exc)


def _repo_chunks(
    collection: str,
    tenant: str | None,
    document_name: str | None,
    include_text: bool,
) -> dict[str, Any]:
    path = f"/repository/collections/{quote(collection, safe='')}/chunks"
    params: dict[str, Any] = {"limit": 500, "offset": 0, "include_text": str(include_text).lower()}
    if tenant:
        params["tenant"] = tenant
    if document_name:
        params["document_name"] = document_name
    return _request("GET", path, params=params, timeout=120)


def _clear_repo_cached_data(*prefixes: str) -> None:
    for key in list(st.session_state.keys()):
        if any(key.startswith(prefix) for prefix in prefixes):
            del st.session_state[key]


def _chunk_rows(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in chunks:
        props = item.get("properties") or {}
        metadata = item.get("metadata") or {}
        row = {
            "uuid": item.get("uuid"),
            "chunk_id": props.get("chunk_id"),
            "doc_name": props.get("doc_name"),
            "page": props.get("page"),
            "section_name": props.get("section_name"),
            "category": props.get("category"),
            "score": metadata.get("score"),
            "distance": metadata.get("distance"),
        }
        if "text" in props:
            row["text"] = props.get("text")
        rows.append(row)
    return rows


def render_document_chunks_viewer(document_id: str, *, key_prefix: str, default_limit: int = 50) -> None:
    st.subheader("Chunks")
    c_limit, c_offset, c_text, c_vector = st.columns([1, 1, 1, 1])
    with c_limit:
        chunk_limit = st.number_input(
            "Limit",
            min_value=1,
            max_value=5000,
            value=default_limit,
            step=10,
            key=f"{key_prefix}::limit",
        )
    with c_offset:
        chunk_offset = st.number_input(
            "Offset",
            min_value=0,
            value=0,
            step=50,
            key=f"{key_prefix}::offset",
        )
    with c_text:
        include_text = st.checkbox("Include text", value=True, key=f"{key_prefix}::text")
    with c_vector:
        include_vector = st.checkbox("Include vector", value=False, key=f"{key_prefix}::vector")

    payload_key = f"{key_prefix}::payload"
    if st.button("Load chunks", key=f"{key_prefix}::load"):
        try:
            st.session_state[payload_key] = _document_chunks(
                document_id,
                limit=int(chunk_limit),
                offset=int(chunk_offset),
                include_text=include_text,
                include_vector=include_vector,
            )
        except requests.HTTPError as exc:
            st.error(f"Failed to load chunks: {_format_http_error(exc)}")
        except Exception as exc:
            st.error(f"Failed to load chunks: {exc}")

    chunks_payload = st.session_state.get(payload_key)
    if chunks_payload:
        st.caption(
            f"Returned {chunks_payload.get('returned', 0)} chunk(s) from "
            f"{chunks_payload.get('collection_name') or 'document collection'}."
        )
        chunks = chunks_payload.get("chunks") or []
        st.dataframe(_chunk_rows(chunks), use_container_width=True, hide_index=True)
        with st.expander("Raw chunks response", expanded=False):
            st.json(chunks_payload)


def _embedding_model_options() -> list[dict[str, Any]]:
    try:
        catalog = _request("GET", "/repositories/embedding-models", timeout=30)
    except Exception:
        catalog = {}

    models_root = str(catalog.get("local_models_root") or "configured mounted models root")
    options: list[dict[str, Any]] = []
    seen_model_ids: set[str] = set()
    for item in catalog.get("options") or catalog.get("models") or []:
        if not isinstance(item, dict) or not isinstance(item.get("embedding_model"), dict):
            continue
        model = dict(item["embedding_model"])
        model_id = model.get("model_id") or item.get("model_id") or "unknown"
        provider = model.get("provider") or item.get("provider") or "local"
        available = item.get("available")
        seen_model_ids.add(str(model_id))
        options.append(
            {
                "label": f"{model_id} ({provider}, {'installed' if available else 'missing'})",
                "embedding_model": model,
                "available": available,
                "models_root": models_root,
            }
        )
    for model_id in TEST_EMBEDDING_MODELS:
        if model_id in seen_model_ids:
            continue
        options.append(
            {
                "label": f"{model_id} (local, missing)",
                "embedding_model": {
                    "provider": "local",
                    "model_id": model_id,
                    "local_model_dir": model_id,
                },
                "available": False,
                "models_root": models_root,
            }
        )
    return options


def _sample_repository_payload(
    *,
    name: str,
    owner_user_id: str,
    owner_user_name: str,
    default_tenant_id: str,
    embedding_model: dict[str, Any],
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    min_content_words: int | None,
    metadata_extraction: bool,
    citation_retainment: bool,
    template_extraction: bool,
    reference_document_extraction: bool,
    conversion_for_rendering: bool,
    intelligent_extraction: bool,
    key_field_extraction_enabled: bool,
    strict_key_field_page_scope: bool,
    key_fields: list[dict[str, Any]] | None,
    validation_enabled: bool,
    validation_confidence_threshold: float,
    reranking: bool,
    retrieval_search_mode: str,
    lexical_composition: bool,
    document_type_id: str,
) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "embedding_model": embedding_model,
        "chunking_strategy": normalize_chunking_strategy(chunking_strategy),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "indexing_strategy": "weaviate_upsert",
        "metadata_extraction": metadata_extraction,
        "citation_retainment": citation_retainment,
        "template_extraction": template_extraction,
        "reference_document_extraction": reference_document_extraction,
        "conversion_for_rendering": conversion_for_rendering,
        "intelligent_extraction": intelligent_extraction,
        "key_field_extraction_enabled": key_field_extraction_enabled,
        "key_field_extraction": key_field_extraction_enabled,
        "strict_key_field_page_scope": strict_key_field_page_scope,
        "validation_enabled": validation_enabled,
        "validation_confidence_threshold": validation_confidence_threshold,
        "reranking": reranking,
        "retrieval_search_mode": retrieval_search_mode,
        "lexical_composition": lexical_composition,
    }
    if min_content_words:
        settings["chunking_config"] = {"min_content_words": min_content_words}
    if document_type_id.strip():
        settings["document_type_id"] = document_type_id.strip()
    if key_field_extraction_enabled and key_fields:
        settings["key_fields"] = key_fields

    payload: dict[str, Any] = {
        "name": name.strip(),
        "owner_user_id": owner_user_id.strip(),
        "settings": settings,
    }
    if key_field_extraction_enabled:
        payload["key_field_extraction_enabled"] = True
        if key_fields:
            payload["key_fields"] = key_fields
    if owner_user_name.strip():
        payload["owner_user_name"] = owner_user_name.strip()
    if default_tenant_id.strip():
        payload["default_tenant_id"] = default_tenant_id.strip()
    return payload


def render_sample_rag_settings() -> None:
    st.subheader("Sample RAG Settings for Testing")
    st.caption("Create a repository with a known-good RAG configuration similar to the app settings screen.")

    default_name = f"rag-test-{uuid.uuid4().hex[:8]}"
    parsed_key_fields: list[dict[str, Any]] = []
    key_fields_parse_error: str | None = None
    with st.form("sample_rag_settings_form"):
        c_name, c_owner = st.columns(2)
        with c_name:
            repo_name = st.text_input("Repository name", value=default_name)
        with c_owner:
            owner_user_id = st.text_input("Owner user ID", value=os.getenv("DMS_API_USER", "admin"))

        c_owner_name, c_tenant = st.columns(2)
        with c_owner_name:
            owner_user_name = st.text_input("Owner display name", value="System Admin")
        with c_tenant:
            default_tenant_id = st.text_input("Default tenant ID", value="")

        st.markdown("**Embedding model**")
        catalog_options = _embedding_model_options()
        embedding_labels = [option["label"] for option in catalog_options] + ["custom"]
        selected_embedding_label = st.selectbox(
            "Embedding model",
            options=embedding_labels,
            index=0,
            help="Select one mounted model at a time. After testing, remove that model folder and select the next model.",
        )
        if selected_embedding_label == "custom":
            custom_embedding_model = st.text_input("Custom local model directory / model ID", value="")
            embedding_model = {
                "provider": "local",
                "model_id": custom_embedding_model.strip(),
                "local_model_dir": custom_embedding_model.strip(),
            }
        else:
            embedding_model = next(
                option["embedding_model"]
                for option in catalog_options
                if option["label"] == selected_embedding_label
            )
            selected_option = next(option for option in catalog_options if option["label"] == selected_embedding_label)
            if selected_option.get("available") is False:
                st.warning(
                    f"{embedding_model['local_model_dir']} is not installed under the mounted models root "
                    f"({selected_option.get('models_root')}) yet. Mount this model first, test it, then remove it before the next model."
                )

        st.markdown("**Chunking strategy**")
        c_strategy, c_chunk, c_overlap = st.columns(3)
        with c_strategy:
            chunking_strategy = st.selectbox("Strategy", options=CHUNKING_STRATEGY_OPTIONS, index=0)
        with c_chunk:
            chunk_size = st.number_input("Chunk size", min_value=200, max_value=2000, value=512, step=50)
        with c_overlap:
            chunk_overlap = st.number_input("Sentence overlap", min_value=0, value=2, step=1)
        min_content_words = st.number_input(
            "Minimum words",
            min_value=0,
            value=60,
            step=5,
            help="Set 0 to omit this optional chunking_config value.",
        )

        st.markdown("**Processing pipeline**")
        p1, p2 = st.columns(2)
        with p1:
            metadata_extraction = st.checkbox("Metadata extraction", value=True)
            template_extraction = st.checkbox("Template extraction", value=False)
            conversion_for_rendering = st.checkbox("Conversion for rendering", value=False)
        with p2:
            citation_retainment = st.checkbox("Citation retention", value=True)
            reference_document_extraction = st.checkbox("Reference document extraction", value=False)
            intelligent_extraction = st.checkbox("Intelligent extraction", value=False)

        st.markdown("**Key fields & validation**")
        k1, k2 = st.columns(2)
        with k1:
            key_field_extraction_enabled = st.checkbox(
                "Key field extraction",
                value=False,
                help="Enqueue key_field_extraction processor jobs using repository key fields.",
            )
            strict_key_field_page_scope = st.checkbox(
                "Strict key-field page scope",
                value=False,
                help="Limit extraction to first 2 and last page; skip when page metadata is missing.",
                disabled=not key_field_extraction_enabled,
            )
        with k2:
            validation_enabled = st.checkbox(
                "Document validation",
                value=False,
                help="Run document_validation after key field extraction completes.",
                disabled=not key_field_extraction_enabled,
            )
            validation_confidence_threshold = st.slider(
                "Validation confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.85,
                step=0.05,
                disabled=not validation_enabled,
            )

        key_fields_text = st.text_area(
            "Repository key fields",
            value=DEFAULT_KEY_FIELDS_TEXT,
            height=140,
            help=(
                "One field per line: name | type | required | description. "
                "Required when key field extraction is enabled."
            ),
            disabled=not key_field_extraction_enabled,
        )
        parsed_key_fields: list[dict[str, Any]] = []
        key_fields_parse_error: str | None = None
        if key_field_extraction_enabled:
            try:
                parsed_key_fields = _parse_key_fields(key_fields_text)
                with st.expander("Parsed key fields payload", expanded=False):
                    st.json(parsed_key_fields)
            except Exception as exc:
                key_fields_parse_error = str(exc)
                st.warning(f"Key fields parse error: {exc}")

        st.markdown("**Retrieval defaults**")
        r1, r2, r3 = st.columns(3)
        with r1:
            retrieval_search_mode = st.selectbox(
                "Retrieval mode",
                options=["hybrid", "vector", "keyword"],
                index=0,
                help="Hybrid combines dense vector search with lexical/BM25 search.",
            )
        with r2:
            reranking = st.checkbox("Reranking", value=False)
        with r3:
            lexical_composition = st.checkbox("Lexical composition (BM25)", value=True)

        document_type_id = st.text_input(
            "Default document type UUID",
            value="",
            help="Optional. Leave blank to use the platform default document type.",
        )
        activate_after_create = st.checkbox("Activate repository after create", value=False)

        payload = _sample_repository_payload(
            name=repo_name,
            owner_user_id=owner_user_id,
            owner_user_name=owner_user_name,
            default_tenant_id=default_tenant_id,
            embedding_model=embedding_model,
            chunking_strategy=chunking_strategy,
            chunk_size=int(chunk_size),
            chunk_overlap=int(chunk_overlap),
            min_content_words=int(min_content_words) if min_content_words else None,
            metadata_extraction=metadata_extraction,
            citation_retainment=citation_retainment,
            template_extraction=template_extraction,
            reference_document_extraction=reference_document_extraction,
            conversion_for_rendering=conversion_for_rendering,
            intelligent_extraction=intelligent_extraction,
            key_field_extraction_enabled=key_field_extraction_enabled,
            strict_key_field_page_scope=strict_key_field_page_scope,
            key_fields=parsed_key_fields or None,
            validation_enabled=validation_enabled,
            validation_confidence_threshold=float(validation_confidence_threshold),
            reranking=reranking,
            retrieval_search_mode=retrieval_search_mode,
            lexical_composition=lexical_composition,
            document_type_id=document_type_id,
        )

        with st.expander("Generated repository create payload", expanded=False):
            st.json(payload)

        submitted = st.form_submit_button("Create Sample Repository", type="primary")

    if not submitted:
        return
    if not payload["name"] or not payload["owner_user_id"]:
        st.error("Repository name and owner user ID are required.")
        return
    if not str(payload["settings"]["embedding_model"].get("model_id") or "").strip():
        st.error("Embedding model is required. Select a catalog model or enter a custom local model directory.")
        return
    settings = payload.get("settings") or {}
    if settings.get("key_field_extraction_enabled") and not settings.get("key_fields"):
        st.error(
            "Key field extraction is enabled but no key fields were parsed. "
            "Add at least one field or disable key field extraction."
        )
        return
    if key_fields_parse_error:
        st.error(f"Fix key field definitions before creating the repository: {key_fields_parse_error}")
        return
    if settings.get("validation_enabled") and not settings.get("key_field_extraction_enabled"):
        st.error("Document validation requires key field extraction to be enabled.")
        return

    try:
        created = _request("POST", "/repositories", json=payload, timeout=60)
        repo_id = created.get("repository_id") or created.get("id")
        if activate_after_create and repo_id:
            created = _request("POST", f"/repositories/{quote(str(repo_id), safe='')}/activate", timeout=60)
        st.success(f"Repository ready: {repo_id or payload['name']}")
        st.session_state["last_sample_repository"] = created
        _clear_repo_cached_data("repo_docs::", "repo_catalog::")
    except requests.HTTPError as exc:
        st.error(f"Create repository failed: {_format_http_error(exc)}")
    except Exception as exc:
        st.error(f"Create repository failed: {exc}")

    if st.session_state.get("last_sample_repository"):
        st.json(st.session_state["last_sample_repository"])


@st.fragment()
def _repository_tab_fragment() -> None:
    try:
        payload = _request("GET", "/repositories", timeout=60)
        repositories = payload.get("repositories", []) if isinstance(payload, dict) else payload
        repositories = [repo for repo in repositories if isinstance(repo, dict)]
    except requests.HTTPError as exc:
        st.error(f"Repository list failed: {_format_http_error(exc)}")
        return
    except Exception as exc:
        st.error(f"Repository list failed: {exc}")
        return

    try:
        all_documents = _load_documents()
    except Exception:
        all_documents = []
    repo_doc_counts: dict[str, dict[str, int]] = {}
    for document in all_documents:
        repo_id = str(document.get("repository_id") or (document.get("metadata") or {}).get("repository_id") or "")
        if not repo_id:
            continue
        counts = repo_doc_counts.setdefault(repo_id, {"completed": 0, "failed": 0, "tracked": 0})
        counts["tracked"] += 1
        status = str(document.get("status") or "").lower()
        if status == "completed":
            counts["completed"] += 1
        elif status == "failed":
            counts["failed"] += 1

    c0, c1, c2 = st.columns(3)
    with c0:
        st.metric("Repositories", len(repositories))
    with c1:
        active_count = sum(1 for repo in repositories if str(repo.get("status", "")).lower() == "active")
        st.metric("Active", active_count)
    with c2:
        if st.button("Refresh repositories", key="repo_btn_repositories"):
            _clear_repo_cached_data("repo_docs::", "repo_catalog::")
            _clear_document_type_caches()
            _rerun_fragment_only()

    if not repositories:
        st.warning("No repositories found.")
        return

    try:
        repository_document_types = _load_document_types()
    except Exception:
        repository_document_types = []
    document_type_name_by_id = _document_type_name_by_id(repository_document_types)

    st.dataframe(
        [
            {
                "indexed_documents": repo_doc_counts.get(str(repo.get("repository_id")), {}).get("completed", 0),
                "failed_documents": repo_doc_counts.get(str(repo.get("repository_id")), {}).get("failed", 0),
                "tracked_documents": repo_doc_counts.get(str(repo.get("repository_id")), {}).get(
                    "tracked",
                    repo.get("document_count"),
                ),
                "repository_id": repo.get("repository_id"),
                "name": repo.get("name"),
                "status": repo.get("status"),
                "weaviate_collection": repo.get("weaviate_collection"),
                "default_document_type": document_type_name_by_id.get(
                    _repository_default_document_type_id(repo) or "",
                    _repository_default_document_type_id(repo) or "not set",
                ),
                "owner_user_id": repo.get("owner_user_id"),
                "created_at": repo.get("created_at"),
            }
            for repo in repositories
        ],
        use_container_width=True,
        hide_index=True,
    )

    options = [str(repo.get("repository_id")) for repo in repositories if repo.get("repository_id")]
    labels = {
        str(repo.get("repository_id")): f"{repo.get('name') or repo.get('repository_id')} ({repo.get('status')})"
        for repo in repositories
        if repo.get("repository_id")
    }
    repository_id = st.selectbox(
        "Repository",
        options=options,
        format_func=lambda value: labels.get(value, value),
        key="repo_repository_id",
    )
    selected_repo = next(repo for repo in repositories if str(repo.get("repository_id")) == repository_id)

    repo_settings = selected_repo.get("settings") if isinstance(selected_repo.get("settings"), dict) else {}
    try:
        settings_payload = _request("GET", f"/repositories/{quote(repository_id, safe='')}/settings", timeout=60)
        repo_settings = settings_payload.get("settings") if isinstance(settings_payload.get("settings"), dict) else repo_settings
    except Exception:
        pass
    try:
        key_fields_payload = _request("GET", f"/repositories/{quote(repository_id, safe='')}/key-fields", timeout=60)
        repo_key_fields = list(key_fields_payload.get("key_fields") or [])
    except Exception:
        repo_key_fields = list(repo_settings.get("key_fields") or [])

    st.subheader("Processing pipeline settings")
    st.caption("Toggle key field extraction and document validation for this repository.")
    with st.form(f"repo_pipeline_settings::{repository_id}"):
        current_kfe = bool(
            repo_settings.get("key_field_extraction_enabled")
            if repo_settings.get("key_field_extraction_enabled") is not None
            else repo_settings.get("key_field_extraction")
        )
        current_validation = bool(repo_settings.get("validation_enabled"))
        c_pipe1, c_pipe2 = st.columns(2)
        with c_pipe1:
            edit_kfe = st.checkbox("Key field extraction", value=current_kfe)
            edit_strict_scope = st.checkbox(
                "Strict key-field page scope",
                value=bool(repo_settings.get("strict_key_field_page_scope")),
                disabled=not edit_kfe,
            )
        with c_pipe2:
            edit_validation = st.checkbox(
                "Document validation",
                value=current_validation,
                disabled=not edit_kfe,
            )
            edit_validation_threshold = st.slider(
                "Validation confidence threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(repo_settings.get("validation_confidence_threshold") or 0.85),
                step=0.05,
                disabled=not edit_validation,
            )
        edit_key_fields_text = st.text_area(
            "Repository key fields",
            value=_format_key_fields_text(repo_key_fields),
            height=140,
            help="One field per line: name | type | required | description.",
            disabled=not edit_kfe,
        )
        save_pipeline = st.form_submit_button("Save pipeline settings", type="primary")

    if save_pipeline:
        try:
            parsed_fields = _parse_key_fields(edit_key_fields_text) if edit_kfe else []
            if edit_kfe and not parsed_fields:
                st.error("Key field extraction requires at least one key field.")
            elif edit_validation and not edit_kfe:
                st.error("Document validation requires key field extraction.")
            else:
                patch_body: dict[str, Any] = {
                    "key_field_extraction_enabled": edit_kfe,
                    "key_field_extraction": edit_kfe,
                    "strict_key_field_page_scope": edit_strict_scope,
                    "validation_enabled": edit_validation,
                    "validation_confidence_threshold": float(edit_validation_threshold),
                }
                _request(
                    "PATCH",
                    f"/repositories/{quote(repository_id, safe='')}/settings",
                    json=patch_body,
                    timeout=60,
                )
                if edit_kfe:
                    _request(
                        "PATCH",
                        f"/repositories/{quote(repository_id, safe='')}/key-fields",
                        json={"key_fields": parsed_fields},
                        timeout=60,
                    )
                st.success("Repository pipeline settings saved.")
                _clear_repo_cached_data("repo_docs::", "repo_catalog::")
                _rerun_fragment_only()
        except requests.HTTPError as exc:
            st.error(f"Save pipeline settings failed: {_format_http_error(exc)}")
        except Exception as exc:
            st.error(f"Save pipeline settings failed: {exc}")

    _render_repository_document_types_section(
        repository_id,
        repository_name=str(selected_repo.get("name") or repository_id),
    )

    with st.expander("Raw repository record", expanded=False):
        st.json(selected_repo)

    if str(selected_repo.get("status", "")).lower() != "active":
        if st.button("Activate repository", type="primary", key=f"repo_activate::{repository_id}"):
            try:
                st.success(_request("POST", f"/repositories/{quote(repository_id, safe='')}/activate", timeout=60))
                _clear_repo_cached_data("repo_docs::")
                _rerun_fragment_only()
            except requests.HTTPError as exc:
                st.error(f"Activate failed: {_format_http_error(exc)}")
            except Exception as exc:
                st.error(f"Activate failed: {exc}")

    dkey = f"repo_docs::{repository_id}"
    if dkey not in st.session_state:
        try:
            docs_payload = _request("GET", f"/repositories/{quote(repository_id, safe='')}/documents", timeout=60)
            docs = docs_payload.get("documents", []) if isinstance(docs_payload, dict) else docs_payload
            st.session_state[dkey] = [doc for doc in docs if isinstance(doc, dict)]
        except requests.HTTPError as exc:
            st.error(f"Failed to load repository documents: {_format_http_error(exc)}")
            return
        except Exception as exc:
            st.error(f"Failed to load repository documents: {exc}")
            return
    docs = st.session_state[dkey]

    st.subheader("Documents")
    if not docs:
        st.info("No documents linked to this repository.")
    else:
        completed_count = sum(1 for doc in docs if _is_completed_document(doc))
        failed_count = sum(1 for doc in docs if str(doc.get("status", "")).lower() == "failed")
        c_done, c_failed, c_tracked = st.columns(3)
        c_done.metric("Indexed / completed", completed_count)
        c_failed.metric("Failed", failed_count)
        c_tracked.metric("Tracked total", len(docs))
        st.caption("Failed documents are tracked for audit/debugging but are not counted as indexed chunks.")
        st.dataframe(docs, use_container_width=True, hide_index=True)

    st.divider()
    with st.expander("Repository catalogs", expanded=False):
        try:
            st.subheader("Embedding models")
            st.json(_request("GET", "/repositories/embedding-models", timeout=60))
            st.subheader("Chunking strategies")
            st.json(_request("GET", "/repositories/chunking-strategies", timeout=60))
        except Exception as exc:
            st.error(f"Failed to load repository catalogs: {exc}")


def render_repository_page() -> None:
    st.header("Repositories")
    st.caption("Browse DMS repositories, linked documents, and supported model/chunking catalogs.")
    with st.expander("Create sample RAG settings", expanded=True):
        render_sample_rag_settings()
    st.divider()
    _repository_tab_fragment()


def _queue_list(vhost: str | None) -> list[dict[str, Any]]:
    params: dict[str, str] | None = {"vhost": vhost} if vhost else None
    try:
        data = _request("GET", "/queues", params=params, timeout=60)
    except requests.HTTPError as exc:
        if exc.response is None or exc.response.status_code != 404:
            raise
        data = _rabbitmq_management_request(
            "GET",
            f"/queues/{_rabbitmq_vhost(vhost)}" if vhost else "/queues",
            timeout=60,
        )
    if isinstance(data, dict):
        return list(data.get("queues") or [])
    return list(data or [])


def _rabbitmq_vhost(vhost: str | None) -> str:
    if vhost in (None, "", "default", "/"):
        return quote("/", safe="")
    return quote(vhost, safe="")


def _rabbitmq_management_request(method: str, path: str, **kwargs: Any) -> Any:
    base_url = st.session_state.get("rabbitmq_management_url", DEFAULT_RABBITMQ_MANAGEMENT_URL).rstrip("/")
    user = st.session_state.get("rabbitmq_user", DEFAULT_RABBITMQ_USER)
    password = st.session_state.get("rabbitmq_pass", DEFAULT_RABBITMQ_PASS)
    response = requests.request(
        method,
        f"{base_url}/api{path}",
        auth=(user, password),
        timeout=kwargs.pop("timeout", 60),
        **kwargs,
    )
    response.raise_for_status()
    return response.json() if (response.text or "").strip() else None


def _rabbitmq_peek_messages(vhost: str, queue_name: str, count: int) -> dict[str, Any]:
    messages = _rabbitmq_management_request(
        "POST",
        f"/queues/{_rabbitmq_vhost(vhost)}/{quote(queue_name, safe='')}/get",
        json={"count": count, "ackmode": "ack_requeue_true", "encoding": "auto"},
        timeout=60,
    )
    return {
        "vhost": "/" if vhost in ("default", "/", "") else vhost,
        "queue": queue_name,
        "count_returned": len(messages or []),
        "messages": messages or [],
        "note": "Messages were requeued (ack_requeue_true).",
    }


def _rabbitmq_delete_messages(vhost: str, queue_name: str, count: int) -> dict[str, Any]:
    removed = 0
    last_removed: Any = None
    for _ in range(count):
        batch = _rabbitmq_management_request(
            "POST",
            f"/queues/{_rabbitmq_vhost(vhost)}/{quote(queue_name, safe='')}/get",
            json={"count": 1, "ackmode": "ack_requeue_false", "encoding": "auto"},
            timeout=60,
        )
        if not batch:
            break
        removed += 1
        last_removed = batch[-1] if isinstance(batch, list) else batch
    return {
        "vhost": "/" if vhost in ("default", "/", "") else vhost,
        "queue": queue_name,
        "removed": removed,
        "last_removed": last_removed,
    }


def _rabbitmq_publish_test_message(vhost: str, queue_name: str, body: dict[str, Any]) -> dict[str, Any]:
    payload = _rabbitmq_management_request(
        "POST",
        f"/exchanges/{_rabbitmq_vhost(vhost)}/amq.default/publish",
        json={
            "properties": {},
            "routing_key": queue_name,
            "payload": json.dumps(body),
            "payload_encoding": "string",
        },
        timeout=60,
    )
    return {
        "published": bool((payload or {}).get("routed")),
        "queue": queue_name,
        "vhost": "/" if vhost in ("default", "/", "") else vhost,
        "payload": body,
        "rabbitmq_response": payload,
    }


def _scheduler_tcp_request(signal_type: str) -> dict[str, Any]:
    host = st.session_state.get("scheduler_host", DEFAULT_SCHEDULER_HOST)
    port = int(st.session_state.get("scheduler_port", DEFAULT_SCHEDULER_PORT))
    message = json.dumps({"signal_type": signal_type}).encode("utf-8")
    with socket.create_connection((host, port), timeout=10) as sock:
        sock.sendall(struct.pack(">I", len(message)) + message)
        header = sock.recv(4)
        if len(header) != 4:
            raise RuntimeError("No response length received from scheduler.")
        expected = struct.unpack(">I", header)[0]
        chunks: list[bytes] = []
        remaining = expected
        while remaining > 0:
            chunk = sock.recv(remaining)
            if not chunk:
                raise RuntimeError("Scheduler response ended early.")
            chunks.append(chunk)
            remaining -= len(chunk)
    return json.loads(b"".join(chunks).decode("utf-8"))


@st.fragment()
def _queue_tab_fragment() -> None:
    all_vhosts = st.checkbox("List queues from all vhosts", value=True, key="queue_all_vhosts")
    vhost_input = st.text_input("Vhost (when not listing all)", value="default", key="queue_vhost_input")

    if st.button("Refresh queue list", key="queue_refresh_list"):
        st.session_state.pop("queue_list_data", None)

    if st.session_state.get("queue_list_data") is None:
        try:
            st.session_state["queue_list_data"] = _queue_list(None if all_vhosts else vhost_input.strip() or "default")
        except Exception as exc:
            st.error(f"Failed to list queues: {exc}")
            return

    queues = st.session_state["queue_list_data"]
    if not queues:
        st.info("No queues returned.")
        return

    labels = [f"{q.get('vhost','')}/{q.get('name','')}" for q in queues]
    pick = st.selectbox("Queue", options=list(range(len(labels))), format_func=lambda i: labels[i], key="queue_pick")
    q = queues[pick]
    qname = str(q.get("name") or "")
    vhost = str(q.get("vhost") or "/")

    st.write(f"Selected: **{vhost}** / **{qname}**")

    if st.button("Peek messages (requeue)", key="queue_peek"):
        try:
            v_enc = "default" if vhost in ("/", "") else vhost
            path = f"/queues/{quote(v_enc, safe='')}/{quote(qname, safe='')}/messages"
            try:
                st.session_state["queue_last_peek"] = _request("GET", path, params={"count": 50}, timeout=60)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                st.session_state["queue_last_peek"] = _rabbitmq_peek_messages(vhost, qname, 50)
        except Exception as exc:
            st.session_state["queue_last_peek"] = {"error": str(exc)}
    if "queue_last_peek" in st.session_state:
        st.json(st.session_state["queue_last_peek"])

    rm = st.number_input("Remove N messages from head", min_value=1, max_value=100, value=1, key="queue_rm")
    if st.button("Delete from head", type="primary", key="queue_del"):
        try:
            v_enc = "default" if vhost in ("/", "") else vhost
            path = f"/queues/{quote(v_enc, safe='')}/{quote(qname, safe='')}/messages"
            try:
                out = _request("DELETE", path, params={"count": int(rm)}, timeout=60)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                out = _rabbitmq_delete_messages(vhost, qname, int(rm))
            st.success(out or "Removed.")
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Test document message")
    if "queue_test_id" not in st.session_state:
        st.session_state["queue_test_id"] = str(uuid.uuid4())
    tid = st.text_input("document_id", value=st.session_state["queue_test_id"], key="queue_tid")
    tname = st.text_input("document_name", value=f"{tid}.pdf", key="queue_tname")
    tcol = st.text_input("collection_name", value=os.getenv("WEAVIATE_COLLECTION", "DocumentChunk"), key="queue_tcol")
    ttenant = st.text_input("tenant_id (optional)", value="", key="queue_ttenant")
    if st.button("Post test message", key="queue_test"):
        body = {
            "document_id": tid.strip(),
            "document_name": tname.strip(),
            "collection_name": tcol.strip(),
            "tenant_id": ttenant.strip() or None,
        }
        try:
            v_enc = "default" if vhost in ("/", "") else vhost
            path = f"/queues/{quote(v_enc, safe='')}/{quote(qname, safe='')}/test-document-message"
            try:
                out = _request("POST", path, json=body, timeout=60)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                out = _rabbitmq_publish_test_message(vhost, qname, body)
            st.success(out or "Published.")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 403:
                st.error("Blocked in production (RAG_API_ENV / APP_ENV).")
            else:
                st.error(exc.response.text if exc.response is not None else str(exc))
        except Exception as exc:
            st.error(str(exc))


def render_queue_page() -> None:
    st.header("Queue (RabbitMQ)")
    st.caption("Management API. Use vhost `default` for `/`. Peek uses ack_requeue_true (messages stay on the queue).")
    _queue_tab_fragment()


@st.fragment()
def _scheduler_tab_fragment() -> None:
    if st.button("Refresh", key="sched_btn"):
        st.session_state.pop("sched_health", None)
        st.session_state.pop("sched_pool", None)

    try:
        if "sched_health" not in st.session_state:
            try:
                st.session_state["sched_health"] = _request("GET", "/scheduler/health", timeout=30)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                st.session_state["sched_health"] = _scheduler_tcp_request("healthcheck")
        health = st.session_state["sched_health"]
    except Exception as exc:
        st.error(f"Scheduler health failed: {exc}")
        health = {}

    status = (health or {}).get("status", "unknown")
    ok = str(status).lower() == "healthy"
    st.markdown(f"**Scheduler TCP:** `{status}` {'🟢' if ok else '🔴'}")

    try:
        if "sched_pool" not in st.session_state:
            try:
                st.session_state["sched_pool"] = _request("GET", "/scheduler/processors", timeout=30)
            except requests.HTTPError as exc:
                if exc.response is None or exc.response.status_code != 404:
                    raise
                st.session_state["sched_pool"] = _scheduler_tcp_request("processor_pool")
        pool = st.session_state["sched_pool"]
    except Exception as exc:
        st.warning(f"Processor pool: {exc}")
        pool = {}

    processors = list((pool or {}).get("processors") or [])
    if not processors:
        st.info("No processor slots in response (is the scheduler running?).")
        return

    rows = []
    for p in processors:
        overall = bool(p.get("overall_healthy"))
        http_ok = bool(p.get("http_healthy"))
        docker = p.get("docker_status")
        ps = str(p.get("processor_status") or "")
        idle = ps.lower() in {"idle", "completed", "failed", "stopped", ""} or not p.get("active_document_name")
        doc = p.get("active_document_name") or "—"
        if not idle and p.get("active_document_id"):
            doc = f"{doc} ({p.get('active_document_id')})"
        rows.append(
            {
                "slot": p.get("slot_id"),
                "container": p.get("container_name"),
                "port": p.get("port"),
                "docker": docker,
                "http": "🟢" if http_ok else "🔴",
                "overall": "🟢" if overall else "🔴",
                "processor": ps,
                "mode": "idle" if idle else "processing",
                "document": doc,
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def render_security_review_page() -> None:
    """Decision-based security review test harness."""
    import tempfile
    import uuid

    from src.features.security.review.review_decisions import decision_model_summary
    from src.features.security.application.upload_security_pipeline import reset_security_pipeline, run_security_pipeline

    st.session_state["_rendered_review_sections"] = set()
    st.header("Security Review — Decision Test Harness")
    st.caption("Run canned DLP scenarios and apply risk-based reviewer decisions.")

    model = decision_model_summary()
    with st.expander("Decision model reference", expanded=False):
        st.json(model)

    harness_doc_id = str(st.session_state.get("harness_scan_doc_id") or "").strip()
    harness_review_id = f"rev_{harness_doc_id}" if harness_doc_id else ""

    st.subheader("Open existing review")
    lookup_id = st.text_input("Review queue id", placeholder="rev_test-case-2", key="harness_lookup_review_id")
    if lookup_id.strip() and lookup_id.strip() != harness_review_id:
        item = _fetch_review_item(lookup_id.strip())
        if item:
            _render_review_queue_section(
                review_id=lookup_id.strip(),
                report={
                    "risk_level": item.get("risk_level"),
                    "severity": item.get("severity"),
                    "dlp_decision": item.get("dlp_decision"),
                    "status": "human_review",
                    "available_reviewer_actions": item.get("available_actions"),
                },
                document_name=str(item.get("document_name") or "document"),
                key_prefix=f"harness_lookup_{lookup_id.strip()}",
            )
        else:
            st.warning(f"No review found for `{lookup_id.strip()}` (API and local queue checked).")
    elif lookup_id.strip() and lookup_id.strip() == harness_review_id:
        st.caption("Same review as the pipeline run below — one form is shown there.")

    st.divider()
    st.subheader("Run canned test case")
    test_cases = {
        "test-case-1-allow": {
            "title": "LOW risk — DLP ALLOW",
            "text": (
                "Standard Operating Procedure for Cleanroom Operations.\n"
                "Purpose: define entry guidelines for the cleanroom facility.\n"
                "Scope: applies to all staff entering Area A.\n"
                "Procedure: personnel must wear garments and perform hand sanitization."
            ),
        },
        "test-case-2-mask-and-allow": {
            "title": "MEDIUM risk — DLP MASK_AND_ALLOW",
            "text": (
                "For manufacturing updates contact employee ID EMP-12345 at "
                "employee.email@example.com or phone number +1-555-019-0199."
            ),
        },
        "test-case-3-human-review": {
            "title": "MEDIUM risk — DLP HUMAN_REVIEW",
            "text": (
                f"NONCE-{uuid.uuid4().hex[:8]} CONFIDENTIAL - INTERNAL USE ONLY\n"
                "Standard cleaning checklist for equipment room Area B."
            ),
        },
        "test-case-4-block": {
            "title": "CRITICAL risk — DLP BLOCK",
            "text": (
                f"NONCE-{uuid.uuid4().hex[:8]} Employee record: "
                "PAN ABCDE1234F and Aadhaar 9876 5432 1098 on file."
            ),
        },
    }

    selected = st.selectbox(
        "Security test case",
        options=list(test_cases.keys()),
        format_func=lambda k: f"{k} — {test_cases[k]['title']}",
    )
    doc_id = st.text_input("Document ID (optional)", value=str(uuid.uuid4()))
    reviewer = st.text_input("Reviewer identity", value="harness-reviewer")

    if st.button("Run security pipeline", type="primary"):
        reset_security_pipeline()
        case = test_cases[selected]
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
            tmp.write(case["text"])
            tmp_path = tmp.name
        result = run_security_pipeline(Path(tmp_path), document_id=doc_id.strip() or None)
        st.session_state["harness_scan_result"] = result
        st.session_state["harness_scan_doc_id"] = doc_id.strip() or result.get("document_id")
        st.session_state["harness_scan_path"] = tmp_path
        st.session_state["harness_test_case"] = selected

    result = st.session_state.get("harness_scan_result")
    if not result:
        pending = _list_pending_reviews()
        if pending:
            st.subheader("Pending reviews")
            st.dataframe(
                [
                    {
                        "review_id": r.get("review_id"),
                        "document": r.get("document_name"),
                        "risk": r.get("risk_level"),
                        "dlp_decision": r.get("dlp_decision"),
                        "review_status": r.get("review_status"),
                    }
                    for r in pending
                ],
                use_container_width=True,
            )
        st.info("Select a test case and run the pipeline, or enter a review id above.")
        return

    review_id = f"rev_{st.session_state.get('harness_scan_doc_id') or ''}"
    _render_review_queue_section(
        review_id=review_id,
        report={
            **result,
            "repository_path": st.session_state.get("harness_scan_path"),
        },
        document_name=str(st.session_state.get("harness_test_case") or "harness-doc"),
        key_prefix=f"harness_run_{review_id}",
    )

    with st.expander("Full pipeline JSON"):
        st.json(result)

    _render_last_review_api_call()

    st.divider()
    pending = _list_pending_reviews()
    if pending:
        st.subheader("All pending reviews")
        st.dataframe(
            [
                {
                    "review_id": r.get("review_id"),
                    "document": r.get("document_name"),
                    "risk": r.get("risk_level"),
                    "dlp_decision": r.get("dlp_decision"),
                    "review_status": r.get("review_status"),
                    "severity": r.get("severity"),
                }
                for r in pending
            ],
            use_container_width=True,
        )
    else:
        st.caption("No pending reviews.")


def render_scheduler_page() -> None:
    st.header("Scheduler")
    _scheduler_tab_fragment()


def render_sidebar() -> None:
    st.sidebar.title(APP_TITLE)
    st.sidebar.caption("Test harness for the RAG API service.")
    if st.session_state.get("api_base_url") == OLD_DEFAULT_API_BASE_URL:
        st.session_state["api_base_url"] = DEFAULT_API_BASE_URL
    st.sidebar.text_input("API base URL", value=DEFAULT_API_BASE_URL, key="api_base_url")
    with st.sidebar.expander("API authentication", expanded=True):
        st.caption("Sign in to call protected Repository and Temporary Repository endpoints.")
        st.text_input("JWT access token", type="password", key="api_access_token_input")
        login_user = st.text_input("User ID", value=os.getenv("PLATFORM_BOOTSTRAP_ADMIN_USER", "admin"), key="api_login_user")
        login_password = st.text_input("Password", type="password", key="api_login_password")
        if st.button("Sign in", key="api_login_button", use_container_width=True):
            try:
                url = f"{_api_base_url()}{_api_path('/auth/token')}"
                response = requests.post(
                    url,
                    json={"user_id": login_user.strip(), "password": login_password},
                    timeout=30,
                )
                response.raise_for_status()
                payload = response.json()
                token = str(payload.get("access_token") or "").strip()
                if not token:
                    raise RuntimeError("Authentication response did not contain an access token.")
                st.session_state["api_access_token"] = token
                for cache_key in ("upload_repositories_cache", "upload_repository_options_cache"):
                    st.session_state.pop(cache_key, None)
                    st.session_state.pop(f"{cache_key}_error", None)
                st.success("Signed in. Protected API requests now use your JWT.")
            except Exception as exc:
                st.error(f"Sign-in failed: {exc}")
    st.sidebar.text_input("Chunking processor URL", value=DEFAULT_CHUNKING_PROCESSOR_URL, key="chunking_processor_url")
    st.sidebar.text_input(
        "Reference processor URL",
        value=DEFAULT_REFERENCE_EXTRACTION_URL,
        key="reference_extraction_url",
    )
    st.sidebar.text_input("Scheduler host", value=DEFAULT_SCHEDULER_HOST, key="scheduler_host")
    st.sidebar.number_input("Scheduler TCP port", value=DEFAULT_SCHEDULER_PORT, step=1, key="scheduler_port")
    if st.sidebar.button("Detect backends", use_container_width=True):
        _discover_backends()
    discovery = st.session_state.get("backend_discovery")
    if discovery is None:
        discovery = _discover_backends()
    with st.sidebar.expander("Detected services", expanded=True):
        for item in discovery:
            marker = "OK" if item.get("ok") else "FAIL"
            st.write(f"{marker} {item.get('label')} - `{item.get('url')}`")
            if not item.get("ok") and item.get("error"):
                st.caption(str(item.get("error")))
    if st.sidebar.button("Check API Health", use_container_width=True):
        try:
            st.sidebar.json(_request("GET", "/health"))
        except requests.ConnectionError:
            st.sidebar.error(f"Backend is not running or is unreachable at {_api_base_url()}.")
        except Exception as exc:
            st.sidebar.error(str(exc))


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.info(
        "For today’s **15-feature release demo**, prefer "
        "`streamlit run tools/simulators/streamlit_release_demo.py` "
        "(feature sidebar with one clickable test panel per feature)."
    )
    render_sidebar()

    tabs = st.tabs(["Upload", "Documents", "Repository", "Queue", "Security Review", "Scheduler", "Chatbot"])
    with tabs[0]:
        render_upload_page()
    with tabs[1]:
        render_documents_page()
    with tabs[2]:
        render_repository_page()
    with tabs[3]:
        render_queue_page()
    with tabs[4]:
        render_security_review_page()
    with tabs[5]:
        render_scheduler_page()
    with tabs[6]:
        render_chatbot_page()


def _citation_label(citation: dict[str, Any]) -> str:
    """Build a compact citation chip: document | section | page."""
    document = (
        str(citation.get("document_name") or citation.get("document") or "").strip()
        or "Unknown document"
    )
    section = str(
        citation.get("section")
        or citation.get("section_name")
        or citation.get("section_path")
        or ""
    ).strip()
    page = citation.get("page")
    page_end = citation.get("page_end")
    if page is None:
        page_label = "page ?"
    elif page_end is not None and int(page_end) != int(page):
        page_label = f"page {page}-{page_end}"
    else:
        page_label = f"page {page}"

    parts = [document]
    if section:
        parts.append(section)
    parts.append(page_label)
    index = citation.get("index")
    prefix = f"[{index}] " if index is not None else ""
    return prefix + " | ".join(parts)


def _render_citation_popups(citations: list[dict[str, Any]]) -> None:
    """Render clickable citation chips; popup body shows the source snippet."""
    if not citations:
        return
    st.write("### Citations")
    st.caption("Click a citation to open the source snippet.")
    # Keep chips in a wrapping row of columns for denser layout.
    columns = st.columns(min(3, max(1, len(citations))))
    for offset, citation in enumerate(citations):
        col = columns[offset % len(columns)]
        label = _citation_label(citation)
        snippet = str(citation.get("snippet") or "").strip()
        line_start = citation.get("line_start")
        line_end = citation.get("line_end")
        with col:
            if hasattr(st, "popover"):
                with st.popover(label, use_container_width=True):
                    st.markdown(f"**{label}**")
                    if line_start is not None or line_end is not None:
                        st.caption(f"Lines: {line_start or '?'}–{line_end or '?'}")
                    if snippet:
                        st.write(snippet)
                    else:
                        st.info("No snippet was returned for this citation.")
                    source_blocks = citation.get("source_blocks") or []
                    if source_blocks:
                        with st.expander("Source blocks"):
                            for block in source_blocks[:8]:
                                preview = str(block.get("text_preview") or "").strip()
                                if preview:
                                    st.markdown(
                                        f"- p.{block.get('page')} "
                                        f"L{block.get('line_start') or block.get('line_number')}:"
                                        f" {preview}"
                                    )
            else:
                with st.expander(label):
                    if snippet:
                        st.write(snippet)
                    else:
                        st.info("No snippet was returned for this citation.")


def render_chatbot_page() -> None:
    st.header("RAG Chatbot with AI Safety Guards")
    st.write("Ask a question about the indexed documents in a repository. Prompt and Output Guards will protect the conversation.")

    try:
        repos_payload = _request("GET", "/repositories", timeout=30)
        repos = repos_payload.get("repositories") or []
        repo_options = {r["repository_id"]: f"{r.get('name') or r['repository_id']} ({r['repository_id']})" for r in repos}
    except Exception:
        repo_options = {}

    if not repo_options:
        st.info("No repositories available. Create a repository first.")
        selected_repo_id = None
    else:
        selected_repo_id = st.selectbox(
            "Select Repository",
            options=list(repo_options.keys()),
            format_func=lambda x: repo_options[x]
        )

    user_query = st.text_input("Enter your question:")

    if st.button("Ask Chatbot"):
        if not selected_repo_id:
            st.error("Select a repository first.")
            return
        if not user_query.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Processing..."):
            try:
                chat_res = _request(
                    "POST",
                    "/chat",
                    json={"query": user_query, "repository_id": selected_repo_id},
                    timeout=60
                )
                st.session_state["chatbot_last_response"] = chat_res
                st.session_state["chatbot_last_query"] = user_query
            except Exception as exc:
                st.session_state.pop("chatbot_last_response", None)
                st.error(f"Chat failed: {str(exc)}")
                return

    chat_res = st.session_state.get("chatbot_last_response")
    if not chat_res:
        return

    if st.session_state.get("chatbot_last_query"):
        st.caption(f"Last question: {st.session_state['chatbot_last_query']}")

    # Check Prompt Guard
    if not chat_res.get("allowed"):
        st.error(f"Prompt Guard Blocked: {chat_res.get('reason')}")
        st.warning(chat_res.get("safe_message"))
        return

    # Display RAG response
    st.write("### Chatbot Response:")
    st.write(chat_res.get("answer"))

    _render_citation_popups(list(chat_res.get("citations") or []))

    if chat_res.get("evidence_sufficient") is False:
        st.info("Evidence was insufficient to ground a document-based answer.")

    # Layered output moderation report
    moderation = chat_res.get("moderation") or {}
    if chat_res.get("action") == "blocked":
        st.error("The response was blocked by output moderation policy.")
    elif not chat_res.get("safe"):
        st.warning("Sensitive data was removed or masked from the response.")
    if moderation:
        with st.expander("Moderation report"):
            st.write(
                f"Final action: `{moderation.get('final_action')}` | "
                f"Severity: `{moderation.get('severity')}` | "
                f"Total time: {moderation.get('total_execution_time_ms')} ms"
            )
            for layer in moderation.get("layers") or []:
                st.markdown(
                    f"- **{layer.get('layer')}** — status `{layer.get('status')}`, "
                    f"action `{layer.get('action')}`, "
                    f"detections {layer.get('detection_count', 0)}, "
                    f"{layer.get('execution_time_ms')} ms"
                )


if __name__ == "__main__":
    main()
