"""Request-scoped context propagation for observability."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("obs_request_id", default=None)
correlation_id_var: ContextVar[str | None] = ContextVar("obs_correlation_id", default=None)
application_name_var: ContextVar[str | None] = ContextVar("obs_application_name", default=None)
user_id_var: ContextVar[str | None] = ContextVar("obs_user_id", default=None)
repository_id_var: ContextVar[str | None] = ContextVar("obs_repository_id", default=None)
document_id_var: ContextVar[str | None] = ContextVar("obs_document_id", default=None)
endpoint_var: ContextVar[str | None] = ContextVar("obs_endpoint", default=None)
http_method_var: ContextVar[str | None] = ContextVar("obs_http_method", default=None)
extra_context_var: ContextVar[dict[str, Any]] = ContextVar("obs_extra_context", default={})


@dataclass
class ObservabilityContext:
    request_id: str | None = None
    correlation_id: str | None = None
    application_name: str | None = None
    user_id: str | None = None
    repository_id: str | None = None
    document_id: str | None = None
    endpoint: str | None = None
    http_method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def get_request_id() -> str | None:
    return request_id_var.get()


def get_correlation_id() -> str | None:
    return correlation_id_var.get()


def get_application_name() -> str | None:
    return application_name_var.get()


def get_user_id() -> str | None:
    return user_id_var.get()


def get_repository_id() -> str | None:
    return repository_id_var.get()


def get_document_id() -> str | None:
    return document_id_var.get()


def current_context() -> ObservabilityContext:
    return ObservabilityContext(
        request_id=get_request_id(),
        correlation_id=get_correlation_id(),
        application_name=get_application_name(),
        user_id=get_user_id(),
        repository_id=get_repository_id(),
        document_id=get_document_id(),
        endpoint=endpoint_var.get(),
        http_method=http_method_var.get(),
        metadata=dict(extra_context_var.get() or {}),
    )


def bind_context(**kwargs: Any) -> list[tuple[ContextVar[Any], Any]]:
    """Set context vars; returns tokens for reset."""
    mapping = {
        "request_id": request_id_var,
        "correlation_id": correlation_id_var,
        "application_name": application_name_var,
        "user_id": user_id_var,
        "repository_id": repository_id_var,
        "document_id": document_id_var,
        "endpoint": endpoint_var,
        "http_method": http_method_var,
    }
    tokens: list[tuple[ContextVar[Any], Any]] = []
    for key, var in mapping.items():
        if key in kwargs and kwargs[key] is not None:
            tokens.append((var, var.set(kwargs[key])))
    if "metadata" in kwargs and isinstance(kwargs["metadata"], dict):
        merged = dict(extra_context_var.get() or {})
        merged.update(kwargs["metadata"])
        tokens.append((extra_context_var, extra_context_var.set(merged)))
    return tokens


def reset_context(tokens: list[tuple[ContextVar[Any], Any]]) -> None:
    for var, token in tokens:
        var.reset(token)


def extract_call_context(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """Resolve repository/document ids from kwargs or common argument shapes."""
    repository_id = kwargs.get("repository_id")
    document_id = kwargs.get("document_id")
    for arg in args:
        if arg is None:
            continue
        if document_id is None:
            document_id = getattr(arg, "document_id", None)
        if repository_id is None:
            repository_id = getattr(arg, "repository_id", None)
        if isinstance(arg, dict):
            document_id = document_id or arg.get("document_id")
            repository_id = repository_id or arg.get("repository_id")
    return {
        "repository_id": str(repository_id) if repository_id else None,
        "document_id": str(document_id) if document_id else None,
    }


def timing_metadata(start: float, end: float, *, func_name: str) -> dict[str, Any]:
    return {
        "function": func_name,
        "start_time": start,
        "end_time": end,
        "duration_ms": (end - start) * 1000.0,
    }


def apply_synthetic_context(
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
    application_name: str | None = None,
    user_id: str | None = None,
    repository_id: str | None = None,
    document_id: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[tuple[ContextVar[Any], Any]]:
    """Bind a synthetic context for scheduler/background jobs."""
    return bind_context(
        request_id=request_id,
        correlation_id=correlation_id,
        application_name=application_name,
        user_id=user_id or "scheduler",
        repository_id=repository_id,
        document_id=document_id,
        metadata=metadata or {},
    )
