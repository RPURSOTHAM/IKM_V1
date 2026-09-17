"""OpenTelemetry tracing with a no-op fallback when the SDK is absent."""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

_logger = logging.getLogger(__name__)
_provider: Any = None
_tracer: Any = None


@dataclass
class SpanRecord:
    name: str
    trace_id: str
    span_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: str | None = None
    duration_ms: float = 0.0


class _NoOpSpan:
    def __init__(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attributes = dict(attributes or {})
        self.trace_id = uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.status = "ok"
        self.error: str | None = None
        self._started = time.perf_counter()

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        self.status = "error"
        self.error = str(exc)

    def end(self) -> SpanRecord:
        return SpanRecord(
            name=self.name,
            trace_id=self.trace_id,
            span_id=self.span_id,
            attributes=self.attributes,
            status=self.status,
            error=self.error,
            duration_ms=(time.perf_counter() - self._started) * 1000,
        )


def configure_tracing(service_name: str | None = None) -> bool:
    """Initialize OpenTelemetry if installed and enabled. Returns True when live."""
    global _provider, _tracer
    enabled = os.getenv("INFRA_OTEL_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        _tracer = None
        return False
    try:
        from opentelemetry import trace  # type: ignore
        from opentelemetry.sdk.resources import Resource  # type: ignore
        from opentelemetry.sdk.trace import TracerProvider  # type: ignore
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter  # type: ignore
    except ImportError:
        _logger.info("OpenTelemetry SDK not installed; using no-op tracer")
        _tracer = None
        return False

    service = service_name or os.getenv("OTEL_SERVICE_NAME") or os.getenv("SERVICE_NAME") or "rag-builder"
    resource = Resource.create({"service.name": service})
    provider = TracerProvider(resource=resource)
    # Prefer OTLP when available; otherwise console exporter for local visibility.
    exporter = None
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # type: ignore

            exporter = OTLPSpanExporter(endpoint=endpoint)
        except ImportError:
            exporter = ConsoleSpanExporter()
    else:
        exporter = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _provider = provider
    _tracer = trace.get_tracer(service)
    return True


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[Any]:
    """Create a span. Uses OpenTelemetry when configured; otherwise a local no-op span."""
    if _tracer is not None:
        with _tracer.start_as_current_span(name) as span:
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    pass
            try:
                yield span
            except Exception as exc:
                try:
                    span.record_exception(exc)
                except Exception:
                    pass
                raise
        return

    span = _NoOpSpan(name, attributes)
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        raise
    finally:
        record = span.end()
        if os.getenv("INFRA_OTEL_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
            _logger.debug("span name=%s duration_ms=%.2f status=%s", record.name, record.duration_ms, record.status)


def tracing_enabled() -> bool:
    return _tracer is not None
