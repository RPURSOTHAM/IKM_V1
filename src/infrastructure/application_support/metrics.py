"""Prometheus metrics with an in-memory fallback when prometheus_client is absent."""

from __future__ import annotations

import os
import threading
import time
from typing import Any


class _FallbackCounter:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._lock = threading.Lock()
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, **labels: str) -> "_FallbackCounterChild":
        key = tuple(labels.get(name, "") for name in self.labelnames)
        return _FallbackCounterChild(self, key)

    def inc(self, amount: float = 1.0) -> None:
        self.labels().inc(amount)

    def collect_lines(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} counter"]
        with self._lock:
            for key, value in self._values.items():
                if self.labelnames:
                    label = ",".join(f'{n}="{v}"' for n, v in zip(self.labelnames, key))
                    lines.append(f"{self.name}{{{label}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return lines


class _FallbackCounterChild:
    def __init__(self, parent: _FallbackCounter, key: tuple[str, ...]) -> None:
        self._parent = parent
        self._key = key

    def inc(self, amount: float = 1.0) -> None:
        with self._parent._lock:
            self._parent._values[self._key] = self._parent._values.get(self._key, 0.0) + amount


class _FallbackHistogram:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._lock = threading.Lock()
        self._sum: dict[tuple[str, ...], float] = {}
        self._count: dict[tuple[str, ...], float] = {}

    def labels(self, **labels: str) -> "_FallbackHistogramChild":
        key = tuple(labels.get(name, "") for name in self.labelnames)
        return _FallbackHistogramChild(self, key)

    def observe(self, amount: float) -> None:
        self.labels().observe(amount)

    def collect_lines(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for key in set(self._sum) | set(self._count):
                if self.labelnames:
                    label = ",".join(f'{n}="{v}"' for n, v in zip(self.labelnames, key))
                    lines.append(f'{self.name}_sum{{{label}}} {self._sum.get(key, 0.0)}')
                    lines.append(f'{self.name}_count{{{label}}} {self._count.get(key, 0.0)}')
                else:
                    lines.append(f"{self.name}_sum {self._sum.get(key, 0.0)}")
                    lines.append(f"{self.name}_count {self._count.get(key, 0.0)}")
        return lines


class _FallbackHistogramChild:
    def __init__(self, parent: _FallbackHistogram, key: tuple[str, ...]) -> None:
        self._parent = parent
        self._key = key

    def observe(self, amount: float) -> None:
        with self._parent._lock:
            self._parent._sum[self._key] = self._parent._sum.get(self._key, 0.0) + amount
            self._parent._count[self._key] = self._parent._count.get(self._key, 0.0) + 1.0


class _FallbackGauge:
    def __init__(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> None:
        self.name = name
        self.documentation = documentation
        self.labelnames = labelnames
        self._lock = threading.Lock()
        self._values: dict[tuple[str, ...], float] = {}

    def labels(self, **labels: str) -> "_FallbackGaugeChild":
        key = tuple(labels.get(name, "") for name in self.labelnames)
        return _FallbackGaugeChild(self, key)

    def set(self, value: float) -> None:
        self.labels().set(value)

    def collect_lines(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.documentation}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for key, value in self._values.items():
                if self.labelnames:
                    label = ",".join(f'{n}="{v}"' for n, v in zip(self.labelnames, key))
                    lines.append(f"{self.name}{{{label}}} {value}")
                else:
                    lines.append(f"{self.name} {value}")
        return lines


class _FallbackGaugeChild:
    def __init__(self, parent: _FallbackGauge, key: tuple[str, ...]) -> None:
        self._parent = parent
        self._key = key

    def set(self, value: float) -> None:
        with self._parent._lock:
            self._parent._values[self._key] = value


class MetricsRegistry:
    """Unified metrics facade. Prefers prometheus_client when installed."""

    def __init__(self, namespace: str = "rag") -> None:
        self.namespace = namespace
        self._started = time.time()
        self._use_prometheus = False
        self._fallback_metrics: list[Any] = []
        self._prometheus: Any = None
        self._init_backend()

        self.request_count = self.counter(
            "requests_total",
            "Total HTTP/service requests",
            ("service", "endpoint", "status"),
        )
        self.request_latency = self.histogram(
            "request_latency_seconds",
            "Request latency in seconds",
            ("service", "endpoint"),
        )
        self.component_errors = self.counter(
            "component_errors_total",
            "Component errors",
            ("component", "error_type"),
        )
        self.pipeline_stage_latency = self.histogram(
            "pipeline_stage_latency_seconds",
            "Pipeline stage latency",
            ("pipeline", "stage"),
        )
        self.up = self.gauge("up", "Service up indicator", ("service",))

    def _init_backend(self) -> None:
        enabled = os.getenv("INFRA_PROMETHEUS_ENABLED", "true").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if not enabled:
            return
        try:
            import prometheus_client

            self._use_prometheus = True
            self._prometheus = prometheus_client
        except ImportError:
            self._use_prometheus = False

    def counter(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> Any:
        full = f"{self.namespace}_{name}"
        if self._use_prometheus:
            return self._prometheus.Counter(full, documentation, labelnames=list(labelnames))
        metric = _FallbackCounter(full, documentation, labelnames)
        self._fallback_metrics.append(metric)
        return metric

    def histogram(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> Any:
        full = f"{self.namespace}_{name}"
        if self._use_prometheus:
            return self._prometheus.Histogram(full, documentation, labelnames=list(labelnames))
        metric = _FallbackHistogram(full, documentation, labelnames)
        self._fallback_metrics.append(metric)
        return metric

    def gauge(self, name: str, documentation: str, labelnames: tuple[str, ...] = ()) -> Any:
        full = f"{self.namespace}_{name}"
        if self._use_prometheus:
            return self._prometheus.Gauge(full, documentation, labelnames=list(labelnames))
        metric = _FallbackGauge(full, documentation, labelnames)
        self._fallback_metrics.append(metric)
        return metric

    def observe_request(self, service: str, endpoint: str, status: str, latency_seconds: float) -> None:
        self.request_count.labels(service=service, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(service=service, endpoint=endpoint).observe(latency_seconds)

    def render(self) -> str:
        if self._use_prometheus:
            return self._prometheus.generate_latest().decode("utf-8")
        lines: list[str] = []
        for metric in self._fallback_metrics:
            lines.extend(metric.collect_lines())
        lines.append(f"# HELP {self.namespace}_process_uptime_seconds Process uptime")
        lines.append(f"# TYPE {self.namespace}_process_uptime_seconds gauge")
        lines.append(f"{self.namespace}_process_uptime_seconds {time.time() - self._started:.3f}")
        return "\n".join(lines) + "\n"

    @property
    def backend(self) -> str:
        return "prometheus_client" if self._use_prometheus else "fallback"


_metrics: MetricsRegistry | None = None
_metrics_lock = threading.Lock()


def get_metrics(namespace: str = "rag") -> MetricsRegistry:
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = MetricsRegistry(namespace=namespace)
        return _metrics


def reset_metrics() -> None:
    global _metrics
    with _metrics_lock:
        _metrics = None
