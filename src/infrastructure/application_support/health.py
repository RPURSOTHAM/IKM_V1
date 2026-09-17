"""Health / readiness registry and FastAPI route helpers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import Response

CheckFn = Callable[[], dict[str, Any] | bool]


@dataclass
class DependencyCheck:
    name: str
    check: CheckFn
    critical: bool = True
    timeout_hint_seconds: float = 2.0


@dataclass
class HealthReport:
    status: str
    service: str
    checks: dict[str, Any] = field(default_factory=dict)
    ready: bool = True
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "service": self.service,
            "ready": self.ready,
            "timestamp": self.timestamp,
            "checks": self.checks,
        }


class HealthRegistry:
    def __init__(self, service: str) -> None:
        self.service = service
        self._checks: list[DependencyCheck] = []

    def add(self, name: str, check: CheckFn, *, critical: bool = True) -> None:
        self._checks.append(DependencyCheck(name=name, check=check, critical=critical))

    def liveness(self) -> HealthReport:
        return HealthReport(status="healthy", service=self.service, ready=True, checks={"process": "up"})

    def readiness(self) -> HealthReport:
        results: dict[str, Any] = {}
        ready = True
        for item in self._checks:
            started = time.perf_counter()
            try:
                raw = item.check()
                if isinstance(raw, bool):
                    ok = raw
                    detail: Any = {"ok": raw}
                else:
                    ok = bool(raw.get("ok", raw.get("status") in {"ok", "healthy", "up", True}))
                    detail = raw
                results[item.name] = {
                    **(detail if isinstance(detail, dict) else {"detail": detail}),
                    "ok": ok,
                    "critical": item.critical,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                if item.critical and not ok:
                    ready = False
            except Exception as exc:
                results[item.name] = {
                    "ok": False,
                    "critical": item.critical,
                    "error": str(exc),
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                }
                if item.critical:
                    ready = False
        return HealthReport(
            status="healthy" if ready else "degraded",
            service=self.service,
            checks=results,
            ready=ready,
        )


def attach_infra_routes(app: Any, *, service: str, health_registry: HealthRegistry | None = None) -> HealthRegistry:
    """Attach /ready, /metrics, and /infra/plugins without replacing existing /health."""
    from src.infrastructure.application_support.metrics import get_metrics
    from src.infrastructure.application_support.registry import get_model_loader

    registry = health_registry or HealthRegistry(service)
    metrics = get_metrics()
    metrics.up.labels(service=service).set(1)

    @app.get("/ready", tags=["Health"])
    def ready() -> dict[str, Any]:
        report = registry.readiness()
        payload = report.to_dict()
        if not report.ready:
            import os

            if os.getenv("INFRA_READY_STRICT", "false").strip().lower() in {"1", "true", "yes", "on"}:
                from fastapi import HTTPException

                raise HTTPException(status_code=503, detail=payload)
        return payload

    @app.get(
        "/metrics",
        tags=["Observability"],
        response_class=Response,
        response_model=None,
    )
    def metrics_endpoint() -> Response:
        body = metrics.render()
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/infra/plugins", tags=["Observability"])
    def plugins_catalog() -> dict[str, Any]:
        return get_model_loader().catalog()

    @app.get("/infra/circuits", tags=["Observability"])
    def circuits() -> dict[str, Any]:
        from src.infrastructure.application_support.resilience import _breakers

        return {"circuits": [breaker.snapshot() for breaker in _breakers.values()]}

    return registry
