"""Collect and record container-level resource metrics for benchmark visibility.

This is a best-effort collector intended for Docker deployments where the
scheduler-service has access to `/var/run/docker.sock` (see deployment compose).

Collected metrics:
- CPU% / memory bytes / block I/O bytes/sec (derived from docker stats)
- storage utilization and database size (best-effort `df -PB1` exec)

Metrics are recorded as gauge-style benchmark metric events via
`get_metrics_service().record_gauge(...)`.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any, Optional

from src.features.observability.metrics.application.metrics_service import get_metrics_service

_last_disk_bytes: dict[str, tuple[float, int]] = {}
_last_df_sample_at: dict[str, float] = {}


@dataclass(frozen=True)
class _ContainerMetricSpec:
    metric_cpu: str
    metric_mem: str
    metric_disk_io: str
    # Optional capacity metrics
    metric_storage_utilization: str | None = None
    metric_database_size: str | None = None
    # Which paths to query for `df -PB1`. We try in order.
    df_paths: tuple[str, ...] = ()


_DOCKER_CONTAINER_SPECS: dict[str, _ContainerMetricSpec] = {
    # container_name -> spec
    "postgres": _ContainerMetricSpec(
        metric_cpu="postgresql_cpu_usage",
        metric_mem="postgresql_memory_usage",
        metric_disk_io="postgresql_disk_io",
        metric_storage_utilization="postgresql_storage_utilization",
        df_paths=("/var/lib/postgresql/data", "/var/lib/postgresql"),
    ),
    "weaviate": _ContainerMetricSpec(
        metric_cpu="weaviate_cpu_usage",
        metric_mem="weaviate_memory_usage",
        metric_disk_io="weaviate_disk_io",
        metric_storage_utilization="weaviate_storage_utilization",
        metric_database_size="weaviate_database_size",
        df_paths=("/var/lib/weaviate", "/data"),
    ),
    "neo4j": _ContainerMetricSpec(
        metric_cpu="neo4j_cpu_usage",
        metric_mem="neo4j_memory_usage",
        metric_disk_io="neo4j_disk_io",
        metric_database_size="neo4j_database_size",
        df_paths=("/var/lib/neo4j", "/data"),
    ),
    "redis": _ContainerMetricSpec(
        metric_cpu="redis_cpu_usage",
        metric_mem="redis_memory_usage",
        metric_disk_io="redis_disk_io",
        # Capacity metrics are handled via Redis INFO in redis_store.
        df_paths=("/data",),
    ),
}


def _cpu_percent(stats: dict[str, Any]) -> Optional[float]:
    """Compute docker stats CPU% based on deltas."""
    try:
        cpu_stats = stats.get("cpu_stats") or {}
        precpu_stats = stats.get("precpu_stats") or {}
        cpu_usage = cpu_stats.get("cpu_usage") or {}
        precpu_usage = precpu_stats.get("cpu_usage") or {}
        total_usage = cpu_usage.get("total_usage")
        prec_total_usage = precpu_usage.get("total_usage")
        system_cpu_usage = cpu_stats.get("system_cpu_usage")
        prec_system_cpu_usage = precpu_stats.get("system_cpu_usage")
        if None in (total_usage, prec_total_usage, system_cpu_usage, prec_system_cpu_usage):
            return None
        cpu_delta = float(total_usage) - float(prec_total_usage)
        sys_delta = float(system_cpu_usage) - float(prec_system_cpu_usage)
        if sys_delta <= 0:
            return None
        percpu_usage = cpu_usage.get("percpu_usage") or []
        cpu_count = len(percpu_usage) or cpu_stats.get("online_cpus") or 1
        return (cpu_delta / sys_delta) * float(cpu_count) * 100.0
    except Exception:
        return None


def _blkio_bytes(stats: dict[str, Any]) -> Optional[int]:
    try:
        blkio = stats.get("blkio_stats") or {}
        items = blkio.get("io_service_bytes_recursive") or []
        total = 0
        for it in items:
            op = str(it.get("op") or "").lower()
            # Docker uses Read/Write labels; we also include async/sync when present.
            if op in {"read", "write", "sync", "async"} or "read" in op or "write" in op:
                val = it.get("value")
                if val is None:
                    continue
                total += int(val)
        return total
    except Exception:
        return None


def _container_df_used_and_percent(container: Any, path: str) -> tuple[int, float] | None:
    """Return (used_bytes, used_percent) for a directory in a container."""
    try:
        # Use -PB1 so sizes are in bytes.
        cmd = f"df -PB1 {path}"
        res = container.exec_run(cmd, stdout=True, stderr=True)
        output = (res.output or b"").decode(errors="ignore") if hasattr(res, "output") else ""
        lines = [ln for ln in output.splitlines() if ln.strip()]
        if len(lines) < 2:
            return None
        # Example columns:
        # Filesystem 1B-blocks Used Available Use% Mounted on
        parts = re.split(r"\s+", lines[1].strip())
        if len(parts) < 6:
            return None
        used_bytes = int(float(parts[2]))
        use_pct_raw = parts[4]
        used_pct = float(use_pct_raw.replace("%", "").strip())
        return used_bytes, used_pct
    except Exception:
        return None


def _try_record_container_specs(
    *,
    docker_client: Any,
    container_name: str,
    spec: _ContainerMetricSpec,
    sample_at: float,
    storage_metrics_interval_sec: float,
) -> None:
    try:
        container = docker_client.containers.get(container_name)
    except Exception:
        return

    try:
        stats = container.stats(stream=False)
    except Exception:
        return

    cpu = _cpu_percent(stats)
    mem = (stats.get("memory_stats") or {}).get("usage")
    disk_bytes = _blkio_bytes(stats)

    svc = get_metrics_service()
    if cpu is not None:
        svc.record_gauge(spec.metric_cpu, cpu, metadata={"container": container_name})
    if mem is not None:
        svc.record_gauge(spec.metric_mem, int(mem), metadata={"container": container_name})

    if disk_bytes is not None:
        prev = _last_disk_bytes.get(container.id)
        if prev is not None:
            prev_at, prev_bytes = prev
            delta_bytes = max(0, int(disk_bytes) - int(prev_bytes))
            dt = max(1e-3, sample_at - prev_at)
            bytes_per_sec = delta_bytes / dt
            svc.record_gauge(spec.metric_disk_io, float(bytes_per_sec), metadata={"container": container_name})
        _last_disk_bytes[container.id] = (sample_at, int(disk_bytes))

    # Capacity metrics via `df` (throttled).
    if spec.metric_storage_utilization or spec.metric_database_size:
        last_df = _last_df_sample_at.get(container.id, 0.0)
        if sample_at - last_df >= storage_metrics_interval_sec:
            for p in spec.df_paths:
                df_res = _container_df_used_and_percent(container, p)
                if not df_res:
                    continue
                used_bytes, used_pct = df_res
                if spec.metric_storage_utilization:
                    svc.record_gauge(
                        spec.metric_storage_utilization,
                        used_pct,
                        metadata={"container": container_name, "df_path": p},
                    )
                if spec.metric_database_size:
                    svc.record_gauge(
                        spec.metric_database_size,
                        used_bytes,
                        metadata={"container": container_name, "df_path": p},
                    )
                break
            _last_df_sample_at[container.id] = sample_at


def record_platform_resource_metrics(
    *,
    docker_mgr: Any,
    sample_at: float | None = None,
    storage_metrics_interval_sec: float = 120.0,
) -> None:
    """Record benchmark resource metrics for scheduler + infra containers."""
    try:
        docker_client = getattr(docker_mgr, "client", None)
        if docker_client is None:
            return
        sample_at = sample_at or time.time()

        svc = get_metrics_service()

        # 1) Scheduler container (this process).
        try:
            sched = None
            for cname in ("rag-scheduler-service", "scheduler-service", "rag-scheduler"):
                try:
                    sched = docker_client.containers.get(cname)
                    break
                except Exception:
                    continue
            if sched is not None:
                stats = sched.stats(stream=False)
                scheduler_cpu = _cpu_percent(stats)
                scheduler_mem = (stats.get("memory_stats") or {}).get("usage")
                if scheduler_cpu is not None:
                    svc.record_gauge("scheduler_cpu_usage", scheduler_cpu, metadata={"container": sched.name})
                if scheduler_mem is not None:
                    svc.record_gauge("scheduler_memory_usage", int(scheduler_mem), metadata={"container": sched.name})
        except Exception:
            pass

        # 2) Processor containers managed by scheduler: aggregate average CPU/mem across running slots.
        processor_containers = []
        try:
            for info in getattr(docker_mgr, "containers", {}).values():
                c = info.get("container")
                if c is None:
                    continue
                try:
                    c.reload()
                except Exception:
                    pass
                if getattr(c, "status", None) == "running":
                    processor_containers.append(c)
        except Exception:
            processor_containers = []

        if processor_containers:
            cpu_vals: list[float] = []
            mem_vals: list[int] = []
            for c in processor_containers:
                try:
                    st = c.stats(stream=False)
                    cpu = _cpu_percent(st)
                    mem = (st.get("memory_stats") or {}).get("usage")
                    if cpu is not None:
                        cpu_vals.append(float(cpu))
                    if mem is not None:
                        mem_vals.append(int(mem))
                except Exception:
                    continue
            if cpu_vals:
                svc.record_gauge(
                    "processor_cpu_usage",
                    sum(cpu_vals) / len(cpu_vals),
                    metadata={"container_count": len(cpu_vals)},
                )
            if mem_vals:
                svc.record_gauge(
                    "processor_memory_usage",
                    int(sum(mem_vals) / len(mem_vals)),
                    metadata={"container_count": len(mem_vals)},
                )

        # 3) Infra containers (postgres/weaviate/neo4j/redis) via stable container names.
        for cname, spec in _DOCKER_CONTAINER_SPECS.items():
            _try_record_container_specs(
                docker_client=docker_client,
                container_name=cname,
                spec=spec,
                sample_at=sample_at,
                storage_metrics_interval_sec=storage_metrics_interval_sec,
            )
    except Exception:
        # Never crash scheduler poll loop due to metrics.
        return

