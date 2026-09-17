"""Environment-driven scheduler tuning (no hardcoded operational limits in call sites)."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    try:
        value = int(str(raw).strip()) if raw is not None and str(raw).strip() else default
    except ValueError:
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


@dataclass(frozen=True)
class SchedulerRuntimeConfig:
    port_range_start: int
    port_range_end: int
    min_parallel_jobs: int
    max_parallel_jobs: int
    queue_check_frequency: int
    container_poll_frequency: int
    job_timeout_seconds: int
    orphan_grace_seconds: int
    stale_queued_seconds: int
    stale_dispatching_seconds: int
    stale_assigned_seconds: int
    queue_scan_limit: int
    republish_cooldown_seconds: int
    stale_queued_republish_limit: int
    processor_health_timeout: float
    dispatch_sla_seconds: int
    queue_priority_batch: int
    unhealthy_restart_threshold: int

    @classmethod
    def from_env(cls) -> SchedulerRuntimeConfig:
        min_parallel = _int_env("SCHEDULER_MIN_PARALLEL_JOBS", 5, minimum=1, maximum=50)
        max_parallel = _int_env("SCHEDULER_MAX_PARALLEL_JOBS", 10, minimum=min_parallel, maximum=50)
        port_start = _int_env("SCHEDULER_PORT_RANGE_START", 3100, minimum=1024, maximum=65500)
        port_end = _int_env("SCHEDULER_PORT_RANGE_END", 3120, minimum=port_start + 1, maximum=65535)
        return cls(
            port_range_start=port_start,
            port_range_end=port_end,
            min_parallel_jobs=min_parallel,
            max_parallel_jobs=max_parallel,
            queue_check_frequency=_int_env("QUEUE_CHECK_FREQUENCY", 1, minimum=1, maximum=60),
            container_poll_frequency=_int_env("CONTAINER_POLLING_FREQUENCY", 3, minimum=1, maximum=120),
            job_timeout_seconds=_int_env("SCHEDULER_JOB_TIMEOUT_SECONDS", 1800, minimum=60),
            orphan_grace_seconds=_int_env("SCHEDULER_ORPHAN_GRACE_SECONDS", 15, minimum=0, maximum=600),
            stale_queued_seconds=_int_env("SCHEDULER_STALE_QUEUED_SECONDS", 25, minimum=10, maximum=3600),
            stale_dispatching_seconds=_int_env("SCHEDULER_STALE_DISPATCHING_SECONDS", 30, minimum=5, maximum=600),
            stale_assigned_seconds=_int_env("SCHEDULER_STALE_ASSIGNED_SECONDS", 120, minimum=30, maximum=7200),
            queue_scan_limit=_int_env("SCHEDULER_QUEUE_SCAN_LIMIT", 100, minimum=10, maximum=2000),
            republish_cooldown_seconds=_int_env("SCHEDULER_REPUBLISH_COOLDOWN_SECONDS", 30, minimum=0, maximum=3600),
            stale_queued_republish_limit=_int_env("SCHEDULER_STALE_QUEUED_REPUBLISH_LIMIT", 10, minimum=0, maximum=100),
            processor_health_timeout=float(_int_env("SCHEDULER_PROCESSOR_HEALTH_TIMEOUT_SECONDS", 5, minimum=1, maximum=60)),
            dispatch_sla_seconds=_int_env("SCHEDULER_DISPATCH_SLA_SECONDS", 30, minimum=10, maximum=300),
            queue_priority_batch=_int_env("SCHEDULER_QUEUE_PRIORITY_BATCH", 25, minimum=5, maximum=200),
            unhealthy_restart_threshold=_int_env("SCHEDULER_UNHEALTHY_RESTART_THRESHOLD", 3, minimum=1, maximum=24),
        )
