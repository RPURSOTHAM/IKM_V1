"""Metrics event data models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any


@dataclass
class MetricEventRecord:
    timestamp: datetime | None = None
    request_id: str | None = None
    correlation_id: str | None = None
    metric_name: str = ""
    duration_ms: float | None = None
    status: str = "success"
    repository_id: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class DailyMetricRecord:
    date: date
    metric_name: str
    count: int = 0
    avg_duration: float | None = None
    p50: float | None = None
    p95: float | None = None
    p99: float | None = None
    max_duration: float | None = None
    failure_count: int = 0


@dataclass
class MetricsQuery:
    page: int = 1
    page_size: int = 50
    metric_name: str | None = None
    metric_names: list[str] | None = None
    exclude_metric_names: list[str] | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    request_id: str | None = None
    repository_id: str | None = None
    status: str | None = None
