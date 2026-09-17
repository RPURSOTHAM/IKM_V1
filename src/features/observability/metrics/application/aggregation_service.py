

from __future__ import annotations

import logging
import statistics
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any

from src.features.observability.metrics.domain.metrics_models import DailyMetricRecord
from src.features.observability.metrics.infrastructure.metrics_repository import get_metrics_repository

logger = logging.getLogger(__name__)


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


class AggregationService:
    def aggregate_day(self, day: date, *, metric_name: str | None = None) -> list[DailyMetricRecord]:
        repo = get_metrics_repository()
        if repo is None:
            return []
        if metric_name:
            names = [metric_name]
        else:
            names = self._distinct_metric_names(day, repo)
        records: list[DailyMetricRecord] = []
        for name in names:
            records.append(self._aggregate_metric_for_day(repo, day, name))
        return records

    def run_daily_aggregation(self, *, days_back: int = 1) -> None:
        repo = get_metrics_repository()
        if repo is None:
            return
        today = datetime.now(timezone.utc).date()
        for offset in range(days_back):
            day = today - timedelta(days=offset)
            for record in self.aggregate_day(day):
                try:
                    repo.upsert_daily(record)
                except Exception:
                    logger.warning(
                        "Daily metrics upsert failed for %s %s",
                        day,
                        record.metric_name,
                        exc_info=True,
                    )

    def schedule_daily_aggregation(self, *, days_back: int = 1) -> None:
        threading.Thread(target=self.run_daily_aggregation, kwargs={"days_back": days_back}, daemon=True).start()

    def summary(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        repository_id: str | None = None,
    ) -> dict[str, Any]:
        repo = get_metrics_repository()
        if repo is None:
            return {"total": 0, "success": 0, "failure": 0, "success_rate": 0.0, "failure_rate": 0.0}
        counts = repo.fetch_status_counts(
            date_from=date_from,
            date_to=date_to,
            repository_id=repository_id,
        )
        success = counts.get("success", 0)
        failure = counts.get("failure", 0)
        total = success + failure
        success_rate = (success / total * 100.0) if total else 0.0
        failure_rate = (failure / total * 100.0) if total else 0.0
        return {
            "total": total,
            "success": success,
            "failure": failure,
            "success_rate": round(success_rate, 2),
            "failure_rate": round(failure_rate, 2),
        }

    def latency_summary(
        self,
        *,
        metric_name: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> dict[str, Any]:
        repo = get_metrics_repository()
        if repo is None:
            return {}
        day = (date_from or datetime.now(timezone.utc)).date()
        rows = repo.fetch_durations_for_day(day, metric_name)
        durations = [d for d, _ in rows]
        if not durations:
            return {"metric_name": metric_name, "count": 0}
        return {
            "metric_name": metric_name,
            "count": len(durations),
            "avg_duration": statistics.mean(durations),
            "p50": _percentile(durations, 50),
            "p95": _percentile(durations, 95),
            "p99": _percentile(durations, 99),
            "max_duration": max(durations),
        }

    def failure_summary(
        self,
        *,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        repository_id: str | None = None,
    ) -> dict[str, Any]:
        repo = get_metrics_repository()
        if repo is None:
            return {"failures": 0, "by_metric": {}}
        counts = repo.fetch_status_counts(
            date_from=date_from,
            date_to=date_to,
            repository_id=repository_id,
            metric_name=None,
        )
        return {"failures": counts.get("failure", 0), "by_status": counts}

    def _aggregate_metric_for_day(self, repo: Any, day: date, metric_name: str) -> DailyMetricRecord:
        rows = repo.fetch_durations_for_day(day, metric_name)
        durations = [d for d, status in rows if d is not None]
        failure_count = sum(1 for _, status in rows if status == "failure")
        count = len(rows) if rows else repo.fetch_status_counts(
            date_from=datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc),
            date_to=datetime.combine(day, datetime.max.time(), tzinfo=timezone.utc),
            metric_name=metric_name,
        ).get("success", 0) + failure_count
        if not durations:
            return DailyMetricRecord(
                date=day,
                metric_name=metric_name,
                count=count,
                failure_count=failure_count,
            )
        return DailyMetricRecord(
            date=day,
            metric_name=metric_name,
            count=count,
            avg_duration=statistics.mean(durations),
            p50=_percentile(durations, 50),
            p95=_percentile(durations, 95),
            p99=_percentile(durations, 99),
            max_duration=max(durations),
            failure_count=failure_count,
        )

    def _distinct_metric_names(self, day: date, repo: Any) -> list[str]:
        from src.features.observability.metrics.application.benchmark_metrics import ALLOWED_METRIC_NAMES

        sql = "SELECT DISTINCT metric_name FROM metrics_events WHERE timestamp::date = %s"
        with repo._db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (day,))
                return [str(r[0]) for r in cur.fetchall() if str(r[0]) in ALLOWED_METRIC_NAMES]


_service: AggregationService | None = None


def get_aggregation_service() -> AggregationService:
    global _service
    if _service is None:
        _service = AggregationService()
    return _service
