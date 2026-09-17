from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Path

from src.features.scheduler_server.client.scheduler_client import SchedulerClient

router = APIRouter(prefix="/scheduler", tags=["Scheduler"])


@router.get(
    "/health",
    summary="Scheduler Server Healthcheck",
    description="Check connectivity and health of the background Scheduler Server daemon over TCP (Port 3200).",
)
def get_scheduler_health() -> Dict[str, Any]:
    client = SchedulerClient()
    result = client.healthcheck()
    if "error" in result:
        raise HTTPException(status_code=503, detail=f"Scheduler daemon unreachable: {result['error']}")
    return result


@router.get(
    "/running-jobs",
    summary="List Running Jobs",
    description="Query all active processor slots and currently executing jobs managed by the Scheduler.",
)
def get_running_jobs() -> List[Dict[str, Any]]:
    client = SchedulerClient()
    return client.get_running_jobs()


@router.get(
    "/jobs/{document_id}",
    summary="Query Job Lifecycle Status",
    description="Query explicit job lifecycle state (QUEUED, DISPATCHING, ASSIGNED, IN_PROGRESS, COMPLETED, FAILED) for a document ID or job ID.",
)
def query_job_status(
    document_id: str = Path(..., description="Document ID or Job ID to query"),
    job_id: Optional[str] = Query(None, description="Optional explicit job ID"),
) -> Dict[str, Any]:
    client = SchedulerClient()
    result = client.query_job_status(document_id=document_id, job_id=job_id)
    if "error" in result and result.get("document_processing_status") == "not_found":
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post(
    "/jobs/{document_id}/kill",
    summary="Kill Active Job",
    description="Instruct the Scheduler Server to stop and kill an active document processing job.",
)
def kill_job(
    document_id: str = Path(..., description="Document ID or Job ID to kill"),
    job_id: Optional[str] = Query(None, description="Optional explicit job ID"),
) -> Dict[str, Any]:
    client = SchedulerClient()
    result = client.kill_job(document_id=document_id, job_id=job_id)
    return result


@router.post(
    "/shutdown",
    summary="Graceful Scheduler Shutdown",
    description="Initiate graceful shutdown sequence of the Scheduler Server daemon.",
)
def shutdown_scheduler() -> Dict[str, Any]:
    client = SchedulerClient()
    return client.shutdown_scheduler()
