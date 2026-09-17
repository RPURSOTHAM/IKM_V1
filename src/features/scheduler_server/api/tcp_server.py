from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Awaitable, Callable, Dict, Any, Optional

from src.features.scheduler_server.api.protocol import encode_frame
from src.features.scheduler_server.application.pool_manager import ProcessorPoolManager
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository
from src.features.scheduler_server.domain.job_state import JobState

logger = logging.getLogger(__name__)


class TCPServerHandler:
    """IPC Control Server over TCP socket on Port 3200."""

    def __init__(
        self,
        host: str,
        port: int,
        pool_mgr: ProcessorPoolManager,
        job_repo: SchedulerJobRepository,
        shutdown_trigger: Callable[[], None],
    ):
        self.host = host
        self.port = port
        self.pool_mgr = pool_mgr
        self.job_repo = job_repo
        self.shutdown_trigger = shutdown_trigger
        self.server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        """Start listening on TCP host:port."""
        self.server = await asyncio.start_server(self._handle_client, self.host, self.port)
        logger.info("TCP Control Server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop TCP server listener."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("TCP Control Server stopped.")

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            header = await reader.readexactly(4)
            length = struct.unpack(">I", header)[0]
            body_bytes = await reader.readexactly(length)
            request = json.loads(body_bytes.decode("utf-8"))

            response = await self._process_request(request)
            response_frame = encode_frame(response)
            writer.write(response_frame)
            await writer.drain()
        except asyncio.IncompleteReadError:
            pass
        except Exception as exc:
            logger.error("Error processing TCP request: %s", exc)
            err_frame = encode_frame({"error": str(exc), "status": "failed"})
            try:
                writer.write(err_frame)
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def _process_request(self, req: Dict[str, Any]) -> Dict[str, Any]:
        signal_type = req.get("signal_type", "").lower()

        if signal_type == "healthcheck":
            return {"status": "Healthy", "service": "SchedulerServer", "version": "2.0.0"}

        elif signal_type == "query":
            job_id = req.get("job_id") or req.get("document_id")
            if not job_id:
                return {"error": "Missing job_id or document_id"}

            record = await asyncio.to_thread(self.job_repo.get_job_by_id, job_id) or await asyncio.to_thread(self.job_repo.get_job_by_document_id, job_id)
            if record:
                return {
                    "document_processing_status": record.get("status"),
                    "document_id": record.get("document_id"),
                    "job_id": record.get("job_id"),
                    "processor_id": record.get("processor_id"),
                    "chunks_processed": record.get("progress_percent") or record.get("chunks_processed"),
                    "error_message": record.get("error_details"),
                    "process_time": str(record.get("updated_at")) if record.get("updated_at") else None,
                }
            return {"error": f"Job {job_id} not found", "document_processing_status": "not_found"}

        elif signal_type == "running_jobs":
            active_jobs = await asyncio.to_thread(self.job_repo.list_running_jobs)
            result = []
            for j in active_jobs:
                result.append({
                    "job_id": j.get("job_id"),
                    "document_id": j.get("document_id"),
                    "processor_id": j.get("processor_id"),
                    "status": j.get("status"),
                })
            return result  # Returns List directly per contract

        elif signal_type == "kill":
            job_id = req.get("job_id") or req.get("document_id")
            if not job_id:
                return {"error": "Missing job_id or document_id"}

            record = await asyncio.to_thread(self.job_repo.get_job_by_id, job_id) or await asyncio.to_thread(self.job_repo.get_job_by_document_id, job_id)
            if not record:
                return {"error": f"Job {job_id} not found"}

            doc_id = record.get("document_id")
            j_id = record.get("job_id")
            # Stop HTTP job on matching slot
            for slot in self.pool_mgr.slots.values():
                if slot.current_job_id == j_id or slot.current_document_id == doc_id:
                    await asyncio.to_thread(
                        self.pool_mgr.docker_mgr.stop_processor_job_http,
                        slot.host_port,
                        slot.container_name,
                        doc_id,
                    )
                    slot.mark_idle()
                    break

            # State transition: STOPPED
            await asyncio.to_thread(self.job_repo.update_job_status, j_id, JobState.STOPPED)
            return {"status": "killed", "job_id": j_id, "document_id": doc_id}

        elif signal_type == "shutdown":
            logger.info("Scheduler shutdown signal received via TCP.")
            self.shutdown_trigger()
            return {"status": "shutdown_initiated", "message": "Scheduler is shutting down gracefully"}

        return {"error": f"Unknown signal_type: {signal_type}"}
