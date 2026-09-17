from __future__ import annotations

import json
import socket
import struct
import logging
from typing import Any, Dict, List, Optional

from src.features.scheduler_server.api.protocol import encode_frame, decode_frame

logger = logging.getLogger(__name__)


class SchedulerClient:
    """Client for parent API service to communicate with Scheduler Server over TCP."""

    def __init__(self, host: str | None = None, port: int | None = None, timeout: float = 30.0):
        if host is None:
            import os
            try:
                from src.features.configuration.platform_settings import get_settings
                host = get_settings().scheduler_host
            except Exception:
                host = os.getenv("SCHEDULER_HOST", "localhost")
        if port is None:
            import os
            try:
                from src.features.configuration.platform_settings import get_settings
                port = get_settings().scheduler_port
            except Exception:
                port = int(os.getenv("SCHEDULER_TCP_PORT", "3200"))

        self.host = host
        self.port = int(port)
        self.timeout = timeout

    def send_request(self, message: Dict[str, Any]) -> Any:
        """Send length-prefixed JSON request to scheduler TCP server."""
        hosts_to_try = [self.host]
        for fallback in ("127.0.0.1", "localhost"):
            if fallback not in hosts_to_try:
                hosts_to_try.append(fallback)

        last_exc = None
        for host in hosts_to_try:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.timeout)
                sock.connect((host, self.port))

                frame = encode_frame(message)
                sock.sendall(frame)

                header = sock.recv(4)
                if not header or len(header) < 4:
                    sock.close()
                    return {"error": "Empty or incomplete response header"}

                length = struct.unpack(">I", header)[0]
                body_bytes = b""
                while len(body_bytes) < length:
                    chunk = sock.recv(min(4096, length - len(body_bytes)))
                    if not chunk:
                        break
                    body_bytes += chunk

                sock.close()
                return json.loads(body_bytes.decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                logger.debug("TCP Communication with Scheduler %s:%d failed: %s", host, self.port, exc)

        logger.error("TCP Communication with Scheduler %s:%d failed on all attempted hosts: %s", self.host, self.port, last_exc)
        return {"error": f"Communication failed: {str(last_exc)}"}

    def query_job_status(self, document_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        return self.send_request({"signal_type": "query", "document_id": document_id, "job_id": job_id})

    def kill_job(self, document_id: str, job_id: Optional[str] = None) -> Dict[str, Any]:
        return self.send_request({"signal_type": "kill", "document_id": document_id, "job_id": job_id})

    def healthcheck(self) -> Dict[str, Any]:
        return self.send_request({"signal_type": "healthcheck"})

    def get_running_jobs(self) -> List[Dict[str, Any]]:
        resp = self.send_request({"signal_type": "running_jobs"})
        return resp if isinstance(resp, list) else []

    def shutdown_scheduler(self) -> Dict[str, Any]:
        return self.send_request({"signal_type": "shutdown"})
