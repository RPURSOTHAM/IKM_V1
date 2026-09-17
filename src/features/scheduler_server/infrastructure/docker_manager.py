from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
try:
    import docker
    from docker.errors import APIError, NotFound
except ImportError:
    docker = None  # type: ignore

    class APIError(Exception):  # type: ignore
        pass

    class NotFound(Exception):  # type: ignore
        pass

logger = logging.getLogger(__name__)


class DockerServiceManager:
    """Docker Abstraction Layer for processor container slots."""

    def __init__(
        self,
        docker_host: Optional[str] = None,
        processor_http_mode: str = "localhost",
        processor_http_host: str = "localhost",
        docker_network: str = "docnet",
        processor_health_timeout: float = 5.0,
    ):
        self.docker_host = docker_host
        self.processor_health_timeout = float(processor_health_timeout)
        self.processor_http_mode = (
            processor_http_mode.lower()
            if processor_http_mode
            else ("docker_network" if os.getenv("RUNNING_IN_DOCKER", "").lower() in {"1", "true"} else "localhost")
        )
        self.processor_http_host = processor_http_host
        self.docker_network = docker_network
        self.client = self._connect_to_docker()

    def _connect_to_docker(self) -> Optional[docker.DockerClient]:
        try:
            if self.docker_host:
                client = docker.DockerClient(base_url=self.docker_host)
            else:
                client = docker.from_env()
            client.ping()
            logger.info("Connected to Docker daemon successfully.")
            return client
        except Exception as exc:
            logger.warning("Docker daemon unavailable (%s); scheduler continuing in standalone mode.", exc)
            return None

    def resolve_processor_http_url(self, slot_port: int, container_name: Optional[str] = None) -> str:
        """Resolve processor base HTTP URL depending on environment mode."""
        if self.processor_http_mode == "docker_network" and container_name:
            return f"http://{container_name}:3100"
        return f"http://{self.processor_http_host}:{slot_port}"

    def check_processor_health(self, slot_port: int, container_name: Optional[str] = None, timeout: float = 3.0) -> Dict[str, Any]:
        """Perform HTTP healthcheck on GET /health."""
        base_url = self.resolve_processor_http_url(slot_port, container_name)
        health_url = f"{base_url}/health"
        try:
            resp = requests.get(health_url, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "health": data.get("health", "healthy"),
                    "state": data.get("state", "idle"),
                    "details": data,
                    "reachable": True,
                }
            return {"health": "unhealthy", "state": "unknown", "reachable": True, "status_code": resp.status_code}
        except Exception as exc:
            logger.debug("Healthcheck request to %s failed: %s", health_url, exc)
            return {"health": "unhealthy", "state": "unknown", "reachable": False, "error": str(exc)}

    def dispatch_job_http(
        self,
        slot_port: int,
        container_name: Optional[str],
        job_payload: Dict[str, Any],
        timeout: float = 60.0,
    ) -> tuple[bool, int, str]:
        """Issue non-blocking POST /process request to processor HTTP server."""
        base_url = self.resolve_processor_http_url(slot_port, container_name)
        process_url = f"{base_url}/process"
        try:
            resp = requests.post(process_url, json=job_payload, timeout=timeout)
            if resp.status_code in {200, 202}:
                try:
                    body = resp.json()
                    status_text = str(body.get("status") or "").lower()
                    if status_text in {"human_review", "upload_blocked", "failed", "blocked"}:
                        reason = body.get("reason") or f"Processor rejected job with status: {status_text}"
                        logger.error("Processor rejected job dispatch (%s): %s", status_text, reason)
                        return False, resp.status_code, reason
                except Exception:
                    pass
                logger.info("Successfully dispatched job %s to %s", job_payload.get("document_id"), process_url)
                return True, resp.status_code, ""
            logger.error("Failed to dispatch job to %s: HTTP %s - %s", process_url, resp.status_code, resp.text)
            return False, resp.status_code, resp.text
        except Exception as exc:
            logger.error("Exception dispatching job to %s: %s", process_url, exc)
            return False, 500, str(exc)

    def stop_processor_job_http(self, slot_port: int, container_name: Optional[str], document_id: str, timeout: float = 5.0) -> bool:
        """Issue POST /stop request to cancel active job."""
        base_url = self.resolve_processor_http_url(slot_port, container_name)
        stop_url = f"{base_url}/stop"
        try:
            resp = requests.post(stop_url, json={"document_id": document_id}, timeout=timeout)
            return resp.status_code in {200, 202, 404}
        except Exception as exc:
            logger.warning("Stop request to %s failed: %s", stop_url, exc)
            return False

    def find_slot_containers(self, project_prefix: str = "doc_processor_") -> List[Any]:
        """Discover existing processor slot containers."""
        if not self.client:
            return []
        try:
            containers = self.client.containers.list(all=True)
            return [c for c in containers if c.name and (c.name.startswith("processor_slot_") or c.name.startswith(project_prefix))]
        except Exception as exc:
            logger.error("Failed to list Docker containers: %s", exc)
            return []

    def start_slot_container(
        self,
        slot_name: str,
        image_name: str,
        host_port: int,
        document_host_path: str,
        document_container_path: str = "/app/documents",
        models_host_path: Optional[str] = None,
        models_container_path: str = "/app/src/models",
    ) -> Optional[Any]:
        """Start a new processor container for the given slot."""
        if not self.client:
            logger.warning("Cannot start slot container %s: Docker client unavailable", slot_name)
            return None
        try:
            # Remove existing container with same name if present
            try:
                old = self.client.containers.get(slot_name)
                logger.info("Removing stale container %s", slot_name)
                old.remove(force=True)
            except NotFound:
                pass

            volumes = {
                document_host_path: {"bind": document_container_path, "mode": "rw"},
            }
            if models_host_path:
                volumes[models_host_path] = {"bind": models_container_path, "mode": "ro"}

            host_repo = (os.getenv("HOST_REPO_ROOT") or os.getenv("REPO_ROOT") or "").rstrip("/\\").replace("\\", "/")
            if host_repo:
                volumes[f"{host_repo}/src"] = {"bind": "/app/src", "mode": "ro"}
                volumes[f"{host_repo}/configs/security"] = {"bind": "/app/configs/security", "mode": "ro"}
                volumes[f"{host_repo}/configs/processors.yaml"] = {"bind": "/app/configs/processors.yaml", "mode": "ro"}

            from src.shared.networking.processor_env import processor_infra_env

            environment = {
                "PROCESSOR_PORT": "3100",
                "SLOT_HOST_PORT": str(host_port),
                "CONTAINER_NAME": slot_name,
                "DEPLOYMENT_MODE": "docker",
                "RUNNING_IN_DOCKER": "1",
                "DOCUMENT_ROOT": document_container_path,
                "PROCESSOR_DOCUMENTS_DIR_IN_CONTAINER": document_container_path,
                "DOCUMENT_CLASSIFIER_PRELOAD": "false",
                "SECURITY_PIPELINE_LITE": "true",
                "PROCESSOR_SKIP_SECURITY_WHEN_PREVALIDATED": os.getenv(
                    "PROCESSOR_SKIP_SECURITY_WHEN_PREVALIDATED",
                    "true",
                ),
            }
            environment.update(processor_infra_env())

            container = self.client.containers.run(
                image=image_name,
                name=slot_name,
                command="sh -c 'uvicorn src.workers.document_processor.api_server:app --host 0.0.0.0 --port 3100'",
                detach=True,
                ports={"3100/tcp": host_port},
                volumes=volumes,
                environment=environment,
                network=self.docker_network,
                restart_policy={"Name": "unless-stopped"},
            )
            logger.info("Started processor slot container %s on host port %s (ID: %s)", slot_name, host_port, container.id[:12])
            return container
        except Exception as exc:
            logger.error("Failed to start slot container %s: %s", slot_name, exc)
            return None

    def stop_and_remove_container(self, container_id_or_name: str, timeout: int = 5) -> bool:
        """Stop and force remove a container."""
        if not self.client:
            return True
        try:
            container = self.client.containers.get(container_id_or_name)
            container.stop(timeout=timeout)
            container.remove(force=True)
            logger.info("Stopped and removed container %s", container_id_or_name)
            return True
        except NotFound:
            return True
        except Exception as exc:
            logger.error("Error stopping container %s: %s", container_id_or_name, exc)
            return False
