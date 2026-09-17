from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(REPO_ROOT / "deploy" / "application" / ".env", override=False)

from src.shared.networking.bind_paths import ensure_host_bind_path_exists, host_repo_root, normalize_bind_mount_host_path
from src.shared.networking.document_paths import processor_documents_container_dir
from src.shared.networking.repo_root import resolve_document_host_path, resolve_repository_root
from src.shared.networking.hosts import detect_deployment_mode, infra_hosts_for_mode

from src.features.scheduler_server.infrastructure.docker_manager import DockerServiceManager
from src.features.scheduler_server.infrastructure.job_repository import SchedulerJobRepository
from src.features.scheduler_server.application.pool_manager import ProcessorPoolManager
from src.features.scheduler_server.application.queue_consumer import RabbitMQQueueConsumer
from src.features.scheduler_server.application.health_monitor import ContainerHealthMonitor
from src.features.scheduler_server.api.tcp_server import TCPServerHandler
from src.features.scheduler_server.scheduler_config import SchedulerRuntimeConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SCHEDULER_DISPLAY_NAME = "rag scheduler daemon"


def _configure_process_display_name(title: str = SCHEDULER_DISPLAY_NAME) -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(title)
        except Exception:
            pass


class SchedulerDaemon:
    """Main Daemon for the Scheduler Server Feature."""

    def __init__(self):
        infra = infra_hosts_for_mode(detect_deployment_mode())
        _running_in_docker = os.getenv("RUNNING_IN_DOCKER", "").lower() in {"1", "true", "yes"} or os.path.exists("/.dockerenv")
        default_processor_http_mode = "docker_network" if _running_in_docker else "localhost"

        repo_root = resolve_repository_root(REPO_ROOT)
        self.runtime = SchedulerRuntimeConfig.from_env()
        self.config = {
            "port_range_start": self.runtime.port_range_start,
            "port_range_end": self.runtime.port_range_end,
            "min_parallel_jobs": self.runtime.min_parallel_jobs,
            "max_parallel_jobs": self.runtime.max_parallel_jobs,
            "processor_image": os.getenv("PROCESSOR_IMAGE_NAME", "rag-processor:latest"),
            "queue_check_freq": self.runtime.queue_check_frequency,
            "container_poll_freq": self.runtime.container_poll_frequency,
            "job_timeout_seconds": self.runtime.job_timeout_seconds,
            "rabbitmq": {
                "host": os.getenv("RABBITMQ_HOST") or infra["rabbitmq"],
                "port": int(os.getenv("RABBITMQ_PORT", 5672)),
                "user": os.getenv("RABBITMQ_USER", "rabbitmq_user"),
                "pass": os.getenv("RABBITMQ_PASS", "rabbitmq_password"),
                "queue_name": os.getenv("RABBITMQ_QUEUE_NAME", "document_processing_queue"),
            },
            "docker_host": os.getenv("DOCKER_HOST"),
            "tcp_port": int(os.getenv("SCHEDULER_TCP_PORT", 3200)),
            "document_host_path": resolve_document_host_path(repo_root / "data" / "documents"),
            "document_container_path": processor_documents_container_dir(),
            "models_host_path": normalize_bind_mount_host_path(
                os.getenv("SCHEDULER_MODELS_HOST_PATH") or str(repo_root / "models"),
                repo_root=host_repo_root(),
            ),
            "models_container_path": os.getenv("SCHEDULER_MODELS_CONTAINER_PATH", "/app/src/models"),
            "processor_http_mode": os.getenv("SCHEDULER_PROCESSOR_HTTP_MODE", default_processor_http_mode),
            "processor_http_host": os.getenv("SCHEDULER_PROCESSOR_HTTP_HOST", "localhost"),
            "docker_network": os.getenv("SCHEDULER_DOCKER_NETWORK", "docnet"),
        }

        ensure_host_bind_path_exists(self.config["document_host_path"])
        ensure_host_bind_path_exists(self.config["models_host_path"])

        logger.info(
            "Scheduler initialized: parallel=%d-%d ports=%d-%d queue_poll=%ds health_poll=%ds scan_limit=%d",
            self.config["min_parallel_jobs"],
            self.config["max_parallel_jobs"],
            self.config["port_range_start"],
            self.config["port_range_end"],
            self.runtime.queue_check_frequency,
            self.runtime.container_poll_frequency,
            self.runtime.queue_scan_limit,
        )

        # Initialize core components
        self.docker_mgr = DockerServiceManager(
            docker_host=self.config["docker_host"],
            processor_http_mode=self.config["processor_http_mode"],
            processor_http_host=self.config["processor_http_host"],
            docker_network=self.config["docker_network"],
            processor_health_timeout=self.runtime.processor_health_timeout,
        )
        self.job_repo = SchedulerJobRepository()
        self.pool_mgr = ProcessorPoolManager(
            docker_mgr=self.docker_mgr,
            job_repo=self.job_repo,
            port_range=(self.config["port_range_start"], self.config["port_range_end"]),
            min_parallel_jobs=self.config["min_parallel_jobs"],
            max_parallel_jobs=self.config["max_parallel_jobs"],
            document_host_path=self.config["document_host_path"],
            document_container_path=self.config["document_container_path"],
            models_host_path=self.config["models_host_path"],
            models_container_path=self.config["models_container_path"],
            processor_image=self.config["processor_image"],
            unhealthy_restart_threshold=self.runtime.unhealthy_restart_threshold,
        )
        self.queue_consumer = RabbitMQQueueConsumer(
            rabbitmq_config=self.config["rabbitmq"],
            pool_mgr=self.pool_mgr,
            job_repo=self.job_repo,
            check_frequency=self.config["queue_check_freq"],
            queue_scan_limit=self.runtime.queue_scan_limit,
            dispatch_sla_seconds=self.runtime.dispatch_sla_seconds,
            queue_priority_batch=self.runtime.queue_priority_batch,
        )
        self.pool_mgr.set_requeue_callback(self.queue_consumer.republish_job_by_id)
        self.health_monitor = ContainerHealthMonitor(
            pool_mgr=self.pool_mgr,
            job_repo=self.job_repo,
            runtime_config=self.runtime,
            config_refresh_callback=self.refresh_platform_config,
            queue_consumer=self.queue_consumer,
        )
        self.tcp_server = TCPServerHandler(
            host="0.0.0.0",
            port=self.config["tcp_port"],
            pool_mgr=self.pool_mgr,
            job_repo=self.job_repo,
            shutdown_trigger=self.request_shutdown,
        )
        self._shutdown_event = asyncio.Event()

    async def refresh_platform_config(self) -> None:
        """Apply hot-reloaded platform settings."""
        pass

    def request_shutdown(self) -> None:
        """Trigger graceful daemon shutdown."""
        logger.info("Initiating graceful shutdown sequence...")
        self._shutdown_event.set()

    async def run(self) -> None:
        """Run all daemon tasks concurrently until shutdown signal."""
        _configure_process_display_name()
        logger.info("Starting Scheduler Server Daemon on TCP port %d...", self.config["tcp_port"])

        # Recover stale jobs interrupted by a previous restart (CH-03 resilience)
        try:
            self.job_repo.recover_stale_dispatching_jobs(timeout_seconds=0)
            self.job_repo.recover_stale_in_progress_jobs(timeout_seconds=self.config["job_timeout_seconds"])
            self.pool_mgr.reconcile_pool()

            live_job_ids: set[str] = set()
            for slot in self.pool_mgr.slots.values():
                if slot.current_job_id:
                    live_job_ids.add(str(slot.current_job_id))
                if slot.current_document_id:
                    live_job_ids.add(str(slot.current_document_id))

            self.job_repo.recover_stale_assigned_jobs(
                timeout_seconds=300,
                exclude_job_ids=live_job_ids,
            )
            orphan_grace = int(os.getenv("SCHEDULER_ORPHAN_GRACE_SECONDS", "30"))
            orphaned = self.job_repo.recover_orphaned_active_jobs(
                live_job_ids,
                heartbeat_grace_seconds=orphan_grace,
                recently_released_document_ids=set(),
            )
            for jid in orphaned:
                record = self.job_repo.get_job_record(jid)
                did = str((record or {}).get("document_id") or jid)
                self.queue_consumer.republish_job_by_id(jid, did)
        except Exception as exc:
            logger.warning("Startup stale job recovery warning: %s", exc)

        # Start TCP server and async background loops
        await self.tcp_server.start()
        queue_task = asyncio.create_task(self.queue_consumer.start_polling())
        health_task = asyncio.create_task(self.health_monitor.start_monitoring())

        logger.info("Scheduler Daemon fully operational.")
        await self._shutdown_event.wait()

        # Graceful shutdown sequence
        logger.info("Executing graceful shutdown...")
        self.queue_consumer.stop()
        self.health_monitor.stop()

        queue_task.cancel()
        health_task.cancel()

        await self.tcp_server.stop()
        self.pool_mgr.stop_all_slots()
        logger.info("Scheduler Daemon shutdown complete.")


def main() -> None:
    daemon = SchedulerDaemon()
    try:
        asyncio.run(daemon.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler process terminated.")


if __name__ == "__main__":
    main()
