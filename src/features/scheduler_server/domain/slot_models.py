from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class SlotStatus(str, Enum):
    """Runtime Health & Activity Status of Processor Slots."""
    STARTING = "STARTING"
    HEALTHY_IDLE = "HEALTHY_IDLE"
    RESERVED = "RESERVED"
    BUSY = "BUSY"
    UNHEALTHY = "UNHEALTHY"
    STOPPING = "STOPPING"
    TERMINATED = "TERMINATED"


@dataclass
class ProcessorSlot:
    """Represents a managed processor container slot."""
    slot_id: str
    host_port: int
    container_id: Optional[str] = None
    container_name: Optional[str] = None
    status: SlotStatus = SlotStatus.STARTING
    health: str = "unhealthy"
    state: str = "idle"
    current_job_id: Optional[str] = None
    current_document_id: Optional[str] = None
    current_processor_type: Optional[str] = None
    assigned_at: Optional[datetime] = None
    released_at: Optional[datetime] = None
    last_released_document_id: Optional[str] = None
    last_health_check: Optional[datetime] = None
    failure_count: int = 0
    is_dynamic: bool = False  # True if created beyond MIN_PARALLEL_JOBS for dynamic scaling

    @property
    def is_available_for_job(self) -> bool:
        """Slot is ready only when healthy AND state is idle."""
        return (
            self.health == "healthy"
            and self.state == "idle"
            and self.status in {SlotStatus.HEALTHY_IDLE, SlotStatus.STARTING}
            and self.current_job_id is None
        )

    def mark_reserved(
        self,
        job_id: str,
        document_id: str,
        processor_type: Optional[str] = None,
    ) -> None:
        self.status = SlotStatus.RESERVED
        self.current_job_id = job_id
        self.current_document_id = document_id
        self.current_processor_type = processor_type
        self.assigned_at = datetime.utcnow()

    def mark_busy(
        self,
        job_id: str,
        document_id: str,
        processor_type: Optional[str] = None,
    ) -> None:
        self.status = SlotStatus.BUSY
        self.state = "busy"
        self.current_job_id = job_id
        self.current_document_id = document_id
        self.current_processor_type = processor_type
        self.assigned_at = datetime.utcnow()

    def mark_idle(self) -> None:
        self.status = SlotStatus.HEALTHY_IDLE
        self.state = "idle"
        self.released_at = datetime.utcnow()
        self.last_released_document_id = self.current_document_id
        self.current_job_id = None
        self.current_document_id = None
        self.current_processor_type = None
        self.assigned_at = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "host_port": self.host_port,
            "container_id": self.container_id,
            "container_name": self.container_name,
            "status": self.status.value,
            "health": self.health,
            "state": self.state,
            "current_job_id": self.current_job_id,
            "current_document_id": self.current_document_id,
            "current_processor_type": self.current_processor_type,
            "assigned_at": self.assigned_at.isoformat() if self.assigned_at else None,
            "released_at": self.released_at.isoformat() if self.released_at else None,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "is_dynamic": self.is_dynamic,
        }
