from __future__ import annotations

from enum import Enum
from typing import Set


class JobState(str, Enum):
    """Authoritative Lifecycle States for Document Jobs."""
    QUEUED = "QUEUED"
    DISPATCHING = "DISPATCHING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMED_OUT = "TIMED_OUT"
    STOPPED = "STOPPED"

    @classmethod
    def terminal_states(cls) -> Set[JobState]:
        return {cls.COMPLETED, cls.FAILED, cls.TIMED_OUT, cls.STOPPED}

    @property
    def is_terminal(self) -> bool:
        return self in self.terminal_states()

    @property
    def is_active(self) -> bool:
        return self in {JobState.DISPATCHING, JobState.ASSIGNED, JobState.IN_PROGRESS}


def is_valid_state_transition(current: JobState | str, target: JobState | str) -> bool:
    """Validate lifecycle state machine transition."""
    try:
        curr_enum = JobState(current) if isinstance(current, str) else current
        targ_enum = JobState(target) if isinstance(target, str) else target
    except ValueError:
        return False

    if curr_enum == targ_enum:
        return True

    if curr_enum.is_terminal:
        return False

    allowed: dict[JobState, Set[JobState]] = {
        JobState.QUEUED: {JobState.DISPATCHING, JobState.FAILED, JobState.STOPPED},
        JobState.DISPATCHING: {JobState.ASSIGNED, JobState.QUEUED, JobState.FAILED, JobState.STOPPED},
        JobState.ASSIGNED: {JobState.IN_PROGRESS, JobState.FAILED, JobState.TIMED_OUT, JobState.STOPPED},
        JobState.IN_PROGRESS: {JobState.COMPLETED, JobState.FAILED, JobState.TIMED_OUT, JobState.STOPPED},
    }

    return targ_enum in allowed.get(curr_enum, set())
