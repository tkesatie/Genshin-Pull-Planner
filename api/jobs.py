"""Simulation jobs (Design Document §11, §18 Phase 6).

A Monte Carlo run is the one planner operation that is genuinely slow: the
default 10,000 histories (§11) are seconds of work, not milliseconds. The
design document's endpoint list reflects that with a submit/poll pair:

    POST /accounts/{id}/simulation/run   ->  job id
    GET  /accounts/{id}/simulation/{job_id}

Two rules keep the contract honest:

* the *plan* is validated synchronously at submission
  (`SpendPlan.require_valid_for`, §12), so an unexecutable plan fails
  immediately rather than after sampling;
* the job carries `runs` and `seed`, so a result is always reproducible and
  never mistaken for an exact value (§2, §11 invariant 10).

The store is in-process and non-durable: jobs are lost on restart. That is
deliberate for Phase 6 - a queue and worker replace `SimulationJobStore`
without changing the HTTP contract, the same way a database replaces
`InMemoryAccountRepository`.
"""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from simulation import SimulationResult


class JobNotFound(LookupError):
    """Raised when a job id is unknown; routers map this to 404."""


class JobStatus(str, Enum):
    """Lifecycle of one simulation job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class SimulationJob:
    """One submitted simulation run (§11).

    Attributes:
        id: server-assigned job identifier.
        account_id: the account the run belongs to; a job is only visible
            under its own account.
        runs: how many histories were requested.
        seed: the rng seed (None draws entropy from the OS).
        status: see JobStatus.
        submitted_at / started_at / finished_at: UTC timestamps.
        result: the aggregated simulation, present only when SUCCEEDED.
        error: the failure message, present only when FAILED.
    """

    id: str
    account_id: str
    runs: int
    seed: int | None
    status: JobStatus
    submitted_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    result: SimulationResult | None = None
    error: str | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class SimulationJobStore:
    """Process-local job storage and execution (§18 Phase 6).

    `run` is what a background task calls: it moves the job through its
    lifecycle and captures a failure as job state rather than letting it
    escape into a background task where no client could ever see it.
    """

    _jobs: dict[str, SimulationJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def submit(
        self, account_id: str, runs: int, seed: int | None
    ) -> SimulationJob:
        """Record a queued job and return it."""
        job = SimulationJob(
            id=uuid4().hex,
            account_id=account_id,
            runs=runs,
            seed=seed,
            status=JobStatus.QUEUED,
            submitted_at=_now(),
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> SimulationJob:
        """One job.

        Raises:
            JobNotFound: if the job id is unknown.
        """
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise JobNotFound(f"no simulation job with id {job_id!r}")
        return job

    def _update(self, job_id: str, **changes: object) -> SimulationJob:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise JobNotFound(f"no simulation job with id {job_id!r}")
            updated = replace(job, **changes)
            self._jobs[job_id] = updated
        return updated

    def run(self, job_id: str, work: Callable[[], SimulationResult]) -> None:
        """Execute `work` for a queued job and store its outcome.

        A raised exception becomes FAILED with its message; nothing escapes
        into the background task, where the client could never observe it.
        """
        self._update(job_id, status=JobStatus.RUNNING, started_at=_now())
        try:
            result = work()
        except Exception as error:  # noqa: BLE001 - recorded, not swallowed
            self._update(
                job_id,
                status=JobStatus.FAILED,
                finished_at=_now(),
                error=str(error) or error.__class__.__name__,
            )
            return
        self._update(
            job_id,
            status=JobStatus.SUCCEEDED,
            finished_at=_now(),
            result=result,
        )
