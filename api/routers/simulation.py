"""Simulation endpoints (Design Document §11, §18 Phase 6).

    POST /accounts/{id}/simulation/run       submit a run
    GET  /accounts/{id}/simulation/{job_id}  poll it

Submission validates the plan synchronously - `SpendPlan.require_valid_for`
rejects entries for banners outside the roadmap or before the current one
(§12) - so an unexecutable plan is a 422 at submit time rather than a failed
job later. Only the sampling runs in the background.

The context and plan are captured at submission, so a run answers the
question that was asked: editing the account while a job is in flight cannot
change what that job simulated.

Jobs are visible only under their own account, and a result always carries
its `runs` and `seed` so it can be reproduced exactly (§11 invariant 10).
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from api.dependencies import ContextOverrides, context_overrides, get_jobs, get_record
from api.jobs import JobNotFound, SimulationJobStore
from api.repository import AccountRecord
from api.schemas.simulation import SimulationJobView, SimulationRunRequest
from simulation import simulate

router = APIRouter(tags=["simulation"])


@router.post(
    "/accounts/{account_id}/simulation/run",
    response_model=SimulationJobView,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a simulation run",
    description=(
        "Simulates the plan across `runs` possible futures (§11) and returns "
        "a queued job. The plan is validated immediately: every entry must "
        "name a roadmap banner at or after the current one (§12). Poll the "
        "job for the aggregated result."
    ),
)
def run_simulation(
    payload: SimulationRunRequest,
    background_tasks: BackgroundTasks,
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
    jobs: SimulationJobStore = Depends(get_jobs),
) -> SimulationJobView:
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    plan = payload.plan.to_domain()
    # Fail fast on a plan that could never be executed, rather than after
    # sampling thousands of histories (§12).
    plan.require_valid_for(context)
    if payload.runs < 1:
        raise ValueError(f"runs must be >= 1, got {payload.runs}")

    job = jobs.submit(record.id, runs=payload.runs, seed=payload.seed)
    background_tasks.add_task(
        jobs.run,
        job.id,
        lambda: simulate(context, plan, runs=payload.runs, seed=payload.seed),
    )
    return SimulationJobView.from_domain(job)


@router.get(
    "/accounts/{account_id}/simulation/{job_id}",
    response_model=SimulationJobView,
    summary="Read a simulation job",
    description=(
        "The job's status and, once it has succeeded, the aggregated result: "
        "per-goal satisfaction probabilities, per-banner spending behavior "
        "and end-of-history wish pools (§11)."
    ),
)
def read_simulation(
    job_id: str,
    record: AccountRecord = Depends(get_record),
    jobs: SimulationJobStore = Depends(get_jobs),
) -> SimulationJobView:
    try:
        job = jobs.get(job_id)
    except JobNotFound as missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(missing)
        ) from missing
    if job.account_id != record.id:
        # A job belongs to the account it was submitted under; leaking its
        # existence through a different account would be a cross-account read.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"no simulation job with id {job_id!r} for this account",
        )
    return SimulationJobView.from_domain(job)
