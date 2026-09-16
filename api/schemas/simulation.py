"""Wire shapes for the simulation engine (Design Document §11, §12).

A plan entry names its banner by (character, version, phase) - the same
identity the roadmap uses (§7) - and carries a *desired resulting
constellation*, never a copy count (§4.2, §12). The simulator derives copies
from the simulated account when the banner arrives (§10.4).

Results always carry `runs` and `seed`: an identical pair reproduces the
result exactly (§11 invariant 10), and a probability without them would look
like an exact value.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from api.jobs import SimulationJob
from api.schemas.domain import BannerModel, GoalModel, StrictModel
from simulation import (
    BannerAggregate,
    GoalProbability,
    PlannedSpend,
    SimulationResult,
    SpendPlan,
)


class PlannedSpendModel(StrictModel):
    """One banner's spending decision (§12)."""

    banner: BannerModel
    target_constellation: int = Field(
        ...,
        description=(
            "Desired resulting constellation for the banner's character "
            '("pursue C2" is 2), never a copy count (§4.2).'
        ),
    )
    budget: int = Field(
        ..., description="Maximum wishes this entry permits - a cap, not a commitment."
    )

    def to_domain(self) -> PlannedSpend:
        return PlannedSpend(
            banner=self.banner.to_domain(),
            target_constellation=self.target_constellation,
            budget=self.budget,
        )

    @classmethod
    def from_domain(cls, entry: PlannedSpend) -> "PlannedSpendModel":
        return cls(
            banner=BannerModel.from_domain(entry.banner),
            target_constellation=entry.target_constellation,
            budget=entry.budget,
        )


class SpendPlanModel(StrictModel):
    """A strategy as per-banner decisions (§12).

    A banner without an entry is skipped: processed chronologically for
    income and pity/guarantee, spending nothing (§11).
    """

    entries: list[PlannedSpendModel] = Field(default_factory=list)

    def to_domain(self) -> SpendPlan:
        return SpendPlan(entries=tuple(entry.to_domain() for entry in self.entries))

    @classmethod
    def from_domain(cls, plan: SpendPlan) -> "SpendPlanModel":
        return cls(
            entries=[
                PlannedSpendModel.from_domain(entry) for entry in plan.entries
            ]
        )


class SimulationRunRequest(StrictModel):
    """Submit a Monte Carlo run (§11)."""

    plan: SpendPlanModel
    runs: int = Field(10_000, description="Histories to simulate.")
    seed: int | None = Field(
        0, description="Rng seed; null draws entropy from the OS."
    )


class GoalProbabilityView(BaseModel):
    """How often one goal was satisfied across histories (§11)."""

    goal: GoalModel
    probability: float

    @classmethod
    def from_domain(cls, entry: GoalProbability) -> "GoalProbabilityView":
        return cls(
            goal=GoalModel.from_domain(entry.goal), probability=entry.probability
        )


class BannerAggregateView(BaseModel):
    """One banner's behavior across histories (§11)."""

    banner: BannerModel
    target_constellation: int | None
    planned_budget: int
    mean_income_credited: float
    mean_wishes_spent: float
    target_met_probability: float
    mean_copies_obtained: float

    @classmethod
    def from_domain(cls, aggregate: BannerAggregate) -> "BannerAggregateView":
        return cls(
            banner=BannerModel.from_domain(aggregate.banner),
            target_constellation=aggregate.target_constellation,
            planned_budget=aggregate.planned_budget,
            mean_income_credited=aggregate.mean_income_credited,
            mean_wishes_spent=aggregate.mean_wishes_spent,
            target_met_probability=aggregate.target_met_probability,
            mean_copies_obtained=aggregate.mean_copies_obtained,
        )


class SimulationResultView(BaseModel):
    """Aggregated outcome of many simulated histories (§11)."""

    runs: int
    seed: int | None
    plan: SpendPlanModel
    goals: list[GoalProbabilityView]
    banners: list[BannerAggregateView]
    all_goals_probability: float
    final_wishes_mean: float
    final_wishes_min: int
    final_wishes_max: int

    @classmethod
    def from_domain(cls, result: SimulationResult) -> "SimulationResultView":
        return cls(
            runs=result.runs,
            seed=result.seed,
            plan=SpendPlanModel.from_domain(result.plan),
            goals=[GoalProbabilityView.from_domain(goal) for goal in result.goals],
            banners=[
                BannerAggregateView.from_domain(banner) for banner in result.banners
            ],
            all_goals_probability=result.all_goals_probability,
            final_wishes_mean=result.final_wishes_mean,
            final_wishes_min=result.final_wishes_min,
            final_wishes_max=result.final_wishes_max,
        )


class SimulationJobView(BaseModel):
    """A submitted simulation run and, once finished, its result (§11)."""

    job_id: str
    account_id: str
    status: str = Field(
        ..., description='"queued", "running", "succeeded" or "failed".'
    )
    runs: int
    seed: int | None
    submitted_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    result: SimulationResultView | None
    error: str | None

    @classmethod
    def from_domain(cls, job: SimulationJob) -> "SimulationJobView":
        return cls(
            job_id=job.id,
            account_id=job.account_id,
            status=job.status.value,
            runs=job.runs,
            seed=job.seed,
            submitted_at=job.submitted_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            result=(
                None
                if job.result is None
                else SimulationResultView.from_domain(job.result)
            ),
            error=job.error,
        )
