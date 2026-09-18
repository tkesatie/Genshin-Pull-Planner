"""Wire shapes for the planner and optimizer.

Sampled numbers always travel with their provenance (runs, seed).
"""

from pydantic import BaseModel, Field

from api.schemas.domain import BannerModel, GoalModel
from api.schemas.simulation import SpendPlanModel
from optimizer import (
    GoalStanding,
    OutcomeOption,
    Recommendation,
    RejectedOutcome,
    StopConditions,
)
from planner import GoalEvaluation, ProtectedGoalOutcome


class GoalEvaluationView(BaseModel):
    """One goal's planner state."""

    goal: GoalModel
    copies_needed: int
    state: str = Field(..., description='"satisfied", "active" or "blocked".')
    blocked_by: GoalModel | None
    next_banner: BannerModel | None
    relevant: bool
    actionable: bool

    @classmethod
    def from_domain(cls, evaluation: GoalEvaluation, *, relevant: bool, actionable: bool):
        return cls(
            goal=GoalModel.from_domain(evaluation.goal),
            copies_needed=evaluation.copies_needed,
            state=evaluation.state.value,
            blocked_by=None if evaluation.blocked_by is None else GoalModel.from_domain(evaluation.blocked_by),
            next_banner=None if evaluation.next_banner is None else BannerModel.from_domain(evaluation.next_banner),
            relevant=relevant,
            actionable=actionable,
        )


class GoalEvaluationsView(BaseModel):
    current_banner: BannerModel | None
    available_banners: list[BannerModel]
    goals: list[GoalEvaluationView]


class ProtectedGoalOutcomeView(BaseModel):
    goal: GoalModel
    banner: BannerModel
    budget_at_banner: int
    required_wishes: int
    confidence: float
    meets_threshold: bool

    @classmethod
    def from_domain(cls, outcome: ProtectedGoalOutcome):
        return cls(
            goal=GoalModel.from_domain(outcome.goal),
            banner=BannerModel.from_domain(outcome.banner),
            budget_at_banner=outcome.budget_at_banner,
            required_wishes=outcome.required_wishes,
            confidence=outcome.confidence,
            meets_threshold=outcome.meets_threshold,
        )


class SafeSpendView(BaseModel):
    current_banner: BannerModel | None
    available_banners: list[BannerModel]
    safe_spend: int
    account_wishes: int
    confidence: float
    approximation: str = Field(
        "sequential independent reserves (Phase 3); the recommendation endpoint evaluates spending through simulation",
        description="How this number was produced.",
    )
    protected: list[ProtectedGoalOutcomeView]


class OutcomeView(BaseModel):
    """One current-banner target."""

    character: str
    constellation: int = Field(..., description="Desired resulting constellation.")
    rank: int
    weapon_refinement: int | None
    label: str

    @classmethod
    def from_domain(cls, outcome: OutcomeOption):
        return cls(
            character=outcome.character,
            constellation=outcome.constellation,
            rank=outcome.rank,
            weapon_refinement=outcome.weapon_refinement,
            label=outcome.label,
        )


class OutcomeProbabilityView(BaseModel):
    """Probability of reaching one current-banner constellation milestone."""

    outcome: OutcomeView
    probability: float


class SpendRowView(BaseModel):
    """One simulated current-banner spend scenario."""

    wishes_spent: int
    outcomes: list[OutcomeProbabilityView]
    protected: list["GoalStandingView"]
    all_protected_meet_threshold: bool


class SpendTableView(BaseModel):
    """How spending changes current-banner milestones and future protection."""

    current_banner: BannerModel
    outcomes: list[OutcomeView]
    step: int
    confidence: float
    runs: int
    seed: int | None
    rows: list[SpendRowView]


class GoalStandingView(BaseModel):
    goal: GoalModel
    banner: BannerModel
    probability: float
    meets_threshold: bool
    constraining: bool = Field(
        ...,
        description="Whether this goal's threshold gates the decision.",
    )

    @classmethod
    def from_domain(cls, standing: GoalStanding):
        return cls(
            goal=GoalModel.from_domain(standing.goal),
            banner=BannerModel.from_domain(standing.banner),
            probability=standing.probability,
            meets_threshold=standing.meets_threshold,
            constraining=standing.constraining,
        )


class RejectedOutcomeView(BaseModel):
    outcome: OutcomeView
    best_budget: int
    outcome_probability: float
    min_protected_probability: float | None = None
    shortfalls: list[GoalStandingView]

    @classmethod
    def from_domain(cls, rejected: RejectedOutcome):
        return cls(
            outcome=OutcomeView.from_domain(rejected.outcome),
            best_budget=rejected.best.budget,
            outcome_probability=rejected.best.outcome_probability,
            min_protected_probability=rejected.best.min_protected_probability,
            shortfalls=[GoalStandingView.from_domain(standing) for standing in rejected.shortfalls],
        )


class StopConditionsView(BaseModel):
    action: str = Field(..., description='"pursue", "discretionary" or "skip".')
    outcome_label: str | None
    spend_cap: int
    rules: list[str]

    @classmethod
    def from_domain(cls, stops: StopConditions):
        return cls(
            action=stops.action,
            outcome_label=stops.outcome_label,
            spend_cap=stops.spend_cap,
            rules=list(stops.rules),
        )


class RecommendationView(BaseModel):
    banner: BannerModel
    action: str = Field(..., description='"pursue", "discretionary" or "skip".')
    outcome: OutcomeView | None
    budget: int
    plan: SpendPlanModel | None
    outcome_probability: float
    confidence: float
    minimum_outcome_probability: float
    protected: list[GoalStandingView]
    rejected: list[RejectedOutcomeView]
    skip_reason: str | None
    discretionary_reason: str | None = None
    stops: StopConditionsView
    runs: int
    seed: int | None

    @classmethod
    def from_domain(cls, recommendation: Recommendation, *, confidence: float, minimum_outcome_probability: float):
        return cls(
            banner=BannerModel.from_domain(recommendation.banner),
            action=recommendation.action,
            outcome=None if recommendation.outcome is None else OutcomeView.from_domain(recommendation.outcome),
            budget=recommendation.budget,
            plan=None if recommendation.plan is None else SpendPlanModel.from_domain(recommendation.plan),
            outcome_probability=recommendation.outcome_probability,
            confidence=confidence,
            minimum_outcome_probability=minimum_outcome_probability,
            protected=[GoalStandingView.from_domain(standing) for standing in recommendation.protected],
            rejected=[RejectedOutcomeView.from_domain(rejected) for rejected in recommendation.rejected],
            skip_reason=recommendation.skip_reason,
            discretionary_reason=recommendation.discretionary_reason,
            stops=StopConditionsView.from_domain(recommendation.stops),
            runs=recommendation.runs,
            seed=recommendation.seed,
        )
