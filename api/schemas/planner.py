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
    PullStrategy,
    StrategyStep,
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




class CachedGoalProbabilityView(BaseModel):
    goal: GoalModel
    probability: float


class CachedGoalEvidenceView(BaseModel):
    goal: GoalModel
    probability: float
    goal_probabilities: list[CachedGoalProbabilityView]
    joint_probability: float | None
    runs: int
    safe_spend_remaining: int | None
    protected_probability: float | None
    protected_starting_wishes: int | None
    protected_total_wishes: int | None


class CachedPlannerRefreshView(BaseModel):
    account_wishes: int
    confidence: float
    evidence: list[CachedGoalEvidenceView]
    recommendation_budget: int | None = None
    recommendation_constellation: int | None = None
    recommendation_probability: float | None = None
    recommendation_goal_probabilities: list[CachedGoalProbabilityView] = Field(default_factory=list)


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


class StrategyStepView(BaseModel):
    action: str
    goal: GoalModel
    banner: BannerModel
    reserve_wishes: int | None = None
    safe_spend: int | None = None
    outcome_probability: float | None = None
    protected_probability: float | None = None
    future_income: int | None = None
    protected_starting_wishes: int | None = None
    protected_total_wishes: int | None = None

    @classmethod
    def from_domain(cls, step: StrategyStep):
        return cls(
            action=step.action,
            goal=GoalModel.from_domain(step.goal),
            banner=BannerModel.from_domain(step.banner),
            reserve_wishes=step.reserve_wishes,
            safe_spend=step.safe_spend,
            outcome_probability=step.outcome_probability,
            protected_probability=step.protected_probability,
            future_income=step.future_income,
            protected_starting_wishes=step.protected_starting_wishes,
            protected_total_wishes=step.protected_total_wishes,
        )


class PullStrategyView(BaseModel):
    steps: list[StrategyStepView]
    reserve_goal: GoalModel | None
    reserve_wishes: int | None
    reserve_probability: float | None
    confidence: float
    runs: int
    seed: int | None
    starting_wishes: int
    future_income: int

    @classmethod
    def from_domain(cls, strategy: PullStrategy, *, confidence: float):
        return cls(
            steps=[StrategyStepView.from_domain(step) for step in strategy.steps],
            reserve_goal=(
                None
                if strategy.reserve_goal is None
                else GoalModel.from_domain(strategy.reserve_goal)
            ),
            reserve_wishes=strategy.reserve_wishes,
            reserve_probability=strategy.reserve_probability,
            confidence=confidence,
            runs=strategy.runs,
            seed=strategy.seed,
            starting_wishes=strategy.starting_wishes,
            future_income=strategy.future_income,
        )


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
    all_goals_probability: float = Field(
        ...,
        description=(
            "Fraction of simulated histories in which EVERY roadmap goal ended "
            "satisfied under this recommendation - not just the protected ones. "
            "Under a skip, this is the do-nothing baseline's own roadmap-wide "
            "figure. Not the product of the individual protected probabilities: "
            "goals are not independent across one shared history."
        ),
    )
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
            all_goals_probability=recommendation.all_goals_probability,
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
