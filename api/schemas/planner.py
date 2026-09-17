"""Wire shapes for the planner and optimizer (Design Document §9, §13-§15).

Every view here is a translation of a planner or optimizer dataclass. The
field meanings live in those dataclasses' docstrings and are not restated
or reinterpreted: a `budget_at_banner` on the wire means exactly what
`planner.ProtectedGoalOutcome.budget_at_banner` means.

Sampled numbers always travel with their provenance (`runs`, `seed`) so no
client can mistake a Monte Carlo estimate for an exact value (§2, §11
invariant 10).
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
from planner import GoalEvaluation, ProtectedGoalOutcome, SpendRow


class GoalEvaluationView(BaseModel):
    """One goal's planner state (§9)."""

    goal: GoalModel
    copies_needed: int
    state: str = Field(..., description='"satisfied", "active" or "blocked".')
    blocked_by: GoalModel | None
    next_banner: BannerModel | None = Field(
        None,
        description=(
            "First banner for this character at or after the current banner; "
            "null when the roadmap schedules none - the goal exists but is "
            "not schedulable (§8)."
        ),
    )
    relevant: bool = Field(
        ..., description="Matches the current banner's featured character (§9)."
    )
    actionable: bool = Field(
        ..., description="Relevant and active: what can be pursued now (§9)."
    )

    @classmethod
    def from_domain(
        cls, evaluation: GoalEvaluation, *, relevant: bool, actionable: bool
    ) -> "GoalEvaluationView":
        return cls(
            goal=GoalModel.from_domain(evaluation.goal),
            copies_needed=evaluation.copies_needed,
            state=evaluation.state.value,
            blocked_by=(
                None
                if evaluation.blocked_by is None
                else GoalModel.from_domain(evaluation.blocked_by)
            ),
            next_banner=(
                None
                if evaluation.next_banner is None
                else BannerModel.from_domain(evaluation.next_banner)
            ),
            relevant=relevant,
            actionable=actionable,
        )


class GoalEvaluationsView(BaseModel):
    """Every roadmap goal in priority order, against the current banner (§9)."""

    current_banner: BannerModel
    goals: list[GoalEvaluationView]


class ProtectedGoalOutcomeView(BaseModel):
    """One protected goal under the Phase 3 approximation (§13 step 5, §14)."""

    goal: GoalModel
    banner: BannerModel
    budget_at_banner: int
    required_wishes: int
    confidence: float
    meets_threshold: bool

    @classmethod
    def from_domain(
        cls, outcome: ProtectedGoalOutcome
    ) -> "ProtectedGoalOutcomeView":
        return cls(
            goal=GoalModel.from_domain(outcome.goal),
            banner=BannerModel.from_domain(outcome.banner),
            budget_at_banner=outcome.budget_at_banner,
            required_wishes=outcome.required_wishes,
            confidence=outcome.confidence,
            meets_threshold=outcome.meets_threshold,
        )


class SafeSpendView(BaseModel):
    """The safe-spend approximation and what it is protecting (§14).

    `safe_spend` is the sequential independent-reserve approximation, not a
    simulated boundary: `planner.safe_spend` documents the assumptions, and
    the optimizer's recommendation is the simulation-based answer (§14).
    """

    current_banner: BannerModel
    safe_spend: int
    account_wishes: int
    confidence: float
    approximation: str = Field(
        "sequential independent reserves (Phase 3); the recommendation "
        "endpoint evaluates spending through simulation (§14)",
        description="How this number was produced.",
    )
    protected: list[ProtectedGoalOutcomeView]


class SpendRowView(BaseModel):
    """One candidate spend on the current banner (§14)."""

    wishes_spent: int
    goal_confidence: float
    all_protected_meet_threshold: bool
    protected: list[ProtectedGoalOutcomeView]

    @classmethod
    def from_domain(cls, row: SpendRow) -> "SpendRowView":
        return cls(
            wishes_spent=row.wishes_spent,
            goal_confidence=row.goal_confidence,
            all_protected_meet_threshold=row.all_protected_meet_threshold,
            protected=[
                ProtectedGoalOutcomeView.from_domain(outcome)
                for outcome in row.protected
            ],
        )


class SpendTableView(BaseModel):
    """The spending table for the current banner's single active goal (§14)."""

    current_banner: BannerModel
    goal: GoalModel
    copies_needed: int
    step: int
    confidence: float
    rows: list[SpendRowView]


class OutcomeView(BaseModel):
    """One outcome the optimizer may pursue (§13 step 2, §15)."""

    character: str
    constellation: int = Field(
        ..., description="Desired resulting constellation, never a copy count."
    )
    rank: int
    weapon_refinement: int | None
    label: str

    @classmethod
    def from_domain(cls, outcome: OutcomeOption) -> "OutcomeView":
        return cls(
            character=outcome.character,
            constellation=outcome.constellation,
            rank=outcome.rank,
            weapon_refinement=outcome.weapon_refinement,
            label=outcome.label,
        )


class GoalStandingView(BaseModel):
    """One protected goal's simulated standing (§13 steps 4-5)."""

    goal: GoalModel
    banner: BannerModel
    probability: float
    meets_threshold: bool
    constraining: bool = Field(
        ...,
        description=(
            "Whether this goal's threshold gates the decision: its priority "
            "outranks the current banner's goal (§2). A non-constraining "
            "goal is reported, not ignored - it is still pursued, but a "
            "lower-priority goal cannot veto spending on a higher-priority "
            "current one."
        ),
    )

    @classmethod
    def from_domain(cls, standing: GoalStanding) -> "GoalStandingView":
        return cls(
            goal=GoalModel.from_domain(standing.goal),
            banner=BannerModel.from_domain(standing.banner),
            probability=standing.probability,
            meets_threshold=standing.meets_threshold,
            constraining=standing.constraining,
        )


class RejectedOutcomeView(BaseModel):
    """A more-preferred outcome that failed feasibility (§13 step 7)."""

    outcome: OutcomeView
    best_budget: int = Field(
        ..., description="The cap of the candidate closest to feasible."
    )
    outcome_probability: float
    min_protected_probability: float | None = Field(
        None,
        description=(
            "The weakest *constraining* protected standing of the closest "
            "candidate; null when nothing outranks the decision."
        ),
    )
    shortfalls: list[GoalStandingView] = Field(
        ...,
        description=(
            "Constraining protected goals still below the threshold - the why."
        ),
    )

    @classmethod
    def from_domain(cls, rejected: RejectedOutcome) -> "RejectedOutcomeView":
        return cls(
            outcome=OutcomeView.from_domain(rejected.outcome),
            best_budget=rejected.best.budget,
            outcome_probability=rejected.best.outcome_probability,
            min_protected_probability=rejected.best.min_protected_probability,
            shortfalls=[
                GoalStandingView.from_domain(standing)
                for standing in rejected.shortfalls
            ],
        )


class StopConditionsView(BaseModel):
    """When to stop pursuing, and what to do then (§1)."""

    action: str
    outcome_label: str | None
    spend_cap: int
    rules: list[str]

    @classmethod
    def from_domain(cls, stops: StopConditions) -> "StopConditionsView":
        return cls(
            action=stops.action,
            outcome_label=stops.outcome_label,
            spend_cap=stops.spend_cap,
            rules=list(stops.rules),
        )


class RecommendationView(BaseModel):
    """The planner's decision at the current banner (§1, §13, §20)."""

    banner: BannerModel
    action: str = Field(..., description='"pursue" or "skip".')
    outcome: OutcomeView | None
    budget: int = Field(
        ..., description="Largest feasible spend cap - a cap, not a commitment (§12)."
    )
    plan: SpendPlanModel | None
    outcome_probability: float
    confidence: float = Field(
        ..., description="The threshold the decision was made against."
    )
    protected: list[GoalStandingView]
    rejected: list[RejectedOutcomeView]
    skip_reason: str | None
    stops: StopConditionsView
    runs: int
    seed: int | None

    @classmethod
    def from_domain(
        cls, recommendation: Recommendation, *, confidence: float
    ) -> "RecommendationView":
        return cls(
            banner=BannerModel.from_domain(recommendation.banner),
            action=recommendation.action,
            outcome=(
                None
                if recommendation.outcome is None
                else OutcomeView.from_domain(recommendation.outcome)
            ),
            budget=recommendation.budget,
            plan=(
                None
                if recommendation.plan is None
                else SpendPlanModel.from_domain(recommendation.plan)
            ),
            outcome_probability=recommendation.outcome_probability,
            confidence=confidence,
            protected=[
                GoalStandingView.from_domain(standing)
                for standing in recommendation.protected
            ],
            rejected=[
                RejectedOutcomeView.from_domain(rejected)
                for rejected in recommendation.rejected
            ],
            skip_reason=recommendation.skip_reason,
            stops=StopConditionsView.from_domain(recommendation.stops),
            runs=recommendation.runs,
            seed=recommendation.seed,
        )
