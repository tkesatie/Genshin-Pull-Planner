"""Planner endpoints."""

from fastapi import APIRouter, Depends, Query

from api.dependencies import ContextOverrides, context_overrides, get_record
from api.repository import AccountRecord
from api.schemas.domain import BannerModel, GoalModel
from api.schemas.planner import (
    GoalEvaluationsView,
    GoalEvaluationView,
    GoalStandingView,
    OutcomeProbabilityView,
    OutcomeView,
    ProtectedGoalOutcomeView,
    RecommendationView,
    SafeSpendView,
    SpendRowView,
    SpendTableView,
    StopConditionsView,
)
from optimizer import (
    DEFAULT_RUNS,
    MINIMUM_OUTCOME_PROBABILITY,
    OutcomeOption,
    available_outcomes,
    evaluate_candidate,
    recommend,
)
from planner import (
    actionable_goals,
    current_banner,
    evaluate_goals,
    protected_goal_outcomes,
    relevant_goal_evaluations,
    safe_spend,
)
from simulation import DEFAULT_SEED

router = APIRouter(tags=["planner"])


@router.get("/accounts/{account_id}/planner/goals", response_model=GoalEvaluationsView)
def planner_goals(
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> GoalEvaluationsView:
    context = record.context(confidence=overrides.confidence, income_scenario=overrides.income_scenario)
    relevant = {evaluation.goal for evaluation in relevant_goal_evaluations(context)}
    actionable = {evaluation.goal for evaluation in actionable_goals(context)}
    return GoalEvaluationsView(
        current_banner=BannerModel.from_domain(current_banner(context)),
        goals=[
            GoalEvaluationView.from_domain(
                evaluation,
                relevant=evaluation.goal in relevant,
                actionable=evaluation.goal in actionable,
            )
            for evaluation in evaluate_goals(context)
        ],
    )


@router.get("/accounts/{account_id}/planner/safe-spend", response_model=SafeSpendView)
def planner_safe_spend(
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> SafeSpendView:
    context = record.context(confidence=overrides.confidence, income_scenario=overrides.income_scenario)
    return SafeSpendView(
        current_banner=BannerModel.from_domain(current_banner(context)),
        safe_spend=safe_spend(context),
        account_wishes=context.account.wishes,
        confidence=context.confidence,
        protected=[
            ProtectedGoalOutcomeView.from_domain(outcome)
            for outcome in protected_goal_outcomes(context, spent=0)
        ],
    )


@router.get(
    "/accounts/{account_id}/planner/spend-table",
    response_model=SpendTableView,
    summary="How spending changes current-banner milestones",
    description=(
        "Shows how candidate current-banner spending changes the probability "
        "of reaching each constellation milestone offered for the current "
        "banner, alongside the simulated probability of every protected "
        "future goal."
    ),
)
def planner_spend_table(
    step: int = Query(10, description="Spend increment between rows."),
    runs: int = Query(DEFAULT_RUNS, description="Histories per candidate."),
    seed: int | None = Query(DEFAULT_SEED, description="Rng seed; null draws entropy from the OS."),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> SpendTableView:
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")

    outcomes = _spend_table_outcomes(context, record.preferences)
    if not outcomes:
        return SpendTableView(
            current_banner=BannerModel.from_domain(current_banner(context)),
            outcomes=[],
            step=step,
            confidence=context.confidence,
            runs=runs,
            seed=seed,
            rows=[],
        )

    budgets = list(range(0, context.account.wishes + 1, step))
    if budgets[-1] != context.account.wishes:
        budgets.append(context.account.wishes)

    rows = []
    for budget in budgets:
        outcome_probabilities = [
            OutcomeProbabilityView(
                outcome=OutcomeView.from_domain(outcome),
                probability=evaluate_candidate(
                    context, outcome, budget, runs=runs, seed=seed
                ).outcome_probability,
            )
            for outcome in outcomes
        ]

        # Use the same protected-goal evaluation for every milestone. The
        # future-roadmap tradeoff does not depend on which current-banner
        # constellation column the user is looking at.
        reference_candidate = evaluate_candidate(
            context, outcomes[0], budget, runs=runs, seed=seed
        )
        rows.append(
            SpendRowView(
                wishes_spent=budget,
                outcomes=outcome_probabilities,
                protected=[
                    GoalStandingView.from_domain(standing)
                    for standing in reference_candidate.protected
                ],
                all_protected_meet_threshold=all(
                    standing.meets_threshold
                    for standing in reference_candidate.protected
                    if standing.constraining
                ),
            )
        )

    return SpendTableView(
        current_banner=BannerModel.from_domain(current_banner(context)),
        outcomes=[OutcomeView.from_domain(outcome) for outcome in outcomes],
        step=step,
        confidence=context.confidence,
        runs=runs,
        seed=seed,
        rows=rows,
    )


def _spend_table_outcomes(context, preferences) -> tuple[OutcomeOption, ...]:
    """Return unique current-banner constellation milestones for the table.

    The optimizer's outcome list can contain weapon-labelled variants of the
    same constellation. The spend analysis is character-only, so the table
    collapses those variants and reports the probability of each resulting
    constellation, from C0 upward.
    """
    available = available_outcomes(context, preferences)
    by_constellation: dict[int, OutcomeOption] = {}
    for outcome in available:
        by_constellation.setdefault(
            outcome.constellation,
            OutcomeOption(
                character=outcome.character,
                constellation=outcome.constellation,
                rank=outcome.rank,
            ),
        )
    return tuple(sorted(by_constellation.values(), key=lambda outcome: outcome.constellation))


def _recommendation(
    record: AccountRecord,
    overrides: ContextOverrides,
    runs: int,
    seed: int | None,
    budgets: list[int] | None,
    minimum_outcome_probability: float,
):
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    decision = recommend(
        context,
        record.preferences,
        runs=runs,
        seed=seed,
        budgets=budgets,
        minimum_outcome_probability=minimum_outcome_probability,
    )
    return context, decision


@router.get("/accounts/{account_id}/planner/recommendation", response_model=RecommendationView)
def planner_recommendation(
    runs: int = Query(DEFAULT_RUNS),
    seed: int | None = Query(DEFAULT_SEED),
    budgets: list[int] | None = Query(None),
    minimum_outcome_probability: float = Query(MINIMUM_OUTCOME_PROBABILITY),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> RecommendationView:
    context, decision = _recommendation(record, overrides, runs, seed, budgets, minimum_outcome_probability)
    return RecommendationView.from_domain(
        decision,
        confidence=context.confidence,
        minimum_outcome_probability=minimum_outcome_probability,
    )


@router.get("/accounts/{account_id}/planner/stop-conditions", response_model=StopConditionsView)
def planner_stop_conditions(
    runs: int = Query(DEFAULT_RUNS),
    seed: int | None = Query(DEFAULT_SEED),
    budgets: list[int] | None = Query(None),
    minimum_outcome_probability: float = Query(MINIMUM_OUTCOME_PROBABILITY),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> StopConditionsView:
    _, decision = _recommendation(record, overrides, runs, seed, budgets, minimum_outcome_probability)
    return StopConditionsView.from_domain(decision.stops)
