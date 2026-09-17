"""Planner endpoints (Design Document §9, §13, §14, §18 Phase 6).

    GET /accounts/{id}/planner/goals             goal states (§9)
    GET /accounts/{id}/planner/safe-spend        approximation (§14)
    GET /accounts/{id}/planner/spend-table       spend vs. roadmap (§14)
    GET /accounts/{id}/planner/recommendation    the decision (§13)
    GET /accounts/{id}/planner/stop-conditions   the decision's discipline (§1)

Two views of spending coexist here on purpose, and each says which it is:

* `/safe-spend` and `/spend-table` are the Phase 3 sequential
  independent-reserve approximation - fast, analytical, and explicitly
  provisional (§14);
* `/recommendation` is the Phase 5 optimizer running the Monte Carlo
  simulator, which counts carried pity, guarantee, actual spending and
  income timing (§14), and gates spending only on protected goals that
  outrank the current banner's goal (§2).

They can disagree; the recommendation is the answer, and the approximation
is the cheap explanation. §14 warns against letting the approximation
become a permanent assumption, so it is labelled rather than hidden. The
priority gate is one place they diverge by design: Phase 3 answers "what is
protected" for every future goal alike, while the optimizer answers "what
may constrain this decision".

`/stop-conditions` returns the stops of a real `recommend()` call. Stop
conditions belong to a recommendation (Phase 5 invariant 12) - composing
rules here would let the two drift apart.
"""

from fastapi import APIRouter, Depends, Query

from api.dependencies import ContextOverrides, context_overrides, get_record
from api.repository import AccountRecord
from api.schemas.domain import BannerModel, GoalModel
from api.schemas.planner import (
    GoalEvaluationsView,
    GoalEvaluationView,
    ProtectedGoalOutcomeView,
    RecommendationView,
    SafeSpendView,
    SpendRowView,
    SpendTableView,
    StopConditionsView,
)
from optimizer import DEFAULT_RUNS, MINIMUM_OUTCOME_PROBABILITY, recommend
from planner import (
    actionable_goals,
    current_banner,
    evaluate_goals,
    protected_goal_outcomes,
    relevant_goal_evaluations,
    safe_spend,
    single_copy_active_goal,
    spend_table,
)
from simulation import DEFAULT_SEED

router = APIRouter(tags=["planner"])


@router.get(
    "/accounts/{account_id}/planner/goals",
    response_model=GoalEvaluationsView,
    summary="Goal states against the current banner",
    description=(
        "Every roadmap goal in priority order with its planner state (§9): "
        "satisfied, active, or blocked behind a lower-constellation goal for "
        "the same character. Goals with no upcoming banner stay visible - "
        '"not schedulable" is not "does not exist" (§8).'
    ),
)
def planner_goals(
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> GoalEvaluationsView:
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    relevant = {
        evaluation.goal for evaluation in relevant_goal_evaluations(context)
    }
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


@router.get(
    "/accounts/{account_id}/planner/safe-spend",
    response_model=SafeSpendView,
    summary="Safe-spend approximation",
    description=(
        "The largest current-banner spend that keeps every protected future "
        "goal at or above the confidence threshold, under the Phase 3 "
        "sequential independent-reserve approximation (§14). A safe spend of "
        "0 means the roadmap cannot be protected even by spending nothing - "
        '"do not spend" is a legitimate answer (§1).'
    ),
)
def planner_safe_spend(
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> SafeSpendView:
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
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
    summary="Spend table for the current banner",
    description=(
        "One row per candidate spend: the probability of the active goal's "
        "copy, and what that spend does to every protected future goal - "
        "spending now is an explicit tradeoff against future probability "
        "(§1). Requires exactly one active single-copy goal on the current "
        "banner; multi-copy targets are rejected rather than approximated "
        "(§10.4)."
    ),
)
def planner_spend_table(
    step: int = Query(1, description="Spend increment between rows."),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> SpendTableView:
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    evaluation = single_copy_active_goal(context)
    return SpendTableView(
        current_banner=BannerModel.from_domain(current_banner(context)),
        goal=GoalModel.from_domain(evaluation.goal),
        copies_needed=evaluation.copies_needed,
        step=step,
        confidence=context.confidence,
        rows=[
            SpendRowView.from_domain(row) for row in spend_table(context, step=step)
        ],
    )


def _recommendation(
    record: AccountRecord,
    overrides: ContextOverrides,
    runs: int,
    seed: int | None,
    budgets: list[int] | None,
    minimum_outcome_probability: float,
):
    """Run the optimizer once for the endpoints that need a decision (§13)."""
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


@router.get(
    "/accounts/{account_id}/planner/recommendation",
    response_model=RecommendationView,
    summary="The current recommendation",
    description=(
        "The highest-preference outcome that can be pursued while every "
        "protected future goal that outranks this banner's goal stays at or "
        "above the confidence threshold, and the largest feasible spend cap "
        "for it (§13, §14). Lower-priority future goals are still pursued and "
        "reported, but they cannot force the spend down (§2). Skipping is a "
        "legitimate recommendation and comes with per-outcome rejections "
        "(§1).\n\n"
        "This runs the Monte Carlo simulator once per candidate cap. A skip "
        "scans every cap of every outcome, which is seconds of work at the "
        "defaults; pass a coarser `budgets` list when latency matters, "
        "keeping in mind that a coarse list can miss a narrow feasible "
        "window. Results are reproducible for an identical `runs`/`seed`.\n\n"
        "A feasible outcome whose probability at its largest safe cap falls "
        "below `minimum_outcome_probability` is reported as "
        '`action="discretionary"` rather than `"pursue"`: the budget is '
        "still safe to spend, but not recommended (§14)."
    ),
)
def planner_recommendation(
    runs: int = Query(DEFAULT_RUNS, description="Histories per candidate."),
    seed: int | None = Query(
        DEFAULT_SEED, description="Rng seed; null draws entropy from the OS."
    ),
    budgets: list[int] | None = Query(
        None,
        description=(
            "Candidate spend caps to scan. Defaults to every spend from the "
            "account's wishes down to 0."
        ),
    ),
    minimum_outcome_probability: float = Query(
        MINIMUM_OUTCOME_PROBABILITY,
        description=(
            "Minimum outcome probability for an ordinary recommendation "
            '(§14); below it a feasible outcome is reported as '
            '"discretionary" instead of "pursue".'
        ),
    ),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> RecommendationView:
    context, decision = _recommendation(
        record, overrides, runs, seed, budgets, minimum_outcome_probability
    )
    return RecommendationView.from_domain(
        decision,
        confidence=context.confidence,
        minimum_outcome_probability=minimum_outcome_probability,
    )


@router.get(
    "/accounts/{account_id}/planner/stop-conditions",
    response_model=StopConditionsView,
    summary="Stop conditions for the current recommendation",
    description=(
        "The discipline attached to the current recommendation: stop on the "
        "outcome, never exceed the cap, re-run after the banner resolves "
        "(§1, §2). These are the stops of an actual recommendation, so they "
        "cost the same as `/planner/recommendation`."
    ),
)
def planner_stop_conditions(
    runs: int = Query(DEFAULT_RUNS, description="Histories per candidate."),
    seed: int | None = Query(
        DEFAULT_SEED, description="Rng seed; null draws entropy from the OS."
    ),
    budgets: list[int] | None = Query(
        None, description="Candidate spend caps to scan."
    ),
    minimum_outcome_probability: float = Query(
        MINIMUM_OUTCOME_PROBABILITY,
        description=(
            "Minimum outcome probability for an ordinary recommendation "
            '(§14); below it the stops describe a "discretionary" gamble '
            'instead of a "pursue".'
        ),
    ),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> StopConditionsView:
    _, decision = _recommendation(
        record, overrides, runs, seed, budgets, minimum_outcome_probability
    )
    return StopConditionsView.from_domain(decision.stops)
