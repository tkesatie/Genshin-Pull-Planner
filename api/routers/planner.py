"""Planner endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

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
    evaluate_candidate,
    recommend,
)
from planner import (
    actionable_goals,
    available_banners,
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
    banners = available_banners(context)
    relevant = {evaluation.goal for evaluation in relevant_goal_evaluations(context)}
    actionable = {evaluation.goal for evaluation in actionable_goals(context)}
    return GoalEvaluationsView(
        current_banner=(BannerModel.from_domain(banners[0]) if len(banners) == 1 else None),
        available_banners=[BannerModel.from_domain(banner) for banner in banners],
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
    character: str | None = Query(None),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> SafeSpendView:
    context = record.context(confidence=overrides.confidence, income_scenario=overrides.income_scenario)
    banners = available_banners(context)
    selected = _select_banner(banners, character)
    return SafeSpendView(
        current_banner=BannerModel.from_domain(selected) if selected else None,
        available_banners=[BannerModel.from_domain(banner) for banner in banners],
        safe_spend=safe_spend(context, banner=selected),
        account_wishes=context.account.wishes,
        confidence=context.confidence,
        protected=[
            ProtectedGoalOutcomeView.from_domain(outcome)
            for outcome in protected_goal_outcomes(context, spent=0, banner=selected)
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
    character: str | None = Query(None),
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

    selected = _select_banner(available_banners(context), character)
    outcomes = _spend_table_outcomes(context, record.preferences, selected)
    if not outcomes:
        return SpendTableView(
            current_banner=BannerModel.from_domain(selected),
            available_banners=[BannerModel.from_domain(banner) for banner in available_banners(context)],
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
                    context, outcome, budget, banner=selected, runs=runs, seed=seed
                ).outcome_probability,
            )
            for outcome in outcomes
        ]

        # Use the same protected-goal evaluation for every milestone. The
        # future-roadmap tradeoff does not depend on which current-banner
        # constellation column the user is looking at.
        reference_candidate = evaluate_candidate(
            context, outcomes[0], budget, banner=selected, runs=runs, seed=seed
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
        current_banner=BannerModel.from_domain(selected),
        available_banners=[BannerModel.from_domain(banner) for banner in available_banners(context)],
        outcomes=[OutcomeView.from_domain(outcome) for outcome in outcomes],
        step=step,
        confidence=context.confidence,
        runs=runs,
        seed=seed,
        rows=rows,
    )


def _spend_table_outcomes(context, preferences, banner) -> tuple[OutcomeOption, ...]:
    """Return unsatisfied roadmap milestones for the current banner.

    The recommendation's outcome list is intentionally preference-driven,
    but the spend analysis answers a different question: how does spending
    on this banner affect each milestone in the roadmap for this character?
    Therefore blocked progression goals (such as Vesna C2 behind Vesna C0)
    remain visible here even when the optimizer would not currently choose
    them as its recommendation outcome.
    """
    current_character = banner.character
    preference_ranks = {
        preference.constellation: preference.rank
        for preference in preferences
        if preference.character == current_character
    }

    by_constellation: dict[int, OutcomeOption] = {}
    for evaluation in relevant_goal_evaluations(context, banner):
        if evaluation.copies_needed <= 0:
            continue
        constellation = evaluation.goal.constellation
        by_constellation.setdefault(
            constellation,
            OutcomeOption(
                character=current_character,
                constellation=constellation,
                rank=preference_ranks.get(constellation, evaluation.goal.priority),
            ),
        )
    return tuple(
        sorted(by_constellation.values(), key=lambda outcome: outcome.constellation)
    )



def _select_banner(banners, character):
    if not banners:
        raise HTTPException(status_code=422, detail="no roadmap banner at the current version/phase")
    if character is None:
        if len(banners) > 1:
            raise HTTPException(
                status_code=422,
                detail="multiple banners are active; select a character explicitly: "
                + ", ".join(banner.character for banner in banners),
            )
        return banners[0]
    matches = [banner for banner in banners if banner.character == character]
    if not matches:
        raise HTTPException(
            status_code=422,
            detail=f"character {character!r} is not an available banner at the current version/phase",
        )
    return matches[0]

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
