"""Planner endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query

from api.dependencies import ContextOverrides, context_overrides, get_record
from api.repository import AccountRecord
from api.planner_cache import CachedCandidateEvidence, planner_evidence_cache
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
    PullStrategyView,
    CachedPlannerRefreshView,
    CachedGoalEvidenceView,
    CachedGoalProbabilityView,
)
from optimizer.protection import constraining_goals
from optimizer import (
    DEFAULT_RUNS,
    MINIMUM_OUTCOME_PROBABILITY,
    OutcomeOption,
    evaluate_candidate,
    recommend,
    build_strategy,
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

@router.post("/accounts/{account_id}/planner/timing", include_in_schema=False)
def planner_timing(timing: dict[str, object]) -> None:
    """Print the browser-side planner timing summary to the server terminal."""
    def fmt(value):
        return "n/a" if value is None else f"{value:.2f}s"

    print(
        "Planner timing\n"
        "-------------\n"
        f"Run:                {timing.get('planner_run_id', 'unknown')} ({timing.get('planner_run_source', 'unknown')})\n"
        f"Planner calculation: {fmt(timing.get('planner_calculation_seconds'))}\n"
        f"API response:        {fmt(timing.get('api_response_seconds'))}\n"
        f"Frontend load:       {fmt(timing.get('frontend_load_seconds'))}\n"
        f"Total:               {fmt(timing.get('total_seconds'))}"
    )



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
    def cache_candidate(candidate):
        planner_evidence_cache.put_candidate(
            record.id,
            candidate,
            runs=runs,
            seed=seed,
        )

    decision = recommend(
        context,
        record.preferences,
        runs=runs,
        seed=seed,
        budgets=budgets,
        minimum_outcome_probability=minimum_outcome_probability,
        simulation_sink=cache_candidate,
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


@router.get(
    "/accounts/{account_id}/planner/strategy",
    response_model=PullStrategyView,
    summary="Build the current multi-step pull strategy",
)
def planner_strategy(
    runs: int = Query(DEFAULT_RUNS),
    seed: int | None = Query(DEFAULT_SEED),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> PullStrategyView:
    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    def cache_simulation(goal, result):
        planner_evidence_cache.put(
            record.id,
            goal,
            result,
            runs=runs,
            seed=seed,
            confidence=context.confidence,
            income_scenario=context.income_scenario,
        )

    strategy = build_strategy(
        context,
        runs=runs,
        seed=seed,
        simulation_sink=cache_simulation,
    )
    return PullStrategyView.from_domain(
        strategy,
        confidence=context.confidence,
    )



@router.get(
    "/accounts/{account_id}/planner/cached-refresh",
    response_model=CachedPlannerRefreshView,
    summary="Condition retained planner evidence on a recorded pull",
)
def planner_cached_refresh(
    character: str = Query(...),
    outcome: str = Query(...),
    wishes_used: int = Query(..., ge=1),
    recommendation_budget: int | None = Query(None, ge=1),
    recommendation_constellation: int | None = Query(None, ge=0),
    record: AccountRecord = Depends(get_record),
    overrides: ContextOverrides = Depends(context_overrides),
) -> CachedPlannerRefreshView:
    if outcome not in {"featured", "lost_50_50"}:
        raise HTTPException(status_code=400, detail="outcome must be featured or lost_50_50")

    context = record.context(
        confidence=overrides.confidence,
        income_scenario=overrides.income_scenario,
    )
    conditioned = planner_evidence_cache.condition_account(
        record.id,
        character=character,
        outcome=outcome,
        wishes_used=wishes_used,
    )

    available = available_banners(context)
    current_banner = next(
        (banner for banner in available if banner.character == character),
        None,
    )

    # Rare observations can leave too few matching histories. In that case,
    # use the retained plan as a starting point but simulate from the actual
    # post-pull account state. This avoids both tiny conditional samples and a
    # full optimizer rerun.
    for goal, evidence in tuple(conditioned.items()):
        if evidence.runs >= MIN_CONDITIONED_RUNS or evidence.result.plan is None:
            continue
        if current_banner is None:
            continue
        entries = list(evidence.result.plan.entries)
        index = next((i for i, item in enumerate(entries) if item.banner == current_banner), None)
        if index is None:
            continue
        entry = entries[index]
        entries[index] = replace(entry, budget=max(0, entry.budget - wishes_used))
        shared_budget = evidence.result.plan.shared_current_phase_budget
        if shared_budget is not None:
            shared_budget = max(0, shared_budget - wishes_used)
        refreshed_plan = replace(
            evidence.result.plan,
            entries=tuple(entries),
            shared_current_phase_budget=shared_budget,
        )
        refreshed_result = simulate(
            context,
            refreshed_plan,
            runs=CONDITIONED_FALLBACK_RUNS,
            seed=DEFAULT_SEED,
        )
        planner_evidence_cache.put(
            record.id,
            goal,
            refreshed_result,
            runs=CONDITIONED_FALLBACK_RUNS,
            seed=DEFAULT_SEED,
            confidence=evidence.confidence,
            income_scenario=evidence.income_scenario,
        )
        refreshed = planner_evidence_cache.get(record.id, goal)
        if refreshed is not None:
            conditioned[goal] = refreshed

    # If there were zero matches, condition_account intentionally leaves the
    # original evidence in place. Reuse it for the same targeted fallback.
    if not conditioned and current_banner is not None:
        for goal in tuple(
            evidence.goal
            for evidence in (
                planner_evidence_cache.get(record.id, goal)
                for goal in context.roadmap.goals_in_priority_order()
            )
            if evidence is not None
        ):
            evidence = planner_evidence_cache.get(record.id, goal)
            if evidence is None or evidence.result.plan is None:
                continue
            entries = list(evidence.result.plan.entries)
            index = next((i for i, item in enumerate(entries) if item.banner == current_banner), None)
            if index is None:
                continue
            entry = entries[index]
            entries[index] = replace(entry, budget=max(0, entry.budget - wishes_used))
            shared_budget = evidence.result.plan.shared_current_phase_budget
            if shared_budget is not None:
                shared_budget = max(0, shared_budget - wishes_used)
            refreshed_result = simulate(
                context,
                replace(evidence.result.plan, entries=tuple(entries), shared_current_phase_budget=shared_budget),
                runs=CONDITIONED_FALLBACK_RUNS,
                seed=DEFAULT_SEED,
            )
            planner_evidence_cache.put(
                record.id,
                goal,
                refreshed_result,
                runs=CONDITIONED_FALLBACK_RUNS,
                seed=DEFAULT_SEED,
                confidence=evidence.confidence,
                income_scenario=evidence.income_scenario,
            )
            refreshed = planner_evidence_cache.get(record.id, goal)
            if refreshed is not None:
                conditioned[goal] = refreshed

    evidence_views: list[CachedGoalEvidenceView] = []

    conditioned_recommendation = None
    if recommendation_budget is not None and recommendation_constellation is not None:
        new_budget = recommendation_budget - wishes_used
        if new_budget >= 0 and current_banner is not None:
            conditioned_recommendation = planner_evidence_cache.condition_candidate(
                record.id,
                character=character,
                outcome=outcome,
                constellation=recommendation_constellation,
                banner_version=current_banner.version,
                banner_phase=current_banner.phase,
                new_budget=new_budget,
                wishes_used=wishes_used,
            )
            if conditioned_recommendation is None or conditioned_recommendation.runs < MIN_CONDITIONED_RUNS:
                # The retained candidate may have zero matching histories.
                # Evaluate only this candidate from the post-pull account.
                fresh_candidate = evaluate_candidate(
                    context,
                    conditioned_recommendation.candidate.outcome,
                    new_budget,
                    banner=current_banner,
                    runs=CONDITIONED_FALLBACK_RUNS,
                    seed=DEFAULT_SEED,
                    simulation_sink=lambda candidate: planner_evidence_cache.put_candidate(
                        record.id,
                        candidate,
                        runs=CONDITIONED_FALLBACK_RUNS,
                        seed=DEFAULT_SEED,
                    ),
                )
                conditioned_recommendation = CachedCandidateEvidence(
                    account_id=record.id,
                    character=character,
                    constellation=recommendation_constellation,
                    banner_version=current_banner.version,
                    banner_phase=current_banner.phase,
                    budget=new_budget,
                    candidate=fresh_candidate,
                    runs=CONDITIONED_FALLBACK_RUNS,
                    seed=DEFAULT_SEED,
                    observations=conditioned_recommendation.observations,
                )

    for goal, evidence in conditioned.items():
        probabilities = {item.goal: item.probability for item in evidence.result.goals}
        joint_probability = (
            evidence.result.joint_goal_probability.probability
            if evidence.result.joint_goal_probability is not None
            else None
        )

        safe_spend_remaining = None
        protected_probability = None
        protected_starting_wishes = None
        protected_total_wishes = None

        if current_banner is not None:
            entry = next(
                (
                    item
                    for item in evidence.result.plan.entries
                    if item.banner == current_banner
                    and item.target_constellation == goal.constellation
                ),
                None,
            )
            if entry is not None:
                safe_spend_remaining = max(0, entry.budget - wishes_used)

                protected = constraining_goals(
                    context,
                    priority=goal.priority,
                    banner=current_banner,
                )
                protected_values = [
                    probabilities[protected_goal]
                    for protected_goal in protected
                    if protected_goal in probabilities
                ]
                if protected_values:
                    protected_probability = min(protected_values)

                protected_starting_wishes = context.account.wishes - safe_spend_remaining
                reserve_goal = min(protected, key=lambda item: item.priority) if protected else None
                if reserve_goal is not None:
                    reserve_evaluation = next(
                        item
                        for item in evaluate_goals(context)
                        if item.goal == reserve_goal
                    )
                    future_income = (
                        context.income_available_before(
                            reserve_evaluation.next_banner.version,
                            reserve_evaluation.next_banner.phase,
                        )
                        if reserve_evaluation.next_banner is not None
                        else 0
                    )
                    protected_total_wishes = protected_starting_wishes + future_income

        evidence_views.append(
            CachedGoalEvidenceView(
                goal=GoalModel.from_domain(goal),
                probability=probabilities.get(goal, 0.0),
                goal_probabilities=[
                    CachedGoalProbabilityView(
                        goal=GoalModel.from_domain(item.goal),
                        probability=item.probability,
                    )
                    for item in evidence.result.goals
                ],
                joint_probability=joint_probability,
                runs=evidence.runs,
                safe_spend_remaining=safe_spend_remaining,
                protected_probability=protected_probability,
                protected_starting_wishes=protected_starting_wishes,
                protected_total_wishes=protected_total_wishes,
            )
        )

    recommendation_goal_probabilities = []
    recommendation_probability = None
    recommendation_budget = None
    recommendation_constellation = None
    if conditioned_recommendation is not None:
        result = conditioned_recommendation.candidate.result
        recommendation_probability = conditioned_recommendation.candidate.outcome_probability
        recommendation_budget = conditioned_recommendation.budget
        recommendation_constellation = conditioned_recommendation.constellation
        recommendation_goal_probabilities = [
            CachedGoalProbabilityView(
                goal=GoalModel.from_domain(item.goal),
                probability=item.probability,
            )
            for item in result.goals
        ]

    return CachedPlannerRefreshView(
        account_wishes=context.account.wishes,
        confidence=context.confidence,
        evidence=evidence_views,
        recommendation_budget=recommendation_budget,
        recommendation_constellation=recommendation_constellation,
        recommendation_probability=recommendation_probability,
        recommendation_goal_probabilities=recommendation_goal_probabilities,
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
