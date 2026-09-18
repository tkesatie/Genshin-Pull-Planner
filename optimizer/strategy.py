"""Multi-step pull strategy derived from roadmap priorities and confidence.

This layer answers the UI question "what should I do in sequence?" without
changing optimizer.recommend(), whose job remains evaluating one current
decision.

A reserve is now calculated for the *current decision's priority*: it is the
smallest wish balance that allows every higher-priority future protected goal
to be pursued on its future banner while meeting the configured confidence
threshold. Lower-priority future goals do not consume the protection reserve.

After an actual pull, rerun the planner so pity, guarantee, ownership, and
wishes
are reflected in the new reserve.
"""

from dataclasses import dataclass

from domain import Banner, Goal
from optimizer.protection import constraining_goals, protected_groups
from planner import PlannerContext, evaluate_goals
from planner.banners import available_banners
from simulation import PlannedSpend, SpendPlan, simulate


@dataclass(frozen=True)
class StrategyStep:
    """One user-facing step in the current pull strategy."""

    action: str
    goal: Goal
    banner: Banner
    reserve_wishes: int | None = None
    safe_spend: int | None = None
    outcome_probability: float | None = None
    protected_probability: float | None = None


@dataclass(frozen=True)
class PullStrategy:
    """The current ordered strategy and its confidence-based reserve."""

    steps: tuple[StrategyStep, ...]
    reserve_goal: Goal | None
    reserve_wishes: int | None
    reserve_probability: float | None
    runs: int
    seed: int | None


def _protection_plan(
    context: PlannerContext,
    priority: int,
    *,
    current_banner: Banner,
) -> tuple[SpendPlan, tuple[Goal, ...]]:
    """Build the future plan required to protect a current decision."""
    constraining = constraining_goals(context, priority=priority, banner=current_banner)
    entries: list[PlannedSpend] = []
    protected: list[Goal] = []

    for group in protected_groups(context, banner=current_banner):
        goals = tuple(goal for goal in group.goals if goal in constraining)
        if not goals:
            continue
        protected.extend(goals)
        entries.append(
            PlannedSpend(
                banner=group.banner,
                target_constellation=max(goal.constellation for goal in goals),
                budget=group.uncapped_budget,
            )
        )
    return SpendPlan(entries=tuple(entries)), tuple(protected)


def _evaluate_spend(
    context: PlannerContext,
    goal: Goal,
    banner: Banner,
    spend: int,
    *,
    runs: int,
    seed: int | None,
) -> tuple[SimulationResult, tuple[Goal, ...]]:
    """Evaluate one current spend against all higher-priority future goals."""
    future_plan, protected_goals = _protection_plan(
        context, goal.priority, current_banner=banner
    )
    plan = SpendPlan(
        entries=(
            PlannedSpend(
                banner=banner,
                target_constellation=goal.constellation,
                budget=spend,
            ),
            *future_plan.entries,
        )
    )
    return simulate(context, plan, runs=runs, seed=seed), protected_goals


def _safe_spend(
    context: PlannerContext,
    goal: Goal,
    banner: Banner,
    *,
    runs: int,
    seed: int | None,
) -> tuple[int, SimulationResult, tuple[Goal, ...]]:
    """Find the largest current spend that preserves every constraint."""
    for spend in range(context.account.wishes, -1, -1):
        result, protected_goals = _evaluate_spend(
            context, goal, banner, spend, runs=runs, seed=seed
        )
        if not protected_goals:
            return spend, result, protected_goals
        probabilities = [
            next(item for item in result.goals if item.goal == protected).probability
            for protected in protected_goals
        ]
        if all(probability >= context.confidence for probability in probabilities):
            return spend, result, protected_goals
    raise RuntimeError("safe-spend search must find the zero-spend candidate")

def build_strategy(
    context: PlannerContext,
    *,
    runs: int = 2_000,
    seed: int | None = 0,
) -> PullStrategy:
    """Build the current multi-step strategy and spend frontiers."""
    banners = available_banners(context)
    if not banners:
        return PullStrategy((), None, None, None, runs, seed)

    current_characters = {banner.character for banner in banners}
    evaluations = evaluate_goals(context)
    current_goals = [
        evaluation for evaluation in evaluations
        if evaluation.goal.character in current_characters
        and evaluation.goal.constellation == 0
        and evaluation.copies_needed > 0
    ]
    current_goals.sort(key=lambda evaluation: evaluation.goal.priority)

    steps: list[StrategyStep] = []
    for evaluation in current_goals:
        banner = next(banner for banner in banners if banner.character == evaluation.goal.character)
        steps.append(StrategyStep(
            action="get", goal=evaluation.goal, banner=banner,
            safe_spend=context.account.wishes,
        ))

    progression = [
        evaluation for evaluation in evaluations
        if evaluation.goal.character in current_characters
        and evaluation.goal.constellation > 0
        and evaluation.copies_needed > 0
    ]
    progression.sort(key=lambda evaluation: evaluation.goal.priority)

    reserve_goal = None
    reserve_wishes = None
    reserve_probability = None

    for evaluation in progression:
        banner = next(banner for banner in banners if banner.character == evaluation.goal.character)
        spend, result, protected_goals = _safe_spend(
            context, evaluation.goal, banner, runs=runs, seed=seed
        )
        protected_probability = None
        if protected_goals:
            protected_probability = min(
                next(item for item in result.goals if item.goal == protected).probability
                for protected in protected_goals
            )
        reserve = context.account.wishes - spend
        if reserve_goal is None and protected_goals:
            reserve_goal = min(protected_goals, key=lambda item: item.priority)
            reserve_wishes = reserve
            reserve_probability = protected_probability
        steps.append(StrategyStep(
            action="pursue_until_reserve", goal=evaluation.goal, banner=banner,
            reserve_wishes=reserve,
            safe_spend=spend,
            outcome_probability=result.banners[0].target_met_probability,
            protected_probability=protected_probability,
        ))

    if reserve_goal is not None:
        reserve_banner = min(
            (banner for banner in context.roadmap.banners
             if banner.character == reserve_goal.character
             and banner.order_key > banners[0].order_key),
            key=lambda banner: banner.order_key,
        )
        steps.append(StrategyStep("save", reserve_goal, reserve_banner))

    return PullStrategy(
        steps=tuple(steps), reserve_goal=reserve_goal,
        reserve_wishes=reserve_wishes, reserve_probability=reserve_probability,
        runs=runs, seed=seed,
    )
