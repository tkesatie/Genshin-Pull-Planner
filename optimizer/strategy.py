"""Multi-step pull strategy derived from roadmap priorities and confidence.

This layer answers the UI question "what should I do in sequence?" without
changing optimizer.recommend(), whose job remains evaluating one current
decision.

The reserve is a re-runnable checkpoint: it is the smallest wish balance that
gives the highest-priority future goal enough simulated probability to meet
the configured confidence threshold when that future goal is pursued without
current-banner spending. After an actual pull, rerun the planner so pity,
guarantee, ownership, and wishes are reflected in the new reserve.
"""

from dataclasses import dataclass

from domain import Banner, Goal
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


@dataclass(frozen=True)
class PullStrategy:
    """The current ordered strategy and its confidence-based reserve."""

    steps: tuple[StrategyStep, ...]
    reserve_goal: Goal | None
    reserve_wishes: int | None
    reserve_probability: float | None
    runs: int
    seed: int | None


def _future_banner(context: PlannerContext, goal: Goal, current_order: int) -> Banner:
    return min(
        (
            banner
            for banner in context.roadmap.banners_for(goal.character)
            if banner.order_key > current_order
        ),
        key=lambda banner: banner.order_key,
    )


def _future_reserve(
    context: PlannerContext,
    goal: Goal,
    banner: Banner,
    *,
    runs: int,
    seed: int | None,
) -> tuple[int, float]:
    """Find the smallest wish balance that protects a future goal.

    The current slot is skipped. This is therefore a checkpoint for the
    account after current-banner actions, not a claim about the starting
    account's immediately spendable balance.
    """
    for reserve in range(context.account.wishes + 1):
        plan = SpendPlan(
            entries=(
                PlannedSpend(
                    banner=banner,
                    target_constellation=goal.constellation,
                    budget=reserve,
                ),
            )
        )
        result = simulate(context, plan, runs=runs, seed=seed)
        standing = next(item for item in result.goals if item.goal == goal)
        if standing.probability >= context.confidence:
            return reserve, standing.probability

    plan = SpendPlan(
        entries=(
            PlannedSpend(
                banner=banner,
                target_constellation=goal.constellation,
                budget=context.account.wishes,
            ),
        )
    )
    result = simulate(context, plan, runs=runs, seed=seed)
    standing = next(item for item in result.goals if item.goal == goal)
    return context.account.wishes, standing.probability


def build_strategy(
    context: PlannerContext,
    *,
    runs: int = 2_000,
    seed: int | None = 0,
) -> PullStrategy:
    """Build the current multi-step strategy.

    Current-banner C0 milestones come first. A deeper constellation on a
    current banner can then be pursued before a future goal, but only while
    preserving the future confidence reserve. The planner should rerun after
    meaningful account updates.
    """
    banners = available_banners(context)
    if not banners:
        return PullStrategy((), None, None, None, runs, seed)

    current_characters = {banner.character for banner in banners}
    evaluations = evaluate_goals(context)

    current_goals = [
        evaluation
        for evaluation in evaluations
        if evaluation.goal.character in current_characters
        and evaluation.goal.constellation == 0
        and evaluation.copies_needed > 0
    ]
    current_goals.sort(key=lambda evaluation: evaluation.goal.priority)

    steps: list[StrategyStep] = []
    for evaluation in current_goals:
        banner = next(
            banner for banner in banners
            if banner.character == evaluation.goal.character
        )
        steps.append(StrategyStep("get", evaluation.goal, banner))

    future = [
        evaluation
        for evaluation in evaluations
        if evaluation.copies_needed > 0
        and evaluation.next_banner is not None
        and evaluation.next_banner.order_key > banners[0].order_key
    ]
    future.sort(key=lambda evaluation: evaluation.goal.priority)

    reserve_goal = future[0].goal if future else None
    reserve_wishes = None
    reserve_probability = None
    reserve_banner = None

    if reserve_goal is not None:
        reserve_banner = _future_banner(
            context, reserve_goal, banners[0].order_key
        )
        reserve_wishes, reserve_probability = _future_reserve(
            context,
            reserve_goal,
            reserve_banner,
            runs=runs,
            seed=seed,
        )

    progression = [
        evaluation
        for evaluation in evaluations
        if evaluation.goal.character in current_characters
        and evaluation.goal.constellation > 0
        and evaluation.copies_needed > 0
    ]
    progression.sort(key=lambda evaluation: evaluation.goal.priority)

    for evaluation in progression:
        banner = next(
            banner for banner in banners
            if banner.character == evaluation.goal.character
        )
        steps.append(
            StrategyStep(
                "pursue_until_reserve",
                evaluation.goal,
                banner,
                reserve_wishes=reserve_wishes,
            )
        )

    if reserve_goal is not None:
        steps.append(
            StrategyStep("save", reserve_goal, reserve_banner)
        )

    return PullStrategy(
        steps=tuple(steps),
        reserve_goal=reserve_goal,
        reserve_wishes=reserve_wishes,
        reserve_probability=reserve_probability,
        runs=runs,
        seed=seed,
    )
