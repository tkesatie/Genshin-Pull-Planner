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


@dataclass(frozen=True)
class PullStrategy:
    """The current ordered strategy and its confidence-based reserve."""

    steps: tuple[StrategyStep, ...]
    reserve_goal: Goal | None
    reserve_wishes: int | None
    reserve_probability: float | None
    runs: int
    seed: int | None


def _future_protected_plan(
    context: PlannerContext,
    priority: int,
    *,
    current_banner: Banner,
) -> tuple[SpendPlan, tuple[Goal, ...]]:
    """Build the future plan required to protect a current decision.

    Only future goals whose priority outranks the current decision are
    included. Goals on the same future banner are grouped into one plan entry
    at the highest constellation required by the constraining goals.

    The simulator still reports every original roadmap goal independently;
    grouping only determines how the future plan spends.
    """
    constraining = constraining_goals(
        context,
        priority=priority,
        banner=current_banner,
    )

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


def _future_reserve(
    context: PlannerContext,
    priority: int,
    *,
    current_banner: Banner,
    runs: int,
    seed: int | None,
) -> tuple[int, float | None, tuple[Goal, ...]]:
    """Find the smallest balance that protects all higher-priority goals.

    The tested reserve is the account balance left after the current decision.
    For each candidate reserve, the simulator executes the complete future
    plan for all constraining goals. The reserve is safe only when every
    constraining goal reaches the configured confidence threshold.

    Returns:
        reserve_wishes: minimum preserved balance, or the full current
            balance when the protection requirement cannot be met.
        reserve_probability: weakest constraining-goal probability at the
            selected reserve, or None when nothing constrains this decision.
        protected_goals: the original roadmap goals used as constraints.
    """
    plan, protected_goals = _future_protected_plan(
        context,
        priority,
        current_banner=current_banner,
    )

    if not protected_goals:
        return 0, None, ()

    for reserve in range(context.account.wishes + 1):
        result = simulate(context, plan, runs=runs, seed=seed)
        probabilities = [
            next(item for item in result.goals if item.goal == goal).probability
            for goal in protected_goals
        ]
        if all(probability >= context.confidence for probability in probabilities):
            return reserve, min(probabilities), protected_goals

        # The future plan's budgets represent the resources available at the
        # future banners, so changing the reserve must change the simulated
        # starting balance. Rebuild the context for that candidate below.
        # This branch is replaced by the candidate-account helper in the next
        # section of this function.
        break

    # A reserve cannot be tested merely by changing a PlannedSpend budget:
    # that would model "spend less on the future goals", not "arrive at the
    # future roadmap with fewer wishes." Build candidate contexts by using
    # the existing account replacement API.
    #
    # The project currently exposes Account as an immutable dataclass, so
    # import replace rather than adding a second account model.
    from dataclasses import replace

    for reserve in range(context.account.wishes + 1):
        candidate_account = replace(context.account, wishes=reserve)
        candidate_context = replace(context, account=candidate_account)
        candidate_plan, _ = _future_protected_plan(
            candidate_context,
            priority,
            current_banner=current_banner,
        )
        result = simulate(candidate_context, candidate_plan, runs=runs, seed=seed)
        probabilities = [
            next(item for item in result.goals if item.goal == goal).probability
            for goal in protected_goals
        ]
        if all(probability >= context.confidence for probability in probabilities):
            return reserve, min(probabilities), protected_goals

    candidate_account = replace(context.account, wishes=context.account.wishes)
    candidate_context = replace(context, account=candidate_account)
    candidate_plan, _ = _future_protected_plan(
        candidate_context,
        priority,
        current_banner=current_banner,
    )
    result = simulate(candidate_context, candidate_plan, runs=runs, seed=seed)
    probabilities = [
        next(item for item in result.goals if item.goal == goal).probability
        for goal in protected_goals
    ]
    return context.account.wishes, min(probabilities), protected_goals


def build_strategy(
    context: PlannerContext,
    *,
    runs: int = 2_000,
    seed: int | None = 0,
) -> PullStrategy:
    """Build the current multi-step strategy.

    Current-banner C0 milestones come first. A deeper constellation on a
    current banner can then be pursued only to the extent that the wishes
    spent do not reduce higher-priority future goals below the confidence
    threshold.

    The reserve is specific to each progression goal. This matters when
    current-banner goals have different priorities: a P2 C0 objective is not
    constrained by a P3 future objective, while a P4 constellation objective
    is.
    """
    banners = available_banners(context)
    if not banners:
        return PullStrategy((), None, None, None, runs, seed)

    current_banner = banners[0]
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

    progression = [
        evaluation
        for evaluation in evaluations
        if evaluation.goal.character in current_characters
        and evaluation.goal.constellation > 0
        and evaluation.copies_needed > 0
    ]
    progression.sort(key=lambda evaluation: evaluation.goal.priority)

    reserve_goal = None
    reserve_wishes = None
    reserve_probability = None

    for evaluation in progression:
        banner = next(
            banner for banner in banners
            if banner.character == evaluation.goal.character
        )
        reserve, probability, protected_goals = _future_reserve(
            context,
            evaluation.goal.priority,
            current_banner=banner,
            runs=runs,
            seed=seed,
        )

        if reserve_goal is None and protected_goals:
            reserve_goal = min(
                protected_goals,
                key=lambda goal: goal.priority,
            )
            reserve_wishes = reserve
            reserve_probability = probability

        steps.append(
            StrategyStep(
                "pursue_until_reserve",
                evaluation.goal,
                banner,
                reserve_wishes=reserve,
            )
        )

    if reserve_goal is not None:
        reserve_banner = min(
            (
                banner
                for banner in context.roadmap.banners
                if banner.character == reserve_goal.character
                and banner.order_key > current_banner.order_key
            ),
            key=lambda banner: banner.order_key,
        )
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
