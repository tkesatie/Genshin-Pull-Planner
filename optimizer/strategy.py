"""Multi-step pull strategy derived from roadmap priorities and confidence.

This layer answers the UI question "what should I do in sequence?" without
changing optimizer.recommend(), whose job remains evaluating one current
decision.

A spend frontier is calculated for the current decision's priority. Higher-
priority C0 goals on the current banner are included in the same simulated
plan with an uncapped budget, so their random pull costs are accounted for
before evaluating how much can be spent on the current progression goal.
Higher-priority future goals are then protected at the configured confidence
threshold.

After an actual pull, rerun the planner so pity, guarantee, ownership, and
wishes are reflected in the new frontier.
"""

from dataclasses import dataclass

from domain import Banner, Goal
from optimizer.protection import constraining_goals, protected_groups
from planner import PlannerContext, evaluate_goals
from planner.banners import available_banners
from simulation import PlannedSpend, SimulationResult, SpendPlan, simulate


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
    excluded_goals: frozenset[Goal] = frozenset(),
) -> tuple[SpendPlan, tuple[Goal, ...]]:
    """Build the future plan required to protect a current decision.

    Goals already represented by higher-priority current-banner allocations
    are not also treated as protected future goals. In particular, an
    alternative banner in the current phase may appear in
    ``protected_groups()`` but is already handled by ``_current_required_plan``.
    Adding it again would create duplicate plan entries for the same banner.
    """
    constraining = constraining_goals(context, priority=priority, banner=current_banner)
    entries: list[PlannedSpend] = []
    protected: list[Goal] = []

    for group in protected_groups(context, banner=current_banner):
        goals = tuple(
            goal for goal in group.goals
            if goal in constraining and goal not in excluded_goals
        )
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

def _current_required_plan(
    context: PlannerContext,
    goal: Goal,
    *,
    current_banner: Banner,
) -> SpendPlan:
    """Build the higher-priority allocations that precede the current goal.

    A same-character C0 goal is intentionally omitted because a progression
    target on that banner already pursues the C0 first. Its budget is therefore
    the total cap for that banner, not an additional spend after C0.
    """
    evaluations = evaluate_goals(context)
    entries: list[PlannedSpend] = []

    for evaluation in evaluations:
        required = evaluation.goal
        if required.priority >= goal.priority:
            continue
        if required.character == goal.character:
            continue
        if required.constellation != 0 or evaluation.copies_needed <= 0:
            continue
        banner = next(
            (
                candidate
                for candidate in available_banners(context)
                if candidate.character == required.character
            ),
            None,
        )
        if banner is None or banner == current_banner:
            continue
        entries.append(
            PlannedSpend(
                banner=banner,
                target_constellation=0,
                budget=context.account.wishes,
            )
        )

    return SpendPlan(entries=tuple(entries))


def _evaluate_spend(
    context: PlannerContext,
    goal: Goal,
    banner: Banner,
    spend: int,
    *,
    runs: int,
    seed: int | None,
) -> tuple[SimulationResult, tuple[Goal, ...]]:
    """Evaluate a current spend after higher-priority C0 allocations.

    The current progression goal's budget is a total cap on its banner. If
    that goal is Vesna C2, for example, the simulator spends toward Vesna C0
    first and then continues toward C2 while the same budget remains. This
    avoids pretending we know the exact number of wishes consumed by the C0.
    """
    required_plan = _current_required_plan(
        context, goal, current_banner=banner
    )
    required_goals = frozenset(
        evaluation.goal
        for evaluation in evaluate_goals(context)
        if evaluation.goal.priority < goal.priority
        and evaluation.goal.character != goal.character
        and evaluation.goal.constellation == 0
        and evaluation.copies_needed > 0
    )
    future_plan, protected_goals = _protection_plan(
        context,
        goal.priority,
        current_banner=banner,
        excluded_goals=required_goals,
    )
    plan = SpendPlan(
        entries=(
            *required_plan.entries,
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
    """Find the largest current budget that preserves every constraint.

    The protected probability is empirically monotonic with spend, but Monte
    Carlo introduces small local fluctuations. Use coarse sampling to bracket
    the frontier, binary search to narrow it, then exhaustively test a small
    window around the boundary so the returned value is directly verified.
    """
    max_spend = context.account.wishes
    cache: dict[int, tuple[SimulationResult, tuple[Goal, ...]]] = {}

    def evaluate(spend: int) -> tuple[SimulationResult, tuple[Goal, ...]]:
        if spend not in cache:
            cache[spend] = _evaluate_spend(
                context, goal, banner, spend, runs=runs, seed=seed
            )
        return cache[spend]

    def is_safe(spend: int) -> bool:
        result, protected_goals = evaluate(spend)
        if not protected_goals:
            return True
        probabilities = [
            next(item for item in result.goals if item.goal == protected).probability
            for protected in protected_goals
        ]
        return all(probability >= context.confidence for probability in probabilities)

    # First find a safe/unsafe bracket with a small number of broad samples.
    if is_safe(max_spend):
        safe = max_spend
        unsafe = None
    else:
        safe = 0
        unsafe = max_spend
        step = max(25, max_spend // 8)
        spend = max_spend - step

        while spend > 0:
            if is_safe(spend):
                safe = spend
                break
            unsafe = spend
            spend -= step

        if spend == 0 and safe == 0:
            evaluate(0)
        elif safe == 0 and is_safe(0):
            evaluate(0)
        elif safe == 0:
            raise RuntimeError("safe-spend search must find the zero-spend candidate")

    # Narrow the bracket using the monotonic trend observed in Monte Carlo.
    if unsafe is not None:
        low = safe
        high = unsafe
        while high - low > 1:
            mid = (low + high) // 2
            if is_safe(mid):
                low = mid
            else:
                high = mid
        safe = low

    # Verify the boundary directly. A few noisy Monte Carlo points should not
    # determine the answer solely through the binary search.
    window = 5
    lower = max(0, safe - window)
    upper = min(max_spend, safe + window)
    for spend in range(upper, lower - 1, -1):
        if is_safe(spend):
            result, protected_goals = evaluate(spend)
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
