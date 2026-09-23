"""Multi-step pull strategy derived from roadmap priorities and confidence.

This layer answers the UI question "what should I do in sequence?" without
changing optimizer.recommend(), whose job remains evaluating one current
decision.

A spend frontier is calculated for the current decision's priority. Higher-
priority C0 goals on the current phase are included in the same simulated
plan and share the current-phase spend cap, so their random pull costs are
accounted for before evaluating progression.
Higher-priority future goals are then protected at the configured confidence
threshold.

After an actual pull, rerun the planner so pity, guarantee, ownership, and
wishes are reflected in the new frontier.
"""

from dataclasses import dataclass, replace
from typing import Callable

from domain import WEAPON_EVENT_BANNER, Banner, Goal, TargetKind
from optimizer.protection import constraining_goals, protected_groups
from planner import PlannerContext, character_view, evaluate_goals
from planner.banners import available_banners
from planner.protection import protected_goal_outcomes, weapon_goal_reserve
from probability import refinement_cumulative_probability
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
    future_income: int | None = None
    protected_starting_wishes: int | None = None
    protected_total_wishes: int | None = None
    all_goals_probability: float | None = None


@dataclass(frozen=True)
class PullStrategy:
    """The current ordered strategy and its confidence-based reserve."""

    steps: tuple[StrategyStep, ...]
    reserve_goal: Goal | None
    reserve_wishes: int | None
    reserve_probability: float | None
    runs: int
    seed: int | None
    starting_wishes: int
    future_income: int


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
        # Weapon banners are never character-simulated (Phase 4); their
        # goals stay in `protected` and their reserves are evaluated
        # analytically (_protected_requirement), not by plan entries here.
        if group.banner.target.kind is not TargetKind.CHARACTER:
            continue
        entries.append(
            PlannedSpend(
                banner=group.banner,
                # goal.level unifies constellation and refinement; for
                # character goals this is the same value
                # `goal.constellation` always returned.
                target_constellation=max(goal.level for goal in goals),
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
        # Higher-priority C0 allocations are character-banner business;
        # weapon goals are protected through their own reserves (Phase 4).
        if required.target.kind is not TargetKind.CHARACTER:
            continue
        if required.target == goal.target:
            continue
        if required.constellation != 0 or evaluation.copies_needed <= 0:
            continue
        banner = next(
            (
                candidate
                for candidate in available_banners(context)
                # Target-based matching (Phase 4): a weapon banner has no
                # `.character`, so unified lookup must not use it.
                if candidate.target == required.target
            ),
            None,
        )
        if banner is None:
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
        and evaluation.goal.target.kind is TargetKind.CHARACTER
        and evaluation.goal.target != goal.target
        and evaluation.goal.level == 0
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
        ),
        shared_current_phase_budget=spend,
    )
    joint_goals = tuple(required_goals) + (goal,)
    return simulate(
        character_view(context),
        plan,
        runs=runs,
        seed=seed,
        joint_goals=joint_goals,
    ), protected_goals


def _protected_requirement(
    context: PlannerContext,
    protected_goals: tuple[Goal, ...],
    *,
    runs: int,
    seed: int | None,
) -> int | None:
    """Find the smallest future resource pool that protects every goal.

    The reserve is measured independently of current spending. The future
    state starts at zero pity, with no guarantee or Capturing Radiance. This
    is the conservative post-character state: obtaining the current target
    ends that banner at zero pity, while future guarantee/Radiance advantages
    should not be required to justify spending now.
    """
    if not protected_goals:
        return None

    required = min(protected_goals, key=lambda goal: goal.priority)
    evaluation = next(
        item for item in evaluate_goals(context) if item.goal == required
    )
    banner = evaluation.next_banner
    if banner is None:
        return None

    if required.target.kind is TargetKind.WEAPON:
        # A weapon goal's reserve comes from the exact weapon probability
        # engine (§10 weapon): the character Monte Carlo cannot execute a
        # weapon banner, so the simulated search below would be
        # meaningless. The fresh-state reserve is the same quantity that
        # search looks for (pity 0, no guarantee, no Fate Points) and is
        # drawn from the SAME account wish pool, so character and weapon
        # goals never receive two independent reserves.
        return weapon_goal_reserve(context, evaluation.copies_needed)

    future_context = replace(
        context,
        account=replace(
            context.account,
            wishes=0,
            current_pity=0,
            character_guarantee=False,
            capturing_radiance_counter=0,
        ),
        income=None,
    )

    cache: dict[int, float] = {}

    def probability(wishes: int) -> float:
        if wishes not in cache:
            plan = SpendPlan(
                entries=(
                    PlannedSpend(
                        banner=banner,
                        target_constellation=required.level,
                        budget=wishes,
                    ),
                )
            )
            simulation_context = replace(
                future_context,
                account=replace(future_context.account, wishes=wishes),
            )
            result = simulate(
                character_view(simulation_context),
                plan,
                runs=runs,
                seed=seed,
                joint_goals=(required,),
            )
            cache[wishes] = result.joint_goal_probability.probability
        return cache[wishes]

    # A character target has a finite worst-case guarantee: each missing
    # copy can require at most two hard-pity cycles (one non-featured 5-star
    # followed by the guaranteed featured 5-star). Use that as a hard search
    # bound so a low-confidence/low-run simulation can never make the
    # exponential search grow without limit.
    maximum = (
        evaluation.copies_needed
        * context.mechanics.hard_pity
        * 2
    )
    if probability(maximum) < context.confidence:
        return None

    high = 1
    while high < maximum and probability(high) < context.confidence:
        high = min(high * 2, maximum)

    low = 0
    while low < high:
        mid = (low + high) // 2
        if probability(mid) >= context.confidence:
            high = mid
        else:
            low = mid + 1
    return low


def _safe_spend(
    context: PlannerContext,
    goal: Goal,
    banner: Banner,
    *,
    runs: int,
    seed: int | None,
    simulation_sink: Callable[[Goal, SimulationResult], None] | None = None,
) -> tuple[int, SimulationResult, tuple[Goal, ...]]:
    """Calculate the current spend ceiling from the protected reserve."""
    constraining = constraining_goals(
        context, priority=goal.priority, banner=banner
    )
    required_goals = frozenset(
        evaluation.goal
        for evaluation in evaluate_goals(context)
        if evaluation.goal.priority < goal.priority
        and evaluation.goal.target.kind is TargetKind.CHARACTER
        and evaluation.goal.target != goal.target
        and evaluation.goal.level == 0
        and evaluation.copies_needed > 0
        and any(
            candidate.target == evaluation.goal.target
            for candidate in available_banners(context)
        )
    )
    _, protected_goals = _protection_plan(
        context,
        goal.priority,
        current_banner=banner,
        excluded_goals=required_goals,
    )
    # Only future goals constrain the reserve. Higher-priority goals on a
    # simultaneous current banner are handled by _current_required_plan and
    # consume the same current-phase cap; they are not part of the reserve.
    protected_goals = tuple(
        goal_item for goal_item in protected_goals if goal_item in constraining
    )
    requirement = _protected_requirement(
        context,
        protected_goals,
        runs=runs,
        seed=seed,
    )

    if requirement is None:
        spend = context.account.wishes
    else:
        protected = min(protected_goals, key=lambda item: item.priority)
        evaluation = next(
            item for item in evaluate_goals(context) if item.goal == protected
        )
        assert evaluation.next_banner is not None
        future_income = context.income_available_before(
            evaluation.next_banner.version,
            evaluation.next_banner.phase,
        )
        spend = min(
            context.account.wishes,
            max(
                0,
                context.account.wishes + future_income - requirement,
            ),
        )

    result, _ = _evaluate_spend(
        context,
        goal,
        banner,
        spend,
        runs=runs,
        seed=seed,
    )
    if simulation_sink is not None:
        simulation_sink(goal, result)
    return spend, result, protected_goals

def _weapon_progression_step(
    context: PlannerContext,
    evaluation,
    banner: Banner,
    *,
    runs: int,
    seed: int | None,
) -> tuple[StrategyStep, tuple[Goal, ...]]:
    """The current weapon banner's progression step (Phase 4).

    Weapon banners are never character-simulated (planner.projection), so
    this step is computed with the exact weapon probability engine over the
    SAME shared wish pool the character branch reserves against - the
    reserve itself comes from planner.protection, whose unified sequential
    reserve is target-aware, so a protected character goal and a protected
    weapon goal produce ONE account-level requirement.

    `all_goals_probability` stays None: there is no joint simulation to
    read it from, and reporting a fabricated number would be worse than
    reporting none (§11 provenance).

    Returns the step and the protected goals it reserved for, so the caller
    can apply the same reserve bookkeeping (reserve goal, reserve wishes,
    reserve probability) the character branch performs from its simulation.
    No character simulation runs here, so no `simulation_sink` evidence is
    produced for a weapon step.
    """
    goal = evaluation.goal
    constraining = constraining_goals(
        context, priority=goal.priority, banner=banner
    )
    required_goals = frozenset(
        item.goal
        for item in evaluate_goals(context)
        if item.goal.priority < goal.priority
        and item.goal.target.kind is TargetKind.CHARACTER
        and item.goal.target != goal.target
        and item.goal.level == 0
        and item.copies_needed > 0
        and any(
            candidate.target == item.goal.target
            for candidate in available_banners(context)
        )
    )
    _, protected_goals = _protection_plan(
        context,
        goal.priority,
        current_banner=banner,
        excluded_goals=required_goals,
    )
    protected_goals = tuple(
        item for item in protected_goals if item in constraining
    )

    spend = context.account.wishes
    reserve_income = 0
    requirement = _protected_requirement(
        context, protected_goals, runs=runs, seed=seed
    )
    if requirement is not None:
        protected = min(protected_goals, key=lambda item: item.priority)
        protected_evaluation = next(
            item for item in evaluate_goals(context) if item.goal == protected
        )
        assert protected_evaluation.next_banner is not None
        reserve_income = context.income_available_before(
            protected_evaluation.next_banner.version,
            protected_evaluation.next_banner.phase,
        )
        spend = min(
            context.account.wishes,
            max(0, context.account.wishes + reserve_income - requirement),
        )

    # The protected goals' own confidence at this spend, from the unified
    # sequential reserve (character and weapon goals alike).
    outcomes = {
        outcome.goal: outcome
        for outcome in protected_goal_outcomes(context, spent=spend, banner=banner)
    }
    protected_probability = None
    if protected_goals:
        confidences = [
            outcomes[item].confidence
            for item in protected_goals
            if item in outcomes
        ]
        if confidences:
            protected_probability = min(confidences)

    weapon_state = context.account.weapon_state
    owned = context.account.owned_characters.owned_refinement(goal.target.name)
    curve = refinement_cumulative_probability(
        spend,
        owned,
        goal.level,
        weapon_state.pity,
        weapon_state.guarantee,
        weapon_state.fate_points,
        WEAPON_EVENT_BANNER,
    )

    reserve = context.account.wishes - spend
    return (
        StrategyStep(
            action="pursue_until_reserve",
            goal=goal,
            banner=banner,
            reserve_wishes=reserve,
            safe_spend=spend,
            outcome_probability=float(curve[spend]),
            protected_probability=protected_probability,
            future_income=reserve_income,
            protected_starting_wishes=reserve,
            protected_total_wishes=reserve + reserve_income,
        ),
        protected_goals,
    )


def build_strategy(
    context: PlannerContext,
    *,
    runs: int = 2_000,
    seed: int | None = 0,
    simulation_sink: Callable[[Goal, SimulationResult], None] | None = None,
) -> PullStrategy:
    """Build the current multi-step strategy and spend frontiers."""
    banners = available_banners(context)
    if not banners:
        return PullStrategy((), None, None, None, runs, seed, context.account.wishes, 0)

    current_targets = {banner.target for banner in banners}
    evaluations = evaluate_goals(context)
    # The "get" step names the current banner's first acquisition: the
    # character's C0. A weapon's first acquisition is R1, which is a level
    # above R0, so every current-banner weapon goal is handled by the
    # analytic progression branch below (unified: one strategy, one shared
    # wish pool, globally ordered by priority).
    current_goals = [
        evaluation for evaluation in evaluations
        if evaluation.goal.target in current_targets
        and evaluation.goal.target.kind is TargetKind.CHARACTER
        and evaluation.goal.level == 0
        and evaluation.copies_needed > 0
    ]
    current_goals.sort(key=lambda evaluation: evaluation.goal.priority)

    steps: list[StrategyStep] = []
    for evaluation in current_goals:
        banner = next(
            banner for banner in banners
            if banner.target == evaluation.goal.target
        )
        steps.append(StrategyStep(
            action="get", goal=evaluation.goal, banner=banner,
            safe_spend=context.account.wishes,
        ))

    progression = [
        evaluation for evaluation in evaluations
        if evaluation.goal.target in current_targets
        and (
            evaluation.goal.target.kind is TargetKind.WEAPON
            or evaluation.goal.level > 0
        )
        and evaluation.copies_needed > 0
    ]
    progression.sort(key=lambda evaluation: evaluation.goal.priority)

    reserve_goal = None
    reserve_wishes = None
    reserve_probability = None

    for evaluation in progression:
        banner = next(
            banner for banner in banners
            if banner.target == evaluation.goal.target
        )
        if evaluation.goal.target.kind is TargetKind.WEAPON:
            # One strategy decision, two execution branches: a weapon banner
            # cannot be executed by the character Monte Carlo, so its step is
            # computed with the exact weapon probability engine over the SAME
            # shared wish pool (Phase 4). Protection, reserves and the
            # protected goal all come from the unified planner layer, so the
            # weapon step is not a competing second recommendation.
            weapon_step, weapon_protected = _weapon_progression_step(
                context,
                evaluation,
                banner,
                runs=runs,
                seed=seed,
            )
            steps.append(weapon_step)
            if reserve_goal is None and weapon_protected:
                reserve_goal = min(
                    weapon_protected, key=lambda item: item.priority
                )
                reserve_wishes = weapon_step.reserve_wishes
                reserve_probability = weapon_step.protected_probability
            continue
        spend, result, protected_goals = _safe_spend(
            context,
            evaluation.goal,
            banner,
            runs=runs,
            seed=seed,
            simulation_sink=simulation_sink,
        )
        protected_probability = None
        if protected_goals:
            protected_probability = min(
                next(item for item in result.goals if item.goal == protected).probability
                for protected in protected_goals
            )
        reserve = context.account.wishes - spend
        reserve_income = 0
        if protected_goals:
            reserve_goal_for_display = min(protected_goals, key=lambda item: item.priority)
            reserve_evaluation = next(
                item for item in evaluate_goals(context)
                if item.goal == reserve_goal_for_display
            )
            if reserve_evaluation.next_banner is not None:
                reserve_income = context.income_available_before(
                    reserve_evaluation.next_banner.version,
                    reserve_evaluation.next_banner.phase,
                )
        reserve_total = reserve + reserve_income
        if reserve_goal is None and protected_goals:
            reserve_goal = min(protected_goals, key=lambda item: item.priority)
            reserve_wishes = reserve
            reserve_probability = protected_probability
        steps.append(StrategyStep(
            action="pursue_until_reserve", goal=evaluation.goal, banner=banner,
            reserve_wishes=reserve,
            safe_spend=spend,
            outcome_probability=(
                result.joint_goal_probability.probability
                if result.joint_goal_probability is not None
                else 0.0
            ),
            # This is the roadmap-wide probability from the SAME executable
            # plan that produced outcome_probability. The recommendation
            # endpoint evaluates a different candidate plan, so its
            # all_goals_probability is not the strategy shown here.
            all_goals_probability=result.all_goals_probability,
            protected_probability=protected_probability,
            future_income=reserve_income,
            protected_starting_wishes=reserve,
            protected_total_wishes=reserve_total,
        ))

    if reserve_goal is not None:
        reserve_banner = min(
            (banner for banner in context.roadmap.banners
             if banner.target == reserve_goal.target
             and banner.order_key > banners[0].order_key),
            key=lambda banner: banner.order_key,
        )
        steps.append(StrategyStep("save", reserve_goal, reserve_banner))

    return PullStrategy(
        steps=tuple(steps), reserve_goal=reserve_goal,
        reserve_wishes=reserve_wishes, reserve_probability=reserve_probability,
        runs=runs, seed=seed,
        starting_wishes=context.account.wishes,
        future_income=(
            context.income_available_before(
                reserve_banner.version,
                reserve_banner.phase,
            )
            if reserve_goal is not None
            else 0
        ),
    )
