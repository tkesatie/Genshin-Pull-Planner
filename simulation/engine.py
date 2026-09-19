"""The Monte Carlo engine (Design Document §11, §18 Phase 4).

`simulate_history` produces one possible future account history (§11):

    Starting Account
          ↓
    Banner 1   (plan says what to pursue and the spend cap)
          ↓
    Pull outcomes   (Phase 2 rates; mechanics as data, §17)
          ↓
    Updated Account (pity, guarantee, ownership, wishes)
          ↓
    Income
          ↓
    Banner 2
          ↓
    ...

The layers this package keeps separate:

    Phase 2       "What is the probability?"                   probability/
    Phase 4       "What does one possible future look like?"   this module
    Aggregation   "Across N futures, how often?"               simulation.outcomes
    Phase 5       "Which strategy should we run?"              not here (§13)

Account state transitions (§4, §11): the simulator never mutates domain
state. Every 5-star rebuilds the frozen Account:

    featured copy -> pity 0, guarantee off, ownership +1
                     (NOT_OWNED -1 becomes C0, §4.2)
    lost 50/50    -> pity 0, guarantee on
    no 5-star     -> pity +1, guarantee unchanged

All per-pull rates come from `probability.pull_rate`; no rate mathematics
lives here (§17).

Multi-copy targets (§10.4): the plan's `target_constellation` is a desired
resulting constellation, never a copy count (§4.2). `copies_needed` is
computed from the simulated account when the banner arrives; an account
already at or above the target spends nothing.

Spend cap semantics (§12):

    budget        maximum wishes the plan permits on the banner
    available     wishes in the simulated account when the banner arrives
    wishes_spent  min(budget, available, wishes the target still needs)

so wishes never go negative and a cap never forces overspending.

Income timing (§16) - the Phase 3/4 version-level approximation, stated
explicitly so it never reads as an accident:

    Forecast income for the current version is not available to the current
    phase. It becomes usable when the roadmap advances beyond the current
    version/phase, so it cannot inflate the current banner's spendable wishes.
    Within-version arrival order (phase 1 vs phase 2, commissions, events)
    is deliberately not modeled; that granularity is out of Phase 4 scope.
    Income forecast for versions before the current version is presumed
    banked in `account.wishes` and never credited (planner.context).
"""

import numpy as np
from dataclasses import replace

from domain import Account, Banner, Goal, WishMechanics, copies_needed_for
from planner.banners import available_banners, current_banner
from planner.context import PlannerContext
from probability import pull_rate
from simulation.outcomes import aggregate_runs
from simulation.results import BannerResult, GoalOutcome, RunResult, SimulationResult
from simulation.strategy import SpendPlan

DEFAULT_RUNS = 10_000
DEFAULT_SEED = 0


def _banners_to_process(context: PlannerContext, plan: SpendPlan) -> list[Banner]:
    """Every roadmap banner chronologically at or after the current one (§7)."""
    matches = available_banners(context)
    if not matches:
        current = current_banner(context)
    else:
        current = matches[0]
    plan_order = {entry.banner: index for index, entry in enumerate(plan.entries)}
    return sorted(
        (
            banner
            for banner in context.roadmap.banners_in_chronological_order()
            if banner.order_key >= current.order_key
        ),
        key=lambda banner: (banner.order_key, plan_order.get(banner, len(plan.entries))),
    )


def _pull_toward_target(
    account: Account,
    character: str,
    copies_needed: int,
    budget: int,
    mechanics: WishMechanics,
    rng: np.random.Generator,
) -> tuple[int, int, tuple[int, ...], Account]:
    """Pull until `copies_needed` featured copies or the spend cap is hit.

    The cap is min(budget, available): the plan permits at most `budget`,
    and the account cannot spend what it does not have (§12, §4.1). Each
    featured copy bumps ownership by one constellation (NOT_OWNED -1 becomes
    C0, §4.2) and leaves the account at pity 0 with the guarantee off; a
    lost 50/50 leaves it at pity 0 with the guarantee on (§11).

    Returns:
        (wishes_spent, copies_obtained, copy_wishes, account_after) where
        copy_wishes holds the 1-based within-banner wish index of each
        featured copy.
    """
    available = min(budget, account.wishes)
    pity = account.current_pity
    guarantee = account.character_guarantee
    radiance = account.capturing_radiance_counter
    ownership = account.owned_characters
    owned = ownership.owned_constellation(character)

    spent = 0
    obtained = 0
    copy_wishes: list[int] = []
    five_star_outcomes: list[tuple[int, bool]] = []
    while obtained < copies_needed and spent < available:
        spent += 1
        if rng.random() < pull_rate(pity, mechanics):
            was_guaranteed = guarantee
            if guarantee:
                featured = True
            elif radiance >= 3:
                featured = True
            elif radiance == 2:
                featured = rng.random() < (6.0 / 11.0)
            else:
                featured = rng.random() < mechanics.featured_rate

            five_star_outcomes.append((spent, featured))
            if featured:
                obtained += 1
                copy_wishes.append(spent)
                owned += 1
                ownership = ownership.with_constellation(character, owned)
                pity, guarantee = 0, False
                if was_guaranteed:
                    radiance = radiance
                elif radiance >= 3:
                    radiance = 1
                else:
                    radiance = max(0, radiance - 1)
            else:
                pity, guarantee = 0, True
                radiance = min(3, radiance + 1)
        else:
            pity += 1

    account_after = Account(
        current_pity=pity,
        character_guarantee=guarantee,
        owned_characters=ownership,
        wishes=account.wishes - spent,
        capturing_radiance_counter=radiance,
    )
    return spent, obtained, tuple(copy_wishes), tuple(five_star_outcomes), account_after


def _run_history(
    context: PlannerContext, plan: SpendPlan, rng: np.random.Generator
) -> RunResult:
    """Walk one history without re-validating (the caller validated once)."""
    banners = _banners_to_process(context, plan)
    mechanics = context.mechanics

    account = context.account
    credited_so_far = 0  # cumulative income_credit already folded into `account`
    shared_current_phase_spent = 0
    banner_results: list[BannerResult] = []

    # Roadmap outcome tracking (§11): per goal, whether it is satisfied and
    # the first banner after which it became satisfied (None = from start).
    satisfied: dict[Goal, bool] = {}
    satisfied_after: dict[Goal, Banner | None] = {}
    for goal in context.roadmap.goals_in_priority_order():
        satisfied[goal] = copies_needed_for(account, goal) == 0
        satisfied_after[goal] = None

    for banner in banners:
        # Forecast income is future resource: it cannot fund the current
        # phase, but becomes available when the roadmap reaches a later slot.
        income_credited = 0
        credit = context.income_available_before(banner.version, banner.phase)
        if credit > credited_so_far:
            income_credited = credit - credited_so_far
            credited_so_far = credit
            if income_credited > 0:
                account = replace(account, wishes=account.wishes + income_credited)

        entry = plan.entry_for(banner)
        if entry is None:
            banner_results.append(
                BannerResult(
                    banner=banner,
                    target_constellation=None,
                    budget=0,
                    copies_needed=0,
                    income_credited=income_credited,
                    wishes_spent=0,
                    copies_obtained=0,
                    copy_wishes=(),
                    target_met=False,
                    account_after=account,
                    five_star_outcomes=(),
                )
            )
        else:
            owned = account.owned_constellation(banner.character)
            copies_needed = max(entry.target_constellation - owned, 0)
            budget = entry.budget
            if (
                plan.shared_current_phase_budget is not None
                and banner.version == context.current_version
                and banner.phase == context.current_phase
            ):
                budget = min(
                    budget,
                    max(plan.shared_current_phase_budget - shared_current_phase_spent, 0),
                )
            spent, obtained, copy_wishes, five_star_outcomes, account = _pull_toward_target(
                account,
                banner.character,
                copies_needed,
                budget,
                mechanics,
                rng,
            )
            if (
                plan.shared_current_phase_budget is not None
                and banner.version == context.current_version
                and banner.phase == context.current_phase
            ):
                shared_current_phase_spent += spent
            banner_results.append(
                BannerResult(
                    banner=banner,
                    target_constellation=entry.target_constellation,
                    budget=entry.budget,
                    copies_needed=copies_needed,
                    income_credited=income_credited,
                    wishes_spent=spent,
                    copies_obtained=obtained,
                    copy_wishes=copy_wishes,
                    target_met=obtained >= copies_needed,
                    account_after=account,
                    five_star_outcomes=five_star_outcomes,
                )
            )

        for goal in context.roadmap.goals_in_priority_order():
            if not satisfied[goal] and copies_needed_for(account, goal) == 0:
                satisfied[goal] = True
                satisfied_after[goal] = banner

    goal_outcomes = tuple(
        GoalOutcome(
            goal=goal,
            satisfied=satisfied[goal],
            satisfied_after=satisfied_after[goal],
        )
        for goal in context.roadmap.goals_in_priority_order()
    )
    return RunResult(
        banner_results=tuple(banner_results),
        account_after=account,
        goal_outcomes=goal_outcomes,
    )


def simulate_history(
    context: PlannerContext, plan: SpendPlan, rng: np.random.Generator
) -> RunResult:
    """One possible future account history (§11).

    The plan is validated against the context first (§12). Every stochastic
    decision is driven by `rng`, so the same (context, plan, rng state)
    produce the same history.

    Raises:
        ValueError: via `SpendPlan.require_valid_for`, or when the context
            has no roadmap banner at its (version, phase).
    """
    plan.require_valid_for(context)
    return _run_history(context, plan, rng)


def simulate(
    context: PlannerContext,
    plan: SpendPlan,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
    joint_goals=(),
) -> SimulationResult:
    """Simulate `runs` possible futures and aggregate them (§11).

    The plan is validated once against the context. The default seed makes
    re-runs reproducible (§2); `seed=None` draws entropy from the OS
    instead.

    Raises:
        ValueError: when runs < 1, via `SpendPlan.require_valid_for`, or
            when the context has no roadmap banner at its (version, phase).
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    plan.require_valid_for(context)
    rng = np.random.default_rng(seed)
    histories = [_run_history(context, plan, rng) for _ in range(runs)]
    return aggregate_runs(histories, plan, seed, joint_goals=joint_goals)

