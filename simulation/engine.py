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
    Aggregation   "Across N futures, how often?"                simulation.outcomes
    Phase 5       "Which strategy should we run?"               not here (§13)

Account state transitions (§4, §11): the simulator never mutates domain
state. Every 5-star rebuilds the frozen Account:

    featured copy -> pity 0, guarantee off, ownership +1
                     (NOT_OWNED -1 becomes C0, §4.2)
    lost 50/50    -> pity 0, guarantee on
    no 5-star     -> pity +1, guarantee unchanged

All per-pull rates come from `probability.pull_rate`; no rate mathematics
lives here (§17). Capturing Radiance's win-probability schedule and its
loss-streak-counter transition rule are likewise single-sourced from
`probability.capturing_radiance_rate` and
`domain.next_capturing_radiance_counter` respectively, so this module, the
Phase 2 analytic engine, and the account-update API cannot drift apart on
the mechanic (see those functions' docstrings for the calibration).

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

from domain import (
    Account,
    Banner,
    Goal,
    Ownership,
    WishMechanics,
    copies_needed_for,
    next_capturing_radiance_counter,
)
from planner.banners import available_banners, current_banner
from planner.context import PlannerContext
from probability import capturing_radiance_rate, pull_rate, pull_rate_array
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

    Capturing Radiance's win probability at each state comes from
    `probability.capturing_radiance_rate`, and the counter's transition
    after each pull comes from `domain.next_capturing_radiance_counter` -
    both shared with the Phase 2 analytic engine and the account-update API
    so the three cannot silently disagree on the mechanic.

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
            featured = (
                True
                if was_guaranteed
                else rng.random() < capturing_radiance_rate(radiance, mechanics)
            )

            five_star_outcomes.append((spent, featured))
            radiance = next_capturing_radiance_counter(
                radiance, was_guaranteed=was_guaranteed, featured=featured
            )
            if featured:
                obtained += 1
                copy_wishes.append(spent)
                owned += 1
                ownership = ownership.with_constellation(character, owned)
                pity, guarantee = 0, False
            else:
                pity, guarantee = 0, True
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


def _pull_toward_target_vectorized(
    current_pity: np.ndarray,
    guarantee: np.ndarray,
    radiance: np.ndarray,
    wishes: np.ndarray,
    owned: np.ndarray,
    copies_needed: np.ndarray,
    budget: np.ndarray,
    mechanics: WishMechanics,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[list[tuple[int, bool]]],
]:
    """Simulate one banner across many independent histories.

    This is the vectorized banner boundary: all per-history state is stored
    in NumPy arrays, while Python loops only over the maximum number of
    wishes any history can spend. Result reconstruction belongs to a later
    step.
    """
    current_pity = np.asarray(current_pity, dtype=int).copy()
    guarantee = np.asarray(guarantee, dtype=bool).copy()
    radiance = np.asarray(radiance, dtype=int).copy()
    wishes = np.asarray(wishes, dtype=int).copy()
    owned = np.asarray(owned, dtype=int).copy()
    copies_needed = np.asarray(copies_needed, dtype=int)
    budget = np.asarray(budget, dtype=int)

    if not (
        current_pity.shape
        == guarantee.shape
        == radiance.shape
        == wishes.shape
        == owned.shape
        == copies_needed.shape
        == budget.shape
    ):
        raise ValueError("all vectorized banner state arrays must have the same shape")

    available = np.minimum(wishes, budget)
    spent = np.zeros_like(wishes)
    obtained = np.zeros_like(copies_needed)
    five_star_outcomes: list[list[tuple[int, bool]]] = [
        [] for _ in range(wishes.size)
    ]

    # Capturing Radiance is single-sourced, not re-derived: the featured
    # win probability per counter state comes from
    # probability.capturing_radiance_rate, and the counter transition per
    # (previous counter, was guaranteed, was featured) comes from
    # domain.next_capturing_radiance_counter - the exact functions the
    # scalar engine calls. Both tables are tiny (4 and 4x2x2 entries), so
    # building them once per banner call is free; the per-history work
    # stays fully vectorized array indexing.
    radiance_rates = np.array(
        [capturing_radiance_rate(r, mechanics) for r in range(4)], dtype=float
    )
    radiance_next = np.empty((4, 2, 2), dtype=int)
    for previous in range(4):
        for was_guaranteed in (False, True):
            for featured in (False, True):
                radiance_next[previous, int(was_guaranteed), int(featured)] = (
                    next_capturing_radiance_counter(
                        previous,
                        was_guaranteed=was_guaranteed,
                        featured=featured,
                    )
                )

    max_steps = int(available.max(initial=0))
    for _ in range(max_steps):
        active = (obtained < copies_needed) & (spent < available)
        if not np.any(active):
            break

        spent[active] += 1

        five_star = np.zeros_like(active)
        five_star[active] = (
            rng.random(np.count_nonzero(active))
            < pull_rate_array(current_pity[active], mechanics)
        )

        if np.any(five_star):
            star_indices = np.flatnonzero(five_star)
            was_guaranteed = guarantee[star_indices].copy()

            featured_draw = rng.random(star_indices.size)
            featured = was_guaranteed | (
                featured_draw < radiance_rates[radiance[star_indices]]
            )

            for index, is_featured in zip(star_indices, featured):
                five_star_outcomes[index].append(
                    (int(spent[index]), bool(is_featured))
                )

            previous_radiance = radiance[star_indices]
            radiance[star_indices] = radiance_next[
                previous_radiance,
                was_guaranteed.astype(int),
                featured.astype(int),
            ]
            current_pity[star_indices] = 0
            guarantee[star_indices] = ~featured

            featured_indices = star_indices[featured]
            obtained[featured_indices] += 1
            owned[featured_indices] += 1
        non_five_star = active & ~five_star
        current_pity[non_five_star] += 1

    return (
        spent,
        obtained,
        current_pity,
        guarantee,
        radiance,
        owned,
        five_star_outcomes,
    )


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



def _simulate_scalar(
    context: PlannerContext,
    plan: SpendPlan,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
    joint_goals=(),
) -> SimulationResult:
    """Reference simulation using the original one-history-at-a-time engine."""
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    plan.require_valid_for(context)
    rng = np.random.default_rng(seed)
    histories = [_run_history(context, plan, rng) for _ in range(runs)]
    return aggregate_runs(histories, plan, seed, joint_goals=joint_goals)



def _simulate_vectorized(
    context: PlannerContext,
    plan: SpendPlan,
    runs: int,
    seed: int | None,
    joint_goals=(),
) -> SimulationResult:
    """Simulate many histories with banner-wise vectorization."""
    banners = _banners_to_process(context, plan)
    mechanics = context.mechanics
    rng = np.random.default_rng(seed)

    current_pity = np.full(runs, context.account.current_pity, dtype=int)
    guarantee = np.full(runs, context.account.character_guarantee, dtype=bool)
    radiance = np.full(
        runs, context.account.capturing_radiance_counter, dtype=int
    )
    wishes = np.full(runs, context.account.wishes, dtype=int)

    characters = (
        {banner.character for banner in banners}
        | {goal.character for goal in context.roadmap.goals_in_priority_order()}
        | set(context.account.owned_characters.characters)
    )
    owned_by_character = {
        character: np.full(
            runs,
            context.account.owned_constellation(character),
            dtype=int,
        )
        for character in characters
    }

    credited_so_far = 0
    shared_current_phase_spent = np.zeros(runs, dtype=int)
    banner_results: list[list[BannerResult]] = [[] for _ in range(runs)]

    # Per-run Ownership objects, mirroring the scalar engine's reconstruction
    # exactly (§4.2): every run's Ownership starts as the account's own frozen
    # object (cheap: one shared reference for the all-starting-state case) and
    # is rebuilt only for the runs that actually won a copy on a banner. A
    # character the plan/goals merely mention is NOT auto-added with
    # NOT_OWNED; only characters the run wins join the key set.
    starting_ownership = context.account.owned_characters
    ownership_by_run: list[Ownership] = [starting_ownership] * runs

    goals = context.roadmap.goals_in_priority_order()
    satisfied = np.zeros((len(goals), runs), dtype=bool)
    satisfied_after: list[list[Banner | None]] = [
        [None] * runs for _ in goals
    ]
    for goal_index, goal in enumerate(goals):
        satisfied[goal_index] = (
            owned_by_character[goal.character] >= goal.constellation
        )

    for banner in banners:
        income_credited = 0
        credit = context.income_available_before(banner.version, banner.phase)
        if credit > credited_so_far:
            income_credited = credit - credited_so_far
            credited_so_far = credit
            if income_credited:
                wishes += income_credited

        entry = plan.entry_for(banner)
        copies_needed = np.zeros(runs, dtype=int)
        budget = np.zeros(runs, dtype=int)
        target_constellation: int | None = None

        if entry is not None:
            target_constellation = entry.target_constellation
            owned = owned_by_character[banner.character]
            copies_needed = np.maximum(entry.target_constellation - owned, 0)
            budget.fill(entry.budget)

            if (
                plan.shared_current_phase_budget is not None
                and banner.version == context.current_version
                and banner.phase == context.current_phase
            ):
                budget = np.minimum(
                    budget,
                    np.maximum(
                        plan.shared_current_phase_budget
                        - shared_current_phase_spent,
                        0,
                    ),
                )

            (
                spent,
                obtained,
                current_pity,
                guarantee,
                radiance,
                updated_owned,
                five_star_outcomes,
            ) = _pull_toward_target_vectorized(
                current_pity,
                guarantee,
                radiance,
                wishes,
                owned,
                copies_needed,
                budget,
                mechanics,
                rng,
            )
            wishes -= spent

            # Rebuild each winning run's Ownership exactly once per banner:
            # only runs whose constellation actually changed get a new frozen
            # Ownership object; every other run keeps its previous one.
            wins = np.flatnonzero(updated_owned > owned)
            if wins.size:
                character = banner.character
                for run_index in wins:
                    ownership_by_run[run_index] = Ownership(
                        {
                            **ownership_by_run[run_index].characters,
                            character: int(updated_owned[run_index]),
                        }
                    )
            owned_by_character[banner.character] = updated_owned

            if (
                plan.shared_current_phase_budget is not None
                and banner.version == context.current_version
                and banner.phase == context.current_phase
            ):
                shared_current_phase_spent += spent
        else:
            spent = np.zeros(runs, dtype=int)
            obtained = np.zeros(runs, dtype=int)
            five_star_outcomes = [[] for _ in range(runs)]

        for goal_index, goal in enumerate(goals):
            newly_satisfied = (
                ~satisfied[goal_index]
                & (owned_by_character[goal.character] >= goal.constellation)
            )
            for run_index in np.flatnonzero(newly_satisfied):
                satisfied_after[goal_index][run_index] = banner
            satisfied[goal_index] |= newly_satisfied

        planned_budget = entry.budget if entry is not None else 0
        # One .tolist() per banner beats one NumPy-scalar extraction per
        # element: the per-run loop below reads Python ints only.
        pity_list = current_pity.tolist()
        guarantee_list = guarantee.tolist()
        wishes_list = wishes.tolist()
        radiance_list = radiance.tolist()
        copies_list = copies_needed.tolist()
        spent_list = spent.tolist()
        obtained_list = obtained.tolist()

        for run_index in range(runs):
            ownership = ownership_by_run[run_index]
            account_after = Account(
                current_pity=pity_list[run_index],
                character_guarantee=guarantee_list[run_index],
                owned_characters=ownership,
                wishes=wishes_list[run_index],
                capturing_radiance_counter=radiance_list[run_index],
            )
            outcomes = five_star_outcomes[run_index]
            banner_results[run_index].append(
                BannerResult(
                    banner=banner,
                    target_constellation=target_constellation,
                    budget=planned_budget,
                    copies_needed=copies_list[run_index],
                    income_credited=income_credited,
                    wishes_spent=spent_list[run_index],
                    copies_obtained=obtained_list[run_index],
                    copy_wishes=tuple(
                        wish for wish, featured in outcomes if featured
                    ),
                    target_met=(
                        entry is not None
                        and obtained_list[run_index] >= copies_list[run_index]
                    ),
                    account_after=account_after,
                    five_star_outcomes=tuple(outcomes),
                )
            )

    histories = tuple(
        RunResult(
            banner_results=tuple(banner_results[run_index]),
            account_after=banner_results[run_index][-1].account_after,
            goal_outcomes=tuple(
                GoalOutcome(
                    goal=goal,
                    satisfied=bool(satisfied[goal_index, run_index]),
                    satisfied_after=satisfied_after[goal_index][run_index],
                )
                for goal_index, goal in enumerate(goals)
            ),
        )
        for run_index in range(runs)
    )
    return aggregate_runs(histories, plan, seed, joint_goals=joint_goals)


def simulate(
    context: PlannerContext,
    plan: SpendPlan,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
    joint_goals=(),
) -> SimulationResult:
    """Simulate runs possible futures and aggregate them (§11).

    Banners remain sequential while independent histories are vectorized
    within each banner.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    plan.require_valid_for(context)
    return _simulate_vectorized(
        context, plan, runs, seed, joint_goals=joint_goals
    )
