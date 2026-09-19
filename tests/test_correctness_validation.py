"""Correctness validation for the probability engine and Monte Carlo simulator.

This suite targets the "Math / planner correctness" items from the personal
punch-list:

    * validate simulation against the exact analytical engine (an
      "external calculator" that is exact, not another Monte Carlo)
    * validate multi-copy targets (e.g. Skirk C2) via the exact state-based
      analytical reference
    * exercise Capturing Radiance (soft guarantee at counter==2, hard
      guarantee at counter>=3) and the 50/50 loss transition
    * exercise edge cases: starting guaranteed, losing the 50/50, a goal
      that is already complete, multiple 5-star events in one banner,
      and very low / very high wish budgets
    * sanity-check reserve/income timing (PlannerContext) at the 450-wish
      scale mentioned in the punch list

Every Monte Carlo comparison uses a binomial confidence interval rather
than a fixed epsilon, so the suite does not become flaky as `runs` changes
and does not silently pass by using a huge tolerance.
"""

import math

import numpy as np
import pytest

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
    WishMechanics,
)
from domain.mechanics import CHARACTER_EVENT_BANNER
from planner.context import PlannerContext
from probability import cumulative_probability, multi_copy_cumulative_probability, pull_rate
from simulation import PlannedSpend, SpendPlan, simulate


MECHANICS = CHARACTER_EVENT_BANNER


def _context(
    *,
    wishes: int,
    pity: int = 0,
    guarantee: bool = False,
    radiance: int = 0,
    owned: dict | None = None,
    mechanics: WishMechanics = MECHANICS,
    income: IncomeForecast | None = None,
    current_version: str = "7.0",
) -> PlannerContext:
    account = Account(
        current_pity=pity,
        character_guarantee=guarantee,
        owned_characters=Ownership(owned or {}),
        wishes=wishes,
        capturing_radiance_counter=radiance,
    )
    roadmap = Roadmap(
        goals=[Goal("Target", 5, 1)],  # unreachable dummy goal, unused directly
        banners=[Banner("Target", current_version, 1)],
    )
    return PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version=current_version,
        current_phase=1,
        mechanics=mechanics,
        income=income,
    )


def _wilson_interval(successes: int, n: int, z: float = 4.0) -> tuple[float, float]:
    """A wide (z=4 => ~99.994% CI) Wilson score interval.

    Deliberately wide: this suite must not become flaky because of Monte
    Carlo noise, so it only fails on a real discrepancy between the
    simulator and the exact analytical / combinatorial reference.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = phat + z ** 2 / (2 * n)
    spread = z * math.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)
    low = (centre - spread) / denom
    high = (centre + spread) / denom
    return max(0.0, low), min(1.0, high)


def _assert_matches(simulated_p: float, runs: int, reference_p: float, label: str) -> None:
    successes = round(simulated_p * runs)
    low, high = _wilson_interval(successes, runs)
    assert low - 1e-9 <= reference_p <= high + 1e-9, (
        f"{label}: simulated {simulated_p:.4f} (n={runs}) does not bracket "
        f"the reference probability {reference_p:.4f} (Wilson CI "
        f"[{low:.4f}, {high:.4f}])"
    )


# ---------------------------------------------------------------------------
# Single-copy: simulator must reproduce the exact DP curve.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "pity,guaranteed,wishes",
    [
        (0, False, 90),      # fresh account, one full cycle
        (0, True, 90),       # guaranteed at pity 0
        (73, False, 20),     # right at soft pity start
        (89, False, 1),      # exactly at hard pity: must be deterministic
        (30, True, 60),      # mid-pity, guaranteed
        (0, False, 1),       # single wish, very low budget
    ],
)
def test_single_copy_matches_analytical(pity, guaranteed, wishes):
    runs = 20_000
    context = _context(wishes=wishes, pity=pity, guarantee=guaranteed)
    plan = SpendPlan(
        entries=(
            PlannedSpend(banner=Banner("Target", "7.0", 1), target_constellation=0, budget=wishes),
        )
    )
    result = simulate(context, plan, runs=runs, seed=1)
    simulated = result.banners[0].target_met_probability

    reference = float(
        cumulative_probability(wishes, pity, guaranteed, MECHANICS)[wishes]
    )
    _assert_matches(simulated, runs, reference, f"pity={pity} guarantee={guaranteed} wishes={wishes}")


def test_hard_pity_is_deterministic():
    """At pity 89 the next pull is guaranteed a 5-star (§10.1)."""
    context = _context(wishes=1, pity=89, guarantee=True)
    plan = SpendPlan(
        entries=(PlannedSpend(banner=Banner("Target", "7.0", 1), target_constellation=0, budget=1),)
    )
    result = simulate(context, plan, runs=2_000, seed=2)
    assert result.banners[0].target_met_probability == pytest.approx(1.0)


def test_zero_wishes_never_succeeds():
    """Very low wish count edge case: 0 wishes must yield 0 probability."""
    context = _context(wishes=0, pity=0, guarantee=False)
    plan = SpendPlan(
        entries=(PlannedSpend(banner=Banner("Target", "7.0", 1), target_constellation=0, budget=0),)
    )
    result = simulate(context, plan, runs=1_000, seed=3)
    assert result.banners[0].target_met_probability == 0.0
    assert result.banners[0].mean_wishes_spent == 0.0


# ---------------------------------------------------------------------------
# Multi-copy (e.g. Skirk C2): validate against the exact state-based
# analytical probability model, including Capturing Radiance carryover.
# ---------------------------------------------------------------------------


def test_large_budget_450_wishes_reaches_near_certainty():
    """The 450-wish scale explicitly called out in the punch list."""
    runs = 5_000
    context = _context(wishes=450, pity=0, guarantee=False)
    banner = Banner("Skirk", "7.1", 2)
    context = PlannerContext(
        account=context.account,
        roadmap=Roadmap(goals=[Goal("Skirk", 2, 1)], banners=[banner]),
        current_version="7.1",
        current_phase=2,
        mechanics=MECHANICS,
    )
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=2, budget=450),))
    result = simulate(context, plan, runs=runs, seed=5)

    reference = float(multi_copy_cumulative_probability(450, 3, 0, False, MECHANICS)[450])
    simulated = result.banners[0].target_met_probability
    _assert_matches(simulated, runs, reference, "Skirk C2 (3 copies) within 450 wishes")
    # 450 wishes is ~5 hard-pity cycles for 3 copies: should be very likely,
    # though not the near-certainty a naive "5 cycles for 3 copies" estimate
    # suggests once soft pity and 50/50 losses are modeled exactly.
    assert reference > 0.95


# ---------------------------------------------------------------------------
# Goal already complete: must spend nothing and report success trivially.
# ---------------------------------------------------------------------------

def test_goal_already_satisfied_spends_nothing():
    context = _context(wishes=200, pity=40, guarantee=False, owned={"Skirk": 2})
    banner = Banner("Skirk", "7.0", 1)
    context = PlannerContext(
        account=context.account,
        roadmap=Roadmap(goals=[Goal("Skirk", 1, 1)], banners=[banner]),
        current_version="7.0",
        current_phase=1,
        mechanics=MECHANICS,
    )
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=1, budget=200),))
    result = simulate(context, plan, runs=500, seed=6)
    aggregate = result.banners[0]
    assert aggregate.target_met_probability == 1.0
    assert aggregate.mean_wishes_spent == 0.0
    assert aggregate.mean_copies_obtained == 0.0


# ---------------------------------------------------------------------------
# 50/50 loss transition and Capturing Radiance.
# ---------------------------------------------------------------------------

def test_losing_the_5050_sets_pity_zero_and_guarantee():
    """At hard pity with no guarantee, a lost 50/50 must reset pity to 0,
    set guarantee True, and bump the Capturing Radiance counter - never
    leave pity elevated or guarantee False (§11 transition rules).
    """
    runs = 4_000
    context = _context(wishes=1, pity=89, guarantee=False, radiance=0)
    banner = Banner("Target", "7.0", 1)
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=5, budget=1),))
    result = simulate(context, plan, runs=runs, seed=7)

    histories = result.histories
    lost_count = 0
    for history in histories:
        after = history.banner_results[0].account_after
        obtained = history.banner_results[0].copies_obtained
        if obtained == 0:
            lost_count += 1
            assert after.current_pity == 0
            assert after.character_guarantee is True
            assert after.capturing_radiance_counter == 1
        else:
            assert after.current_pity == 0
            assert after.character_guarantee is False
            assert after.capturing_radiance_counter == 0

    # Roughly half should lose the 50/50 (mechanics.featured_rate == 0.5).
    _assert_matches(lost_count / runs, runs, 0.5, "P(lose 50/50 at hard pity)")


def test_capturing_radiance_soft_guarantee_at_two():
    """At counter==2, a lost 5-star becomes featured with probability 6/11,
    not the plain 50% base rate (§11 Capturing Radiance rule).
    """
    runs = 20_000
    context = _context(wishes=1, pity=89, guarantee=False, radiance=2)
    banner = Banner("Target", "7.0", 1)
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=5, budget=1),))
    result = simulate(context, plan, runs=runs, seed=8)

    featured_count = sum(
        1 for history in result.histories
        if history.banner_results[0].copies_obtained == 1
    )
    _assert_matches(featured_count / runs, runs, 6.0 / 11.0, "Capturing Radiance counter=2")


def test_capturing_radiance_hard_guarantee_at_three():
    """At counter>=3, the next 5-star is always featured, guarantee or not.

    target_constellation=5 with an unowned character means copies_needed=6,
    so the 1-wish budget can never satisfy the whole target - this test is
    about the single forced pull, so it checks copies_obtained directly
    rather than target_met (which also depends on copies_needed).
    """
    context = _context(wishes=1, pity=89, guarantee=False, radiance=3)
    banner = Banner("Target", "7.0", 1)
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=5, budget=1),))
    result = simulate(context, plan, runs=2_000, seed=9)
    for history in result.histories:
        assert history.banner_results[0].copies_obtained == 1
        assert history.banner_results[0].account_after.character_guarantee is False
        assert history.banner_results[0].account_after.current_pity == 0


def test_multiple_five_star_outcomes_recorded_in_order():
    """A multi-copy target on a low-hard-pity banner should regularly hit
    several 5-star events (both losses and wins) within one banner; every
    recorded event must be internally consistent with copies_obtained.
    """
    fast_mechanics = WishMechanics(
        banner_type="test",
        hard_pity=10,
        soft_pity_start=8,
        base_rate=0.3,
        soft_pity_increment=0.2,
        featured_rate=0.5,
    )
    context = _context(wishes=60, pity=0, guarantee=False, mechanics=fast_mechanics)
    banner = Banner("Target", "7.0", 1)
    context = PlannerContext(
        account=context.account,
        roadmap=Roadmap(goals=[Goal("Target", 3, 1)], banners=[banner]),
        current_version="7.0",
        current_phase=1,
        mechanics=fast_mechanics,
    )
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=3, budget=60),))
    result = simulate(context, plan, runs=1_000, seed=10)

    saw_multiple_five_stars = False
    for history in result.histories:
        banner_result = history.banner_results[0]
        outcomes = banner_result.five_star_outcomes
        if len(outcomes) > 1:
            saw_multiple_five_stars = True
        # Wish indices must be strictly increasing and within budget.
        wishes_seen = [wish for wish, _ in outcomes]
        assert wishes_seen == sorted(wishes_seen)
        assert all(1 <= wish <= banner_result.wishes_spent for wish in wishes_seen)
        # Exactly the featured entries correspond to copies obtained.
        featured_events = sum(1 for _, featured in outcomes if featured)
        assert featured_events == banner_result.copies_obtained
        # copy_wishes must be exactly the wish indices of featured events.
        assert banner_result.copy_wishes == tuple(
            wish for wish, featured in outcomes if featured
        )
    assert saw_multiple_five_stars, (
        "expected at least one simulated history with multiple 5-star events "
        "given the fast test mechanics"
    )


# ---------------------------------------------------------------------------
# Reserve / income accounting sanity (PlannerContext.income_available_before).
# ---------------------------------------------------------------------------

def test_current_version_income_not_available_to_current_phase():
    """Forecast income for the current version must not inflate the
    current-phase spendable budget (planner.context docstring / §16).
    """
    income = IncomeForecast(
        versions=[VersionIncome(version="7.1", estimate=IncomeEstimate(low=60, expected=90, high=120))]
    )
    context = _context(wishes=100, current_version="7.1", income=income)
    context = PlannerContext(
        account=context.account,
        roadmap=context.roadmap,
        current_version="7.1",
        current_phase=1,
        income=income,
        mechanics=MECHANICS,
    )
    # Same version/phase as "now": not available yet.
    assert context.income_available_before("7.1", 1) == 0
    # A later phase: this version's income has arrived by then.
    assert context.income_available_before("7.1", 2) == 90
    # A later version: also arrived.
    assert context.income_available_before("7.2", 1) == 90


def test_income_credit_excludes_versions_before_current():
    income = IncomeForecast(
        versions=[
            VersionIncome(version="7.0", estimate=IncomeEstimate(low=10, expected=10, high=10)),
            VersionIncome(version="7.1", estimate=IncomeEstimate(low=60, expected=90, high=120)),
        ]
    )
    context = _context(wishes=0, current_version="7.1", income=income)
    context = PlannerContext(
        account=context.account,
        roadmap=context.roadmap,
        current_version="7.1",
        current_phase=1,
        income=income,
        mechanics=MECHANICS,
    )
    # 7.0 is before the current version and must never be credited.
    assert context.income_credit("7.1") == 90
    assert context.income_credit("7.2") == 90


def test_income_flows_into_simulated_banner_budget():
    """End-to-end: income configured for a future banner actually raises
    the wishes available when the simulator reaches that banner.
    """
    income = IncomeForecast(
        versions=[VersionIncome(version="7.1", estimate=IncomeEstimate(low=90, expected=90, high=90))]
    )
    account = Account(wishes=0, current_pity=0, character_guarantee=False)
    roadmap = Roadmap(
        goals=[Goal("Later", 0, 1)],
        # A banner is required at the current (version, phase) too, even
        # though the plan spends nothing on it - it is just where "now" is.
        banners=[Banner("Now", "7.0", 1), Banner("Later", "7.1", 1)],
    )
    context = PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version="7.0",
        current_phase=1,
        income=income,
        income_scenario="expected",
        mechanics=MECHANICS,
    )
    banner = roadmap.banners[1]
    plan = SpendPlan(entries=(PlannedSpend(banner=banner, target_constellation=0, budget=90),))
    result = simulate(context, plan, runs=2_000, seed=11)
    # All 90 income wishes should have been credited exactly once, on the
    # "Later" banner (index 1) - not on the current "Now" banner.
    assert result.banners[0].mean_income_credited == pytest.approx(0.0)
    assert result.banners[1].mean_income_credited == pytest.approx(90.0)
    reference = float(cumulative_probability(90, 0, False, MECHANICS)[90])
    _assert_matches(result.banners[1].target_met_probability, 2_000, reference, "income-funded banner")

def test_multicopy_exact_model_carries_capturing_radiance_between_copies():
    """Multi-copy probability must preserve Radiance after the first copy."""
    mechanics = WishMechanics(banner_type="test", hard_pity=1, soft_pity_start=1, base_rate=0.5, soft_pity_increment=0.0, featured_rate=0.5)
    curve = multi_copy_cumulative_probability(2, 2, 0, False, mechanics, starting_radiance=2)
    # Wish 1: 6/11 featured -> Radiance 1, then 1/2 featured on wish 2.
    # Otherwise 5/11 loses -> wish 2 is guaranteed. Total = 8/11.
    assert curve[2] == pytest.approx(8.0 / 11.0)