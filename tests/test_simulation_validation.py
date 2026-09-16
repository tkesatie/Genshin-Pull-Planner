"""Validation of the simulator against analytical results (§18 Phase 4).

Two independent references:

1. Single-copy equivalence (probability invariant 10, mirrored): the
   empirical P(one featured copy within N) must reproduce the Phase 2
   curve `cumulative_probability` within sampling error, across starting
   pity and guarantee states.

2. Multi-copy reference (§10.4): derive the single-copy completion-time
   PMF from the Phase 2 CDF, f(n) = F(n) - F(n-1) with f(0) = 0, then
   convolve that PMF K times. The time to the K-th featured copy is the
   K-fold convolution of the PMF - never a convolution of the CDF - because
   after any featured copy the state is deterministically (pity 0,
   guarantee off), which makes inter-copy times independent fresh draws:

       C_1(n) = F(n)
       C_K(n) = sum_{i=1..n} f(i) * C_{K-1}(n - i)     (K >= 2)

   The reference CDF is compared against Monte Carlo results. Tiny
   mechanics (hard_pity=5) keep the horizon small and make off-by-one
   errors loud; P(featured within 10) and P(two featured within 20) are
   exactly 1 under them, giving exact anchors.
"""

import numpy as np
import pytest

from domain import CHARACTER_EVENT_BANNER, Account, Banner, Goal, Roadmap, WishMechanics
from planner import PlannerContext
from probability import cumulative_probability
from simulation import PlannedSpend, SpendPlan, simulate

TINY = WishMechanics(
    banner_type="tiny",
    hard_pity=5,
    soft_pity_start=3,
    base_rate=0.25,
    soft_pity_increment=0.2,
    featured_rate=0.5,
)


def completion_pmf(
    start_pity: int, start_guarantee: bool, mechanics: WishMechanics, horizon: int
) -> np.ndarray:
    """f(n) = P(the first featured copy lands exactly on wish n), Phase 2.

    Derived from the Phase 2 CDF: F(n) = cumulative_probability(n, ...),
    f(n) = F(n) - F(n-1), f(0) = 0.
    """
    cdf = cumulative_probability(horizon, start_pity, start_guarantee, mechanics)
    return np.clip(np.diff(cdf, prepend=0.0), 0.0, None)


def k_copy_cdf(
    start_pity: int,
    start_guarantee: bool,
    k: int,
    mechanics: WishMechanics,
) -> np.ndarray:
    """Exact P(K featured copies within n wishes), by convolving the PMF.

    C_1 = F; each further copy convolves the PMF once (see module
    docstring). The horizon covers the worst case: K cycles of
    lose-50/50-then-guaranteed (2 * hard_pity wishes per copy).
    """
    horizon = 2 * k * mechanics.hard_pity
    pmf = completion_pmf(start_pity, start_guarantee, mechanics, horizon)
    cdf = np.cumsum(pmf)
    for _ in range(k - 1):
        cdf = np.convolve(pmf, cdf)[: horizon + 1]
    return cdf


def engine_probability(
    start_pity: int,
    start_guarantee: bool,
    target_constellation: int,
    budget: int,
    mechanics: WishMechanics,
    runs: int = 20_000,
    seed: int = 99,
) -> float:
    """Empirical P(the plan's target is met) through the real simulator."""
    roadmap = Roadmap(
        goals=[Goal("Target", target_constellation, 1)],
        banners=[Banner("Target", "7.0", 1)],
    )
    account = Account(
        current_pity=start_pity,
        character_guarantee=start_guarantee,
        wishes=budget,
    )
    context = PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0", mechanics=mechanics
    )
    plan = SpendPlan(
        entries=(
            PlannedSpend(
                Banner("Target", "7.0", 1), target_constellation, budget
            ),
        )
    )
    result = simulate(context, plan, runs=runs, seed=seed)
    return result.goals[0].probability


class TestSingleCopyAgainstPhase2:
    """Probability invariant 10, mirrored: simulation must reproduce the
    analytical curve within sampling error (§18 Phase 4 validation)."""

    @pytest.mark.parametrize(
        "pity,guarantee,budget",
        [
            (0, False, 60),
            (37, False, 90),
            (60, True, 50),
        ],
    )
    def test_matches_the_phase2_curve(self, pity, guarantee, budget):
        expected = float(
            cumulative_probability(
                budget, pity, guarantee, CHARACTER_EVENT_BANNER
            )[budget]
        )
        empirical = engine_probability(
            pity, guarantee, 0, budget, CHARACTER_EVENT_BANNER,
            runs=6_000, seed=1234,
        )
        # 6000 trials: SE <= sqrt(0.25/6000) ~ 0.0065, so 0.03 is a
        # generous ~4.6-sigma band - never flaky, still catches bugs.
        assert empirical == pytest.approx(expected, abs=0.03)

    def test_tiny_mechanics_single_copy(self):
        expected = float(cumulative_probability(7, 0, False, TINY)[7])
        empirical = engine_probability(0, False, 0, 7, TINY)
        assert empirical == pytest.approx(expected, abs=0.02)


class TestMultiCopyAgainstPmfConvolution:
    """The independent multi-copy reference (§10.4): PMF convolved K times."""

    @pytest.mark.parametrize(
        "k,budget",
        [
            (1, 7),   # mid distribution
            (1, 10),  # mechanics-guaranteed: exactly 1
            (2, 8),   # mid distribution
            (2, 20),  # mechanics-guaranteed: exactly 1
            (3, 12),  # mid distribution
        ],
    )
    def test_matches_the_convolution_reference(self, k, budget):
        reference = float(k_copy_cdf(0, False, k, TINY)[budget])
        empirical = engine_probability(0, False, k - 1, budget, TINY)
        if budget >= 2 * k * TINY.hard_pity:
            # Mechanics guarantee success here; both must be exactly 1.
            assert reference == pytest.approx(1.0, abs=1e-12)
            assert empirical == 1.0
        else:
            # 20k trials: SE <= 0.0035, so 0.02 is a ~5.7-sigma band.
            assert empirical == pytest.approx(reference, abs=0.02)

    def test_multi_copy_from_a_mid_pity_start(self):
        """The same reference logic works from a non-zero starting pity."""
        reference = float(k_copy_cdf(2, False, 2, TINY)[15])
        empirical = engine_probability(2, False, 1, 15, TINY)
        assert empirical == pytest.approx(reference, abs=0.02)

