"""cumulative_probability: exact single-copy curves (Design Document §10.2).

Validation strategy:

1. Contract tests for the np.ndarray result (§10.2, Phase 2 invariants 3-5).
2. Closed-form anchors, computed independently of the DP.
3. A brute-force reference DP written in a deliberately different style.
4. A seeded vectorized Monte Carlo cross-check (Phase 2 invariant 10):
   the analytical engine must be reproducible by simulation within
   sampling error.
"""

import numpy as np
import pytest

from domain import CHARACTER_EVENT_BANNER, WishMechanics
from probability import cumulative_probability, pull_rate


def with_featured_rate(featured_rate: float) -> WishMechanics:
    """CHARACTER_EVENT_BANNER mechanics with a different 50/50 rate."""
    return WishMechanics(
        banner_type=CHARACTER_EVENT_BANNER.banner_type,
        hard_pity=CHARACTER_EVENT_BANNER.hard_pity,
        soft_pity_start=CHARACTER_EVENT_BANNER.soft_pity_start,
        base_rate=CHARACTER_EVENT_BANNER.base_rate,
        soft_pity_increment=CHARACTER_EVENT_BANNER.soft_pity_increment,
        featured_rate=featured_rate,
    )


class TestResultContract:
    """The np.ndarray contract: cumulative, 0-indexed, monotone (§10.2)."""

    def test_returns_array_of_length_wishes_plus_one(self):
        curve = cumulative_probability(50, 0, False, CHARACTER_EVENT_BANNER)
        assert isinstance(curve, np.ndarray)
        assert curve.shape == (51,)

    def test_index_n_is_probability_within_n_wishes(self):
        # Index 0 means "within 0 wishes": impossible, so exactly 0.
        curve = cumulative_probability(10, 0, False, CHARACTER_EVENT_BANNER)
        assert curve[0] == 0.0
        # Each entry is the *cumulative* probability, never "exactly on
        # wish N": the curve can only grow.
        assert np.all(np.diff(curve) >= -1e-12)

    def test_values_are_probabilities(self):
        curve = cumulative_probability(200, 0, False, CHARACTER_EVENT_BANNER)
        assert np.all(curve >= 0.0)
        assert np.all(curve <= 1.0)

    def test_float_dtype(self):
        curve = cumulative_probability(5, 0, True, CHARACTER_EVENT_BANNER)
        assert curve.dtype == np.float64


class TestClosedFormAnchors:
    """Anchors computed in closed form, independent of the DP.

    These numbers belong to the CHARACTER_EVENT_BANNER mechanics
    configuration, not to the engine itself.
    """

    def test_five_star_within_73_pulls(self):
        # Pulls 1..73 are all base rate: 1 - (1 - 0.006)^73.
        expected = 1.0 - (1.0 - 0.006) ** 73
        curve = cumulative_probability(73, 0, True, CHARACTER_EVENT_BANNER)
        # With a guaranteed 5-star, the featured curve IS the 5-star curve.
        assert curve[73] == pytest.approx(expected, abs=1e-9)

    def test_five_star_within_74_pulls(self):
        # Pull 74 adds the first soft-pity increment: 1 - 0.994^73 * 0.934.
        expected = 1.0 - (1.0 - 0.006) ** 73 * (1.0 - 0.066)
        curve = cumulative_probability(74, 0, True, CHARACTER_EVENT_BANNER)
        assert curve[74] == pytest.approx(expected, abs=1e-9)

    def test_hard_pity_closes_the_cycle_exactly(self):
        curve = cumulative_probability(90, 0, True, CHARACTER_EVENT_BANNER)
        assert curve[90] == 1.0
        assert curve[89] < 1.0

    def test_average_pulls_per_five_star(self):
        # E[T] = sum of tail probabilities. For these exact mechanics the
        # widely quoted consolidated figure is ~62.3 pulls (rate 1.6%).
        curve = cumulative_probability(90, 0, True, CHARACTER_EVENT_BANNER)
        expectation = float(np.sum(1.0 - curve[:90]))
        assert expectation == pytest.approx(62.297, abs=0.01)


class TestGuaranteeSemantics:
    """Guarantee state behaves exactly as the 50/50 rules require (§10)."""

    def test_one_wish_from_full_pity_guarantee_is_certain(self):
        # Pity 89 -> the next pull is the hard-pity 5-star, and it is
        # guaranteed featured.
        curve = cumulative_probability(1, 89, True, CHARACTER_EVENT_BANNER)
        assert curve[1] == 1.0

    def test_guaranteed_curve_dominates_the_50_50_curve(self):
        guaranteed = cumulative_probability(120, 37, True, CHARACTER_EVENT_BANNER)
        not_guaranteed = cumulative_probability(120, 37, False, CHARACTER_EVENT_BANNER)
        assert np.all(guaranteed >= not_guaranteed - 1e-12)

    def test_full_50_50_cycle_is_bounded_by_two_hard_pities(self):
        # Lose the 50/50 at pull 90, then the guarantee lands by pull 180.
        curve = cumulative_probability(180, 0, False, CHARACTER_EVENT_BANNER)
        assert curve[180] == 1.0
        assert curve[179] < 1.0

    def test_fresh_fifty_fifty_within_one_cycle_is_a_bit_over_half(self):
        # Most of the 5-star mass sits at pulls 74-90, leaving little room
        # for a second 5-star: the 90-wish curve must land a bit above 50%.
        curve = cumulative_probability(90, 0, False, CHARACTER_EVENT_BANNER)
        assert 0.5 < curve[90] < 0.6

    def test_featured_rate_of_one_collapses_the_50_50(self):
        # With featured_rate 1.0 every 5-star is featured, so the
        # not-guaranteed curve must equal the standard guaranteed curve.
        always_featured = cumulative_probability(90, 0, False, with_featured_rate(1.0))
        guaranteed = cumulative_probability(90, 0, True, CHARACTER_EVENT_BANNER)
        assert np.allclose(always_featured, guaranteed)


class TestStartingPity:
    """Progressed pity only ever improves the curve (§4.1, §10.2)."""

    def test_higher_pity_never_lowers_the_curve(self):
        fresh = cumulative_probability(90, 0, True, CHARACTER_EVENT_BANNER)
        progressed = cumulative_probability(90, 45, True, CHARACTER_EVENT_BANNER)
        assert np.all(progressed >= fresh - 1e-12)

    def test_pity_shortens_time_to_hard_pity_exactly(self):
        # Pity 45 -> hard pity is 45 wishes away, no matter what.
        curve = cumulative_probability(45, 45, True, CHARACTER_EVENT_BANNER)
        assert curve[45] == 1.0
        assert curve[44] < 1.0


class TestValidation:
    def test_negative_wishes_rejected(self):
        with pytest.raises(ValueError, match="wishes"):
            cumulative_probability(-1, 0, False, CHARACTER_EVENT_BANNER)

    def test_starting_pity_at_hard_pity_rejected(self):
        with pytest.raises(ValueError, match="starting_pity"):
            cumulative_probability(10, 90, False, CHARACTER_EVENT_BANNER)

    def test_negative_starting_pity_rejected(self):
        with pytest.raises(ValueError, match="starting_pity"):
            cumulative_probability(10, -1, True, CHARACTER_EVENT_BANNER)


# ---------------------------------------------------------------------------
# Independent reference implementations used for cross-checking.
# ---------------------------------------------------------------------------


def brute_force_curve(
    wishes: int, starting_pity: int, guaranteed: bool, mechanics: WishMechanics
) -> np.ndarray:
    """Reference DP written in a deliberately different style.

    Walks every `(pity, guarantee)` state individually with dict-based mass
    instead of the engine's vectorized survival tracking.
    """
    states = {(starting_pity, guaranteed): 1.0}
    curve = np.empty(wishes + 1)
    curve[0] = 0.0

    for wish in range(1, wishes + 1):
        success = 0.0
        nxt: dict[tuple[int, bool], float] = {}
        for (pity, guarantee), mass in states.items():
            rate = pull_rate(pity, mechanics)
            featured = 1.0 if guarantee else mechanics.featured_rate
            success += mass * rate * featured
            lost = mass * rate * (1.0 - featured)
            stay = mass * (1.0 - rate)
            if stay > 0.0:
                key = (pity + 1, guarantee)
                nxt[key] = nxt.get(key, 0.0) + stay
            if lost > 0.0:
                key = (0, True)
                nxt[key] = nxt.get(key, 0.0) + lost
        states = nxt
        curve[wish] = min(curve[wish - 1] + success, 1.0)

    return curve


@pytest.mark.parametrize(
    "wishes,starting_pity,guaranteed,mechanics",
    [
        (60, 0, False, CHARACTER_EVENT_BANNER),
        (150, 37, False, CHARACTER_EVENT_BANNER),
        (45, 45, True, CHARACTER_EVENT_BANNER),
        (90, 0, True, CHARACTER_EVENT_BANNER),
        (180, 0, False, CHARACTER_EVENT_BANNER),
        # Small mechanics make off-by-one errors loud.
        (
            12,
            1,
            False,
            WishMechanics(
                banner_type="tiny",
                hard_pity=5,
                soft_pity_start=3,
                base_rate=0.25,
                soft_pity_increment=0.2,
                featured_rate=0.5,
            ),
        ),
    ],
)
def test_matches_brute_force_reference(wishes, starting_pity, guaranteed, mechanics):
    curve = cumulative_probability(wishes, starting_pity, guaranteed, mechanics)
    reference = brute_force_curve(wishes, starting_pity, guaranteed, mechanics)
    assert np.allclose(curve, reference, atol=1e-10)


def vectorized_monte_carlo(
    trials: int, starting_pity: int, guaranteed: bool, mechanics: WishMechanics
) -> np.ndarray:
    """Empirical curve from a vectorized simulation (fixed seed)."""
    rng = np.random.default_rng(20240916)
    horizon = 2 * mechanics.hard_pity
    rates = np.array([pull_rate(p, mechanics) for p in range(mechanics.hard_pity)])

    pity = np.full(trials, starting_pity)
    guarantee = np.full(trials, guaranteed)
    done = np.zeros(trials, dtype=bool)
    success_at = np.full(trials, horizon + 1, dtype=int)

    for wish in range(1, horizon + 1):
        hit = rng.random(trials) < rates[pity]
        lost = hit & ~guarantee & (rng.random(trials) >= mechanics.featured_rate)
        featured = hit & ~lost
        newly = featured & ~done
        success_at[newly] = wish
        done |= newly
        guarantee = np.where(done, guarantee, guarantee | lost)
        pity = np.where(done, 0, np.where(hit, 0, pity + 1))
        if done.all():
            break

    counts = np.bincount(success_at, minlength=horizon + 2)[: horizon + 1]
    return np.cumsum(counts) / trials


def test_monte_carlo_reproduces_the_analytical_curve():
    """Phase 2 invariant 10: simulation must match analysis within error.

    50k trials give a standard error of ~0.002, so 0.02 is a generous
    ~9-sigma band - loose enough to never flake, tight enough to catch a
    broken engine.
    """
    trials = 50_000
    for pity, guaranteed in [(0, False), (37, False), (60, True)]:
        analytical = cumulative_probability(180, pity, guaranteed, CHARACTER_EVENT_BANNER)
        empirical = vectorized_monte_carlo(trials, pity, guaranteed, CHARACTER_EVENT_BANNER)
        assert np.all(np.abs(analytical - empirical) < 0.02), (pity, guaranteed)
