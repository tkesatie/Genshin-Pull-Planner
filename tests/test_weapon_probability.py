"""Phase 4 weapon-probability engine tests.

Covers: weapon pity / 5-star probability, the 75/25 rate-up roll and its
guarantee carryover, Epitomized Path Fate Points, multi-copy refinement
targets, edge cases, Monte Carlo cross-validation, and performance.

Mechanics sources (see domain.mechanics.WEAPON_EVENT_BANNER):

    * Official (in-game Details text): 0.7% base / 1.85% consolidated
      5-star rate, hard pity 80, 75% rate-up share, Epitomized Path
      guarantees the designated weapon at 1 Fate Point (Version 5.0;
      previously 2).
    * Not published by HoYoverse: soft pity. The soft_pity_start=63 /
      increment=0.07 values are community-documented estimates consumed
      as data (§17); tests below avoid anchoring on exact soft-pity
      positions except where the curve shape is asserted only loosely.
    * Assumption (stated in probability.weapon): the two rate-up weapons
      are symmetric, so the designated share before guarantees is 75%/2.
"""

import numpy as np
import pytest

from domain import WEAPON_EVENT_BANNER, WishMechanics
from probability import (
    DEFAULT_FATE_POINTS_REQUIRED,
    designated_rate,
    pull_rate,
    refinement_cumulative_probability,
    weapon_cumulative_probability,
    weapon_wishes_for_confidence,
)

MECHANICS = WEAPON_EVENT_BANNER

# A tiny mechanics set with exact closed forms for tree-derived anchors:
# base rate 0.5 per pull from pull 1, hard pity 2, rate-up 75%.
TOY = WishMechanics(
    banner_type="weapon_event",
    hard_pity=2,
    soft_pity_start=1,
    base_rate=0.5,
    soft_pity_increment=0.1,
    featured_rate=0.75,
)


# ---------------------------------------------------------------------------
# Mechanics data
# ---------------------------------------------------------------------------


def test_weapon_event_banner_official_constants():
    # Official figures only: hard pity 80, base 0.7%, rate-up 75%.
    assert MECHANICS.hard_pity == 80
    assert MECHANICS.base_rate == 0.007
    assert MECHANICS.featured_rate == 0.75
    assert MECHANICS.banner_type == "weapon_event"
    # Distinct from the character banner - weapons are not characters with
    # different constants.
    from domain import CHARACTER_EVENT_BANNER

    assert MECHANICS != CHARACTER_EVENT_BANNER


def test_default_fate_points_required_is_one():
    # Epitomized Path needs 1 Fate Point since Version 5.0.
    assert DEFAULT_FATE_POINTS_REQUIRED == 1


def test_designated_rate_is_half_the_rate_up_share():
    # Symmetric rate-up pair assumption (see probability.weapon docstring).
    assert designated_rate(MECHANICS) == pytest.approx(0.375)


# ---------------------------------------------------------------------------
# Weapon pity / 5-star probability
# ---------------------------------------------------------------------------


def test_weapon_pity_uses_weapon_rates_not_character_rates():
    # The weapon banner's soft pity starts at pull 64 (1-based); the
    # character banner's starts at 74. Base rates differ too.
    assert pull_rate(62, MECHANICS) == pytest.approx(0.007)
    from domain import CHARACTER_EVENT_BANNER

    assert pull_rate(62, CHARACTER_EVENT_BANNER) == pytest.approx(0.006)
    # Past that point both ramp, but from different starts/increments.
    assert pull_rate(63, MECHANICS) == pytest.approx(0.007 + 0.06)
    assert pull_rate(72, CHARACTER_EVENT_BANNER) == pytest.approx(0.006)


def test_weapon_soft_pity_ramp_and_hard_pity():
    # Upcoming pull 64 = first soft-pity pull (soft_pity_start 64).
    assert pull_rate(63, MECHANICS) == pytest.approx(0.007 + 0.06)
    assert pull_rate(64, MECHANICS) == pytest.approx(0.007 + 2 * 0.06)
    # Hard pity: pull 80 is a guaranteed 5-star; pity 79 is valid.
    assert pull_rate(79, MECHANICS) == 1.0
    # The soft-pity ramp saturates exactly at the hard pity.
    from probability import pull_rate as pr

    assert all(pr(p, MECHANICS) < 1.0 for p in range(78))


def test_weapon_pity_validation():
    with pytest.raises(ValueError):
        pull_rate(80, MECHANICS)  # pity == hard_pity is invalid
    with pytest.raises(ValueError):
        pull_rate(-1, MECHANICS)


def test_curve_is_cumulative_monotone_and_bounded():
    curve = weapon_cumulative_probability(50, 0, False, 0, MECHANICS)
    assert isinstance(curve, np.ndarray)
    assert curve.shape == (51,)
    assert curve[0] == 0.0
    assert np.all(np.diff(curve) >= -1e-12)
    assert np.all((curve >= 0.0) & (curve <= 1.0))


def test_zero_wishes_is_zero():
    curve = weapon_cumulative_probability(0, 40, True, 0, MECHANICS)
    assert curve[0] == 0.0


def test_guaranteed_curve_with_fate_point_cap_matches_five_star_curve():
    # With the Fate Point cap reached, every 5-star is the designated
    # weapon, so the designated curve IS the 5-star curve.
    curve = weapon_cumulative_probability(80, 0, True, 1, MECHANICS)
    assert curve[80] == pytest.approx(1.0)
    assert curve[79] < 1.0


def test_rate_up_guarantee_alone_is_not_designated_certain():
    # The guarantee forces a rate-up weapon; the designated one is still
    # only half of the rate-up pair. One hard-pity cycle lands the 5-star
    # for sure but only ~61% of the time is it the designated weapon (a
    # rate-up miss leaves a 50% chance on the next certain cycle, so the
    # curve has NOT reached 1.0 or even 0.5 within one cycle).
    curve = weapon_cumulative_probability(80, 0, True, 0, MECHANICS)
    assert curve[80] == pytest.approx(0.5 + 0.5 * 0.5 * (1 - 0.375) / 0.5 * 0.5, abs=0.15)
    assert curve[80] > 0.5
    assert curve[80] < 1.0
    # With fate points also capped, the designated weapon is certain by 80.
    capped = weapon_cumulative_probability(80, 0, True, 1, MECHANICS)
    assert capped[80] == pytest.approx(1.0)


def test_hard_pity_closes_one_cycle_exactly():
    # Fate Point cap + hard pity: certainty within one cycle.
    curve = weapon_cumulative_probability(80, 0, True, 1, MECHANICS)
    assert curve[80] == pytest.approx(1.0)


def test_average_pulls_per_five_star_matches_consolidated_rate():
    # The official consolidated 1.85% rate implies ~54.05 pulls per 5-star.
    # The exact 5-star curve (fate points capped: designated == 5-star) over
    # one hard-pity cycle must match that figure - the validation anchor for
    # the community-sourced soft-pity schedule (see domain.mechanics).
    curve = weapon_cumulative_probability(80, 0, True, 1, MECHANICS)
    expectation = float(np.sum(1.0 - curve[:80]))
    assert expectation == pytest.approx(1.0 / 0.0185, abs=0.5)


# ---------------------------------------------------------------------------
# Featured weapon: 75/25, guarantee carryover
# ---------------------------------------------------------------------------


# TOY rate note: with hard_pity=2 the only valid soft_pity_start is 1, so
# pull 1's rate is base + one increment = 0.6 and pull 2 is the hard pity
# (rate 1.0). The hand-derived anchors below use those exact rates.

def test_toy_first_five_star_outcomes_partition_correctly():
    # TOY mechanics, pity 0, no guarantee, no fate points:
    #   hit@1: 0.6 * 0.375 = 0.225
    #   miss@1 (0.6 * 0.625 = 0.375 -> fate point capped) then the pull-2
    #   hard-pity 5-star is a certain designated hit: 0.375
    #   -> 0.6 by wish 2. Every path ends by hard pity 2.
    curve = weapon_cumulative_probability(2, 0, False, 0, TOY)
    assert curve[2] == pytest.approx(0.225 + 0.375)


def test_toy_no_guarantee_designated_probability_within_one_wish():
    # One wish from pity 1: guaranteed 5-star (pull 2 = hard pity), then the
    # 75/25 roll: P(designated) = 0.375.
    curve = weapon_cumulative_probability(1, 1, False, 0, TOY)
    assert curve[1] == pytest.approx(0.375)


def test_toy_guarantee_makes_rate_up_certain_but_not_designated():
    # Guarantee armed, fate points 0: the 5-star is a rate-up weapon for
    # sure, but only half the rate-up share is the designated weapon.
    curve = weapon_cumulative_probability(1, 1, True, 0, TOY)
    assert curve[1] == pytest.approx(0.5)


def test_toy_fate_point_cap_makes_designated_certain():
    # Fate Points at the required amount: every 5-star is the designated
    # weapon regardless of the 75/25 roll.
    curve = weapon_cumulative_probability(1, 1, False, 1, TOY)
    assert curve[1] == pytest.approx(1.0)


def test_state_after_losing_the_rate_up_roll():
    # Pity 0 on TOY: the pull-1 5-star (rate 0.6) misses with 0.6*0.625 =
    # 0.375, which caps the Fate Point meter (required=1) - the next 5-star
    # is then a certain designated hit. Hit-by-1 is 0.6*0.375 = 0.225.
    curve = weapon_cumulative_probability(4, 0, False, 0, TOY)
    assert curve[1] == pytest.approx(0.225)
    assert curve[2] == pytest.approx(0.6)  # 0.225 + 0.375 (miss then certain)
    # All mass has completed by wish 2.
    assert curve[4] == pytest.approx(1.0)


def test_each_five_star_is_not_an_independent_fresh_banner():
    # If each 5-star were an independent 0.375 designated roll, two certain
    # 5-stars would give 1 - 0.625^2 = 0.609375. The state-carrying model
    # instead makes a miss cap the Fate Points, so the second 5-star is a
    # certain designated hit and two certain 5-stars complete for sure.
    independent_approx = 1.0 - 0.625 ** 2
    curve = weapon_cumulative_probability(4, 0, False, 0, TOY)
    assert curve[4] != pytest.approx(independent_approx, abs=1e-3)
    assert curve[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Fate Points
# ---------------------------------------------------------------------------


def test_existing_fate_point_of_two_required_makes_next_five_star_certain():
    # With fate_points_required=2 (pre-5.0 rules) and one point already
    # accumulated: the pull-1 5-star (rate 0.6, hard pity) misses with
    # 0.6 * 0.625 = 0.375 and reaches the cap of 2. The pull-2 roll then
    # happens at pity-0 rate 0.6 (pity reset), so P(within 2 wishes) =
    # 0.375 + 0.375 * 0.6 = 0.75. A third wish closes it (hard pity 2).
    curve = weapon_cumulative_probability(2, 1, False, 1, TOY, fate_points_required=2)
    assert curve[2] == pytest.approx(0.75)
    curve3 = weapon_cumulative_probability(3, 1, False, 1, TOY, fate_points_required=2)
    assert curve3[3] == pytest.approx(1.0)


def test_fate_points_accumulate_only_on_misses():
    # One wish from pity 1 (a certain 5-star on TOY):
    #   fp 0 -> designated with 0.375
    #   fp 1 (== required) -> designated with certainty
    assert weapon_cumulative_probability(1, 1, False, 0, TOY)[1] == pytest.approx(0.375)
    assert weapon_cumulative_probability(1, 1, False, 1, TOY)[1] == pytest.approx(1.0)


def test_fate_points_reset_after_the_designated_weapon():
    # Two copies within 2 wishes on TOY from pity 0: both 5-stars must be
    # designated AND the first hit resets fate points for the second cycle.
    # P = (0.6 * 0.375) * (0.6 * 0.375) = 0.050625 - the second factor is
    # the fresh fate-point-0 roll at pity-0 rate 0.6, proving the reset.
    curve = weapon_cumulative_probability(2, 0, False, 0, TOY, copies=2)
    assert curve[2] == pytest.approx(0.6 * 0.375 * 0.6 * 0.375)


def test_fate_point_cap_worst_case_is_two_hard_pities():
    # Starting at pity 0 with the Fate Point cap already reached, the
    # designated weapon arrives within one hard-pity cycle: worst case 80
    # wishes, and the confidence search must agree.
    assert (
        weapon_wishes_for_confidence(1.0, 0, False, 1, MECHANICS) == 80
    )


def test_invalid_fate_point_state_is_rejected():
    with pytest.raises(ValueError):
        weapon_cumulative_probability(10, 0, False, 2, MECHANICS)
    with pytest.raises(ValueError):
        weapon_cumulative_probability(10, 0, False, -1, MECHANICS)
    with pytest.raises(ValueError):
        weapon_wishes_for_confidence(0.5, 0, False, 5, MECHANICS)


# ---------------------------------------------------------------------------
# Refinements / multiple copies
# ---------------------------------------------------------------------------


def test_first_copy_from_unowned_matches_copies_equals_one():
    # Domain convention: NOT_OWNED (-1) -> R0 counts as the first copy,
    # so R1 from unowned needs 2 copies.
    from_refinement = refinement_cumulative_probability(
        100, -1, 1, 0, False, 0, MECHANICS
    )
    from_copies = weapon_cumulative_probability(100, 0, False, 0, MECHANICS, copies=2)
    assert np.allclose(from_refinement, from_copies)
    # And R0 from unowned is exactly one copy.
    r0 = refinement_cumulative_probability(100, -1, 0, 0, False, 0, MECHANICS)
    assert np.allclose(r0, weapon_cumulative_probability(100, 0, False, 0, MECHANICS))


def test_r1_to_r2_is_one_more_copy():
    from_refinement = refinement_cumulative_probability(
        100, 1, 2, 0, False, 0, MECHANICS
    )
    from_copies = weapon_cumulative_probability(100, 0, False, 0, MECHANICS)
    assert np.allclose(from_refinement, from_copies)


def test_refinement_target_already_met_is_all_ones():
    curve = refinement_cumulative_probability(50, 2, 1, 0, False, 0, MECHANICS)
    assert np.all(curve == 1.0)


def test_higher_refinement_needs_more_wishes():
    r1 = refinement_cumulative_probability(160, -1, 1, 0, False, 0, MECHANICS)
    r3 = refinement_cumulative_probability(160, -1, 3, 0, False, 0, MECHANICS)
    assert np.all(r3 <= r1 + 1e-12)
    # Two copies from unowned are ~40% certain in 160 wishes; three copies
    # are far from certain in the same window.
    assert r1[160] == pytest.approx(0.39995010774421647)
    assert r3[160] < 0.05


def test_multi_copy_worst_case_is_bounded_by_copies_times_two_cycles():
    # R4 from unowned = 5 copies (NOT_OWNED -> R0 is the first copy); the
    # model's exact worst case completes within copies * 2 * 80 wishes.
    curve = refinement_cumulative_probability(1000, -1, 4, 0, False, 0, MECHANICS)
    assert curve[1000] == pytest.approx(1.0)
    assert curve[160] < 0.05  # 5 copies cannot all land in 160 wishes


def test_confidence_search_for_two_copies_is_bounded():
    # The model's exact two-copy worst case (derived from the DP itself,
    # not assumed): mass completes within the search horizon.
    n = weapon_wishes_for_confidence(1.0, 0, False, 0, MECHANICS, copies=2)
    assert n <= 2 * 2 * MECHANICS.hard_pity
    # A single copy with the fate-point cap from pity 0 lands exactly at
    # the hard pity.
    assert weapon_wishes_for_confidence(1.0, 0, False, 1, MECHANICS) == 80


def test_confidence_search_monotone_in_confidence():
    low = weapon_wishes_for_confidence(0.5, 0, False, 0, MECHANICS)
    high = weapon_wishes_for_confidence(0.9, 0, False, 0, MECHANICS)
    assert low <= high


def test_unattainable_confidence_raises():
    # Invalid confidence values are rejected by the search.
    with pytest.raises(ValueError):
        weapon_wishes_for_confidence(1.5, 0, False, 0, MECHANICS)
    with pytest.raises(ValueError):
        weapon_wishes_for_confidence(0.0, 0, False, 0, MECHANICS)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_zero_pity_high_pity_ordering():
    zero = weapon_cumulative_probability(80, 0, False, 0, MECHANICS)
    high = weapon_cumulative_probability(80, 79, False, 0, MECHANICS)
    assert np.all(high >= zero - 1e-12)
    # Pity 79 -> pull 80 is the hard-pity 5-star; with fate points capped
    # it is a certain designated hit.
    capped = weapon_cumulative_probability(1, 79, False, 1, MECHANICS)
    assert capped[1] == pytest.approx(1.0)


def test_pity_at_hard_pity_minus_one_is_valid():
    curve = weapon_cumulative_probability(1, 79, True, 1, MECHANICS)
    assert curve[1] == pytest.approx(1.0)


def test_guaranteed_state_dominates_not_guaranteed():
    g = weapon_cumulative_probability(160, 20, True, 0, MECHANICS)
    ng = weapon_cumulative_probability(160, 20, False, 0, MECHANICS)
    assert np.all(g >= ng - 1e-12)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        weapon_cumulative_probability(-1, 0, False, 0, MECHANICS)
    with pytest.raises(ValueError):
        weapon_cumulative_probability(10, 80, False, 0, MECHANICS)
    with pytest.raises(ValueError):
        weapon_cumulative_probability(10, 0, False, 0, MECHANICS, copies=0)
    with pytest.raises(ValueError):
        refinement_cumulative_probability(10, -2, 1, 0, False, 0, MECHANICS)
    with pytest.raises(ValueError):
        refinement_cumulative_probability(10, 0, -1, 0, False, 0, MECHANICS)


# ---------------------------------------------------------------------------
# Monte Carlo cross-validation (probability invariant 10, weapon mirror)
# ---------------------------------------------------------------------------


def _wilson(successes: int, n: int, z: float = 4.0):
    phat = successes / n
    denom = 1 + z ** 2 / n
    centre = phat + z ** 2 / (2 * n)
    spread = z * np.sqrt((phat * (1 - phat) + z ** 2 / (4 * n)) / n)
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def test_simulation_matches_analytic_single_copy():
    from simulation.weapon_engine import simulate_weapon_runs

    wishes = 100
    reference = float(
        weapon_cumulative_probability(wishes, 0, False, 0, MECHANICS)[wishes]
    )
    runs = 20_000
    copies = simulate_weapon_runs(wishes, 0, False, 0, MECHANICS, runs, seed=7)
    successes = int((copies >= 1).sum())
    low, high = _wilson(successes, runs)
    assert low - 1e-9 <= reference <= high + 1e-9, (
        f"simulated {successes / runs:.4f} vs analytic {reference:.4f}"
    )


def test_simulation_matches_analytic_with_fate_point_head_start():
    from simulation.weapon_engine import simulate_weapon_runs

    wishes = 60
    reference = float(
        weapon_cumulative_probability(wishes, 30, True, 1, MECHANICS)[wishes]
    )
    runs = 20_000
    copies = simulate_weapon_runs(wishes, 30, True, 1, MECHANICS, runs, seed=11)
    successes = int((copies >= 1).sum())
    low, high = _wilson(successes, runs)
    assert low - 1e-9 <= reference <= high + 1e-9


def test_simulation_matches_analytic_two_copies():
    from simulation.weapon_engine import simulate_weapon_runs

    wishes = 300
    reference = float(
        weapon_cumulative_probability(
            wishes, 0, False, 0, MECHANICS, copies=2
        )[wishes]
    )
    runs = 20_000
    copies = simulate_weapon_runs(wishes, 0, False, 0, MECHANICS, runs, seed=3)
    successes = int((copies >= 2).sum())
    low, high = _wilson(successes, runs)
    assert low - 1e-9 <= reference <= high + 1e-9


def test_simulation_is_deterministic_for_seed():
    from simulation.weapon_engine import simulate_weapon_runs

    a = simulate_weapon_runs(50, 0, False, 0, MECHANICS, 500, seed=42)
    b = simulate_weapon_runs(50, 0, False, 0, MECHANICS, 500, seed=42)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


def test_analytic_curve_is_fast_enough():
    import time

    start = time.perf_counter()
    for pity in range(0, 80, 7):
        weapon_cumulative_probability(320, pity, False, 0, MECHANICS, copies=2)
    elapsed = time.perf_counter() - start
    # 12 full multi-copy curves; the planner calls these per goal, so each
    # must be well under a second.
    assert elapsed < 6.0, f"12 two-copy 320-wish curves took {elapsed:.3f}s"




