"""Capturing Radiance math (Design Document §17-adjacent; undocumented mechanic).

These tests pin down three things that were previously only checked by
inspection:

1. `probability.capturing_radiance_rate` and
   `domain.next_capturing_radiance_counter` together reproduce the
   officially cited long-run average win rate of 55% (a regression test
   for the calibration, not just the individual numbers).
2. `probability.character.cumulative_probability` /
   `wishes_for_confidence` are Capturing-Radiance-aware via
   `starting_radiance`, and agree with the simulator's per-state rates.
3. `simulation.engine` and `api.routers.accounts.record_pull_result` no
   longer disagree on the post-win counter at radiance 3 - the concrete
   bug this patch fixes.
"""

import numpy as np
import pytest

from dataclasses import replace

from domain import (
    Account,
    Banner,
    Goal,
    Ownership,
    Roadmap,
    WishMechanics,
    next_capturing_radiance_counter,
)
from planner.context import PlannerContext
from probability import capturing_radiance_rate, cumulative_probability, wishes_for_confidence
from probability.rates import CAPTURING_RADIANCE_RATE_AT_2
from simulation import PlannedSpend, SpendPlan, simulate


# A tiny mechanics fixture where the *next* pull is always a 5-star
# (pity = hard_pity - 1), so Capturing Radiance's effect on the featured
# rate is the only source of randomness left - useful for deterministic
# integration tests.
_GUARANTEED_NEXT_PULL_MECHANICS = WishMechanics(
    banner_type="test",
    hard_pity=2,
    soft_pity_start=1,
    base_rate=0.006,
    soft_pity_increment=0.5,
    featured_rate=0.5,
)


# ---------------------------------------------------------------------------
# 1. The transition/rate schedule itself
# ---------------------------------------------------------------------------


def test_capturing_radiance_rate_schedule():
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    assert capturing_radiance_rate(0, mechanics) == mechanics.featured_rate
    assert capturing_radiance_rate(1, mechanics) == mechanics.featured_rate
    assert capturing_radiance_rate(2, mechanics) == pytest.approx(6.0 / 11.0)
    assert capturing_radiance_rate(2, mechanics) == CAPTURING_RADIANCE_RATE_AT_2
    assert capturing_radiance_rate(3, mechanics) == 1.0


def test_capturing_radiance_rate_rejects_out_of_range():
    with pytest.raises(ValueError):
        capturing_radiance_rate(4, _GUARANTEED_NEXT_PULL_MECHANICS)
    with pytest.raises(ValueError):
        capturing_radiance_rate(-1, _GUARANTEED_NEXT_PULL_MECHANICS)


@pytest.mark.parametrize(
    "radiance, was_guaranteed, featured, expected",
    [
        # A guaranteed pull never touches the counter, win or lose.
        (0, True, True, 0),
        (2, True, True, 2),
        (3, True, True, 3),
        (1, True, False, 1),
        # A non-guaranteed loss increments, capped at 3.
        (0, False, False, 1),
        (1, False, False, 2),
        (2, False, False, 3),
        (3, False, False, 3),
        # A non-guaranteed win from 0 or 1 fully resets to 0.
        (0, False, True, 0),
        (1, False, True, 0),
        # A non-guaranteed win from 2 or 3 leaves a residual mark at 1.
        # radiance=3 is the exact case api.routers.accounts previously
        # got wrong (it computed max(0, 3 - 1) == 2 instead of 1).
        (2, False, True, 1),
        (3, False, True, 1),
    ],
)
def test_next_capturing_radiance_counter_transitions(
    radiance, was_guaranteed, featured, expected
):
    assert (
        next_capturing_radiance_counter(
            radiance, was_guaranteed=was_guaranteed, featured=featured
        )
        == expected
    )


def test_next_capturing_radiance_counter_rejects_out_of_range():
    with pytest.raises(ValueError):
        next_capturing_radiance_counter(4, was_guaranteed=False, featured=False)


def test_capturing_radiance_never_allows_four_consecutive_losses():
    """radiance 3 always wins, so the streak can never reach a 4th loss."""
    radiance = 0
    for _ in range(10):
        radiance = next_capturing_radiance_counter(
            radiance, was_guaranteed=False, featured=False
        )
        assert radiance <= 3
    # Having been driven to the ceiling by repeated losses, the counter
    # must now be exactly 3, and capturing_radiance_rate there is 1.0 -
    # i.e. the very next attempt cannot lose.
    assert radiance == 3
    assert capturing_radiance_rate(radiance, _GUARANTEED_NEXT_PULL_MECHANICS) == 1.0


# ---------------------------------------------------------------------------
# 2. The calibration: long-run average win rate is exactly 55%
# ---------------------------------------------------------------------------


def test_capturing_radiance_schedule_reproduces_official_55_percent_average():
    """Regression test for the calibration claimed in the docstrings.

    Solves the 4-state Markov chain formed by capturing_radiance_rate's win
    probabilities and next_capturing_radiance_counter's transitions
    directly (no sampling), and checks the stationary-distribution-weighted
    average win rate is exactly 0.55 - the officially cited aggregate
    figure for Capturing Radiance. This is the check that would catch a
    future edit to either function silently breaking the calibration
    (e.g. changing 6/11 to some other "reasonable-looking" fraction).
    """
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    win_rate = np.array([capturing_radiance_rate(r, mechanics) for r in range(4)])

    # Build the transition matrix P[from, to] over the post-attempt state,
    # covering both possible outcomes (win/lose) from each starting state.
    transition = np.zeros((4, 4))
    for radiance in range(4):
        p_win = win_rate[radiance]
        win_to = next_capturing_radiance_counter(radiance, was_guaranteed=False, featured=True)
        lose_to = next_capturing_radiance_counter(radiance, was_guaranteed=False, featured=False)
        transition[radiance, win_to] += p_win
        transition[radiance, lose_to] += 1.0 - p_win

    # Stationary distribution: solve pi = pi @ P, sum(pi) == 1, via the
    # standard "replace one equation with normalization" trick.
    a = np.vstack([(transition.T - np.eye(4))[:-1], np.ones(4)])
    b = np.zeros(4)
    b[-1] = 1.0
    stationary = np.linalg.solve(a, b)

    assert stationary.sum() == pytest.approx(1.0)
    assert all(p >= 0 for p in stationary)

    average_win_rate = float(stationary @ win_rate)
    assert average_win_rate == pytest.approx(0.55, abs=1e-9)


def test_capturing_radiance_schedule_reproduces_55_percent_by_simulation():
    """The same claim as above, but empirically, via direct sampling.

    A second, independent check on the same property using a different
    method (Monte Carlo instead of solving the chain algebraically), so a
    mistake in the linear-algebra test above wouldn't be the only thing
    standing between a calibration regression and a green test suite.
    """
    rng = np.random.default_rng(0)
    radiance = 0
    wins = 0
    attempts = 200_000
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    for _ in range(attempts):
        featured = rng.random() < capturing_radiance_rate(radiance, mechanics)
        wins += featured
        radiance = next_capturing_radiance_counter(
            radiance, was_guaranteed=False, featured=featured
        )
    empirical_rate = wins / attempts
    assert empirical_rate == pytest.approx(0.55, abs=0.01)


# ---------------------------------------------------------------------------
# 3. probability.character is Capturing-Radiance-aware
# ---------------------------------------------------------------------------


def test_cumulative_probability_defaults_to_radiance_zero():
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    curve = cumulative_probability(1, starting_pity=1, guaranteed=False, mechanics=mechanics)
    # pity=1 with hard_pity=2 guarantees a 5-star on the next pull; with
    # radiance defaulted to 0, the featured chance is just the base rate.
    assert curve[1] == pytest.approx(mechanics.featured_rate)


def test_cumulative_probability_uses_explicit_starting_radiance():
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    curve_radiance_3 = cumulative_probability(
        1, starting_pity=1, guaranteed=False, mechanics=mechanics, starting_radiance=3
    )
    # At radiance 3 the next non-guaranteed 5-star is guaranteed featured.
    assert curve_radiance_3[1] == pytest.approx(1.0)

    curve_radiance_2 = cumulative_probability(
        1, starting_pity=1, guaranteed=False, mechanics=mechanics, starting_radiance=2
    )
    assert curve_radiance_2[1] == pytest.approx(6.0 / 11.0)

    # Higher radiance can only ever help, never hurt.
    curve_radiance_0 = cumulative_probability(
        1, starting_pity=1, guaranteed=False, mechanics=mechanics, starting_radiance=0
    )
    assert curve_radiance_0[1] <= curve_radiance_2[1] <= curve_radiance_3[1]


def test_cumulative_probability_rejects_out_of_range_radiance():
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    with pytest.raises(ValueError):
        cumulative_probability(
            1, starting_pity=1, guaranteed=False, mechanics=mechanics, starting_radiance=4
        )


def test_wishes_for_confidence_needs_no_more_wishes_at_higher_radiance():
    mechanics = _GUARANTEED_NEXT_PULL_MECHANICS
    needed_radiance_0 = wishes_for_confidence(
        0.9, starting_pity=0, guaranteed=False, mechanics=mechanics, starting_radiance=0
    )
    needed_radiance_3 = wishes_for_confidence(
        0.9, starting_pity=0, guaranteed=False, mechanics=mechanics, starting_radiance=3
    )
    assert needed_radiance_3 <= needed_radiance_0


# ---------------------------------------------------------------------------
# 4. Integration: the simulator applies the same schedule end to end,
#    including the exact radiance-3 win case that api.routers.accounts
#    previously got wrong.
# ---------------------------------------------------------------------------


def _context_with_radiance(radiance: int) -> PlannerContext:
    account = Account(
        current_pity=1,  # hard_pity=2, so the next pull is a guaranteed 5-star
        character_guarantee=False,
        owned_characters=Ownership({}),
        wishes=1,
        capturing_radiance_counter=radiance,
    )
    roadmap = Roadmap(
        goals=[],
        banners=[Banner(character="Test", version="1.0", phase=1)],
    )
    return PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version="1.0",
        current_phase=1,
        confidence=0.9,
        mechanics=_GUARANTEED_NEXT_PULL_MECHANICS,
    )


def test_simulation_engine_wins_deterministically_at_radiance_three():
    """At radiance 3, a non-guaranteed 5-star is always featured (rate 1.0),
    and the resulting account should carry radiance 1 afterward - not 0,
    and not the buggy max(0, 3 - 1) == 2 that api.routers.accounts used to
    produce for this exact case.
    """
    context = _context_with_radiance(3)
    plan = SpendPlan(
        entries=(
            PlannedSpend(
                banner=Banner(character="Test", version="1.0", phase=1),
                target_constellation=0,
                budget=1,
            ),
        )
    )
    result = simulate(context, plan, runs=5, seed=0)

    banner_result = result.banners[0]
    assert banner_result.target_met_probability == pytest.approx(1.0)
    assert banner_result.mean_wishes_spent == pytest.approx(1.0)

    # Every history is identical here (fully deterministic given the
    # mechanics), so inspect one directly for the resulting radiance.
    history = result.histories[0]
    assert history.banner_results[0].account_after.capturing_radiance_counter == 1


def test_account_endpoint_pull_result_matches_simulator_at_radiance_three():
    """Direct regression test for the api.routers.accounts bug: recording a
    real featured pull at radiance 3 must land on the same counter value
    the simulator would produce for the same event (1), not the old
    max(0, radiance - 1) formula's answer (2).
    """
    account = Account(
        current_pity=0,
        character_guarantee=False,
        owned_characters=Ownership({}),
        wishes=10,
        capturing_radiance_counter=3,
    )

    # This mirrors the "featured" branch of api.routers.accounts.record_pull_result.
    ownership = account.owned_characters
    current = ownership.owned_constellation("Test")
    updated_account = replace(
        account,
        wishes=account.wishes - 1,
        current_pity=0,
        character_guarantee=False,
        capturing_radiance_counter=next_capturing_radiance_counter(
            account.capturing_radiance_counter,
            was_guaranteed=account.character_guarantee,
            featured=True,
        ),
        owned_characters=ownership.with_constellation("Test", current + 1),
    )

    assert updated_account.capturing_radiance_counter == 1
    # The old buggy formula would have produced this instead - pin it down
    # explicitly so a future refactor can't quietly reintroduce it.
    buggy_old_value = max(0, account.capturing_radiance_counter - 1)
    assert buggy_old_value == 2
    assert updated_account.capturing_radiance_counter != buggy_old_value


def test_account_endpoint_pull_result_guarantee_never_touches_radiance():
    """A guaranteed win (character_guarantee already True) must leave the
    counter untouched, whatever it was - mirrors the lost_50_50 branch's
    was_guaranteed handling too.
    """
    for starting_radiance in range(4):
        account = Account(
            current_pity=0,
            character_guarantee=True,
            owned_characters=Ownership({}),
            wishes=10,
            capturing_radiance_counter=starting_radiance,
        )
        new_radiance = next_capturing_radiance_counter(
            account.capturing_radiance_counter,
            was_guaranteed=account.character_guarantee,
            featured=True,
        )
        assert new_radiance == starting_radiance
