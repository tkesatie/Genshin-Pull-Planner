"""Focused tests for the vectorized single-banner Monte Carlo core."""

import numpy as np
import pytest

from domain import Account, Ownership, WishMechanics, next_capturing_radiance_counter
from probability import capturing_radiance_rate
from simulation.engine import (
    _pull_toward_target,
    _pull_toward_target_vectorized,
)


def forced_mechanics(featured_rate: float = 1.0) -> WishMechanics:
    return WishMechanics(
        banner_type="forced",
        hard_pity=2,
        soft_pity_start=1,
        base_rate=0.5,
        soft_pity_increment=0.5,
        featured_rate=featured_rate,
    )


def slow_mechanics() -> WishMechanics:
    return WishMechanics(
        banner_type="slow",
        hard_pity=90,
        soft_pity_start=74,
        base_rate=1e-12,
        soft_pity_increment=1e-12,
        featured_rate=1.0,
    )


def stochastic_test_mechanics() -> WishMechanics:
    return WishMechanics(
        banner_type="stochastic_test",
        hard_pity=4,
        soft_pity_start=3,
        base_rate=0.2,
        soft_pity_increment=0.2,
        featured_rate=0.5,
    )


class ZeroRng:
    """Test RNG that makes every probabilistic draw succeed."""

    def random(self, size=None):
        if size is None:
            return 0.0
        return np.zeros(size, dtype=float)


class TestVectorizedBanner:
    def test_matches_scalar_statistically(self):
        """Independent RNG streams should produce equivalent distributions."""
        runs = 10_000
        mechanics = stochastic_test_mechanics()

        vectorized = _pull_toward_target_vectorized(
            current_pity=np.zeros(runs, dtype=int),
            guarantee=np.zeros(runs, dtype=bool),
            radiance=np.zeros(runs, dtype=int),
            wishes=np.full(runs, 8, dtype=int),
            owned=np.full(runs, -1, dtype=int),
            copies_needed=np.ones(runs, dtype=int),
            budget=np.full(runs, 8, dtype=int),
            mechanics=mechanics,
            rng=np.random.default_rng(12345),
        )

        scalar_spent = np.empty(runs, dtype=int)
        scalar_obtained = np.empty(runs, dtype=int)
        scalar_rng = np.random.default_rng(67890)

        for i in range(runs):
            account = Account(
                owned_characters=Ownership({"Vesna": -1}),
                wishes=8,
            )
            spent, obtained, _, _, _ = _pull_toward_target(
                account,
                "Vesna",
                1,
                8,
                mechanics,
                scalar_rng,
            )
            scalar_spent[i] = spent
            scalar_obtained[i] = obtained

        vector_spent, vector_obtained, *_ = vectorized

        # The original assertion compared against zero rather than the scalar
        # oracle. The two implementations should agree with each other.
        assert abs(vector_obtained.mean() - scalar_obtained.mean()) < 0.03
        assert abs(vector_spent.mean() - scalar_spent.mean()) < 0.15

    def test_matches_scalar_for_deterministic_featured_outcomes(self):
        """The vectorized state machine matches the scalar oracle when
        randomness is removed from the featured decision."""
        initial = [
            (0, False, 0, 5, -1, 1, 5),
            (1, True, 0, 5, -1, 2, 5),
            (0, False, 2, 5, 0, 2, 5),
            (0, False, 3, 4, 1, 1, 3),
        ]

        current_pity, guarantee, radiance, wishes, owned, copies_needed, budget = map(
            np.array, zip(*initial)
        )

        vectorized = _pull_toward_target_vectorized(
            current_pity=current_pity,
            guarantee=guarantee,
            radiance=radiance,
            wishes=wishes,
            owned=owned,
            copies_needed=copies_needed,
            budget=budget,
            mechanics=forced_mechanics(),
            rng=ZeroRng(),
        )

        scalar = []
        for (
            pity,
            is_guaranteed,
            starting_radiance,
            available_wishes,
            starting_owned,
            needed,
            cap,
        ) in initial:
            account = Account(
                current_pity=int(pity),
                character_guarantee=bool(is_guaranteed),
                owned_characters=Ownership({"Vesna": int(starting_owned)}),
                wishes=int(available_wishes),
                capturing_radiance_counter=int(starting_radiance),
            )
            spent, obtained, copy_wishes, outcomes, account_after = _pull_toward_target(
                account,
                "Vesna",
                int(needed),
                int(cap),
                forced_mechanics(),
                ZeroRng(),
            )
            scalar.append(
                (
                    spent,
                    obtained,
                    account_after.current_pity,
                    account_after.character_guarantee,
                    account_after.capturing_radiance_counter,
                    account_after.owned_constellation("Vesna"),
                    list(outcomes),
                )
            )

        spent, obtained, pity, guarantee, radiance, owned, outcomes = vectorized

        np.testing.assert_array_equal(
            spent, [result[0] for result in scalar]
        )
        np.testing.assert_array_equal(
            obtained, [result[1] for result in scalar]
        )
        np.testing.assert_array_equal(
            pity, [result[2] for result in scalar]
        )
        np.testing.assert_array_equal(
            guarantee, [result[3] for result in scalar]
        )
        np.testing.assert_array_equal(
            radiance, [result[4] for result in scalar]
        )
        np.testing.assert_array_equal(
            owned, [result[5] for result in scalar]
        )
        assert outcomes == [result[6] for result in scalar]

    def test_independent_histories_stop_at_their_own_targets_or_budgets(self):
        spent, obtained, pity, guarantee, radiance, owned, outcomes = _pull_toward_target_vectorized(
            current_pity=np.array([0, 0, 0]),
            guarantee=np.array([False, False, False]),
            radiance=np.array([0, 0, 0]),
            wishes=np.array([5, 5, 5]),
            owned=np.array([-1, -1, -1]),
            copies_needed=np.array([1, 2, 3]),
            budget=np.array([5, 2, 3]),
            mechanics=forced_mechanics(),
            rng=np.random.default_rng(0),
        )

        np.testing.assert_array_equal(spent, [1, 2, 3])
        np.testing.assert_array_equal(obtained, [1, 2, 3])
        np.testing.assert_array_equal(pity, [0, 0, 0])
        np.testing.assert_array_equal(guarantee, [False, False, False])
        np.testing.assert_array_equal(radiance, [0, 0, 0])
        np.testing.assert_array_equal(owned, [0, 1, 2])
        assert outcomes == [
            [(1, True)],
            [(1, True), (2, True)],
            [(1, True), (2, True), (3, True)],
        ]

    def test_guarantee_and_radiance_state_transition_are_per_history(self):
        spent, obtained, pity, guarantee, radiance, owned, outcomes = _pull_toward_target_vectorized(
            current_pity=np.array([0, 0]),
            guarantee=np.array([False, True]),
            radiance=np.array([0, 0]),
            wishes=np.array([2, 1]),
            owned=np.array([-1, -1]),
            copies_needed=np.array([1, 1]),
            budget=np.array([2, 1]),
            mechanics=forced_mechanics(featured_rate=1e-9),
            rng=np.random.default_rng(0),
        )

        np.testing.assert_array_equal(spent, [2, 1])
        np.testing.assert_array_equal(obtained, [1, 1])
        np.testing.assert_array_equal(pity, [0, 0])
        np.testing.assert_array_equal(guarantee, [False, False])
        np.testing.assert_array_equal(radiance, [1, 0])
        np.testing.assert_array_equal(owned, [0, 0])
        assert outcomes[0] == [(1, False), (2, True)]
        assert outcomes[1] == [(1, True)]

    def test_no_five_star_advances_pity_independently(self):
        spent, obtained, pity, guarantee, radiance, owned, outcomes = _pull_toward_target_vectorized(
            current_pity=np.array([0, 1]),
            guarantee=np.array([False, True]),
            radiance=np.array([0, 2]),
            wishes=np.array([1, 1]),
            owned=np.array([-1, -1]),
            copies_needed=np.array([1, 1]),
            budget=np.array([1, 1]),
            mechanics=slow_mechanics(),
            rng=np.random.default_rng(0),
        )

        np.testing.assert_array_equal(spent, [1, 1])
        np.testing.assert_array_equal(obtained, [0, 0])
        np.testing.assert_array_equal(pity, [1, 2])
        np.testing.assert_array_equal(guarantee, [False, True])
        np.testing.assert_array_equal(radiance, [0, 2])
        np.testing.assert_array_equal(owned, [-1, -1])
        assert outcomes == [[], []]

    def test_rejects_mismatched_state_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            _pull_toward_target_vectorized(
                current_pity=np.array([0, 0]),
                guarantee=np.array([False]),
                radiance=np.array([0, 0]),
                wishes=np.array([2, 2]),
                owned=np.array([-1, -1]),
                copies_needed=np.array([1, 1]),
                budget=np.array([2, 2]),
                mechanics=forced_mechanics(),
                rng=np.random.default_rng(0),
            )


class TestCapturingRadianceVectorized:
    """The vectorized engine must not re-derive Capturing Radiance (§6).

    The featured win probability per counter state must come from
    `probability.capturing_radiance_rate` and the counter transition from
    `domain.next_capturing_radiance_counter` - the same functions the
    scalar engine calls.
    """

    def test_transitions_match_next_capturing_radiance_counter(self):
        """Every (starting counter, guarantee, outcome) combination moves
        the counter exactly as the single-sourced domain function says."""
        for win_forced, featured_rate in ((True, 1.0), (False, 1e-9)):
            for guaranteed in (False, True):
                for radiance_start in range(4):
                    runs = 1
                    (
                        _,
                        _,
                        _,
                        guarantee_after,
                        radiance_after,
                        _,
                        outcomes,
                    ) = _pull_toward_target_vectorized(
                        current_pity=np.zeros(runs, dtype=int),
                        guarantee=np.full(runs, guaranteed, dtype=bool),
                        radiance=np.full(runs, radiance_start, dtype=int),
                        wishes=np.ones(runs, dtype=int),
                        owned=np.full(runs, -1, dtype=int),
                        copies_needed=np.ones(runs, dtype=int),
                        budget=np.ones(runs, dtype=int),
                        mechanics=forced_mechanics(featured_rate=featured_rate),
                        rng=np.random.default_rng(0),
                    )
                    # Every pull is a certain 5-star (forced mechanics):
                    # the single wish always produces one outcome.
                    ((_, featured),) = outcomes[0]
                    # The featured outcome must follow the single-sourced
                    # schedule: a guarantee forces it, radiance 3 forces it,
                    # and radiance 0/1 at featured_rate ~0 forces a loss.
                    # (Radiance 2 is the stochastic 6/11 state - covered by
                    # the statistical tests below.)
                    if guaranteed or radiance_start == 3:
                        assert featured is True
                    elif radiance_start <= 1 and not win_forced:
                        assert featured is False
                    assert guarantee_after[0] == (not featured)

                    expected_radiance = next_capturing_radiance_counter(
                        radiance_start,
                        was_guaranteed=guaranteed,
                        featured=featured,
                    )
                    assert radiance_after[0] == expected_radiance

    def test_transitions_are_per_history_in_one_mixed_batch(self):
        """All eight (guarantee, counter) starting states in one vectorized
        call transition independently, each matching the domain function."""
        runs = 8
        guarantee = np.array([False, False, False, False, True, True, True, True])
        radiance_start = np.array([0, 1, 2, 3, 0, 1, 2, 3])

        (
            _,
            _,
            _,
            guarantee_after,
            radiance_after,
            _,
            outcomes,
        ) = _pull_toward_target_vectorized(
            current_pity=np.zeros(runs, dtype=int),
            guarantee=guarantee,
            radiance=radiance_start.copy(),
            wishes=np.ones(runs, dtype=int),
            owned=np.full(runs, -1, dtype=int),
            copies_needed=np.ones(runs, dtype=int),
            budget=np.ones(runs, dtype=int),
            # The guaranteed histories are featured regardless of the rate;
            # the non-guaranteed ones all lose the 50/50 at rate ~0.
            mechanics=forced_mechanics(featured_rate=1e-9),
            rng=np.random.default_rng(0),
        )

        for index in range(runs):
            was_guaranteed = bool(guarantee[index])
            ((_, featured),) = outcomes[index]
            expected_radiance = next_capturing_radiance_counter(
                int(radiance_start[index]),
                was_guaranteed=was_guaranteed,
                featured=featured,
            )
            assert radiance_after[index] == expected_radiance
            assert guarantee_after[index] == (not featured)
            # Schedule-aware outcome check: guaranteed histories and
            # radiance-3 histories always win; radiance 0/1 histories at
            # featured_rate ~0 always lose; radiance 2 is the stochastic
            # 6/11 state (either outcome is valid).
            if was_guaranteed or int(radiance_start[index]) == 3:
                assert featured is True
            elif int(radiance_start[index]) <= 1:
                assert featured is False

    def test_featured_probability_matches_capturing_radiance_rate(self):
        """Statistical check of the featured probability at every counter
        state: 0/1 -> featured_rate, 2 -> 6/11, 3 -> guaranteed."""
        runs = 40_000
        # Every pull is a certain 5-star (rate 1.0) and the raw 50/50 is
        # 0.5, so one wish per history isolates the radiance schedule.
        mechanics = forced_mechanics(featured_rate=0.5)
        rng = np.random.default_rng(7)

        featured_fraction = {}
        for radiance_start in range(4):
            _, _, _, _, _, _, outcomes = _pull_toward_target_vectorized(
                current_pity=np.zeros(runs, dtype=int),
                guarantee=np.zeros(runs, dtype=bool),
                radiance=np.full(runs, radiance_start, dtype=int),
                wishes=np.ones(runs, dtype=int),
                owned=np.full(runs, -1, dtype=int),
                copies_needed=np.ones(runs, dtype=int),
                budget=np.ones(runs, dtype=int),
                mechanics=mechanics,
                rng=rng,
            )
            assert len(outcomes) == runs
            featured_fraction[radiance_start] = np.mean(
                [outcome[0][1] for outcome in outcomes]
            )

        for radiance_start in range(4):
            expected = capturing_radiance_rate(radiance_start, mechanics)
            assert featured_fraction[radiance_start] == pytest.approx(
                expected, abs=0.01
            )

    def test_radiance_2_and_3_transitions_match_the_domain_function(self):
        """Statistical transition check for the two boosted counter states:
        a radiance-2 loss lands on 3, a radiance-2 win leaves the residual
        mark 1, a radiance-3 win leaves 1 - and no radiance-3 loss exists
        (the schedule makes it a guaranteed featured win)."""
        runs = 20_000
        mechanics = forced_mechanics(featured_rate=0.5)

        _, _, _, _, radiance_after, _, outcomes = _pull_toward_target_vectorized(
            current_pity=np.zeros(runs, dtype=int),
            guarantee=np.zeros(runs, dtype=bool),
            radiance=np.full(runs, 2, dtype=int),
            wishes=np.ones(runs, dtype=int),
            owned=np.full(runs, -1, dtype=int),
            copies_needed=np.ones(runs, dtype=int),
            budget=np.ones(runs, dtype=int),
            mechanics=mechanics,
            rng=np.random.default_rng(11),
        )

        wins = losses = 0
        for index, outcome in enumerate(outcomes):
            ((_, featured),) = outcome
            expected = next_capturing_radiance_counter(
                2, was_guaranteed=False, featured=featured
            )
            assert radiance_after[index] == expected
            if featured:
                wins += 1
                assert radiance_after[index] == 1  # residual mark
            else:
                losses += 1
                assert radiance_after[index] == 3  # capped streak

        # Both branches occurred, in radiance_rate(2) proportion.
        assert wins > 0 and losses > 0
        assert wins / runs == pytest.approx(
            capturing_radiance_rate(2, mechanics), abs=0.01
        )

        # Radiance 3 is a guaranteed featured win leaving the residual 1;
        # a loss (which would cap the streak at 3) can never happen.
        _, _, _, _, radiance_after_3, _, outcomes_3 = (
            _pull_toward_target_vectorized(
                current_pity=np.zeros(runs, dtype=int),
                guarantee=np.zeros(runs, dtype=bool),
                radiance=np.full(runs, 3, dtype=int),
                wishes=np.ones(runs, dtype=int),
                owned=np.full(runs, -1, dtype=int),
                copies_needed=np.ones(runs, dtype=int),
                budget=np.ones(runs, dtype=int),
                mechanics=mechanics,
                rng=np.random.default_rng(11),
            )
        )
        for index, outcome in enumerate(outcomes_3):
            ((_, featured),) = outcome
            assert featured is True
            assert radiance_after_3[index] == 1
