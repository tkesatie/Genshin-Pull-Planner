"""Focused tests for the vectorized single-banner Monte Carlo core."""

import numpy as np
import pytest

from domain import Account, Ownership, WishMechanics
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

        assert abs(vector_obtained.mean() - scalar_obtained.mean()) < 0.03
        assert abs(vector_spent.mean() - scalar_spent.mean()) < 0.15
        assert abs(vector_obtained.mean() - 0.0) > 0.1


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
