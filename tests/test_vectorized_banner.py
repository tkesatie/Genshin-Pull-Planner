"""Focused tests for the vectorized single-banner Monte Carlo core."""

import numpy as np
import pytest

from domain import WishMechanics
from simulation.engine import _pull_toward_target_vectorized


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
        hard_pity=3,
        soft_pity_start=2,
        base_rate=1e-9,
        soft_pity_increment=0.5,
        featured_rate=1.0,
    )


class TestVectorizedBanner:
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
