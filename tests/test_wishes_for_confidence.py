"""wishes_for_confidence: curve inversion (Design Document §10.3)."""

import numpy as np
import pytest

from domain import CHARACTER_EVENT_BANNER, WishMechanics
from probability import cumulative_probability, wishes_for_confidence


TINY_MECHANICS = WishMechanics(
    banner_type="tiny",
    hard_pity=3,
    soft_pity_start=2,
    base_rate=0.5,
    soft_pity_increment=0.25,
    featured_rate=0.5,
)


class TestHorizons:
    """Certainty horizons follow from the mechanics, never hard-coded."""

    @pytest.mark.parametrize(
        "pity,guaranteed,expected",
        [
            (0, True, 90),  # guaranteed 5-star by one hard-pity cycle
            (0, False, 180),  # lose the 50/50, then a guaranteed cycle
            (89, True, 1),  # next pull IS the hard-pity 5-star
            (45, True, 45),  # 45 pulls remain to the hard-pity pull
            (45, False, 135),  # cycle remainder + one full cycle
        ],
    )
    def test_full_confidence_horizons(self, pity, guaranteed, expected):
        assert (
            wishes_for_confidence(1.0, pity, guaranteed, CHARACTER_EVENT_BANNER)
            == expected
        )

    def test_tiny_mechanics_scale_the_horizon(self):
        assert wishes_for_confidence(1.0, 0, True, TINY_MECHANICS) == 3
        assert wishes_for_confidence(1.0, 0, False, TINY_MECHANICS) == 6


class TestInversion:
    """The result is the first curve index meeting the confidence (§10.3)."""

    @pytest.mark.parametrize(
        "starting_pity,guaranteed",
        [(0, False), (37, False), (37, True), (80, True)],
    )
    def test_round_trip_against_the_curve(self, starting_pity, guaranteed):
        mechanics = CHARACTER_EVENT_BANNER
        curve = cumulative_probability(180, starting_pity, guaranteed, mechanics)
        for confidence in np.linspace(0.05, 1.0, 20):
            n = wishes_for_confidence(
                float(confidence), starting_pity, guaranteed, mechanics
            )
            assert curve[n] >= confidence
            if n > 0:
                # Minimality: the wish before n must fall short.
                assert curve[n - 1] < confidence

    def test_more_confidence_never_needs_fewer_wishes(self):
        mechanics = CHARACTER_EVENT_BANNER
        previous = 0
        for confidence in [0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]:
            n = wishes_for_confidence(confidence, 0, False, mechanics)
            assert n >= previous
            previous = n

    def test_low_confidence_within_the_first_pull(self):
        # Pull 1 yields featured with probability 0.006 * 0.5 ~ 0.003.
        assert wishes_for_confidence(0.002, 0, False, CHARACTER_EVENT_BANNER) == 1
        # 0.005 is beyond the first pull's reach but under pull 2's cumulative
        # reach (1 - 0.994^2) * 0.5 ~ 0.00598.
        assert wishes_for_confidence(0.005, 0, False, CHARACTER_EVENT_BANNER) == 2

    def test_small_confidence_needs_zero_wishes_is_never_reported(self):
        # 0 and negative confidence are meaningless inputs, not "0 wishes".
        with pytest.raises(ValueError, match="confidence"):
            wishes_for_confidence(0.0, 0, False, CHARACTER_EVENT_BANNER)


class TestValidation:
    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            wishes_for_confidence(1.01, 0, False, CHARACTER_EVENT_BANNER)

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            wishes_for_confidence(-0.1, 0, False, CHARACTER_EVENT_BANNER)

    def test_starting_pity_out_of_range_rejected(self):
        with pytest.raises(ValueError, match="starting_pity"):
            wishes_for_confidence(0.5, 90, False, CHARACTER_EVENT_BANNER)

    def test_unattainable_confidence_rejected(self, monkeypatch):
        """Mechanics-driven unattainability: the curve itself decides.

        With valid pull mechanics every confidence <= 1.0 is attainable
        within the bounded horizon, so this feeds the function a capped
        curve to exercise the branch.
        """
        import probability.character as character

        capped = np.linspace(0.0, 0.9, 181)

        def fake_curve(wishes, starting_pity, guaranteed, mechanics, starting_radiance=0):
            assert wishes <= 180
            return capped[: wishes + 1]

        monkeypatch.setattr(character, "cumulative_probability", fake_curve)
        with pytest.raises(ValueError, match="unattainable"):
            character.wishes_for_confidence(0.95, 0, False, CHARACTER_EVENT_BANNER)
