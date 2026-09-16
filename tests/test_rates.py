"""Per-pull rates: pull_rate and featured_rate_at (Design Document §10.1)."""

import pytest

from domain import CHARACTER_EVENT_BANNER, WishMechanics
from probability import featured_rate_at, pull_rate


class TestPullRateConvention:
    """The intended convention, spelled out pull by pull (§10.1)."""

    def test_first_pull_is_base_rate(self):
        assert pull_rate(0, CHARACTER_EVENT_BANNER) == pytest.approx(0.006)

    def test_rate_before_soft_pity_is_flat_base_rate(self):
        # Pulls 1..73 all sit at the base rate.
        assert pull_rate(72, CHARACTER_EVENT_BANNER) == pytest.approx(0.006)

    def test_soft_pity_start_applies_one_increment(self):
        # Pull 74 is the first soft-pity pull: base + one increment.
        assert pull_rate(73, CHARACTER_EVENT_BANNER) == pytest.approx(0.066)

    def test_each_soft_pity_pull_adds_another_increment(self):
        assert pull_rate(74, CHARACTER_EVENT_BANNER) == pytest.approx(0.126)
        assert pull_rate(75, CHARACTER_EVENT_BANNER) == pytest.approx(0.186)

    def test_last_pity_before_hard_pity_is_below_one(self):
        # Pull 89: base + 16 increments = 0.966 - high, but not yet certain.
        assert pull_rate(88, CHARACTER_EVENT_BANNER) == pytest.approx(0.966)

    def test_hard_pity_pull_is_certain(self):
        # Pity 89 is valid: the next pull is pull 90, the hard-pity pull.
        assert pull_rate(89, CHARACTER_EVENT_BANNER) == 1.0

    def test_hard_pity_clamps_rate_at_one(self):
        mechanics = WishMechanics(
            banner_type="test",
            hard_pity=10,
            soft_pity_start=5,
            base_rate=0.1,
            soft_pity_increment=0.5,
            featured_rate=0.5,
        )
        # Pull 10 would naively be 0.1 + 6*0.5 = 3.1; hard pity caps at 1.0.
        assert pull_rate(9, mechanics) == 1.0
        # Pull 7: 0.1 + 3*0.5 = 1.6 -> clamped to 1.0.
        assert pull_rate(6, mechanics) == 1.0
        # Pull 6: 0.1 + 2*0.5 = 1.1 -> clamped to 1.0.
        assert pull_rate(5, mechanics) == 1.0


class TestPullRateValidation:
    """Pity is 0-based and resets on every 5-star: 0 <= pity < hard_pity."""

    def test_pity_89_is_valid(self):
        assert pull_rate(89, CHARACTER_EVENT_BANNER) == 1.0

    def test_pity_90_is_rejected(self):
        with pytest.raises(ValueError, match="hard_pity"):
            pull_rate(90, CHARACTER_EVENT_BANNER)

    def test_negative_pity_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            pull_rate(-1, CHARACTER_EVENT_BANNER)


class TestFeaturedRateAt:
    """Conditional probability that the next 5-star is featured (§10.1)."""

    def test_guarantee_forces_the_featured_unit(self):
        # Conditional on a 5-star: the guarantee makes it certain, whatever
        # the featured_rate says.
        assert featured_rate_at(0, True, CHARACTER_EVENT_BANNER) == 1.0
        assert featured_rate_at(89, True, CHARACTER_EVENT_BANNER) == 1.0

    def test_no_guarantee_is_the_raw_featured_rate(self):
        assert featured_rate_at(0, False, CHARACTER_EVENT_BANNER) == 0.5

    def test_is_conditional_not_unconditional(self):
        # P(next wish featured) = pull_rate * featured_rate_at, so the
        # conditional rate alone must NOT be read as per-wish probability.
        unconditional = pull_rate(0, CHARACTER_EVENT_BANNER) * 0.5
        assert featured_rate_at(0, False, CHARACTER_EVENT_BANNER) != pytest.approx(
            unconditional
        )
