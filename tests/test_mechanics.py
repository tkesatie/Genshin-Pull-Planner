"""Wish mechanics data (Design Document §17)."""

import pytest

from domain import CHARACTER_EVENT_BANNER, WishMechanics


def test_character_event_banner_defaults():
    assert CHARACTER_EVENT_BANNER == WishMechanics(
        banner_type="character_event",
        hard_pity=90,
        soft_pity_start=74,
        base_rate=0.006,
        soft_pity_increment=0.06,
        featured_rate=0.5,
    )


class TestValidation:
    def make(self, **overrides) -> WishMechanics:
        values = dict(
            banner_type="character_event",
            hard_pity=90,
            soft_pity_start=74,
            base_rate=0.006,
            soft_pity_increment=0.06,
            featured_rate=0.5,
        )
        values.update(overrides)
        return WishMechanics(**values)

    def test_hard_pity_must_be_positive(self):
        with pytest.raises(ValueError, match="hard_pity"):
            self.make(hard_pity=0)

    def test_soft_pity_must_be_inside_hard_pity(self):
        with pytest.raises(ValueError, match="soft_pity_start"):
            self.make(soft_pity_start=90)
        with pytest.raises(ValueError, match="soft_pity_start"):
            self.make(soft_pity_start=0)

    def test_base_rate_is_a_probability(self):
        with pytest.raises(ValueError, match="base_rate"):
            self.make(base_rate=0.0)
        with pytest.raises(ValueError, match="base_rate"):
            self.make(base_rate=1.0)

    def test_soft_pity_increment_is_a_probability_step(self):
        with pytest.raises(ValueError, match="soft_pity_increment"):
            self.make(soft_pity_increment=0.0)
        with pytest.raises(ValueError, match="soft_pity_increment"):
            self.make(soft_pity_increment=1.5)

    def test_featured_rate_is_a_probability(self):
        with pytest.raises(ValueError, match="featured_rate"):
            self.make(featured_rate=0.0)
        with pytest.raises(ValueError, match="featured_rate"):
            self.make(featured_rate=1.5)

    def test_boundary_increment_of_one_is_allowed(self):
        mechanics = self.make(soft_pity_increment=1.0)
        assert mechanics.soft_pity_increment == 1.0
