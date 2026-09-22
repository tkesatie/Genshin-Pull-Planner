"""Wish mechanics (Design Document §17).

Mechanics are data rather than hard-coded assumptions: the probability
engine (Phase 2) must be able to consume updated mechanics without changing
logic. No rate mathematics lives here - `pull_rate` and friends are Phase 2.

Field semantics (character event banner convention):

    hard_pity: the pull (1-based) at which a 5-star is guaranteed.
    soft_pity_start: the pull (1-based) at which the soft-pity increment
        first applies on top of the base rate.
    base_rate: 5-star chance per pull before soft pity.
    soft_pity_increment: additional 5-star chance per pull once past
        soft_pity_start.
    featured_rate: chance that an obtained 5-star is the featured character
        (0.5 = "50/50" on the character event banner).

The exact weapon-banner mechanics are deliberately absent until verified
independently (§17).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class WishMechanics:
    """Pull mechanics for one banner type (§17)."""

    banner_type: str
    hard_pity: int
    soft_pity_start: int
    base_rate: float
    soft_pity_increment: float
    featured_rate: float

    def __post_init__(self) -> None:
        if self.hard_pity <= 0:
            raise ValueError(f"hard_pity must be >= 1, got {self.hard_pity}")
        if not 0 < self.soft_pity_start < self.hard_pity:
            raise ValueError(
                "soft_pity_start must satisfy 0 < soft_pity_start < hard_pity "
                f"({self.hard_pity}), got {self.soft_pity_start}"
            )
        if not 0.0 < self.base_rate < 1.0:
            raise ValueError(f"base_rate must be in (0, 1), got {self.base_rate}")
        if not 0.0 < self.soft_pity_increment <= 1.0:
            raise ValueError(
                f"soft_pity_increment must be in (0, 1], got {self.soft_pity_increment}"
            )
        if not 0.0 < self.featured_rate <= 1.0:
            raise ValueError(
                f"featured_rate must be in (0, 1], got {self.featured_rate}"
            )


WEAPON_EVENT_BANNER = WishMechanics(
    banner_type="weapon_event",
    hard_pity=80,
    soft_pity_start=64,
    base_rate=0.007,
    soft_pity_increment=0.06,
    featured_rate=0.75,
)
# Weapon-banner constants (§17):
#   * Official (in-game Details text): base 5-star rate 0.7%, consolidated
#     1.85%, hard pity 80, 75% rate-up share, Epitomized Path at 1 Fate
#     Point (Version 5.0; previously 2).
#   * Soft pity is NOT published by HoYoverse. soft_pity_start=64 with a
#     0.06 increment is a community-documented schedule; it was selected
#     because (a) the ramp saturates exactly at the pull-80 hard pity and
#     (b) the resulting mean of ~54.1 pulls per 5-star matches the
#     officially published 1.85% consolidated rate (implied 54.05) to
#     within ~0.06 pulls. This validates the choice against known reference
#     data without inventing unpublished values as fact - if HoYoverse
#     ever publishes the real schedule, update this data and the probability
#     logic follows (mechanics as data, §17).


CHARACTER_EVENT_BANNER = WishMechanics(
    banner_type="character_event",
    hard_pity=90,
    soft_pity_start=74,
    base_rate=0.006,
    soft_pity_increment=0.06,
    featured_rate=0.5,
)
