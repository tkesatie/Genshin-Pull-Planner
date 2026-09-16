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


CHARACTER_EVENT_BANNER = WishMechanics(
    banner_type="character_event",
    hard_pity=90,
    soft_pity_start=74,
    base_rate=0.006,
    soft_pity_increment=0.06,
    featured_rate=0.5,
)
