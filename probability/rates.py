"""Per-pull rates (Design Document §10.1).

Rate mathematics as data-driven functions over `WishMechanics` (§17), so
mechanics can be updated without changing probability logic.

Pity convention matches `Account.current_pity` (§4.1): 0-based pulls already
made since the last 5-star. The *upcoming* wish is therefore pull
`pity + 1` (1-based).
"""

from domain.mechanics import WishMechanics


def _validate_pity(pity: int, mechanics: WishMechanics) -> None:
    """Pity is 0-based and a 5-star resets it, so pity < hard_pity always.

    With hard_pity=90, pity 89 is valid: the next pull is pull 90 and its
    rate is 1.0. pity=90 and negative pity are rejected.
    """
    if pity < 0:
        raise ValueError(f"pity must be non-negative, got {pity}")
    if pity >= mechanics.hard_pity:
        raise ValueError(
            f"pity must be < hard_pity ({mechanics.hard_pity}), got {pity}"
        )


def pull_rate(pity: int, mechanics: WishMechanics) -> float:
    """Probability that the next wish yields a 5-star (§10.1).

    The next wish is pull `pity + 1` (1-based):

        pull <= soft_pity_start - 1   ->  base_rate
        pull >= soft_pity_start       ->  base_rate +
            (pull - soft_pity_start + 1) * soft_pity_increment
        pull == hard_pity             ->  1.0 (guaranteed)

    For CHARACTER_EVENT_BANNER this gives the intended convention:

        pull 73 -> 0.006
        pull 74 -> 0.066
        pull 75 -> 0.126
        pull 90 -> 1.0
    """
    _validate_pity(pity, mechanics)
    upcoming_pull = pity + 1
    if upcoming_pull >= mechanics.hard_pity:
        return 1.0
    if upcoming_pull < mechanics.soft_pity_start:
        return mechanics.base_rate
    rate = mechanics.base_rate + (
        upcoming_pull - mechanics.soft_pity_start + 1
    ) * mechanics.soft_pity_increment
    return min(rate, 1.0)


def featured_rate_at(
    pity: int, guarantee: bool, mechanics: WishMechanics
) -> float:
    """Probability that the next *5-star* is the featured character (§10.1).

    This is conditional on a 5-star occurring - it is not the probability
    that the next wish is the featured character. That distinction matters
    once the planner composes these rates:

        guarantee=True   ->  1.0   (the guarantee forces the featured unit)
        guarantee=False  ->  mechanics.featured_rate (0.5 = the "50/50")

    The unconditional probability that the next wish is featured is
    `pull_rate(pity, mechanics) * featured_rate_at(pity, guarantee, mechanics)`.
    """
    _validate_pity(pity, mechanics)
    if guarantee:
        return 1.0
    return mechanics.featured_rate
