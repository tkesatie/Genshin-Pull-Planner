"""Per-pull rates (Design Document §10.1).

Rate mathematics as data-driven functions over `WishMechanics` (§17), so
mechanics can be updated without changing probability logic.

Pity convention matches `Account.current_pity` (§4.1): 0-based pulls already
made since the last 5-star. The *upcoming* wish is therefore pull
`pity + 1` (1-based).
"""

from domain.mechanics import WishMechanics

# The win probability at Capturing Radiance counter 2 (see
# capturing_radiance_rate). Not an arbitrary guess: calibrated so the 4-state
# Markov chain formed with domain.account.next_capturing_radiance_counter's
# transitions has a long-run average win rate of exactly 0.55, matching the
# officially cited "55% overall" aggregate figure for the mechanic.
CAPTURING_RADIANCE_RATE_AT_2 = 6.0 / 11.0


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


def pull_rate_array(pity, mechanics: WishMechanics):
    """Vectorized equivalent of pull_rate for an array of pity values.

    pity is the same 0-based state used by pull_rate. The returned array
    contains the probability of a 5-star on the next wish for each history
    independently.

    This deliberately mirrors the scalar implementation rather than
    introducing a separate rate model. It is intended for the vectorized
    Monte Carlo engine, where one NumPy operation can evaluate all active
    histories at once.
    """
    import numpy as np

    pity = np.asarray(pity)
    if np.any(pity < 0):
        raise ValueError("pity must be non-negative")
    if np.any(pity >= mechanics.hard_pity):
        raise ValueError(
            f"pity must be < hard_pity ({mechanics.hard_pity})"
        )

    upcoming_pull = pity + 1
    rates = np.full(pity.shape, mechanics.base_rate, dtype=float)

    soft_pity = upcoming_pull >= mechanics.soft_pity_start
    rates = np.where(
        soft_pity,
        mechanics.base_rate
        + (upcoming_pull - mechanics.soft_pity_start + 1)
        * mechanics.soft_pity_increment,
        rates,
    )

    return np.where(
        upcoming_pull >= mechanics.hard_pity, 1.0, np.minimum(rates, 1.0)
    )


def featured_rate_at(
    pity: int, guarantee: bool, mechanics: WishMechanics
) -> float:
    """Probability that the next *5-star* is the featured character (§10.1).

    This is conditional on a 5-star occurring - it is not the probability
    that the next wish is the featured character. This variant does not
    account for Capturing Radiance (see `capturing_radiance_rate` for a
    radiance-aware version); it answers the simpler pre-Capturing-Radiance
    question:

        guarantee=True   ->  1.0   (the guarantee forces the featured unit)
        guarantee=False  ->  mechanics.featured_rate (0.5 = the "50/50")

    The unconditional probability that the next wish is featured is
    `pull_rate(pity, mechanics) * featured_rate_at(pity, guarantee, mechanics)`.
    """
    _validate_pity(pity, mechanics)
    if guarantee:
        return 1.0
    return mechanics.featured_rate


def capturing_radiance_rate(radiance: int, mechanics: WishMechanics) -> float:
    """P(the next non-guaranteed 5-star is featured), Capturing Radiance-aware.

    `radiance` is the Capturing Radiance loss-streak counter (0-3; see
    `domain.account.Account.capturing_radiance_counter` and
    `domain.account.next_capturing_radiance_counter`). This is the
    radiance-aware counterpart to `featured_rate_at`'s guarantee-only
    question, and should be used instead of it whenever Capturing Radiance
    applies - i.e. whenever the pull is not already guaranteed.

        radiance 0 or 1  ->  mechanics.featured_rate (the base 50/50)
        radiance 2       ->  6/11 (~54.5%)
        radiance 3       ->  1.0 (guaranteed)

    Single-sourced here so `simulation.engine` and `probability.character`
    cannot drift apart on the schedule (they previously each hardcoded
    these thresholds independently). See `CAPTURING_RADIANCE_RATE_AT_2` for
    where the 6/11 figure comes from.

    Raises:
        ValueError: if `radiance` is outside [0, 3].
    """
    if not 0 <= radiance <= 3:
        raise ValueError(f"radiance must be in [0, 3], got {radiance}")
    if radiance < 2:
        return mechanics.featured_rate
    if radiance == 2:
        return CAPTURING_RADIANCE_RATE_AT_2
    return 1.0
