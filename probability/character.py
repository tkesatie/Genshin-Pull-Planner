"""Single-copy character probability (Design Document §10.2, §10.3).

Exact dynamic programming over the `(pity, guarantee)` state space - no
sampling. State space is tiny (hard_pity x 2), so exact probability is both
faster and more accurate than simulation, and gives the Phase 4 Monte Carlo
simulator something deterministic to validate against.

The engine answers only: "starting at this pity/guarantee state, how likely
am I to obtain ONE featured copy within N wishes?" (§10.4 multi-copy targets
are Phase 4.) It knows nothing about characters, goals, or roadmaps.
"""

from dataclasses import dataclass

import numpy as np

from domain.mechanics import WishMechanics
from probability.rates import featured_rate_at, pull_rate


@dataclass(frozen=True)
class _State:
    """Survival probability mass over `(pity, guarantee)` states."""

    no_guarantee: np.ndarray  # shape (hard_pity, 4)
    guaranteed: np.ndarray  # shape (hard_pity, 4)

    def total(self) -> float:
        return float(self.no_guarantee.sum() + self.guaranteed.sum())


@dataclass(frozen=True)
class _Transitions:
    """Per-pity vectors used to advance the distribution one wish.

    Precomputed once per curve instead of per wish:

        rates:         P(5-star on the next wish) at each pity
        survive:       P(no 5-star on the next wish) at each pity
        lost_cond:     P(5-star occurs and is NOT featured) at each pity,
                       for a not-guaranteed state
    """

    rates: np.ndarray
    survive: np.ndarray
    featured: np.ndarray


def _transitions(mechanics: WishMechanics) -> _Transitions:
    pities = np.arange(mechanics.hard_pity)
    rates = np.array([pull_rate(p, mechanics) for p in pities])
    survive = 1.0 - rates
    # A guaranteed 5-star is always featured, so only not-guaranteed states
    # can lose the 50/50.
    featured = np.array(
        [
            mechanics.featured_rate
            if radiance < 2
            else (6.0 / 11.0 if radiance == 2 else 1.0)
            for p in pities
            for radiance in range(4)
        ]
    ).reshape(mechanics.hard_pity, 4)
    return _Transitions(rates, survive, featured)


def _initial_state(
    starting_pity: int, guaranteed: bool, hard_pity: int
) -> _State:
    no_guarantee = np.zeros((hard_pity, 4))
    guaranteed_mass = np.zeros((hard_pity, 4))
    if guaranteed:
        guaranteed_mass[starting_pity, 0] = 1.0
    else:
        no_guarantee[starting_pity, 0] = 1.0
    return _State(no_guarantee, guaranteed_mass)


def _advance(state: _State, moves: _Transitions, hard_pity: int) -> _State:
    """Apply one wish while retaining Capturing Radiance state."""

    carried = state.no_guarantee * moves.survive[:, None]
    carried_g = state.guaranteed * moves.survive[:, None]
    lost = state.no_guarantee * moves.rates[:, None] * (1.0 - moves.featured)

    new_no_guarantee = np.zeros((hard_pity, 4))
    new_guaranteed = np.zeros((hard_pity, 4))
    new_no_guarantee[1:, :] = carried[:-1, :]
    new_guaranteed[1:, :] = carried_g[:-1, :]

    for radiance in range(4):
        next_radiance = min(3, radiance + 1)
        new_guaranteed[0, next_radiance] += lost[:, radiance].sum()

    return _State(new_no_guarantee, new_guaranteed)


def cumulative_probability(
    wishes: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
) -> np.ndarray:
    """P(at least one featured copy within N wishes), for each N (§10.2).

    Returns a float array of length `wishes + 1`:

        result[N] = probability of obtaining the featured copy within N
        additional wishes

    The result is *cumulative*: `result[N]` is not the probability of
    succeeding exactly on wish N. Index 0 is always 0.0 and the curve is
    monotonically non-decreasing.

    Args:
        wishes: how many additional wishes to look ahead (>= 0).
        starting_pity: 0-based pulls since the last 5-star
            (0 <= starting_pity < hard_pity).
        guaranteed: True when the next 5-star is guaranteed featured.
        mechanics: mechanics data for the banner type (§17).
    """
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )

    moves = _transitions(mechanics)
    state = _initial_state(starting_pity, guaranteed, mechanics.hard_pity)
    curve = np.empty(wishes + 1)
    curve[0] = 0.0

    for wish in range(1, wishes + 1):
        state = _advance(state, moves, mechanics.hard_pity)
        curve[wish] = 1.0 - state.total()

    return curve


def wishes_for_confidence(
    confidence: float,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
) -> int:
    """Smallest N with P(featured within N wishes) >= confidence (§10.3).

    Answers "how many wishes are required to reach at least `confidence`
    probability?" The horizon is derived from the mechanics (two hard-pity
    cycles bound the 50/50), so no wish count is hard-coded:

        curve = cumulative probability up to the mechanics horizon
        if max(curve) < confidence: raise ValueError
        else: return the first N with curve[N] >= confidence

    Confidence 1.0 therefore lands on the exact worst case the mechanics
    allow - the first wish count at which the curve is exactly 1 - because
    the search compares against the curve itself, with no tolerance.

    Raises:
        ValueError: if confidence is outside (0, 1], or if the mechanics
            cannot reach the requested confidence within the bounded
            horizon (max of the cumulative curve).
    """
    if not 0.0 < confidence <= 1.0:
        raise ValueError(
            f"confidence must be in (0, 1], got {confidence}"
        )
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )

    # Worst case: reach the hard-pity 5-star, lose the 50/50, then reach the
    # next guaranteed 5-star. No more than 2 * hard_pity wishes from pity 0.
    horizon = 2 * mechanics.hard_pity
    curve = cumulative_probability(
        horizon, starting_pity, guaranteed, mechanics
    )

    best = float(curve.max())
    if best < confidence:
        raise ValueError(
            f"confidence {confidence} is unattainable; the curve reaches at "
            f"most {best:.6f} within {horizon} wishes"
        )

    # Exact comparison: for valid mechanics the hard-pity rate of 1.0 ends
    # every path, so the curve is exactly 1.0 at the worst-case wish count.
    return int(np.searchsorted(curve, confidence, side="left"))
