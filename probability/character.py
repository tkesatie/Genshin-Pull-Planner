"""Single-copy character probability (Design Document §10.2, §10.3).

Exact dynamic programming over the `(pity, guarantee, radiance)` state
space - no sampling. `radiance` is the Capturing Radiance loss-streak
counter (0-3; see `domain.account`). State space is tiny
(hard_pity x 2 x 4), so exact probability is both faster and more accurate
than simulation, and gives the Phase 4 Monte Carlo simulator something
deterministic to validate against - including for accounts that currently
carry a nonzero Capturing Radiance counter, not just the radiance=0 case.

The engine answers only: "starting at this pity/guarantee/radiance state,
how likely am I to obtain ONE featured copy within N wishes?" (§10.4
multi-copy targets are Phase 4.) It knows nothing about characters, goals,
or roadmaps.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from domain.mechanics import WishMechanics
from probability.rates import capturing_radiance_rate, pull_rate


@dataclass(frozen=True)
class _State:
    """Survival probability mass over `(pity, guarantee, radiance)` states."""

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
        featured:      P(5-star is featured | 5-star occurs, not guaranteed)
                       at each (pity, radiance) - constant across pity, but
                       broadcast to that shape for uniform indexing
    """

    rates: np.ndarray
    survive: np.ndarray
    featured: np.ndarray


def _transitions(mechanics: WishMechanics) -> _Transitions:
    pities = np.arange(mechanics.hard_pity)
    rates = np.array([pull_rate(p, mechanics) for p in pities])
    survive = 1.0 - rates
    # A guaranteed 5-star is always featured, so only not-guaranteed states
    # can lose the 50/50. The featured rate depends only on the Capturing
    # Radiance counter, not on pity, so the same row is used at every pity
    # (see probability.rates.capturing_radiance_rate for the single-sourced
    # schedule shared with simulation.engine).
    featured_by_radiance = np.array(
        [capturing_radiance_rate(radiance, mechanics) for radiance in range(4)]
    )
    featured = np.tile(featured_by_radiance, (mechanics.hard_pity, 1))
    return _Transitions(rates, survive, featured)


def _initial_state(
    starting_pity: int, guaranteed: bool, starting_radiance: int, hard_pity: int
) -> _State:
    no_guarantee = np.zeros((hard_pity, 4))
    guaranteed_mass = np.zeros((hard_pity, 4))
    if guaranteed:
        guaranteed_mass[starting_pity, starting_radiance] = 1.0
    else:
        no_guarantee[starting_pity, starting_radiance] = 1.0
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


def _validate_radiance(starting_radiance: int) -> None:
    if not 0 <= starting_radiance <= 3:
        raise ValueError(
            f"starting_radiance must be in [0, 3], got {starting_radiance}"
        )


def _validate_character_inputs(
    wishes: int,
    starting_pity: int,
    mechanics: WishMechanics,
    starting_radiance: int,
) -> None:
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )
    _validate_radiance(starting_radiance)


def _cumulative_probability_uncached(
    wishes: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int,
) -> np.ndarray:
    """Compute a validated character curve without memoization."""
    moves = _transitions(mechanics)
    state = _initial_state(
        starting_pity, guaranteed, starting_radiance, mechanics.hard_pity
    )
    curve = np.empty(wishes + 1)
    curve[0] = 0.0

    for wish in range(1, wishes + 1):
        state = _advance(state, moves, mechanics.hard_pity)
        curve[wish] = 1.0 - state.total()

    return curve


@lru_cache(maxsize=128)
def _cached_cumulative_probability(
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int,
) -> tuple[float, ...]:
    """Memoize one complete curve; callers receive a sliced copy."""
    horizon = 2 * mechanics.hard_pity
    curve = _cumulative_probability_uncached(
        horizon, starting_pity, guaranteed, mechanics, starting_radiance
    )
    return tuple(float(value) for value in curve)


def cumulative_probability(
    wishes: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int = 0,
) -> np.ndarray:
    """P(at least one featured copy within N wishes), for each N (§10.2).

    Returns a fresh float array of length ``wishes + 1``. Curves are memoized
    internally at the deterministic full-cycle horizon and sliced for the
    requested prefix, so repeated planner cap probes do not repeat the DP.
    """
    _validate_character_inputs(wishes, starting_pity, mechanics, starting_radiance)
    horizon = 2 * mechanics.hard_pity
    curve = _cached_cumulative_probability(
        starting_pity, guaranteed, mechanics, starting_radiance
    )
    prefix = np.asarray(curve[: min(wishes + 1, horizon + 1)], dtype=float)
    if wishes <= horizon:
        return prefix
    return np.concatenate((prefix, np.ones(wishes - horizon, dtype=float)))



def _multi_copy_certainty_horizon(
    copies: int, mechanics: WishMechanics
) -> int:
    """Wishes needed for the exact curve to reach certainty."""
    return 2 * mechanics.hard_pity * copies


def _multi_copy_probability_uncached(
    horizon: int,
    copies: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int,
) -> np.ndarray:
    """Compute a validated multi-copy curve without memoization."""
    moves = _transitions(mechanics)
    state = np.zeros((copies, mechanics.hard_pity, 4, 2))
    state[0, starting_pity, starting_radiance, int(guaranteed)] = 1.0
    curve = np.empty(horizon + 1)
    curve[0] = 0.0

    for wish in range(1, horizon + 1):
        new_state = np.zeros_like(state)
        new_state[:, 1:, :, :] += (
            state[:, :-1, :, :] * moves.survive[None, :-1, None, None]
        )

        for copy_count in range(copies):
            for radiance in range(4):
                rate = moves.rates
                guaranteed_mass = state[copy_count, :, radiance, 1] * rate
                if copy_count + 1 < copies:
                    new_state[copy_count + 1, 0, radiance, 0] += guaranteed_mass.sum()

                nonguaranteed = state[copy_count, :, radiance, 0]
                featured_mass = nonguaranteed * rate * moves.featured[:, radiance]
                if copy_count + 1 < copies:
                    next_radiance = 0 if radiance <= 1 else 1
                    new_state[copy_count + 1, 0, next_radiance, 0] += featured_mass.sum()

                lost_mass = nonguaranteed * rate * (1.0 - moves.featured[:, radiance])
                next_radiance = min(3, radiance + 1)
                new_state[copy_count, 0, next_radiance, 1] += lost_mass.sum()

        state = new_state
        curve[wish] = 1.0 - state.sum()

    return curve


@lru_cache(maxsize=128)
def _cached_multi_copy_probability(
    copies: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int,
) -> tuple[float, ...]:
    """Memoize the exact certainty-horizon curve for one starting state."""
    horizon = 2 * mechanics.hard_pity * copies
    curve = _multi_copy_probability_uncached(
        horizon, copies, starting_pity, guaranteed, mechanics, starting_radiance
    )
    return tuple(float(value) for value in curve)


def multi_copy_cumulative_probability(
    wishes: int,
    copies: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int = 0,
) -> np.ndarray:
    """Exact probability of obtaining at least the requested featured copies.

    Generalizes `cumulative_probability` to a target of `copies` featured
    copies, returning P(at least `copies` within N wishes) for every N. The
    state space gains a completed-copies axis:

        state[c, p, r, g] = survival mass with c copies obtained, pity p,
        Capturing Radiance counter r and guarantee g

    The Radiance counter is carried across copies exactly as
    `domain.next_capturing_radiance_counter` describes - a featured win at
    Radiance 2 or 3 leaves the residual mark at 1 rather than resetting to 0
    - so copies are NOT independent draws and convolving the single-copy
    completion-time pmf only approximates this curve (see
    tests/test_reserve_accounting.py, whose reference walks the same pull
    tree independently).

    Raises:
        ValueError: if `wishes` or `copies` is negative, if
            `starting_pity` is outside [0, hard_pity), or if
            `starting_radiance` is outside [0, 3].
    """

    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if copies < 0:
        raise ValueError(f"copies must be non-negative, got {copies}")
    if copies == 0:
        return np.ones(wishes + 1)
    _validate_character_inputs(wishes, starting_pity, mechanics, starting_radiance)

    horizon = _multi_copy_certainty_horizon(copies, mechanics)
    curve = _cached_multi_copy_probability(
        copies, starting_pity, guaranteed, mechanics, starting_radiance
    )
    prefix = np.asarray(curve[: min(wishes + 1, horizon + 1)], dtype=float)
    if wishes <= horizon:
        return prefix
    return np.concatenate((prefix, np.ones(wishes - horizon, dtype=float)))


def multi_copy_wishes_for_confidence(
    confidence: float,
    copies: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int = 0,
) -> int:
    """Smallest wish count reaching the requested multi-copy confidence.

    The search horizon is `2 * hard_pity * copies` wishes: the worst case in
    which every one of the `copies` copies costs a lost 50/50 followed by a
    guaranteed 5-star at hard pity, so the curve is exactly 1.0 there and no
    wish count beyond it can be the first to clear `confidence`.
    """

    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1], got {confidence}")
    if copies < 0:
        raise ValueError(f"copies must be non-negative, got {copies}")
    if copies == 0:
        return 0
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )
    _validate_radiance(starting_radiance)

    horizon = 2 * mechanics.hard_pity * copies
    curve = multi_copy_cumulative_probability(
        horizon, copies, starting_pity, guaranteed, mechanics, starting_radiance
    )

    best = float(curve.max())
    if best < confidence:
        raise ValueError(
            f"confidence {confidence} is unattainable for {copies} copies; "
            f"the curve reaches at most {best:.6f} within {horizon} wishes"
        )

    return int(np.searchsorted(curve, confidence, side="left"))

def wishes_for_confidence(
    confidence: float,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int = 0,
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

    Args:
        starting_radiance: the Capturing Radiance loss-streak counter
            (0-3) to start from; see `cumulative_probability` for why this
            matters. Defaults to 0.

    Raises:
        ValueError: if confidence is outside (0, 1], if starting_radiance
            is outside [0, 3], or if the mechanics cannot reach the
            requested confidence within the bounded horizon (max of the
            cumulative curve).
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
    _validate_radiance(starting_radiance)

    # Worst case: reach the hard-pity 5-star, lose the 50/50, then reach the
    # next guaranteed 5-star. No more than 2 * hard_pity wishes from pity 0.
    horizon = 2 * mechanics.hard_pity
    curve = cumulative_probability(
        horizon, starting_pity, guaranteed, mechanics, starting_radiance
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
