from dataclasses import dataclass

import numpy as np

from domain.mechanics import WishMechanics
from probability.rates import featured_rate_at, pull_rate


@dataclass(frozen=True)
class _State:
    no_guarantee: np.ndarray
    guaranteed: np.ndarray

    def total(self) -> float:
        return float(self.no_guarantee.sum() + self.guaranteed.sum())


@dataclass(frozen=True)
class _Transitions:
    rates: np.ndarray
    survive: np.ndarray
    featured: np.ndarray


def _transitions(mechanics: WishMechanics) -> _Transitions:
    pities = np.arange(mechanics.hard_pity)
    rates = np.array([pull_rate(p, mechanics) for p in pities])
    survive = 1.0 - rates
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
    starting_radiance: int = 0,
) -> np.ndarray:
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )

    if not 0 <= starting_radiance <= 3:
        raise ValueError(f"starting_radiance must satisfy 0 <= starting_radiance <= 3, got {starting_radiance}")

    moves = _transitions(mechanics)
    state = _initial_state(starting_pity, guaranteed, starting_radiance, mechanics.hard_pity)
    curve = np.empty(wishes + 1)
    curve[0] = 0.0

    for wish in range(1, wishes + 1):
        state = _advance(state, moves, mechanics.hard_pity)
        curve[wish] = 1.0 - state.total()

    return curve


def multi_copy_cumulative_probability(
    wishes: int, copies: int, starting_pity: int, guaranteed: bool,
    mechanics: WishMechanics, starting_radiance: int = 0,
) -> np.ndarray:
    """Exact multi-copy probability using the full pull state."""
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if copies < 0:
        raise ValueError(f"copies must be non-negative, got {copies}")
    if copies == 0:
        return np.ones(wishes + 1)
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(f"starting_pity must satisfy 0 <= starting_pity < {mechanics.hard_pity}, got {starting_pity}")
    if not 0 <= starting_radiance <= 3:
        raise ValueError(f"starting_radiance must satisfy 0 <= starting_radiance <= 3, got {starting_radiance}")

    moves = _transitions(mechanics)
    # state[c, p, r, g]: mass with c copies, pity p, Radiance r, guarantee g.
    state = np.zeros((copies, mechanics.hard_pity, 4, 2))
    state[0, starting_pity, starting_radiance, int(guaranteed)] = 1.0
    curve = np.empty(wishes + 1)
    curve[0] = 0.0

    for wish in range(1, wishes + 1):
        new_state = np.zeros_like(state)

        # Only non-guaranteed states can survive a wish without a 5-star.
        new_state[:, 1:, :, 0] += (
            state[:, :-1, :, 0] * moves.survive[None, :-1, None]
        )

        for copy_count in range(copies):
            for radiance in range(4):
                # A guarantee means the next 5-star is the featured character;
                # the pity rate itself does not apply to the guaranteed state.
                guaranteed_mass = state[copy_count, :, radiance, 1]
                if copy_count + 1 < copies:
                    new_state[copy_count + 1, 0, radiance, 0] += guaranteed_mass.sum()

                nonguaranteed = state[copy_count, :, radiance, 0]
                rate = moves.rates
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
def wishes_for_confidence(
    confidence: float,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int = 0,
) -> int:
    if not 0.0 < confidence <= 1.0:
        raise ValueError(
            f"confidence must be in (0, 1], got {confidence}"
        )
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )

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

    return int(np.searchsorted(curve, confidence, side="left"))


def multi_copy_wishes_for_confidence(
    confidence: float,
    copies: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
    starting_radiance: int = 0,
) -> int:
    """Smallest N with P(`copies` copies within N wishes) >= confidence.

    The multi-copy counterpart of `wishes_for_confidence` (§10.3, §10.4):
    exact, not sampled, via `multi_copy_cumulative_probability`.

    Args:
        copies: how many featured copies are needed. 0 wishes always
            satisfies `copies=0` (the goal is already complete).

    Raises:
        ValueError: if confidence is outside (0, 1], or if the mechanics
            cannot reach the requested confidence within the search
            horizon (scaled by `copies`, since each additional copy can
            add up to two more hard-pity cycles in the worst case).
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
