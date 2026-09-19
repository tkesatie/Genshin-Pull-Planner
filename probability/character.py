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


def multi_copy_cumulative_probability(
    wishes: int,
    copies: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
) -> np.ndarray:
    """P(obtaining `copies` featured copies within N wishes), for each N.

    Exact - not sampled - by convolving the single-copy completion-time
    PMF `copies` times: every copy after the first is drawn starting fresh
    at pity 0 with no guarantee, mirroring the simulator's post-copy reset
    (§11: a featured copy resets pity to 0 and turns the guarantee off).

    This is the reserve math a multi-copy protected goal (a constellation
    reached from an unowned or partially-owned character) actually needs.
    Reusing the single-copy curve as-is for such a goal silently answers
    "how likely is exactly one copy" for a goal that needs several - which
    is what `planner.protection.protected_goal_outcomes` did before this
    function existed, and why it could report a multi-copy goal as fully
    protected when its true probability was far lower (see that module's
    docstring and its regression test).

    Args:
        wishes: how many additional wishes to look ahead (>= 0).
        copies: how many featured copies are needed (>= 0). 0 means the
            goal is already satisfied: the curve is 1.0 everywhere,
            including at 0 wishes.
        starting_pity / guaranteed: the state for the FIRST copy only;
            every subsequent copy starts fresh (see above).
        mechanics: mechanics data for the banner type (§17).
    """
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if copies < 0:
        raise ValueError(f"copies must be non-negative, got {copies}")

    if copies == 0:
        return np.ones(wishes + 1)

    first_cdf = cumulative_probability(wishes, starting_pity, guaranteed, mechanics)
    if copies == 1:
        return first_cdf

    first_pmf = np.diff(first_cdf, prepend=0.0)
    fresh_cdf = cumulative_probability(wishes, 0, False, mechanics)
    fresh_pmf = np.diff(fresh_cdf, prepend=0.0)

    total_pmf = first_pmf
    for _ in range(copies - 1):
        total_pmf = np.convolve(total_pmf, fresh_pmf)[: wishes + 1]
    return np.cumsum(total_pmf)[: wishes + 1]


def wishes_for_confidence(
    confidence: float,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
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
        horizon, starting_pity, guaranteed, mechanics
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
        horizon, copies, starting_pity, guaranteed, mechanics
    )

    best = float(curve.max())
    if best < confidence:
        raise ValueError(
            f"confidence {confidence} is unattainable for {copies} copies; "
            f"the curve reaches at most {best:.6f} within {horizon} wishes"
        )

    return int(np.searchsorted(curve, confidence, side="left"))
