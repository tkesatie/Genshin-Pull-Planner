"""Exact weapon-banner probability (Design Document §10, §17; Phase 4).

Exact dynamic programming over the ``(copies, pity, guarantee, fate_points)``
state space - no sampling. It mirrors ``probability.character`` (§10.2/§10.4)
but implements the weapon-banner's own mechanics rather than reusing the
character 50/50 model:

Weapon Event Wish (the published, official rules):

    * 5-star base rate 0.7% (1.85% consolidated), hard pity at 80 wishes
      (a 5-star is guaranteed by pull 80).
    * Soft pity: not published by HoYoverse. Community measurement puts the
      ramp near pull 63-65 with ~6-7% increments; this engine consumes
      whatever the mechanics data says (§17: mechanics are data), so the
      WEAPON_EVENT_BANNER constants can be refined without touching logic.
    * Each 5-star has a 75% chance to be one of the two rate-up weapons.
    * Losing the 75/25 (a non-rate-up 5-star) arms the guarantee: the NEXT
      5-star is one of the rate-up weapons. Pity resets on every 5-star.
    * Epitomized Path / Fate Points (§10 weapon): the player designates one
      of the two rate-up weapons. Every 5-star that is NOT the designated
      weapon adds a Fate Point; since Version 5.0 the designated weapon is
      guaranteed at 1 Fate Point (reduced from 2 pre-5.0). Reaching the
      designated weapon (by rate-up, by guarantee, or by Fate Points)
      resets the Fate Point counter to 0. Worst case is therefore 160
      wishes for one designated copy.

Rate-up semantics: ``featured_rate`` (0.75) is the chance the 5-star is one
of the two rate-up weapons. The designated weapon is then one specific
weapon, and this engine assumes the two rate-up weapons are symmetric
(50/50 within the rate-up pair), giving ``0.75 / 2 = 0.375`` for the
designated weapon before Fate-Point guarantees. This symmetry assumption is
stated explicitly because HoYoverse publishes only the aggregated 75%.

The engine answers: "starting at this pity/guarantee/fate-point state, how
likely am I to obtain ``copies`` designated copies within N wishes?" It
knows nothing about weapons, goals, or roadmaps, exactly like the character
engine.

Note the Fate Point asymmetry against the character 50/50: losing the
rate-up roll (a standard 5-star) adds a Fate Point AND arms the guarantee,
and Fate Points persist across 5-stars until the designated weapon drops.
At fate_points == fate_points_required every 5-star is the designated
weapon regardless of the rate-up roll, so the two guarantee mechanisms
converge there.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from domain.mechanics import WishMechanics
from probability.rates import _validate_pity, pull_rate

DEFAULT_FATE_POINTS_REQUIRED = 1


@dataclass(frozen=True)
class _Transitions:
    """Per-pity vectors used to advance the distribution one wish."""

    rates: np.ndarray
    survive: np.ndarray


def _transitions(mechanics: WishMechanics) -> _Transitions:
    pities = np.arange(mechanics.hard_pity)
    rates = np.array([pull_rate(p, mechanics) for p in pities])
    return _Transitions(rates, 1.0 - rates)


def designated_rate(mechanics: WishMechanics) -> float:
    """P(a 5-star is the designated weapon | 5-star, no guarantee, FP 0).

    ``featured_rate`` covers both rate-up weapons; the designated weapon is
    one of them, so the pre-guarantee designated share is half the rate-up
    share under the rate-up-symmetry assumption (see module docstring).
    """
    return mechanics.featured_rate / 2.0


def _validate_fate_points(fate_points: int, required: int) -> None:
    if fate_points < 0:
        raise ValueError(f"fate_points must be non-negative, got {fate_points}")
    if fate_points > required:
        raise ValueError(
            f"fate_points must be <= the required amount ({required}), "
            f"got {fate_points}"
        )


def _validate_common(
    wishes: int,
    starting_pity: int,
    starting_fate_points: int,
    mechanics: WishMechanics,
    fate_points_required: int,
) -> None:
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    _validate_pity(starting_pity, mechanics)
    _validate_fate_points(starting_fate_points, fate_points_required)


def _advance(
    state: np.ndarray,
    moves: _Transitions,
    copies: int,
    mechanics: WishMechanics,
    fate_points_required: int,
) -> np.ndarray:
    """Apply one wish to the (copies, hard_pity, 2, required+1) state.

    Axes: completed designated copies, pity, guarantee flag, fate points.
    Only the first ``copies`` planes are tracked; mass that completes the
    final copy leaves the survival state (completion = 1 - total).
    """
    new_state = np.zeros_like(state)
    d_rate = designated_rate(mechanics)
    max_fp = fate_points_required

    # No 5-star: pity advances, guarantee/fate points unchanged.
    new_state[:, 1:, :, :] += state[:, :-1, :, :] * moves.survive[None, :-1, None, None]

    for copy in range(copies):
        for guarantee in (0, 1):
            for fp in range(max_fp + 1):
                mass = state[copy, :, guarantee, fp]
                # Only the 5-star fraction of the mass takes an outcome
                # transition; the rest advanced one pity above.
                five_star_mass = float((mass * moves.rates).sum())
                if five_star_mass == 0.0:
                    continue
                # P(the 5-star is the designated weapon | a 5-star occurs).
                # Fate Point cap: certain. Rate-up guarantee: the 5-star is
                # one of the two rate-up weapons, so half of them (symmetry
                # assumption) is the designated one. Otherwise: the full
                # 75/25 roll, half of the rate-up share.
                if fp >= max_fp:
                    hit = 1.0
                elif guarantee == 1:
                    hit = 0.5
                else:
                    hit = d_rate
                hit_mass = five_star_mass * hit
                if copy + 1 < copies:
                    # Designated copy obtained: pity 0, guarantee cleared,
                    # fate points reset. Guarantee/fate points do NOT carry
                    # a residual mark (unlike Capturing Radiance).
                    new_state[copy + 1, 0, 0, 0] += hit_mass
                # else: mass leaves the survival state.

                miss_mass = five_star_mass * (1.0 - hit)
                if fp < max_fp:
                    # A non-designated 5-star adds a Fate Point (Epitomized
                    # Path accumulates across 5-stars until the designated
                    # weapon drops). The rate-up guarantee is consumed by
                    # obtaining a rate-up weapon (miss from a guaranteed
                    # state) and armed by losing the 75/25 (miss from a
                    # not-guaranteed state, i.e. a standard 5-star).
                    next_guarantee = 0 if guarantee == 1 else 1
                    new_state[copy, 0, next_guarantee, fp + 1] += miss_mass
                else:
                    # fp == max_fp: the Fate Point guarantee forces a hit, so
                    # miss_mass is exactly 0 here. The branch documents the
                    # invariant rather than silently dropping mass.
                    assert miss_mass == 0.0

    return new_state


def _weapon_certainty_horizon(
    copies: int,
    mechanics: WishMechanics,
    fate_points_required: int,
) -> int:
    """Wishes needed for the exact weapon curve to reach certainty.

    Before a designated weapon can drop, at most ``fate_points_required``
    non-designated 5-stars can occur.  Each 5-star is bounded by one hard-pity
    cycle, followed by the designated 5-star, so the safe per-copy bound is
    ``(fate_points_required + 1) * hard_pity``.  The default one-Fate-Point
    rule therefore retains the familiar ``2 * hard_pity`` horizon, while the
    optional pre-5.0 two-point rule remains correct as well.
    """
    return copies * (fate_points_required + 1) * mechanics.hard_pity


def _weapon_probability_uncached(
    horizon: int,
    starting_pity: int,
    guarantee: bool,
    starting_fate_points: int,
    mechanics: WishMechanics,
    copies: int,
    fate_points_required: int,
) -> np.ndarray:
    """Compute a validated weapon curve without memoization."""
    moves = _transitions(mechanics)
    state = np.zeros(
        (copies, mechanics.hard_pity, 2, fate_points_required + 1)
    )
    state[0, starting_pity, int(guarantee), starting_fate_points] = 1.0
    curve = np.empty(horizon + 1)
    curve[0] = 0.0

    for wish in range(1, horizon + 1):
        state = _advance(state, moves, copies, mechanics, fate_points_required)
        curve[wish] = 1.0 - state.sum()

    return curve


@lru_cache(maxsize=128)
def _cached_weapon_probability(
    starting_pity: int,
    guarantee: bool,
    starting_fate_points: int,
    mechanics: WishMechanics,
    copies: int,
    fate_points_required: int,
) -> tuple[float, ...]:
    """Memoize the exact certainty-horizon weapon curve for one state."""
    horizon = _weapon_certainty_horizon(
        copies, mechanics, fate_points_required
    )
    curve = _weapon_probability_uncached(
        horizon,
        starting_pity,
        guarantee,
        starting_fate_points,
        mechanics,
        copies,
        fate_points_required,
    )
    return tuple(float(value) for value in curve)


def weapon_cumulative_probability(
    wishes: int,
    starting_pity: int,
    guarantee: bool,
    starting_fate_points: int,
    mechanics: WishMechanics,
    *,
    copies: int = 1,
    fate_points_required: int = DEFAULT_FATE_POINTS_REQUIRED,
) -> np.ndarray:
    """P(at least ``copies`` designated copies within N wishes) for each N.

    Returns a float array of length ``wishes + 1`` with the same cumulative
    contract as ``probability.character.cumulative_probability`` (§10.2):
    index 0 is 0.0 and the curve is monotonically non-decreasing.

    Args:
        wishes: how many additional wishes to look ahead (>= 0).
        starting_pity: 0-based pulls since the last 5-star
            (0 <= starting_pity < hard_pity).
        guarantee: True when the next 5-star is a rate-up weapon.
        starting_fate_points: current Epitomized Path Fate Points toward
            the designated weapon (0..fate_points_required).
        mechanics: mechanics data for the weapon banner (§17).
        copies: designated copies to obtain (>= 1).
        fate_points_required: Fate Points that guarantee the designated
            weapon (officially 1 since Version 5.0; 2 pre-5.0).

    Raises:
        ValueError: if any argument is out of range.
    """
    if copies < 1:
        raise ValueError(f"copies must be >= 1, got {copies}")
    if fate_points_required < 1:
        raise ValueError(
            f"fate_points_required must be >= 1, got {fate_points_required}"
        )
    _validate_common(
        wishes, starting_pity, starting_fate_points, mechanics, fate_points_required
    )

    horizon = _weapon_certainty_horizon(
        copies, mechanics, fate_points_required
    )
    curve = _cached_weapon_probability(
        starting_pity,
        guarantee,
        starting_fate_points,
        mechanics,
        copies,
        fate_points_required,
    )
    prefix = np.asarray(curve[: min(wishes + 1, horizon + 1)], dtype=float)
    if wishes <= horizon:
        return prefix
    return np.concatenate((prefix, np.ones(wishes - horizon, dtype=float)))


def weapon_wishes_for_confidence(
    confidence: float,
    starting_pity: int,
    guarantee: bool,
    starting_fate_points: int,
    mechanics: WishMechanics,
    *,
    copies: int = 1,
    fate_points_required: int = DEFAULT_FATE_POINTS_REQUIRED,
) -> int:
    """Smallest N with P(copies within N wishes) >= confidence.

    The search horizon is ``copies * (fate_points_required + 1) * hard_pity``
    wishes: the worst case for each copy is a sequence of non-designated
    5-stars filling the Fate Point requirement, followed by the designated
    5-star.  With the current one-Fate-Point rule this is
    ``copies * 2 * hard_pity``.

    Raises:
        ValueError: if confidence is outside (0, 1], any state argument is
            invalid, or the confidence is unattainable within the horizon.
    """
    if not 0.0 < confidence <= 1.0:
        raise ValueError(f"confidence must be in (0, 1], got {confidence}")
    horizon = _weapon_certainty_horizon(
        copies, mechanics, fate_points_required
    )
    curve = weapon_cumulative_probability(
        horizon,
        starting_pity,
        guarantee,
        starting_fate_points,
        mechanics,
        copies=copies,
        fate_points_required=fate_points_required,
    )
    best = float(curve.max())
    if best < confidence:
        raise ValueError(
            f"confidence {confidence} is unattainable for {copies} copies; "
            f"the curve reaches at most {best:.6f} within {horizon} wishes"
        )
    return int(np.searchsorted(curve, confidence, side="left"))


def refinement_cumulative_probability(
    wishes: int,
    owned_refinement: int,
    target_refinement: int,
    starting_pity: int,
    guarantee: bool,
    starting_fate_points: int,
    mechanics: WishMechanics,
    *,
    fate_points_required: int = DEFAULT_FATE_POINTS_REQUIRED,
) -> np.ndarray:
    """P(reaching ``target_refinement`` within N wishes) for each N.

    Refinements map to designated copies: from NOT_OWNED (-1) to R1 needs 1
    copy, R1 to R2 needs 1 more, and ``owned_refinement`` >=
    ``target_refinement`` means the target is already met (the curve is
    all ones). This is a thin, domain-semantic wrapper over
    ``weapon_cumulative_probability``; it validates refinement inputs and
    derives the copy count, but contains no probability mathematics of its
    own.
    """
    if owned_refinement < -1:
        raise ValueError(
            f"owned_refinement must be >= -1 (NOT_OWNED), got {owned_refinement}"
        )
    if target_refinement < 0:
        raise ValueError(
            f"target_refinement must be >= 0 (R0 is unowned-but-targeted), "
            f"got {target_refinement}"
        )
    copies = max(target_refinement - owned_refinement, 0)
    if copies == 0:
        return np.ones(wishes + 1)
    return weapon_cumulative_probability(
        wishes,
        starting_pity,
        guarantee,
        starting_fate_points,
        mechanics,
        copies=copies,
        fate_points_required=fate_points_required,
    )
