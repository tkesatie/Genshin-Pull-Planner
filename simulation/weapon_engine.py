"""Weapon-banner Monte Carlo simulation (Design Document §11, §17; Phase 4).

A seeded, vectorized cross-check for ``probability.weapon`` - the weapon
counterpart of the Phase-2 invariant that the Monte Carlo simulator must
reproduce the exact analytic curves within sampling error (probability
invariant 10, mirrored here for the weapon banner).

This is deliberately a small, standalone sampler rather than an extension
of ``simulation.engine``: the character simulator is driven by roadmaps,
plans, income and ownership (§11-§12), none of which exist for weapons
until the strategy phase. Per-pull 5-star rates still come from the shared
``probability.pull_rate`` / ``pull_rate_array`` (§17: no rate mathematics
here), so the two simulators cannot drift on pity mathematics; only the
5-star outcome rule (75/25 + Epitomized Path) is weapon-specific and it
mirrors ``probability.weapon._advance`` exactly.
"""

import numpy as np

from domain.mechanics import WishMechanics
from probability.rates import pull_rate_array


def simulate_weapon_runs(
    wishes: int,
    starting_pity: int,
    guarantee: bool,
    starting_fate_points: int,
    mechanics: WishMechanics,
    runs: int,
    seed: int | None = 0,
    *,
    fate_points_required: int = 1,
) -> np.ndarray:
    """Simulate ``runs`` independent wish sequences; return a copy count.

    Returns an int array of shape ``(runs,)`` holding the number of
    designated-weapon copies obtained within ``wishes`` wishes per run.
    Deterministic for identical arguments (seeded Generator).

    Raises:
        ValueError: if runs < 1 or any starting state is invalid.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if wishes < 0:
        raise ValueError(f"wishes must be non-negative, got {wishes}")
    if not 0 <= starting_pity < mechanics.hard_pity:
        raise ValueError(
            f"starting_pity must satisfy 0 <= starting_pity < "
            f"{mechanics.hard_pity}, got {starting_pity}"
        )
    if starting_fate_points < 0 or starting_fate_points > fate_points_required:
        raise ValueError(
            f"starting_fate_points must be in [0, {fate_points_required}], "
            f"got {starting_fate_points}"
        )

    rng = np.random.default_rng(seed)
    pity = np.full(runs, starting_pity, dtype=np.int64)
    guarantee = np.full(runs, int(guarantee), dtype=bool)
    fate_points = np.full(runs, starting_fate_points, dtype=np.int64)
    copies = np.zeros(runs, dtype=np.int64)

    for _ in range(wishes):
        rates = pull_rate_array(pity, mechanics)
        hit_5star = rng.random(runs) < rates

        if np.any(hit_5star):
            # P(designated | 5-star): 1.0 on guarantee or at the Fate Point
            # cap, otherwise half the rate-up share (symmetric rate-up pair,
            # see probability.weapon.designated_rate).
            designated_rate = np.where(
                guarantee | (fate_points >= fate_points_required),
                1.0,
                mechanics.featured_rate / 2.0,
            )
            designated = hit_5star & (rng.random(runs) < designated_rate)

            copies += designated
            pity = np.where(hit_5star, 0, pity + 1)
            guarantee = np.where(hit_5star, ~designated, guarantee)
            fate_points = np.where(
                hit_5star,
                np.where(designated, 0, np.minimum(fate_points + 1, fate_points_required)),
                fate_points,
            )
        else:
            pity = pity + 1

    return copies
