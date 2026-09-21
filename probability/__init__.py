"""Analytical probability engine (Design Document §10, Phase 2).

Isolated probability questions only: given a starting pity/guarantee/
Capturing Radiance state and mechanics data, how likely is one featured
copy within N wishes?

The engine is deliberately independent from roadmap strategy logic (§10):
it knows nothing about goals, priorities, preferences, banners, or income.
It consumes `domain.mechanics.WishMechanics` as pure data and produces
deterministic numbers.

Phase 2 invariants:

1. `pull_rate()` describes the probability of a 5-star on the next wish.
2. `featured_rate_at()` describes the conditional probability that the next
   5-star is the featured character, given the current guarantee state.
   `capturing_radiance_rate()` is the Capturing-Radiance-aware counterpart,
   keyed on the loss-streak counter instead of (or in addition to) the
   guarantee flag.
3. `cumulative_probability()` describes the probability of obtaining at least
   one featured copy within N wishes (cumulative, not "exactly on wish N").
4. Index 0 of the cumulative curve is always 0.
5. The cumulative curve is monotonically non-decreasing.
6. The probability engine never mutates Account, Roadmap, or other domain
   objects; it takes plain numbers and mechanics data as inputs.
7. The engine operates in target copies, not constellation levels (§10.4);
   multi-copy targets are Phase 4.
8. No roadmap, priority, preference, banner schedule, or income logic
   belongs in this package.
9. Analytical results are deterministic for identical inputs.
10. The Phase 4 Monte Carlo simulator must reproduce these results for
    equivalent single-copy scenarios (including equal starting Capturing
    Radiance state) within sampling error.
"""

from probability.character import (
    cumulative_probability,
    multi_copy_cumulative_probability,
    multi_copy_wishes_for_confidence,
    wishes_for_confidence,
)
from probability.rates import (
    capturing_radiance_rate,
    featured_rate_at,
    pull_rate,
    pull_rate_array,
)

__all__ = [
    "capturing_radiance_rate",
    "cumulative_probability",
    "featured_rate_at",
    "multi_copy_cumulative_probability",
    "multi_copy_wishes_for_confidence",
    "pull_rate",
    "pull_rate_array",
    "wishes_for_confidence",
]
