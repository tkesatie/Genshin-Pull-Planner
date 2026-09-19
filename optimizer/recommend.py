"""Strategy optimization: selection and recommendation (§13 steps 6-7, §14, §15)."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Callable

from domain import Banner, Preference
from planner import PlannerContext
from planner.banners import available_banners
from planner.banners import current_banner
from simulation import DEFAULT_SEED, SpendPlan

from optimizer.evaluation import (
    DEFAULT_RUNS,
    CandidateStrategy,
    GoalStanding,
    evaluate_candidate,
    evaluate_skip_baseline,
)
from optimizer.outcomes import OutcomeOption, available_outcomes
from optimizer.protection import constraining_goals, priority_for_outcome
from optimizer.stops import StopConditions, for_discretionary, for_pursue, for_skip

MINIMUM_OUTCOME_PROBABILITY = 0.25

# Type of the optional cache lookup `recommend()` consults before running a
# fresh Monte Carlo simulation for a given (outcome, cap, banner). Returning
# None means "not cached" - the caller (usually the API layer's
# PlannerEvidenceCache, gated on a context fingerprint so a stale entry from
# before a settings/roadmap change is never handed back) falls through to a
# real simulation exactly as before this existed.
CandidateLookup = Callable[[OutcomeOption, int, Banner], "CandidateStrategy | None"]


@dataclass(frozen=True)
class RejectedOutcome:
    outcome: OutcomeOption
    best: CandidateStrategy
    shortfalls: tuple["GoalStanding", ...]


@dataclass(frozen=True)
class Recommendation:
    banner: Banner
    action: str
    outcome: OutcomeOption | None
    budget: int
    plan: SpendPlan | None
    outcome_probability: float
    protected: tuple[GoalStanding, ...]
    rejected: tuple[RejectedOutcome, ...]
    skip_reason: str | None
    stops: StopConditions
    runs: int
    seed: int | None
    discretionary_reason: str | None = None


def _skip(
    context: PlannerContext,
    reason: str,
    runs: int,
    seed: int | None,
    rejected: tuple[RejectedOutcome, ...] = (),
    skip_baseline_lookup: "Callable[[Banner], tuple[GoalStanding, ...] | None] | None" = None,
    skip_baseline_sink: "Callable[[Banner, tuple[GoalStanding, ...]], None] | None" = None,
) -> Recommendation:
    banners = available_banners(context)
    banner = banners[0] if banners else current_banner(context)

    protected = None
    if skip_baseline_lookup is not None:
        protected = skip_baseline_lookup(banner)
    if protected is None:
        protected = evaluate_skip_baseline(context, runs=runs, seed=seed)
        if skip_baseline_sink is not None:
            skip_baseline_sink(banner, protected)

    return Recommendation(
        banner=banner,
        action="skip",
        outcome=None,
        budget=0,
        plan=None,
        outcome_probability=0.0,
        protected=protected,
        rejected=rejected,
        skip_reason=reason,
        stops=for_skip(reason),
        runs=runs,
        seed=seed,
        discretionary_reason=None,
    )


def _discretionary_reason(
    outcome_label: str, outcome_probability: float, minimum_outcome_probability: float
) -> str:
    return (
        f"{outcome_label} is available but not recommended: the largest "
        "spend that still protects every higher-priority goal gives only a "
        f"{outcome_probability:.0%} chance of achieving it, below the "
        f"{minimum_outcome_probability:.0%} minimum for an ordinary "
        "recommendation. You may still spend up to the reported budget if "
        "you want to take the gamble, but success is unlikely."
    )


def _diagnostic_key(candidate: CandidateStrategy) -> tuple[float, float, int]:
    floor = (
        candidate.min_protected_probability
        if candidate.min_protected_probability is not None
        else 1.0
    )
    return (floor, candidate.outcome_probability, candidate.budget)


def _caps(context: PlannerContext, budgets: Iterable[int] | None) -> list[int]:
    wishes = context.account.wishes
    if budgets is None:
        return list(range(wishes, -1, -1))
    caps = sorted(set(int(budget) for budget in budgets), reverse=True)
    if not caps:
        raise ValueError("budgets must contain at least one cap")
    for cap in caps:
        if not 0 <= cap <= wishes:
            raise ValueError(
                f"every cap must satisfy 0 <= cap <= account wishes "
                f"({wishes}), got {cap}"
            )
    return caps


def _evaluate_cap(
    context: PlannerContext,
    outcome: OutcomeOption,
    cap: int,
    banner: Banner,
    runs: int,
    seed: int | None,
    simulation_sink,
    candidate_lookup: CandidateLookup | None,
) -> CandidateStrategy:
    """One (outcome, cap) evaluation, consulting the cache first.

    A cache hit skips the Monte Carlo simulation entirely - this is the
    fast path for "re-run the planner with nothing changed" and for the
    second half of a coarse-then-fine frontend scan re-visiting caps the
    first pass already computed. `candidate_lookup` is expected to already
    have applied its own freshness check (a context fingerprint) before
    returning anything, so a stale entry from a different confidence,
    mechanics, income, or roadmap is never handed back here - a miss looks
    identical to "never cached" and falls through to a real simulation,
    which is then offered to `simulation_sink` exactly as before.
    """
    if candidate_lookup is not None:
        cached = candidate_lookup(outcome, cap, banner)
        if cached is not None:
            return cached
    candidate = evaluate_candidate(
        context, outcome, cap, banner=banner, runs=runs, seed=seed,
        simulation_sink=simulation_sink,
    )
    return candidate


def recommend(
    context: PlannerContext,
    preferences: Iterable[Preference] = (),
    *,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
    budgets: Iterable[int] | None = None,
    minimum_outcome_probability: float = MINIMUM_OUTCOME_PROBABILITY,
    simulation_sink=None,
    candidate_lookup: CandidateLookup | None = None,
    skip_baseline_lookup: "Callable[[Banner], tuple[GoalStanding, ...] | None] | None" = None,
    skip_baseline_sink: "Callable[[Banner, tuple[GoalStanding, ...]], None] | None" = None,
) -> Recommendation:
    """The highest-preference feasible outcome and its largest feasible cap.

    `candidate_lookup`, when given, is consulted before every simulation
    (see `_evaluate_cap`) so identical work already done - in this call or
    a previous one, via a cache the caller wires up - is never repeated.
    It changes nothing about which outcome or cap wins: a cache hit and a
    fresh simulation of the same (context, outcome, cap, runs, seed) are
    required to be interchangeable (§11 invariant 10's determinism), so
    the caller is responsible for only ever returning a hit that is
    actually equivalent - typically by gating it on a fingerprint of the
    parts of `context` that affect simulated probabilities.
    """
    if not 0.0 <= minimum_outcome_probability <= 1.0:
        raise ValueError(
            "minimum_outcome_probability must be in [0, 1], got "
            f"{minimum_outcome_probability}"
        )
    banners = available_banners(context)
    if not banners:
        return _skip(
            context,
            "no roadmap banner at the current version/phase",
            runs,
            seed,
        )

    all_caps = _caps(context, budgets)
    opportunities = []
    rejected: list[RejectedOutcome] = []

    for banner in banners:
        outcomes = available_outcomes(context, preferences, banner=banner)
        for outcome_index, outcome in enumerate(outcomes):
            outcome_priority = priority_for_outcome(context, outcome)
            caps = (
                all_caps
                if constraining_goals(context, priority=outcome_priority, banner=banner)
                else all_caps[:1]
            )
            diagnostic_best: CandidateStrategy | None = None
            first_feasible: CandidateStrategy | None = None
            for cap in caps:
                candidate = _evaluate_cap(
                    context, outcome, cap, banner, runs, seed, simulation_sink,
                    candidate_lookup,
                )
                if (
                    diagnostic_best is None
                    or _diagnostic_key(candidate) > _diagnostic_key(diagnostic_best)
                ):
                    diagnostic_best = candidate
                if candidate.feasible:
                    first_feasible = candidate
                    break

            if first_feasible is None:
                shortfalls = tuple(
                    standing
                    for standing in diagnostic_best.protected
                    if standing.constraining and not standing.meets_threshold
                )
                rejected.append(
                    RejectedOutcome(
                        outcome=outcome, best=diagnostic_best, shortfalls=shortfalls
                    )
                )
                continue

            opportunities.append(
                (banner, outcome_index, outcome, first_feasible, outcome_priority)
            )

    if not opportunities:
        if not any(available_outcomes(context, preferences, banner=banner) for banner in banners):
            return _skip(
                context, "no active goal or preferred outcome to pursue", runs, seed,
                skip_baseline_lookup=skip_baseline_lookup, skip_baseline_sink=skip_baseline_sink,
            )
        return _skip(
            context,
            f"no preferred outcome can be pursued while keeping every "
            f"higher-priority protected goal at or above the "
            f"{context.confidence:.0%} confidence threshold",
            runs,
            seed,
            rejected=tuple(rejected),
            skip_baseline_lookup=skip_baseline_lookup,
            skip_baseline_sink=skip_baseline_sink,
        )

    def opportunity_key(item):
        banner, outcome_index, outcome, candidate, priority = item
        return (
            priority is None,
            priority if priority is not None else 10**9,
            banner.order_key,
            outcome_index,
        )

    opportunities.sort(key=opportunity_key)

    winner_banner, _, winner_outcome, winner, winner_priority = opportunities[0]

    for banner, _, outcome, candidate, priority in opportunities[1:]:
        if (
            banner == winner_banner
            and outcome.character == winner_outcome.character
            and outcome.constellation > winner_outcome.constellation
            and candidate.outcome_probability >= minimum_outcome_probability
        ):
            winner_banner, winner_outcome, winner = banner, outcome, candidate

    if winner.outcome_probability >= minimum_outcome_probability:
        return Recommendation(
            banner=winner_banner,
            action="pursue",
            outcome=winner_outcome,
            budget=winner.budget,
            plan=winner.plan,
            outcome_probability=winner.outcome_probability,
            protected=winner.protected,
            rejected=tuple(rejected),
            skip_reason=None,
            stops=for_pursue(winner_outcome.label, winner.budget),
            runs=runs,
            seed=seed,
            discretionary_reason=None,
        )
    reason = _discretionary_reason(
        winner_outcome.label,
        winner.outcome_probability,
        minimum_outcome_probability,
    )
    return Recommendation(
        banner=winner_banner,
        action="discretionary",
        outcome=winner_outcome,
        budget=winner.budget,
        plan=winner.plan,
        outcome_probability=winner.outcome_probability,
        protected=winner.protected,
        rejected=tuple(rejected),
        skip_reason=None,
        stops=for_discretionary(
            winner_outcome.label,
            winner.budget,
            winner.outcome_probability,
            minimum_outcome_probability,
        ),
        runs=runs,
        seed=seed,
        discretionary_reason=reason,
    )
