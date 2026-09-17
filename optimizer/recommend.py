"""Strategy optimization: selection and recommendation (§13 steps 6-7, §14, §15).

The decision is lexicographic - never a global "best" score (§2, §13):

    1. Which outcome do I prefer?           outcome order (§15, see below)
    2. Can I pursue it while protecting
       the roadmap?                          feasibility (§13 step 5)
    3. How much may I spend on it?          largest feasible cap (§14)

`optimizer.outcomes.available_outcomes` already restricts every outcome
it returns to the current banner's single character, and resulting
constellations for one character are strictly nested (reaching C2
necessarily reaches C0). That tuple is therefore ordered by descending
constellation - the furthest target first - not by the user's stated
preference rank: a same-character chain is a progression of how far to
go, never a set of mutually exclusive alternatives, so the outcome that
subsumes the most of the chain is tried first. This function simply
walks that order front-to-back; a nearer target never displaces a
further one this function has already found feasible. Within the
winning outcome, the candidate caps are scanned from the account's
wishes down to 0 and the first feasible cap is recommended: the largest
feasible cap (§14's safe spending, evaluated at roadmap level through
simulation).

The scan is exhaustive over the given caps and never binary-searches:
feasibility is not monotone in the cap. Spending one more wish can lose
the 50/50 and carry a guarantee into the next banner - a state that can
protect a future goal more than the wish cost it (§14: pity/guarantee
carry must count). A scan's first feasible cap is the largest feasible
cap among the scanned caps regardless of monotonicity.

Priority enters here, at the decision boundary (§2): only protected goals
that outrank the objective a specific candidate outcome serves gate
spending on it (optimizer.protection.priority_for_outcome,
constraining_goals). A future goal the user ranked below the current
one is still pursued and reported, but it cannot force the spend down to
nothing - otherwise a Priority 2 goal sitting two versions away would
veto a Priority 1 goal available today. This gate is computed per
outcome, not once for the whole decision: a same-character preference
chain can reach past its roadmap-anchored objective into a deeper
constellation that the roadmap itself ranks lower (§2's own worked
example - Vesna C0 at Priority 1, Vesna C2 at Priority 3), and an
intervening goal (Vodynista at Priority 2) can then gate that deeper
reach without gating the shallower one.

Skipping is a legitimate recommendation (§1): when no outcome can be
pursued without violating protection - or there is nothing to pursue at
all - the planner says "do not spend" and reports the per-outcome
diagnostics, never a fabricated strategy (§13).

Provenance: every candidate is simulated with the same (runs, seed) and
both travel on the Recommendation (§2, §11 invariant 10) so reported
probabilities are never mistaken for exact values.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from domain import Banner, Preference
from planner import PlannerContext
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
from optimizer.stops import StopConditions, for_pursue, for_skip


@dataclass(frozen=True)
class RejectedOutcome:
    """Why a more-preferred outcome was not recommended (§13 step 7).

    Attributes:
        outcome: the rejected preferred outcome.
        best: the candidate closest to feasibility among the scanned caps:
            the highest minimum probability among the protected goals that
            actually constrain the decision, ties broken by higher outcome
            probability, then larger cap. With no constraining goals the
            floor is 1.0 by definition, so the outcome probability decides -
            the honest "closest to feasible" view.
        shortfalls: the constraining protected goals still below the
            threshold at `best` - the concrete "why".
    """

    outcome: OutcomeOption
    best: CandidateStrategy
    shortfalls: tuple[GoalStanding, ...]


@dataclass(frozen=True)
class Recommendation:
    """The planner's decision at the current banner (§1, §13, §20).

    Attributes:
        banner: the current banner the decision is about.
        action: "pursue" or "skip" (§1: "do not spend" is a legitimate
            answer).
        outcome: the pursued outcome - in outcome order (see module
            docstring: descending constellation for the current
            character's chain), the first feasible one; None when
            skipping.
        budget: the largest feasible cap for the outcome (§14); 0 when
            skipping.
        plan: the executable strategy for the recommendation (§12); None
            when skipping.
        outcome_probability: the empirical probability of achieving the
            outcome under the recommendation.
        protected: each protected goal's standing under the recommendation;
            under a skip, the do-nothing baseline (§2). Every standing
            carries whether it gates the decision (§2's priority gate), so
            a lower-priority goal shows its probability without vetoing
            the spend.
        rejected: outcomes tried before the winner (or before giving up),
            in the same order `available_outcomes` returns them - furthest
            constellation first - the "why" behind the recommendation.
        skip_reason: why nothing can be pursued; None when pursuing.
        stops: the stop conditions attached to the recommendation (§1).
        runs / seed: simulation provenance (§2).
    """

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



def _skip(
    context: PlannerContext,
    reason: str,
    runs: int,
    seed: int | None,
    rejected: tuple[RejectedOutcome, ...] = (),
) -> Recommendation:
    """A do-not-spend recommendation with the do-nothing baseline (§1, §2)."""
    return Recommendation(
        banner=current_banner(context),
        action="skip",
        outcome=None,
        budget=0,
        plan=None,
        outcome_probability=0.0,
        protected=evaluate_skip_baseline(context, runs=runs, seed=seed),
        rejected=rejected,
        skip_reason=reason,
        stops=for_skip(reason),
        runs=runs,
        seed=seed,
    )


def _diagnostic_key(candidate: CandidateStrategy) -> tuple[float, float, int]:
    """Rejection ordering: closest to feasible first (see RejectedOutcome).

    The floor is the weakest *constraining* protection; None means nothing
    gates the candidate, which sorts as a vacuous 1.0 so the outcome
    probability decides.
    """
    floor = (
        candidate.min_protected_probability
        if candidate.min_protected_probability is not None
        else 1.0
    )
    return (floor, candidate.outcome_probability, candidate.budget)


def _caps(context: PlannerContext, budgets: Iterable[int] | None) -> list[int]:
    """Candidate caps, largest first; the default is every spend 0..wishes.

    A caller-provided coarse list is honored as-is: because feasibility is
    not monotone in the cap (module docstring), a coarse list can miss
    narrow feasible windows - pass the full range (the default) for the
    exact boundary.
    """
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


def recommend(
    context: PlannerContext,
    preferences: Iterable[Preference] = (),
    *,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
    budgets: Iterable[int] | None = None,
) -> Recommendation:
    """The highest-preference feasible outcome and its largest feasible
    cap (§13, §14).

    Deterministic for identical (context, preferences, runs, seed,
    budgets). For an outcome with no *constraining* protected goals the
    scan short-circuits to the largest cap: nothing future outranks that
    outcome's objective, and a larger cap never lowers the outcome's
    probability (§12's cap semantics). This is decided per outcome, not
    once for the whole call (module docstring) - a deeper same-character
    reach can be gated even when the shallower one is not. Lower-priority
    protected goals may still exist and are still reported - they simply
    do not gate the spend (§2).

    Cost: a pursue recommendation stops at its first feasible cap, so it
    probes only the rejected upper caps. A skip must scan every cap of
    every outcome to report the rejections, which costs
    `outcomes x caps x runs` histories - seconds, not milliseconds, at the
    defaults. Pass a coarser `budgets` list when latency matters, keeping
    in mind that a coarse list can miss a narrow feasible window
    (module docstring).

    Raises:
        ValueError: via `_caps`, when the context has no roadmap banner
            at its (version, phase), or for degenerate duplicate active
            goals in the no-preference fallback (via
            `available_outcomes`).
    """
    outcomes = available_outcomes(context, preferences)
    all_caps = _caps(context, budgets)
    if not outcomes:
        return _skip(
            context,
            "no outcome to pursue on this banner: no preference chain and "
            "no active goal for the current banner character",
            runs,
            seed,
        )

    rejected: list[RejectedOutcome] = []
    for outcome in outcomes:
        # The cap-scan shortcut is per outcome, not global (§14): a deeper
        # reach on the same character can be its own, lower-priority
        # objective (optimizer.protection.priority_for_outcome), so one
        # outcome may have nothing constraining it while another does.
        outcome_priority = priority_for_outcome(context, outcome)
        caps = (
            all_caps
            if constraining_goals(context, priority=outcome_priority)
            else all_caps[:1]  # the largest cap dominates (see docstring)
        )
        diagnostic_best: CandidateStrategy | None = None
        for cap in caps:
            candidate = evaluate_candidate(
                context, outcome, cap, runs=runs, seed=seed
            )
            if (
                diagnostic_best is None
                or _diagnostic_key(candidate) > _diagnostic_key(diagnostic_best)
            ):
                diagnostic_best = candidate
            if candidate.feasible:
                # First feasible cap in a descending scan = the largest
                # feasible cap among the scanned caps (§14).
                return Recommendation(
                    banner=current_banner(context),
                    action="pursue",
                    outcome=outcome,
                    budget=cap,
                    plan=candidate.plan,
                    outcome_probability=candidate.outcome_probability,
                    protected=candidate.protected,
                    rejected=tuple(rejected),
                    skip_reason=None,
                    stops=for_pursue(outcome.label, cap),
                    runs=runs,
                    seed=seed,
                )
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

    return _skip(
        context,
        f"no preferred outcome can be pursued while keeping every "
        f"higher-priority protected goal at or above the "
        f"{context.confidence:.0%} confidence threshold",
        runs,
        seed,
        rejected=tuple(rejected),
    )
