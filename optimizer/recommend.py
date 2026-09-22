"""Strategy optimization: selection and recommendation (§13 steps 6-7, §14, §15).

The decision is lexicographic - never a global "best" score (§2, §13):

    1. Which outcome do I prefer?           outcome order (§15, see below)
    2. Can I pursue it while protecting
       the roadmap?                          feasibility (§13 step 5)
    3. How much may I spend on it?          largest feasible cap (§14)
    4. Is that cap actually worth
       recommending?                        minimum_outcome_probability (§14)

Step 4 is a separate concept from step 2's protection threshold (Phase 5
correction). The confidence threshold answers "does this leave my
higher-priority future goals safe?" `MINIMUM_OUTCOME_PROBABILITY` answers a
different question: "is this specific outcome, at the largest cap that
keeps the roadmap safe, actually likely enough to call it a recommendation?"
A candidate can be entirely feasible (protection holds, the outcome is
empirically reachable at all) and still have a low empirical
`outcome_probability` - reaching a deep constellation, or reaching any
target on very few wishes. Below the minimum, the winning outcome does not
change (§13 step 7's outcome order and feasibility are untouched - a low
probability at the winning outcome's own cap is never a reason to fall
through to the next, more-conservative outcome), but the action reported is
"discretionary" rather than "pursue": the wishes are available to spend
without endangering anything higher priority, but success is unlikely
enough that presenting it as an ordinary recommendation would overstate it.
"You are allowed to spend these wishes" is not the same claim as "I
recommend spending these wishes."

`optimizer.outcomes.available_outcomes` already restricts every outcome
it returns to the current banner's single character, and resulting
constellations for one character are strictly nested (reaching C2
necessarily reaches C0). That tuple is therefore ordered by descending
constellation - the furthest target first - not by the user's stated
preference rank: a same-character chain is a progression of how far to
go, never a set of mutually exclusive alternatives, so the outcome that
subsumes the most of the chain is tried first. One scheduling exception
(optimizer.outcomes' later-progression rule): a chain constellation whose
roadmap goal is BLOCKED (§9) and whose character has a strictly later
banner is a later progression objective - the roadmap schedules that
reach for the later banner - so it sorts behind the current-banner
objectives and carries a `later_progression` tag. Selection is
two-phase. Phase one evaluates every outcome to its first feasible cap
in a descending scan - for an unconstrained outcome the cap-scan
shortcut makes that the whole pool. Phase two picks the winner: the
first feasible outcome in menu order, except that a tagged
later-progression outcome takes the lead when pursuing it NOW is an
ordinary recommendation (its probability at that cap is at or above
`minimum_outcome_probability`) and it is the deeper constellation. That
is the cumulative-progression rule (§4.2, §12): pulling toward the
active milestone is also progress toward the later objective, and one
plan entry targeting the deeper constellation lets the Phase 4
simulator pull through C0 toward C2 within each history - each
future's C0 acquisition point varies naturally and the remaining spend
continues toward C2. When the surplus makes the progression likely
enough at a protection-respecting cap, the plan targets it; below that
bar the reach stays the gamble the roadmap scheduled for later, the
active milestone leads, and the objective is pursued on its own banner
after the re-run (§2). The winning outcome's reported cap is its first
feasible cap: the largest cap (§14's safe spending, evaluated at
roadmap level through simulation).

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
    evaluate_skip_baseline_full,
    SkipBaseline,
)
from optimizer.outcomes import OutcomeOption, available_outcomes
from optimizer.protection import constraining_goals, priority_for_outcome
from optimizer.stops import StopConditions, for_discretionary, for_pursue, for_skip

# The confidence threshold (PlannerContext.confidence) protects
# *higher-priority future goals*; this constant is a different question -
# whether the current-banner objective itself, at the largest cap that
# keeps those future goals safe, is likely enough to present as a normal
# recommendation rather than a disclosed gamble (module docstring, §14).
MINIMUM_OUTCOME_PROBABILITY = 0.25

# Optional recommendation-cache lookup supplied by the API layer.
CandidateLookup = Callable[[OutcomeOption, int, Banner], "CandidateStrategy | None"]


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
        action: "pursue", "discretionary" or "skip" (§1: "do not spend" is
            a legitimate answer). "discretionary" is a feasible candidate -
            protection holds and the outcome is empirically reachable -
            whose `outcome_probability` falls below
            `MINIMUM_OUTCOME_PROBABILITY`: the wishes may be spent without
            endangering anything higher priority, but the chance of success
            is too low to present as an ordinary recommendation (module
            docstring, §14).
        outcome: the pursued outcome - in outcome order (see module
            docstring: descending constellation for the current
            character's chain), the first feasible one; None when
            skipping. Set for both "pursue" and "discretionary".
        budget: the largest feasible cap for the outcome (§14); 0 when
            skipping. Set for both "pursue" and "discretionary" - a
            discretionary action still names how much may safely be spent.
        plan: the executable strategy for the recommendation (§12); None
            when skipping.
        outcome_probability: the empirical probability of achieving the
            outcome under the recommendation.
        all_goals_probability: fraction of simulated histories in which
            EVERY roadmap goal ended satisfied under this recommendation -
            not just the protected ones (mirrors
            SimulationResult.all_goals_probability, §11). Under a skip,
            this is the do-nothing baseline's own roadmap-wide figure
            (§1, §2): "if I spend nothing here, what's my chance of
            completing the whole roadmap?" This is a different question
            from any single goal's `probability` in `protected` - it is
            the probability that ALL of them (and every satisfied,
            non-protected goal) hold at once, which is not the product of
            the individual probabilities (they are not independent across
            one shared history).
        protected: each protected goal's standing under the recommendation;
            under a skip, the do-nothing baseline (§2). Every standing
            carries whether it gates the decision (§2's priority gate), so
            a lower-priority goal shows its probability without vetoing
            the spend.
        rejected: outcomes that could not be pursued at any scanned cap,
            in the order `available_outcomes` returns them - the "why"
            behind the recommendation. A feasible outcome is never listed
            here: not a "discretionary" winner (the 25% check decides how
            to present the winning outcome, module docstring) and not a
            later-progression outcome that yielded to the winner -
            yielding is the outcome-order eligibility rule, not a
            feasibility verdict.
        skip_reason: why nothing can be pursued; None unless skipping.
        discretionary_reason: why a feasible outcome was downgraded to a
            disclosed gamble instead of an ordinary recommendation; None
            unless action is "discretionary".
        stops: the stop conditions attached to the recommendation (§1).
        runs / seed: simulation provenance (§2).
    """

    banner: Banner
    action: str
    outcome: OutcomeOption | None
    budget: int
    plan: SpendPlan | None
    outcome_probability: float
    all_goals_probability: float
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
    skip_baseline_lookup: "Callable[[Banner], SkipBaseline | None] | None" = None,
    skip_baseline_sink: "Callable[[Banner, SkipBaseline], None] | None" = None,
) -> Recommendation:
    """A do-not-spend recommendation with the do-nothing baseline (§1, §2)."""
    banners = available_banners(context)
    banner = banners[0] if banners else current_banner(context)

    baseline = skip_baseline_lookup(banner) if skip_baseline_lookup is not None and banner is not None else None
    if baseline is None:
        baseline = evaluate_skip_baseline_full(context, runs=runs, seed=seed)
        if skip_baseline_sink is not None and banner is not None:
            skip_baseline_sink(banner, baseline)
    return Recommendation(
        banner=banner,
        action="skip",
        outcome=None,
        budget=0,
        plan=None,
        outcome_probability=0.0,
        all_goals_probability=baseline.all_goals_probability,
        protected=baseline.protected,
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
    """The "available but not recommended" explanation (module docstring).

    Named at the label level (never the detection mechanics beyond the two
    numbers themselves): the largest cap that still protects every
    higher-priority goal, and the empirical chance that spending it achieves
    the outcome.
    """
    return (
        f"{outcome_label} is available but not recommended: the largest "
        "spend that still protects every higher-priority goal gives only a "
        f"{outcome_probability:.0%} chance of achieving it, below the "
        f"{minimum_outcome_probability:.0%} minimum for an ordinary "
        "recommendation. You may still spend up to the reported budget if "
        "you want to take the gamble, but success is unlikely."
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
    """Evaluate one cap, using the recommendation cache when available."""
    if candidate_lookup is not None:
        cached = candidate_lookup(outcome, cap, banner)
        if cached is not None:
            return cached
    return evaluate_candidate(
        context, outcome, cap, banner=banner, runs=runs, seed=seed,
        simulation_sink=simulation_sink,
    )


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
    minimum_outcome_probability: float = MINIMUM_OUTCOME_PROBABILITY,
    simulation_sink=None,
    candidate_lookup: CandidateLookup | None = None,
    skip_baseline_lookup: "Callable[[Banner], tuple[GoalStanding, ...] | None] | None" = None,
    skip_baseline_sink: "Callable[[Banner, tuple[GoalStanding, ...]], None] | None" = None,
) -> Recommendation:
    """The highest-preference feasible outcome and its largest feasible
    cap (§13, §14).

    Deterministic for identical (context, preferences, runs, seed,
    budgets, minimum_outcome_probability). For an outcome with no
    *constraining* protected goals the scan short-circuits to the largest
    cap: nothing future outranks that outcome's objective, and a larger cap
    never lowers the outcome's probability (§12's cap semantics). This is
    decided per outcome, not once for the whole call (module docstring) - a
    deeper same-character reach can be gated even when the shallower one is
    not. Lower-priority protected goals may still exist and are still
    reported - they simply do not gate the spend (§2).

    Selection is two-phase (module docstring): every outcome is evaluated
    to its first feasible cap - the largest cap that keeps the roadmap
    safe - and the winner is the first feasible outcome in menu order,
    unless a tagged later-progression outcome is the deeper constellation
    AND an ordinary recommendation at its cap, in which case it leads.

    `minimum_outcome_probability` never changes which outcome a chosen
    winner is, or its cap (module docstring: for a chosen winner it is not
    an outcome-selection rule) - it only decides whether the winning
    candidate is reported as `action="pursue"` or the disclosed-gamble
    `action="discretionary"`. Its one ordering interaction is the
    later-progression eligibility rule: a roadmap-scheduled later
    objective leads only when pursuing it now clears the minimum at its
    protection-respecting cap.

    Cost: each outcome is probed down to its first feasible cap, so a
    pursue or discretionary recommendation probes the rejected upper caps
    of every outcome it considered. A skip must scan every cap of every
    outcome to report the rejections, which costs
    `outcomes x caps x runs` histories - seconds, not milliseconds, at the
    defaults. Pass a coarser `budgets` list when latency matters, keeping
    in mind that a coarse list can miss a narrow feasible window
    (module docstring).

    Raises:
        ValueError: via `_caps`, when the context has no roadmap banner
            at its (version, phase), or for degenerate duplicate active
            goals in the no-preference fallback (via
            `available_outcomes`); or if `minimum_outcome_probability` is
            outside [0, 1].
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
            skip_baseline_lookup=skip_baseline_lookup,
            skip_baseline_sink=skip_baseline_sink,
        )

    all_caps = _caps(context, budgets)
    opportunities = []
    rejected: list[RejectedOutcome] = []

    # Each current banner is an independent decision opportunity. The
    # roadmap priority of its active goal determines which opportunity gets
    # first consideration; preferences still determine how far to pursue
    # within that character.
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

            # Lower priority number is the higher-priority roadmap objective.
            # Within one banner, preserve the existing outcome menu order.
            opportunities.append(
                (banner, outcome_index, outcome, first_feasible, outcome_priority)
            )

    if not opportunities:
        if not any(available_outcomes(context, preferences, banner=banner) for banner in banners):
            return _skip(context, "no active goal or preferred outcome to pursue", runs, seed)
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

    # A deeper same-character outcome is cumulative progress: reaching
    # C2 necessarily reaches C0. If the deeper outcome is feasible and
    # likely enough to be an ordinary recommendation at its protection-safe
    # cap, it can supersede a shallower outcome even when the shallower
    # roadmap goal has higher priority. A tagged later-progression outcome
    # uses the same rule, but is additionally kept behind the current
    # milestone when it is only a gamble.
    for banner, _, outcome, candidate, priority in opportunities[1:]:
        if (
            banner == winner_banner
            and outcome.character == winner_outcome.character
            and outcome.constellation > winner_outcome.constellation
            and candidate.outcome_probability >= minimum_outcome_probability
        ):
            winner_banner, winner_outcome, winner = banner, outcome, candidate

    # The presentation check (step 4) settles only how the winner is
    # reported - never which outcome or cap won (module docstring).
    if winner.outcome_probability >= minimum_outcome_probability:
        return Recommendation(
            banner=winner_banner,
            action="pursue",
            outcome=winner_outcome,
            budget=winner.budget,
            plan=winner.plan,
            outcome_probability=winner.outcome_probability,
            all_goals_probability=winner.result.all_goals_probability,
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
        all_goals_probability=winner.result.all_goals_probability,
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
