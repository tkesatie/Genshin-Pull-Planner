"""Candidate evaluation through the Phase 4 simulator (§13 steps 4-5, §14).

Every candidate strategy is evaluated by Monte Carlo simulation, not by
the Phase 3 analytic approximation: §14 requires the final planner to
account for carried pity/guarantee, actual spending decisions and income
timing - exactly what simulation.engine models. The Phase 3 approximation
remains its own (documented) view.

Feasibility (§13 step 5) is the conjunction of:

* PROTECTION: every *constraining* protected goal's simulated satisfaction
  probability meets the context's confidence threshold (§1: "pursuing this
  outcome leaves me with a 92% probability of satisfying the protected
  roadmap"). Which protected goals constrain is §2's priority question and
  lives in optimizer.protection.constraining_goals: a future goal the user
  ranked below the current objective is still pursued and reported, but it
  does not veto spending on the higher-priority current goal;
* PURSUIT: the outcome's probability is empirically nonzero.

The pursuit criterion is empirical by construction: `outcome_probability`
is the fraction of simulated histories in which the current-banner target
was met. With finite runs, a mechanically possible but extremely unlikely
outcome can observe zero successes and be reported unpursueable - Monte
Carlo noise, not mathematical impossibility (acknowledged Phase 5 scope).
It exists because a cap of 0 legitimately pursues nothing: a "strategy"
that cannot achieve its outcome must not win the preference comparison
(§13 step 7).

The confidence comparison is `probability >= threshold` (§10.3 semantics).
"""

from dataclasses import dataclass

from domain import Banner, Goal
from planner import PlannerContext
from simulation import DEFAULT_SEED, PlannedSpend, SimulationResult, SpendPlan, simulate

from optimizer.candidates import candidate_plan
from optimizer.outcomes import OutcomeOption

# One candidate probe costs ~2 ms per 50 histories at real mechanics, and the
# optimizer probes many caps per outcome, so its default is deliberately
# smaller than simulation.DEFAULT_RUNS (10_000): at 2_000 runs the standard
# error on a probability is at most sqrt(0.25 / 2000) ~ 1.1%. Probabilities
# travel with their `runs` and `seed` (§2, §11 invariant 10) so they are never
# mistaken for exact values; callers wanting tighter estimates pass a larger
# `runs` without changing any logic.
DEFAULT_RUNS = 2_000


@dataclass(frozen=True)
class GoalStanding:
    """One protected goal's simulated standing under a candidate (§13).

    Attributes:
        goal: the protected roadmap goal - never a merged surrogate; the
            simulator evaluates every original Goal independently (§11).
        banner: the strictly-future banner where the goal is pursued.
        probability: fraction of simulated histories in which the goal
            ended satisfied.
        meets_threshold: probability >= the context's confidence threshold.
        constraining: whether this goal gates feasibility - i.e. whether
            its priority outranks the current decision (§2, see
            optimizer.protection.constraining_goals). A non-constraining
            standing is reported, not ignored: the recommendation still
            shows where the roadmap lands, it just does not let a
            lower-priority goal veto spending on a higher-priority one.
    """

    goal: Goal
    banner: Banner
    probability: float
    meets_threshold: bool
    constraining: bool


@dataclass(frozen=True)
class CandidateStrategy:
    """One evaluated (outcome, cap) candidate (§13 steps 4-5).

    Attributes:
        outcome: the pursued preferred outcome (§15).
        budget: the current-banner spend cap - a cap, not a commitment
            (§12).
        plan: the executed SpendPlan (current + protected entries).
        result: the aggregated simulation (§11) - full provenance for
            audits (per-banner spending, final wishes, seeds).
        outcome_probability: empirical P(the outcome's target is met on
            the current banner within the cap).
        protected: each protected goal's standing, in the protected
            grouping's order (chronological banners, priority within),
            each flagged with whether it gates this candidate (§2).
        min_protected_probability: the weakest *constraining* protected
            standing, or None when no protected goal outranks the current
            decision - with nothing gating, the floor is vacuous and the
            rejection diagnostic falls to the outcome probability.
        feasible: constraining protection AND empirical pursuit (module
            docstring).
    """

    outcome: OutcomeOption
    budget: int
    plan: SpendPlan
    result: SimulationResult
    outcome_probability: float
    protected: tuple[GoalStanding, ...]
    min_protected_probability: float | None
    feasible: bool


def _standings(
    context: PlannerContext,
    result: SimulationResult,
    constraining: frozenset,
) -> tuple[GoalStanding, ...]:
    """Per-goal simulated standings, flagged with the §2 priority gate.

    Per-goal report (§11): every original roadmap Goal keeps its own
    standing - grouping onto banners never merges goals (§13). `constraining`
    is the resolved set of goals that gate THIS decision (see
    optimizer.protection.priority_for_outcome for why that can differ per
    candidate outcome); the standings themselves always describe every
    protected goal, higher- and lower-priority alike.
    """
    from optimizer.protection import protected_groups

    probability_by_goal = {
        probability.goal: probability.probability for probability in result.goals
    }
    return tuple(
        GoalStanding(
            goal=goal,
            banner=group.banner,
            probability=probability_by_goal[goal],
            meets_threshold=probability_by_goal[goal] >= context.confidence,
            constraining=goal in constraining,
        )
        for group in protected_groups(context)
        for goal in group.goals
    )


def evaluate_candidate(
    context: PlannerContext,
    outcome: OutcomeOption,
    budget: int,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
) -> CandidateStrategy:
    """Simulate one (outcome, cap) candidate and assess feasibility (§13).

    Deterministic for identical (context, outcome, budget, runs, seed)
    (§11 invariant 10).

    Raises:
        ValueError: if `budget` is outside [0, account wishes] (via
            `candidate_plan`).
    """
    from optimizer.protection import constraining_goals, priority_for_outcome

    plan = candidate_plan(context, outcome, budget)
    result = simulate(context, plan, runs=runs, seed=seed)

    # The first processed banner is the current banner (§11): its
    # target_met aggregate is the outcome's probability.
    outcome_probability = result.banners[0].target_met_probability

    # Which goals gate THIS outcome (§2): a deeper reach on the current
    # banner's own character can be its own, lower-priority objective
    # (optimizer.protection.priority_for_outcome), so the anchor is
    # computed per outcome rather than once for the whole decision.
    constraining = constraining_goals(
        context, priority=priority_for_outcome(context, outcome)
    )
    standings = _standings(context, result, constraining)

    # Feasibility (§13 step 5): only the goals that outrank the decision
    # gate it (§2); the rest are reported and pursued but never veto.
    gating = [standing for standing in standings if standing.constraining]
    protected_ok = all(standing.meets_threshold for standing in gating)
    min_protected = (
        min(standing.probability for standing in gating) if gating else None
    )
    return CandidateStrategy(
        outcome=outcome,
        budget=budget,
        plan=plan,
        result=result,
        outcome_probability=outcome_probability,
        protected=standings,
        min_protected_probability=min_protected,
        feasible=protected_ok and outcome_probability > 0.0,
    )


def evaluate_skip_baseline(
    context: PlannerContext,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
) -> tuple[GoalStanding, ...]:
    """Protected standings under a do-nothing decision (§1, §2).

    The skip baseline processes every future banner with its protected
    pursuits and no current-banner spending: "what does my roadmap look
    like if I do not spend?" - the diagnostic behind a do-not-spend
    recommendation (§1). There is no specific outcome being pursued here,
    so the whole-decision anchor (`current_goal_priority`, via
    `constraining_goals`'s default) applies, same as before this module
    started gating per outcome.
    """
    from optimizer.protection import constraining_goals, protected_groups

    plan = SpendPlan(
        entries=tuple(
            PlannedSpend(
                banner=group.banner,
                target_constellation=group.target_constellation,
                budget=group.uncapped_budget,
            )
            for group in protected_groups(context)
        )
    )
    result = simulate(context, plan, runs=runs, seed=seed)
    return _standings(context, result, constraining_goals(context))
