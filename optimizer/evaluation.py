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

from dataclasses import dataclass, replace

from domain import WEAPON_EVENT_BANNER, Banner, Goal, TargetKind
from planner import PlannerContext, character_view, evaluate_goal, evaluate_goals
from planner.banners import available_banners
from planner.protection import (
    protected_goal_outcomes,
    weapon_goal_confidence,
)
from probability import (
    multi_copy_cumulative_probability,
    refinement_cumulative_probability,
)
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
            audits (per-banner spending, final wishes, seeds), and the
            source of `result.all_goals_probability` - the chance every
            roadmap goal, not just the protected ones, ends satisfied.
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


@dataclass(frozen=True)
class SkipBaseline:
    """The do-nothing baseline, in full (§1, §2).

    `evaluate_skip_baseline` only ever returned the protected standings;
    a do-not-spend recommendation also wants the roadmap-wide number -
    "if I spend nothing here, what's my chance of completing the WHOLE
    roadmap?" - which requires the underlying SimulationResult. This
    carries both out of one simulation so callers needing the roadmap-wide
    figure don't have to duplicate the run.

    Attributes:
        protected: each protected goal's standing under the baseline -
            identical to what `evaluate_skip_baseline` returns.
        all_goals_probability: fraction of simulated histories in which
            EVERY roadmap goal was satisfied under the baseline, not just
            the protected ones (mirrors SimulationResult.all_goals_probability,
            §11).
    """

    protected: tuple[GoalStanding, ...]
    all_goals_probability: float


def _analytic_goal_standing(
    context: PlannerContext,
    goal: Goal,
    spent: int,
    current_banner: Banner | None,
) -> tuple[float, bool] | None:
    """One goal's analytic standing over the shared wish pool (Phase 4).

    The character Monte Carlo cannot execute weapon banners and cannot
    evaluate weapon goals, so goals absent from a simulation result are
    evaluated through planner.protection's unified sequential-reserve
    model: per-target-type probability (character curves for character
    goals, the exact weapon DP for weapon goals) over ONE account budget.

    Returns (probability, meets_threshold), or None when the goal has no
    upcoming banner at all (not schedulable, therefore not protectable -
    the same rule planner.protection applies).

    Goals whose next banner is at the current order key (a same-slot
    alternative banner) are pursued with whatever remains after the
    current spend - the analytic analog of the simulator's same-slot
    alternative handling (uncapped budget on the remaining pool).
    """
    evaluation = evaluate_goal(context, goal)
    if evaluation.copies_needed == 0:
        return (1.0, True)
    if evaluation.next_banner is None:
        return None

    # Forward the caller's banner: with a character banner and a weapon
    # banner in the SAME version/phase, `current_banner(context)` would
    # raise "ambiguous current banner". The slot is not ambiguous here -
    # the caller already knows which banner this decision is about.
    outcomes = protected_goal_outcomes(context, spent=spent, banner=current_banner)
    by_goal = {outcome.goal: outcome for outcome in outcomes}
    if goal in by_goal:
        outcome = by_goal[goal]
        return (outcome.confidence, outcome.confidence >= context.confidence)

    # Same-slot alternative: the remaining pool after the current spend,
    # from a fresh state for the goal's own target type.
    budget = max(0, context.account.wishes - spent)
    copies = evaluation.copies_needed
    if goal.target.kind is TargetKind.WEAPON:
        probability = weapon_goal_confidence(context, copies, budget)
    else:
        if budget <= 0:
            probability = 0.0
        else:
            probability = float(
                multi_copy_cumulative_probability(
                    budget, copies, 0, False, context.mechanics
                )[budget]
            )
    return (probability, probability >= context.confidence)


def _weapon_goals_factor(
    context: PlannerContext, spent: int, current_banner: Banner | None
) -> float:
    """The weapon goals' share of the roadmap-wide probability figure.

    The character simulator reports `all_goals_probability` over its
    character goals only. This factor multiplies in every incomplete
    weapon goal's analytic probability over the shared pool, so the
    roadmap-wide figure stays honest for mixed roadmaps without forcing
    weapon probability through the character engine. A weapon goal with
    no upcoming banner contributes 0 - it can never be satisfied, the
    same treatment unschedulable character goals get inside the simulator.
    """
    factor = 1.0
    for evaluation in evaluate_goals(context):
        goal = evaluation.goal
        if goal.target.kind is not TargetKind.WEAPON:
            continue
        if evaluation.copies_needed == 0:
            continue  # already satisfied: contributes 1.0
        standing = _analytic_goal_standing(
            context, goal, spent, current_banner
        )
        factor *= standing[0] if standing is not None else 0.0
    return factor


def _standings(
    context: PlannerContext,
    result: SimulationResult | None,
    constraining: frozenset,
    banner=None,
    spent: int | None = None,
) -> tuple[GoalStanding, ...]:
    """Per-goal standings, flagged with the §2 priority gate.

    Per-goal report (§11): every original roadmap Goal keeps its own
    standing - grouping onto banners never merges goals (§13). `constraining`
    is the resolved set of goals that gate THIS decision (see
    optimizer.protection.priority_for_outcome for why that can differ per
    candidate outcome); the standings themselves always describe every
    protected goal, higher- and lower-priority alike.

    Character goals come from the simulation result when one ran; goals
    the character simulator cannot evaluate (weapon goals, and character
    goals in fully analytic weapon-branch candidates) come from
    `_analytic_goal_standing` over the same shared wish pool (`spent`).
    `result=None` means a fully analytic candidate: every standing is
    analytic.
    """
    from optimizer.protection import protected_groups

    probability_by_goal = (
        {probability.goal: probability.probability for probability in result.goals}
        if result is not None
        else {}
    )
    standings: list[GoalStanding] = []
    for group in protected_groups(context, banner=banner):
        for goal in group.goals:
            if goal in probability_by_goal:
                probability = probability_by_goal[goal]
                meets = probability >= context.confidence
            elif spent is not None:
                analytic = _analytic_goal_standing(
                    context, goal, spent, banner
                )
                if analytic is None:
                    continue
                probability, meets = analytic
            else:
                continue
            standings.append(
                GoalStanding(
                    goal=goal,
                    banner=group.banner,
                    probability=probability,
                    meets_threshold=meets,
                    constraining=goal in constraining,
                )
            )
    return tuple(standings)


def evaluate_candidate(
    context: PlannerContext,
    outcome: OutcomeOption,
    budget: int,
    banner=None,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
    simulation_sink=None,
) -> CandidateStrategy:
    """Simulate one (outcome, cap) candidate and assess feasibility (§13).

    Character outcomes run the Phase 4 Monte Carlo (§14). Weapon outcomes
    cannot execute on the character simulator and are evaluated by
    `_evaluate_weapon_candidate` with the exact weapon probability engine
    instead - never forced through the character probability API.

    Deterministic for identical (context, outcome, budget, runs, seed)
    (§11 invariant 10).

    Raises:
        ValueError: if `budget` is outside [0, account wishes] (via
            `candidate_plan`).
    """
    from optimizer.protection import constraining_goals, priority_for_outcome

    # Dispatch on the OUTCOME's target type, not only on a caller-supplied
    # banner: a weapon outcome evaluated with banner=None (single-banner
    # slot, the same rule available_outcomes applies) must reach the weapon
    # branch, never the character simulator. Resolve the outcome's own
    # weapon banner at the current slot so the weapon branch has the banner
    # whose target it evaluates; a weapon outcome without one is caller
    # error, reported as such instead of an AttributeError deep in a plan.
    if banner is None and outcome.target is not None and outcome.target.kind is TargetKind.WEAPON:
        matches = [
            candidate_banner
            for candidate_banner in available_banners(context)
            if candidate_banner.target == outcome.target
        ]
        if not matches:
            raise ValueError(
                f"weapon outcome {outcome.character!r} requires its banner: no "
                "banner for this weapon is available at the current "
                "version/phase"
            )
        banner = matches[0]
    if banner is not None and banner.target.kind is TargetKind.WEAPON:
        return _evaluate_weapon_candidate(
            context, outcome, budget, banner, runs, seed, simulation_sink
        )

    plan = candidate_plan(context, outcome, budget, banner=banner)
    result = simulate(character_view(context), plan, runs=runs, seed=seed)

    # The first processed banner is the current banner (§11): its
    # target_met aggregate is the outcome's probability.
    outcome_probability = result.banners[0].target_met_probability

    # Which goals gate THIS outcome (§2): a deeper reach on the current
    # banner's own character can be its own, lower-priority objective
    # (optimizer.protection.priority_for_outcome), so the anchor is
    # computed per outcome rather than once for the whole decision.
    constraining = constraining_goals(
        context,
        priority=priority_for_outcome(context, outcome),
        banner=banner,
    )
    standings = _standings(
        context, result, constraining, banner=banner, spent=budget
    )

    # Feasibility (§13 step 5): only the goals that outrank the decision
    # gate it (§2); the rest are reported and pursued but never veto.
    gating = [standing for standing in standings if standing.constraining]
    protected_ok = all(standing.meets_threshold for standing in gating)
    min_protected = (
        min(standing.probability for standing in gating) if gating else None
    )

    # Roadmap-wide figure for mixed roadmaps: the simulator covers the
    # character goals; incomplete weapon goals contribute their analytic
    # share of the same wish pool. Pure-character roadmaps keep the
    # simulator's figure untouched (the factor is exactly 1.0).
    weapon_factor = _weapon_goals_factor(context, budget, banner)
    if weapon_factor != 1.0:
        result = replace(
            result,
            all_goals_probability=result.all_goals_probability * weapon_factor,
        )

    candidate = CandidateStrategy(
        outcome=outcome,
        budget=budget,
        plan=plan,
        result=result,
        outcome_probability=outcome_probability,
        protected=standings,
        min_protected_probability=min_protected,
        feasible=protected_ok and outcome_probability > 0.0,
    )
    if simulation_sink is not None:
        simulation_sink(candidate)
    return candidate


def _evaluate_weapon_candidate(
    context: PlannerContext,
    outcome: OutcomeOption,
    budget: int,
    banner: Banner,
    runs: int,
    seed: int | None,
    simulation_sink=None,
) -> CandidateStrategy:
    """One (weapon outcome, cap) candidate, evaluated analytically (Phase 4).

    The character Monte Carlo cannot execute weapon banners (character
    pity/guarantee/Radiance and character ownership only), so the weapon
    branch consumes the exact weapon probability engine (probability.weapon)
    directly:

    * the outcome's probability is the designated-copy refinement curve at
      the account's ACTUAL weapon state (pity, guarantee, Fate Points);
    * feasibility is planner.protection's unified sequential-reserve model
      over the ONE shared wish pool - every constraining protected goal,
      character or weapon, must stay at or above the confidence threshold.

    The roadmap-wide figure is the analytic composite of the outcome and
    each remaining roadmap goal's standing (an independence approximation,
    unlike the character branch's joint Monte Carlo histories; the per-goal
    standings and the feasibility decision are the same either way).
    """
    from optimizer.protection import constraining_goals, priority_for_outcome

    plan = candidate_plan(context, outcome, budget, banner=banner)
    weapon_state = context.account.weapon_state
    owned = context.account.owned_characters.owned_refinement(banner.target.name)
    copies = max(outcome.constellation - owned, 0)
    if copies == 0:
        outcome_probability = 1.0
    elif budget <= 0:
        outcome_probability = 0.0
    else:
        curve = refinement_cumulative_probability(
            budget,
            owned,
            outcome.constellation,
            weapon_state.pity,
            weapon_state.guarantee,
            weapon_state.fate_points,
            WEAPON_EVENT_BANNER,
        )
        outcome_probability = float(curve[budget])

    constraining = constraining_goals(
        context,
        priority=priority_for_outcome(context, outcome),
        banner=banner,
    )
    standings = _standings(
        context, None, constraining, banner=banner, spent=budget
    )

    # Feasibility (§13 step 5): identical semantics to the character branch.
    gating = [standing for standing in standings if standing.constraining]
    protected_ok = all(standing.meets_threshold for standing in gating)
    min_protected = (
        min(standing.probability for standing in gating) if gating else None
    )

    # Analytic roadmap-wide composite over every remaining roadmap goal.
    all_goals_probability = outcome_probability
    for evaluation in evaluate_goals(context):
        goal = evaluation.goal
        if evaluation.copies_needed == 0:
            continue  # already satisfied: contributes 1.0
        if goal.target == banner.target:
            if goal.level > outcome.constellation:
                # A deeper same-weapon objective is not subsumed by this
                # candidate: its own curve within the same cap.
                deeper = refinement_cumulative_probability(
                    budget,
                    owned,
                    goal.level,
                    weapon_state.pity,
                    weapon_state.guarantee,
                    weapon_state.fate_points,
                    WEAPON_EVENT_BANNER,
                )
                all_goals_probability *= float(deeper[budget])
            continue  # level <= outcome: subsumed by the pursued outcome
        standing = next((s for s in standings if s.goal == goal), None)
        if standing is not None:
            all_goals_probability *= standing.probability
        else:
            # No upcoming banner: can never be satisfied, mirroring the
            # simulator's treatment of unschedulable roadmap goals.
            all_goals_probability = 0.0
            break

    # End-of-history wishes are deterministic in this analytic view: the
    # current cap is spent, future reserves are required but not pre-spent
    # (planner.protection treats reserves as required, not consumed).
    remaining = max(0, context.account.wishes - budget)
    result = SimulationResult(
        runs=runs,
        seed=seed,
        plan=plan,
        goals=(),
        banners=(),
        all_goals_probability=all_goals_probability,
        final_wishes_mean=remaining,
        final_wishes_min=remaining,
        final_wishes_max=remaining,
    )
    candidate = CandidateStrategy(
        outcome=outcome,
        budget=budget,
        plan=plan,
        result=result,
        outcome_probability=outcome_probability,
        protected=standings,
        min_protected_probability=min_protected,
        feasible=protected_ok and outcome_probability > 0.0,
    )
    if simulation_sink is not None:
        simulation_sink(candidate)
    return candidate


def evaluate_skip_baseline_full(
    context: PlannerContext,
    runs: int = DEFAULT_RUNS,
    seed: int | None = DEFAULT_SEED,
) -> SkipBaseline:
    """The do-nothing baseline, protected standings plus the roadmap-wide
    probability, from a single simulation (§1, §2).

    The skip baseline processes every future banner with its protected
    pursuits and no current-banner spending: "what does my roadmap look
    like if I do not spend?" - both the per-goal protected view (§1) and
    the "chance of completing the whole roadmap" figure
    (`all_goals_probability`) come from the same run, so this never pays
    for two simulations to answer one question.

    `evaluate_skip_baseline` wraps this and returns only `.protected`, so
    existing callers of that function are unaffected by this addition.
    """
    from optimizer.protection import constraining_goals, protected_groups

    from planner.banners import available_banners
    matches = available_banners(context)
    selected = matches[0] if matches else None
    groups = protected_groups(context, banner=selected, include_same_slot=False)
    plan = SpendPlan(
        entries=tuple(
            PlannedSpend(
                banner=group.banner,
                target_constellation=group.target_constellation,
                budget=group.uncapped_budget,
            )
            for group in groups
            # Weapon banners are never character-simulated (Phase 4);
            # their goals are evaluated analytically below.
            if group.banner.target.kind is TargetKind.CHARACTER
        )
    )
    result = simulate(character_view(context), plan, runs=runs, seed=seed)
    standings = _standings(
        context,
        result,
        constraining_goals(context, banner=selected),
        banner=selected,
        spent=0,
    )

    # Roadmap-wide figure for mixed roadmaps: multiply in the analytic
    # share of every incomplete weapon goal over the shared pool (the
    # pure-character factor is exactly 1.0).
    weapon_factor = _weapon_goals_factor(context, 0, selected)
    all_goals_probability = result.all_goals_probability * weapon_factor
    return SkipBaseline(
        protected=standings, all_goals_probability=all_goals_probability
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

    See `evaluate_skip_baseline_full` for the same computation plus the
    roadmap-wide `all_goals_probability`, from one simulation instead of
    two.
    """
    return evaluate_skip_baseline_full(context, runs=runs, seed=seed).protected
