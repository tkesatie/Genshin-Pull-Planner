"""Strategy optimizer (Design Document §13, §18 Phase 5).

The layer that connects account (§4), roadmap (§8), preferences (§15) and
simulation (§11) into the actual decision (§20): the highest-preference
outcome that can be pursued now while every protected future goal stays at
or above the confidence threshold.

Conceptual flow:

    PlannerContext + Preferences
                |
                v
        available_outcomes()            (§13 step 2, §15)
                |
                v
         OutcomeOption list
                |                \
                v                 v
      protected_groups()    candidate caps   (§13 step 3, §14)
                |                |
                +-------+--------+
                        v
                candidate_plan()         (§13 step 3)
                        v
              evaluate_candidate()     (§13 steps 4-5, §14)
                        v
                 CandidateStrategy
                        v
                    recommend()          (§13 steps 6-7)
                        |
          +-------------+-------------+
          v             v             v
       pursue     discretionary      skip
          |             |
          +------+------+
                 v
        StopConditions            (§1, §2)

Phase 5 invariants:

1. The optimizer consumes domain and planner state and never mutates
   Account, Roadmap, Preference or any other domain object.
2. Outcomes come only from the user's preference chain for the current
   banner character (§15); the planner never invents an outcome (§2). The
   documented fallback - the ACTIVE roadmap goal when no preference chain
   exists - is user-defined through the roadmap (§1, §5).
3. An outcome's constellation is a desired *resulting* constellation,
   never a copy count (§4.2, §12); copies are derived by the simulator
   (§10.4).
4. A candidate strategy is one (outcome, spend cap) decision, executed as
   a SpendPlan the Phase 4 simulator runs faithfully (§12, §13 step 3).
   Future protected goals carry strategic budgets sized so the entry's
   budget is never the limiting factor; `min(budget, available)` (§12) -
   not the budget number - bounds actual spending, so no plan entry can
   overspend or drive wishes negative (§4.1).
5. Protection keeps one classification across phases: unsatisfied goals
   whose next banner is strictly after the current banner (§13 step 5).
   Grouping goals onto a shared banner is a strategy-execution
   optimization only; the simulator evaluates every original Goal
   independently (§11) and feasibility scores each goal separately.
6. Priority enters at the decision boundary, not in the classification
   (§2): only protected goals whose priority outranks the objective a
   specific candidate outcome serves constrain spending on it
   (`priority_for_outcome`, `constraining_goals`). A lower-priority future
   goal is still pursued and still reported, but it cannot veto spending
   on a higher-priority current goal - and the inverse (a future Priority 1
   goal constraining a current Priority 2 goal) is exactly what the gate
   preserves. The anchor is per outcome, not per decision: a preference
   chain reaching past its roadmap-anchored goal into a deeper,
   lower-priority constellation of the same character (§2's own example -
   Vesna C0 at Priority 1, Vesna C2 at Priority 3) is gated by its own
   priority, so an intervening goal (Vodynista at Priority 2) can protect
   itself from the deeper reach without blocking the shallower one.
   `planner.protection` stays priority-independent: its Phase 3 contract
   answers "what is protected", never "what may veto this decision".
7. Feasibility = every constraining protected goal's simulated
   satisfaction probability >= the context threshold AND a nonzero
   empirical outcome probability (§13 step 5). The latter is a Monte
   Carlo criterion - acknowledged sampling noise, not a proof of
   impossibility.
8. Evaluation runs the Phase 4 simulator (§14): carried pity/guarantee,
   actual spending decisions, income timing and multi-copy targets all
   count. No rate mathematics lives in this package.
9. Selection is lexicographic, never a global score (§2, §13): outcome
   order, then feasibility, then the largest feasible cap. Outcome order
   is descending constellation within the current character's chain, not
   raw preference rank - a same-character chain is a progression (reaching
   C2 necessarily reaches C0), so the most-inclusive current-banner target
   is tried first. One scheduling exception: a chain constellation whose
   roadmap goal is blocked (§9) and whose character has a strictly later
   banner is a later progression objective - it sorts behind the nearer
   objectives and takes the lead only when pursuing it now is an ordinary
   recommendation (probability at or above MINIMUM_OUTCOME_PROBABILITY at
   the largest cap that keeps every higher-priority protected goal safe)
   and it is the deeper constellation. Then one plan entry targets it and
   the Phase 4 simulator pulls through the nearer milestones toward it -
   the cumulative progression (§4.2, §12); below that bar the active
   milestone leads and the later objective is pursued on its own banner
   after the re-run (§2). Rank still decides which constellations are in
   scope and how duplicates collapse (optimizer.outcomes). The cap scan
   is exhaustive over the given caps - feasibility is not monotone in the
   cap (a lost 50/50 carried forward can protect a future goal), so no
   binary search.
10. Monte Carlo results are seed-deterministic; the Recommendation carries
   runs and seed so its probabilities are never mistaken for exact values
   (§2, §11 invariant 10).
11. When no outcome is feasible, the recommendation is "do not spend"
    with per-outcome rejections - never a fabricated strategy (§1, §13).
12. Stop conditions express the cap's real-world execution: stop on the
    outcome, never exceed the cap, re-run after updates (§1, §2).
13. Weapon refinement is displayed from preference labels but never
    simulated (§17, §19).
14. Feasibility (invariant 7) and `MINIMUM_OUTCOME_PROBABILITY` are separate
    questions (§2, §14). The confidence threshold protects higher-priority
    future goals; `MINIMUM_OUTCOME_PROBABILITY` (default 0.25) asks whether
    the winning outcome's own empirical probability, at its largest feasible
    cap, is high enough to present as an ordinary recommendation. For a
    chosen winner it is never an outcome-selection rule - it does not
    reopen the choice of outcome or cap, and it never causes a fall-through
    to a different, more-conservative outcome (optimizer.recommend module
    docstring). The one ordering interaction is the later-progression
    eligibility rule (invariant 9): a roadmap-scheduled later objective
    yields to the nearer objectives unless pursuing it now clears the
    minimum at its protection-respecting cap - decided before a winner
    exists, and never changing the presentation of a chosen winner. Below
    the minimum, the same winning (outcome, cap) is reported as
    `action="discretionary"`: the spend is disclosed as available and safe
    for the roadmap, but not recommended, because "allowed to spend" and
    "recommended to spend" are different claims.
15. `Recommendation.all_goals_probability` (and, for skip decisions,
    `SkipBaseline.all_goals_probability` via `evaluate_skip_baseline_full`)
    is the roadmap-wide figure: the fraction of simulated histories in
    which EVERY roadmap goal ended satisfied, not just the protected ones.
    It comes straight off `SimulationResult.all_goals_probability` (§11)
    for the winning candidate's own simulation, so it is never a separate
    computation and never double-simulates.
"""

from optimizer.candidates import candidate_plan
from optimizer.evaluation import (
    DEFAULT_RUNS,
    CandidateStrategy,
    GoalStanding,
    SkipBaseline,
    evaluate_candidate,
    evaluate_skip_baseline,
    evaluate_skip_baseline_full,
)
from optimizer.outcomes import OutcomeOption, available_outcomes, goal_label
from optimizer.protection import (
    ProtectedGroup,
    constraining_goals,
    current_goal_priority,
    priority_for_outcome,
    protected_groups,
)
from optimizer.recommend import (
    MINIMUM_OUTCOME_PROBABILITY,
    Recommendation,
    RejectedOutcome,
    recommend,
)
from optimizer.stops import StopConditions, for_discretionary, for_pursue, for_skip
from optimizer.strategy import PullStrategy, StrategyStep, build_strategy

__all__ = [
    "DEFAULT_RUNS",
    "MINIMUM_OUTCOME_PROBABILITY",
    "CandidateStrategy",
    "GoalStanding",
    "OutcomeOption",
    "ProtectedGroup",
    "Recommendation",
    "RejectedOutcome",
    "SkipBaseline",
    "StopConditions",
    "available_outcomes",
    "candidate_plan",
    "constraining_goals",
    "current_goal_priority",
    "evaluate_candidate",
    "evaluate_skip_baseline",
    "evaluate_skip_baseline_full",
    "for_discretionary",
    "for_pursue",
    "for_skip",
    "priority_for_outcome",
    "protected_groups",
    "recommend",
    "PullStrategy",
    "StrategyStep",
    "build_strategy",
]
