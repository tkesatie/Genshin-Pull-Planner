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
              +---------+---------+
              v                   v
           pursue               skip
              |
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
6. Feasibility = every protected goal's simulated satisfaction
   probability >= the context threshold AND a nonzero empirical outcome
   probability (§13 step 5). The latter is a Monte Carlo criterion -
   acknowledged sampling noise, not a proof of impossibility.
7. Evaluation runs the Phase 4 simulator (§14): carried pity/guarantee,
   actual spending decisions, income timing and multi-copy targets all
   count. No rate mathematics lives in this package.
8. Selection is lexicographic, never a global score (§2, §13): preference
   rank, then feasibility, then the largest feasible cap. The cap scan is
   exhaustive over the given caps - feasibility is not monotone in the
   cap (a lost 50/50 carried forward can protect a future goal), so no
   binary search.
9. Monte Carlo results are seed-deterministic; the Recommendation carries
   runs and seed so its probabilities are never mistaken for exact values
   (§2, §11 invariant 10).
10. When no outcome is feasible, the recommendation is "do not spend"
    with per-outcome rejections - never a fabricated strategy (§1, §13).
11. Stop conditions express the cap's real-world execution: stop on the
    outcome, never exceed the cap, re-run after updates (§1, §2).
12. Weapon refinement is displayed from preference labels but never
    simulated (§17, §19).
"""

from optimizer.candidates import candidate_plan
from optimizer.evaluation import (
    DEFAULT_RUNS,
    CandidateStrategy,
    GoalStanding,
    evaluate_candidate,
    evaluate_skip_baseline,
)
from optimizer.outcomes import OutcomeOption, available_outcomes
from optimizer.protection import ProtectedGroup, protected_groups
from optimizer.recommend import (
    Recommendation,
    RejectedOutcome,
    recommend,
)
from optimizer.stops import StopConditions, for_pursue, for_skip

__all__ = [
    "DEFAULT_RUNS",
    "CandidateStrategy",
    "GoalStanding",
    "OutcomeOption",
    "ProtectedGroup",
    "Recommendation",
    "RejectedOutcome",
    "StopConditions",
    "available_outcomes",
    "candidate_plan",
    "evaluate_candidate",
    "evaluate_skip_baseline",
    "for_pursue",
    "for_skip",
    "protected_groups",
    "recommend",
]
