"""Basic planner (Design Document §18 Phase 3).

Connects the account (§4), the roadmap (§8) and the current banner (§7)
into the first planner outputs: current-banner identification, goal
matching and dependencies (§9), basic protected-goal calculation (§13),
a safe-spend approximation and a spending table (§14).

Phase 3 invariants:

1. The planner consumes domain state; it never mutates Account, Roadmap,
   or other domain objects.
2. All probability comes from the Phase 2 engine; no rate mathematics
   lives here.
3. Goal dependencies are constellation-based on the same character (§9)
   and deliberately independent of priorities (§2).
4. Protected goals are unsatisfied goals whose character's next banner
   is strictly after the current banner.
5. Protection uses the sequential independent-reserve approximation:
   each protected goal gets a full reserve at pity 0 / no guarantee and
   each reserve is consumed completely before the next goal. §14 forbids
   this from becoming a permanent architectural assumption.
6. Future income is interpreted as income arriving after the current
   account state, including income forecast for the current version;
   forecast for earlier versions is presumed banked and never credited.
7. safe_spend is equivalent to the protection model only under
   invariant 5's assumptions; Phase 4 breaks the equivalence. When even
   zero spending cannot protect the roadmap, safe_spend is 0 and every
   spend is unsafe - "do not spend" is a legitimate answer (§1).
8. The spend table requires exactly one active goal matching the current
   banner character, requiring exactly one copy; multi-copy targets are
   Phase 4 and are rejected, not approximated (§10.4, §18).
9. Goals that cannot be scheduled (no upcoming banner) remain visible
   through evaluate_goals() - "not protectable" is not "does not exist"
   (§8).
10. Preference chains, recommendations and stop conditions are Phase 5
    (§15, §13) and deliberately absent here.
"""

from planner.banners import current_banner
from planner.context import PlannerContext
from planner.goals import (
    GoalEvaluation,
    GoalState,
    actionable_goals,
    evaluate_goal,
    evaluate_goals,
    relevant_goal_evaluations,
)
from planner.protection import ProtectedGoalOutcome, protected_goal_outcomes
from planner.safe_spend import safe_spend
from planner.spend_table import SpendRow, single_copy_active_goal, spend_table

__all__ = [
    "GoalEvaluation",
    "GoalState",
    "PlannerContext",
    "ProtectedGoalOutcome",
    "SpendRow",
    "actionable_goals",
    "current_banner",
    "evaluate_goal",
    "evaluate_goals",
    "protected_goal_outcomes",
    "relevant_goal_evaluations",
    "safe_spend",
    "single_copy_active_goal",
    "spend_table",
]