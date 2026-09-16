"""Spending table (Design Document §18 Phase 3).

For each candidate spend on the current banner, one row shows the
probability of achieving the active goal's copy and what that spend does
to every protected future goal (§1: spending more now is an explicit
tradeoff against future probability).

Phase 3 restriction (§18 "initially restrict this to simpler single-copy
scenarios"): the table requires exactly one ACTIVE goal matching the
current banner character, and that goal must require exactly one copy.
Multi-copy targets (§10.4) are Phase 4 and are rejected here rather than
approximated.
"""

from dataclasses import dataclass

from planner.context import PlannerContext
from planner.goals import GoalEvaluation, actionable_goals
from planner.protection import ProtectedGoalOutcome, protected_goal_outcomes
from probability import cumulative_probability


@dataclass(frozen=True)
class SpendRow:
    """One candidate spend on the current banner.

    Attributes:
        wishes_spent: the candidate spend.
        goal_confidence: P(active goal's copy within wishes_spent), from
            the account's actual current pity/guarantee state (§10.2).
        protected: each protected goal's outcome under exactly this spend
            (§13 step 5).
        all_protected_meet_threshold: True when no protected goal falls
            below the confidence threshold. Under the Phase 3
            approximation this is equivalent to
            wishes_spent <= safe_spend (planner.safe_spend).
    """

    wishes_spent: int
    goal_confidence: float
    protected: tuple[ProtectedGoalOutcome, ...]
    all_protected_meet_threshold: bool


def single_copy_active_goal(context: PlannerContext) -> GoalEvaluation:
    """The one active, single-copy goal matching the current banner.

    Raises:
        ValueError: if zero active goals match the current banner
            character, if more than one does (degenerate duplicate
            goals; the planner will not silently choose), or if the goal
            requires more than one copy (Phase 4 multi-copy territory).
    """
    active = actionable_goals(context)
    if not active:
        raise ValueError(
            "no active goal matches the current banner; the spend table "
            "evaluates spending toward an actionable goal (§9)"
        )
    if len(active) > 1:
        listed = ", ".join(
            f"{evaluation.goal.character} C{evaluation.goal.constellation} "
            f"(priority {evaluation.goal.priority})"
            for evaluation in active
        )
        raise ValueError(
            "ambiguous roadmap: multiple active goals match the current "
            f"banner ({listed}); the planner will not silently choose"
        )
    evaluation = active[0]
    if evaluation.copies_needed > 1:
        raise ValueError(
            f"goal {evaluation.goal.character} "
            f"C{evaluation.goal.constellation} requires "
            f"{evaluation.copies_needed} copies; the Phase 3 spend table "
            "supports single-copy goals only (§18, §10.4)"
        )
    return evaluation


def spend_table(context: PlannerContext, step: int = 1) -> list[SpendRow]:
    """Candidate spends from 0 to account wishes, in `step` increments.

    Each row is independent: the goal confidence comes from the
    single-copy curve at the account's current state (computed once and
    indexed), and protection is evaluated with exactly that much spent.

    Raises:
        ValueError: if step < 1, or via `single_copy_active_goal` when
            the current banner does not have exactly one active
            single-copy goal.
    """
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    single_copy_active_goal(context)

    account = context.account
    curve = cumulative_probability(
        account.wishes,
        account.current_pity,
        account.character_guarantee,
        context.mechanics,
    )
    rows: list[SpendRow] = []
    for spend in range(0, account.wishes + 1, step):
        protected = protected_goal_outcomes(context, spent=spend)
        rows.append(
            SpendRow(
                wishes_spent=spend,
                goal_confidence=float(curve[spend]),
                protected=tuple(protected),
                all_protected_meet_threshold=all(
                    outcome.meets_threshold for outcome in protected
                ),
            )
        )
    return rows