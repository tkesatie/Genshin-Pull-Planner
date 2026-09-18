"""Planner input and analysis helpers.

The planner may encounter multiple simultaneous banners at the current
version/phase. Use available_banners when making decisions; the strict
current_banner helper remains for legacy single-banner analysis.
"""

from planner.banners import available_banners, current_banner
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
    "available_banners",
    "current_banner",
    "evaluate_goal",
    "evaluate_goals",
    "protected_goal_outcomes",
    "relevant_goal_evaluations",
    "safe_spend",
    "single_copy_active_goal",
    "spend_table",
]
