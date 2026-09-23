"""Planner input and analysis helpers.

The planner may encounter multiple simultaneous banners at the current
version/phase. Use available_banners when making decisions, upcoming_banners
and next_banner_for for later opportunities, and banners_active_at when a real
timestamp is available; the strict current_banner helper remains for legacy
single-banner analysis.
"""

from planner.banners import (
    available_banners,
    banners_active_at,
    current_banner,
    current_position,
    next_banner_for,
    position_anchor,
    upcoming_banners,
)
from planner.context import PlannerContext
from planner.goals import (
    GoalEvaluation,
    GoalState,
    actionable_goals,
    evaluate_goal,
    evaluate_goals,
    relevant_goal_evaluations,
)
from planner.projection import character_view
from planner.protection import (
    ProtectedGoalOutcome,
    protected_goal_outcomes,
    weapon_goal_confidence,
    weapon_goal_reserve,
)
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
    "banners_active_at",
    "character_view",
    "current_banner",
    "current_position",
    "evaluate_goal",
    "evaluate_goals",
    "next_banner_for",
    "position_anchor",
    "protected_goal_outcomes",
    "relevant_goal_evaluations",
    "safe_spend",
    "single_copy_active_goal",
    "spend_table",
    "upcoming_banners",
    "weapon_goal_confidence",
    "weapon_goal_reserve",
]
