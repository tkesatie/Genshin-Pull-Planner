"""Domain model for the Genshin Pull Strategy Planner (Phase 1).

Phase 1 describes the state of the problem (account, goals, roadmap,
preferences, income, mechanics); it does not solve it. Probability (Phase 2),
actionable goals and dependencies (Phase 3), simulation (Phase 4) and
strategy optimization (Phase 5) build on this layer.

All types are plain frozen dataclasses holding simple data, so serialization
can be added later wherever it is actually needed without reshaping them.
"""

from domain.account import NOT_OWNED, Account, Ownership
from domain.banners import Banner, sorted_chronologically
from domain.goals import Goal, GoalStatus, copies_needed_for, goal_status, goal_statuses
from domain.income import (
    SCENARIOS,
    IncomeEstimate,
    IncomeForecast,
    IncomeSource,
    VersionIncome,
)
from domain.mechanics import CHARACTER_EVENT_BANNER, WishMechanics
from domain.preference import Preference, sort_by_rank
from domain.roadmap import Roadmap
from domain.versions import parse_version

__all__ = [
    "NOT_OWNED",
    "SCENARIOS",
    "CHARACTER_EVENT_BANNER",
    "Account",
    "Banner",
    "Goal",
    "GoalStatus",
    "IncomeEstimate",
    "IncomeForecast",
    "IncomeSource",
    "Ownership",
    "Preference",
    "Roadmap",
    "VersionIncome",
    "WishMechanics",
    "copies_needed_for",
    "goal_status",
    "goal_statuses",
    "parse_version",
    "sort_by_rank",
    "sorted_chronologically",
]
