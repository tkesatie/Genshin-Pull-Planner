"""Goals and goal status (Design Document §5, §6).

A goal is a roadmap objective ("I want Vesna at C2"). Goal status is the raw
remaining work relative to the account ("2 copies remain"). Phase 1 does not
decide whether a goal should be pursued, whether it is actionable, or how
goals depend on each other (§9) - that is planner logic (Phase 3+).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from domain.account import Account


@dataclass(frozen=True)
class Goal:
    """A specific objective in the user's prioritized roadmap (§5).

    A goal contains no banner information (§7) and no preference information
    (§15). The same character may appear in several goals: Vesna C0 and
    Vesna C2 are separate objectives (§5).

    Attributes:
        character: the character the objective is about.
        constellation: the goal constellation ("C2" -> 2). Not a copy count.
        priority: position in the roadmap; lower value = protected first.
    """

    character: str
    constellation: int
    priority: int

    def __post_init__(self) -> None:
        if self.constellation < 0:
            raise ValueError(f"constellation must be >= 0, got {self.constellation}")
        if self.priority < 1:
            raise ValueError(f"priority must be >= 1, got {self.priority}")


@dataclass(frozen=True)
class GoalStatus:
    """The raw status of a goal relative to the account (§6).

    Answers "how much remains to satisfy this goal?" - nothing more. It does
    not decide whether the user should pursue the goal (§6), and it does not
    model goal dependencies such as Vesna C2 depending on Vesna C0 (§9).
    Dependency and actionability logic is added by the planner layer
    (Phase 3).

    Attributes:
        goal: the goal being measured.
        copies_needed: featured copies still required, >= 0. The Phase 4
            simulator will consume this as its target copies (§10.4); a
            constellation is not a number of copies (§4.2).
    """

    goal: Goal
    copies_needed: int


def copies_needed_for(account: Account, goal: Goal) -> int:
    """Raw remaining copies (§6): max(goal_constellation - owned_constellation, 0)."""
    owned_constellation = account.owned_characters.owned_constellation(goal.character)
    return max(goal.constellation - owned_constellation, 0)


def goal_status(account: Account, goal: Goal) -> GoalStatus:
    """Compute the GoalStatus for one goal against the account (§6)."""
    return GoalStatus(goal=goal, copies_needed=copies_needed_for(account, goal))


def goal_statuses(account: Account, goals: Iterable[Goal]) -> list[GoalStatus]:
    """Compute GoalStatus for each goal, preserving input order."""
    return [goal_status(account, goal) for goal in goals]
