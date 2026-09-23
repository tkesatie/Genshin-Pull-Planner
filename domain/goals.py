"""Goals and goal status.

Phase 2 makes goals target-agnostic: the same Goal type can represent a
character constellation or a weapon refinement. Character properties remain
as compatibility accessors for the existing character planner.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from domain.account import Account
from domain.targets import CharacterTarget, GoalTarget, TargetKind


@dataclass(frozen=True, init=False)
class Goal:
    """A prioritized objective for either a character or weapon."""

    target: GoalTarget
    level: int
    priority: int

    def __init__(
        self,
        character: str | None = None,
        constellation: int | None = None,
        priority: int = 1,
        *,
        target: GoalTarget | None = None,
        level: int | None = None,
    ) -> None:
        if target is None:
            if character is None:
                raise TypeError("Goal requires target or character")
            if constellation is None:
                raise TypeError("character goals require constellation")
            target = CharacterTarget(character)
            level = constellation if level is None else level
        elif level is None:
            if constellation is not None:
                level = constellation
            else:
                raise TypeError("Goal requires level")
        elif constellation is not None:
            raise TypeError("provide level or constellation, not both")

        if level is None or level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        if priority < 1:
            raise ValueError(f"priority must be >= 1, got {priority}")

        object.__setattr__(self, "target", target)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "priority", priority)

    @property
    def character(self) -> str:
        if self.target.kind is not TargetKind.CHARACTER:
            raise AttributeError("weapon goals do not have a character")
        return self.target.name

    @property
    def constellation(self) -> int:
        if self.target.kind is not TargetKind.CHARACTER:
            raise AttributeError("weapon goals do not have a constellation")
        return self.level

    @property
    def weapon(self) -> str:
        if self.target.kind is not TargetKind.WEAPON:
            raise AttributeError("character goals do not have a weapon")
        return self.target.name

    @property
    def refinement(self) -> int:
        if self.target.kind is not TargetKind.WEAPON:
            raise AttributeError("character goals do not have a refinement")
        return self.level

    @property
    def label(self) -> str:
        """Display label for the goal: "Vesna C2" or "Wolf Fang R1".

        Target-agnostic: it names whichever level the goal uses, so
        formatting code that must describe an arbitrary goal (error
        messages, evidence summaries) never reaches for the
        character-only accessors.
        """
        if self.target.kind is TargetKind.WEAPON:
            return f"{self.target.name} R{self.level}"
        return f"{self.target.name} C{self.level}"


@dataclass(frozen=True)
class GoalStatus:
    """Raw remaining work for one goal."""

    goal: Goal
    copies_needed: int


def copies_needed_for(account: Account, goal: Goal) -> int:
    if goal.target.kind is TargetKind.CHARACTER:
        owned = account.owned_characters.owned_constellation(goal.target.name)
    else:
        owned = account.owned_characters.owned_refinement(goal.target.name)
        # An unowned weapon (-1) requires one copy to become R1.
        # Once owned, refinement levels are R0..R5 and each additional
        # refinement requires one more copy.
        if owned < 0:
            return max(goal.level, 1)
    return max(goal.level - owned, 0)


def goal_status(account: Account, goal: Goal) -> GoalStatus:
    return GoalStatus(goal=goal, copies_needed=copies_needed_for(account, goal))


def goal_statuses(account: Account, goals: Iterable[Goal]) -> list[GoalStatus]:
    return [goal_status(account, goal) for goal in goals]
