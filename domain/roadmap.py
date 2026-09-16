"""Roadmap (Design Document §8)."""

from dataclasses import dataclass, field

from domain.banners import Banner, sorted_chronologically
from domain.goals import Goal


@dataclass(frozen=True)
class Roadmap:
    """The user's objectives and expected banner schedule (§8).

    The roadmap represents user assumptions, not guaranteed future game
    information (§8).

    Goal priorities must be unique within a roadmap: the priority list is a
    strict protection order (§2), while the same character may legitimately
    appear in several goals (§5).
    """

    goals: list[Goal] = field(default_factory=list)
    banners: list[Banner] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[int] = set()
        duplicates: set[int] = set()
        for goal in self.goals:
            if goal.priority in seen:
                duplicates.add(goal.priority)
            seen.add(goal.priority)
        if duplicates:
            raise ValueError(
                "goal priorities must be unique within a roadmap; "
                f"duplicates: {sorted(duplicates)}"
            )

    def goals_in_priority_order(self) -> list[Goal]:
        """Goals sorted by priority (priority 1 first)."""
        return sorted(self.goals, key=lambda goal: goal.priority)

    def banners_in_chronological_order(self) -> list[Banner]:
        """Banners sorted chronologically (§7): version first, then phase."""
        return sorted_chronologically(self.banners)

    def banners_for(self, character: str) -> list[Banner]:
        """All banners featuring the character, in chronological order."""
        return [
            banner
            for banner in self.banners_in_chronological_order()
            if banner.character == character
        ]
