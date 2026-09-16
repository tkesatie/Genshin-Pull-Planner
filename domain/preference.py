"""Preferences (Design Document §15).

Preferences answer "if I'm going after this character, what outcomes do I
prefer?". They are separate from goals (§5), which answer "where does this
objective sit in my roadmap?" - the two concepts must not be collapsed.

The `tier` field mentioned in §15 is intentionally omitted in Phase 1 until
its meaning is decided.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Preference:
    """One acceptable outcome for a character, ordered by the user (§15).

    Attributes:
        character: the character this preference is about.
        rank: position in the user's preference chain; 1 is most preferred.
        constellation: the wanted constellation ("C2" -> 2).
        weapon_refinement: wanted weapon refinement, following constellation
            numbering conventions: 0 means no refinement wanted (no "R" in
            the label), 1 means R1, and so on.
        notes: free-form user notes.
    """

    character: str
    rank: int
    constellation: int
    weapon_refinement: int = 0
    notes: str = ""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.constellation < 0:
            raise ValueError(f"constellation must be >= 0, got {self.constellation}")
        if self.weapon_refinement < 0:
            raise ValueError(
                f"weapon_refinement must be >= 0, got {self.weapon_refinement}"
            )

    @property
    def label(self) -> str:
        """Derived display label such as "C0", "C2" or "C2R1" (§15).

        Labels are derived for display and never stored as authoritative
        data (§15).
        """
        label = f"C{self.constellation}"
        if self.weapon_refinement > 0:
            label += f"R{self.weapon_refinement}"
        return label


def sort_by_rank(preferences: Iterable[Preference]) -> list[Preference]:
    """Return the preferences ordered by rank (rank 1 first).

    Ties keep their input order (stable sort). Full preference-chain
    evaluation is planner logic (§13, Phase 5) and is not attempted here.
    """
    return sorted(preferences, key=lambda preference: preference.rank)
