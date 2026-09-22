"""Common target types for character and weapon goals/banners."""

from dataclasses import dataclass
from enum import StrEnum


class TargetKind(StrEnum):
    CHARACTER = "character"
    WEAPON = "weapon"


@dataclass(frozen=True)
class GoalTarget:
    """A pullable target identified by kind and stable name."""

    kind: TargetKind
    name: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("target name must not be empty")


@dataclass(frozen=True)
class CharacterTarget(GoalTarget):
    """A character target."""

    kind: TargetKind = TargetKind.CHARACTER


@dataclass(frozen=True)
class WeaponTarget(GoalTarget):
    """A weapon target."""

    kind: TargetKind = TargetKind.WEAPON
