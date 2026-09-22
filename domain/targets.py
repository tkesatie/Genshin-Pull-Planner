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


@dataclass(frozen=True, init=False)
class CharacterTarget(GoalTarget):
    """A character target."""

    def __init__(self, name: str) -> None:
        super().__init__(TargetKind.CHARACTER, name)


@dataclass(frozen=True, init=False)
class WeaponTarget(GoalTarget):
    """A weapon target."""

    def __init__(self, name: str) -> None:
        super().__init__(TargetKind.WEAPON, name)
