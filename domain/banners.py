"""Unified character/weapon banner representation."""

from collections.abc import Iterable
from dataclasses import dataclass

from domain.targets import CharacterTarget, GoalTarget, TargetKind
from domain.versions import parse_version


@dataclass(frozen=True, init=False)
class Banner:
    """A chronological opportunity to pull for a character or weapon."""

    target: GoalTarget
    version: str
    phase: int = 1

    def __init__(
        self,
        character: str | None = None,
        version: str | None = None,
        phase: int = 1,
        *,
        target: GoalTarget | None = None,
    ) -> None:
        if version is None:
            raise TypeError("Banner requires version")
        if target is None:
            if character is None:
                raise TypeError("Banner requires target or character")
            target = CharacterTarget(character)
        elif character is not None:
            raise TypeError("provide target or character, not both")
        parse_version(version)
        if phase < 1:
            raise ValueError(f"phase must be >= 1, got {phase}")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "phase", phase)

    @property
    def character(self) -> str:
        if self.target.kind is not TargetKind.CHARACTER:
            raise AttributeError("weapon banners do not have a character")
        return self.target.name

    @property
    def weapon(self) -> str:
        if self.target.kind is not TargetKind.WEAPON:
            raise AttributeError("character banners do not have a weapon")
        return self.target.name

    @property
    def order_key(self) -> tuple[int, int, int]:
        major, minor = parse_version(self.version)
        return (major, minor, self.phase)


def sorted_chronologically(banners: Iterable[Banner]) -> list[Banner]:
    return sorted(banners, key=lambda banner: banner.order_key)
