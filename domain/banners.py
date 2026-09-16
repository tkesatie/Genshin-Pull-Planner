"""Banners (Design Document §7).

A banner is a chronological opportunity to pull. Banner order and priority
order are deliberately allowed to differ (§7).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from domain.versions import parse_version


@dataclass(frozen=True)
class Banner:
    """An opportunity to pull for a character (§7).

    Attributes:
        character: the featured character.
        version: game version in "<major>.<minor>" form, e.g. "7.0".
        phase: 1-based phase within the version.
    """

    character: str
    version: str
    phase: int = 1

    def __post_init__(self) -> None:
        parse_version(self.version)  # validates the format
        if self.phase < 1:
            raise ValueError(f"phase must be >= 1, got {self.phase}")

    @property
    def order_key(self) -> tuple[int, int, int]:
        """Chronological sort key: (major, minor, phase)."""
        major, minor = parse_version(self.version)
        return (major, minor, self.phase)


def sorted_chronologically(banners: Iterable[Banner]) -> list[Banner]:
    """Return banners in chronological order (§7): version first, then phase."""
    return sorted(banners, key=lambda banner: banner.order_key)
