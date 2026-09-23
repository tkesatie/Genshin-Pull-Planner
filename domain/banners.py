"""Unified character/weapon banner representation (§7).

A banner is one featured opportunity to pull: a featured 5-star character, or
the featured weapon a weapon banner is epitomized toward. Simultaneous
opportunities - two character banners in one phase, or a character banner
next to a weapon banner - are separate `Banner` entries in the same
(version, phase) slot, never one banner with several targets; the planner
reads the whole slot through `planner.banners.available_banners`.

Dates (`start`/`end`) are optional and auxiliary: they describe when the
banner is live in real time, while (version, phase) stays the primary
deterministic planning input. See `Banner` for the timezone and interval
rules.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from domain.targets import CharacterTarget, GoalTarget, TargetKind
from domain.versions import position_key


def _validate_interval(start: datetime | None, end: datetime | None) -> None:
    """Enforce the date rules for a banner interval.

    Raises:
        ValueError: if only one bound is given, if a bound is naive (no UTC
            offset), or if the interval is not strictly ordered.
    """
    if (start is None) != (end is None):
        raise ValueError(
            "banner dates must be provided together: give both start and end "
            "or neither"
        )
    for label, value in (("start", start), ("end", end)):
        if value is None:
            continue
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(
                f"banner {label} must be timezone-aware (include a UTC "
                f"offset), got naive {value.isoformat()!r}"
            )
    if start is not None and end is not None and not start < end:
        raise ValueError(
            f"banner start must be before end, got {start.isoformat()} and "
            f"{end.isoformat()}"
        )



@dataclass(frozen=True, init=False)
class Banner:
    """A chronological opportunity to pull for a character or weapon.

    Attributes:
        target: the single featured target, character or weapon. One target
            per banner is deliberate: it mirrors what a banner actually
            offers and it is what the goal model and the probability engines
            reason about. A weapon banner's second featured 5-star is not
            modelled for the beta, because the planner pursues one
            designated weapon per banner.
        version: the version the banner runs in, e.g. "7.0".
        phase: 1-based phase within the version.
        start: when the banner goes live, or None when unknown.
        end: when the banner stops being live, or None when unknown.

    Dates, when supplied, must be timezone-aware. The planner compares
    instants, so it will not assume UTC or local time for a naive wall-clock
    value - that guess would silently shift every boundary. Any real offset
    is accepted (a client following a server's own zone, e.g. UTC+8, stores
    that offset); equal instants written with different offsets therefore
    compare as the same moment. The interval is half-open - `start`
    inclusive, `end` exclusive - so a hand-off instant belongs to exactly one
    banner and consecutive banners neither overlap nor leave a gap.

    Raises:
        TypeError: if neither a character nor a target is given, or both.
        ValueError: for a malformed version, a phase below 1, or dates that
            violate the rules above.
    """

    target: GoalTarget
    version: str
    phase: int = 1
    start: datetime | None = None
    end: datetime | None = None

    def __init__(
        self,
        character: str | None = None,
        version: str | None = None,
        phase: int = 1,
        *,
        target: GoalTarget | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> None:
        if version is None:
            raise TypeError("Banner requires version")
        if target is None:
            if character is None:
                raise TypeError("Banner requires target or character")
            target = CharacterTarget(character)
        elif character is not None:
            raise TypeError("provide target or character, not both")
        position_key(version, phase)
        _validate_interval(start, end)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

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
        """Numeric (major, minor, phase) position of this banner (§7)."""
        return position_key(self.version, self.phase)

    @property
    def has_dates(self) -> bool:
        """True when the banner's real-time interval is known."""
        return self.start is not None and self.end is not None

    def is_active_at(self, moment: datetime) -> bool:
        """Whether this banner is live at `moment` (`start <= moment < end`).

        Raises:
            ValueError: if the banner has no dates (its activity cannot be
                determined from a timestamp - the caller must know that
                rather than receive a guess), or if `moment` is naive.
        """
        if self.start is None or self.end is None:
            raise ValueError(
                f"banner {self.target.name} {self.version} phase {self.phase} "
                "has no dates; activity cannot be determined from a timestamp"
            )
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError(
                f"moment must be timezone-aware (include a UTC offset), got "
                f"naive {moment.isoformat()!r}"
            )
        return self.start <= moment < self.end


def sorted_chronologically(banners: Iterable[Banner]) -> list[Banner]:
    return sorted(banners, key=lambda banner: banner.order_key)
