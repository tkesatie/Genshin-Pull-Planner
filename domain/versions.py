"""Shared version-string parsing (Design Document §7, §16).

Versions look like "7.0". Numeric parsing matters because chronological
order is fundamental to the planner, and plain string comparison would
mis-order versions such as "7.10" and "7.9".
"""

import re

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)$")


def parse_version(version: str) -> tuple[int, int]:
    """Parse "7.0" into (7, 0).

    Raises:
        ValueError: if the version is not in "<major>.<minor>" form.
    """
    match = _VERSION_PATTERN.match(version)
    if match is None:
        raise ValueError(f"version must look like '7.0', got {version!r}")
    return int(match.group(1)), int(match.group(2))


def position_key(version: str, phase: int) -> tuple[int, int, int]:
    """Numeric ordering key for a roadmap position (§7): (major, minor, phase).

    A position is a version plus a 1-based phase within it. Both version
    numbers are compared numerically, so "7.9" precedes "7.10", and the phase
    breaks ties inside a version, so 7.0 phase 1 precedes 7.0 phase 2. This is
    the one place the two components are composed into a comparable key;
    banner ordering and position comparisons both use it rather than
    re-deriving the tuple (or, worse, comparing version strings).

    Raises:
        ValueError: if the version is malformed or the phase is below 1.
    """
    if phase < 1:
        raise ValueError(f"phase must be >= 1, got {phase}")
    major, minor = parse_version(version)
    return (major, minor, phase)

