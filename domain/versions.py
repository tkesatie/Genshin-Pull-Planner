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
