"""Income (Design Document §16).

Future income is modeled as a range (Low / Expected / High), associated with
future versions. Sources (Commissions, Events, Abyss, Exploration) are
optional supporting detail; the simulation engine will use the aggregate
forecast (§16).
"""

from dataclasses import dataclass, field

from domain.versions import parse_version

SCENARIOS: tuple[str, ...] = ("low", "expected", "high")


@dataclass(frozen=True)
class IncomeEstimate:
    """A ranged wish estimate for one period or source (§16)."""

    low: int
    expected: int
    high: int

    def __post_init__(self) -> None:
        if min(self.low, self.expected, self.high) < 0:
            raise ValueError("income estimates must be non-negative")
        if not self.low <= self.expected <= self.high:
            raise ValueError(
                "income estimates must satisfy low <= expected <= high, "
                f"got low={self.low}, expected={self.expected}, high={self.high}"
            )

    def scenario(self, name: str) -> int:
        """Return the value under a named scenario ("low", "expected" or "high")."""
        if name not in SCENARIOS:
            raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIOS}")
        return getattr(self, name)

    def __add__(self, other: "IncomeEstimate") -> "IncomeEstimate":
        """Combine two estimates scenario-wise (used to aggregate sources)."""
        return IncomeEstimate(
            low=self.low + other.low,
            expected=self.expected + other.expected,
            high=self.high + other.high,
        )


@dataclass(frozen=True)
class IncomeSource:
    """Optional income breakdown by source (§16), e.g. Commissions or Events."""

    name: str
    estimate: IncomeEstimate


@dataclass(frozen=True)
class VersionIncome:
    """Income expected for one version (§16).

    Provide either `estimate` directly or `sources`; when `sources` are
    given, the aggregate is derived as their sum. If both are provided they
    must agree, because there is only one aggregate forecast per version.
    """

    version: str
    estimate: IncomeEstimate | None = None
    sources: list[IncomeSource] = field(default_factory=list)

    def __post_init__(self) -> None:
        parse_version(self.version)
        if self.estimate is None and not self.sources:
            raise ValueError("provide an estimate or at least one source")
        if self.estimate is not None and self.sources:
            derived = sum((s.estimate for s in self.sources), IncomeEstimate(0, 0, 0))
            if self.estimate != derived:
                raise ValueError(
                    f"explicit estimate {self.estimate} does not match the sum "
                    f"of sources {derived}"
                )

    @property
    def aggregate(self) -> IncomeEstimate:
        """The aggregate forecast for this version (§16)."""
        if self.estimate is not None:
            return self.estimate
        return sum((s.estimate for s in self.sources), IncomeEstimate(0, 0, 0))


@dataclass(frozen=True)
class IncomeForecast:
    """Income expected across future versions (§16).

    Version ordering is numeric ("7.10" is after "7.9"), and versions must
    be unique within the forecast.
    """

    versions: list[VersionIncome] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for entry in self.versions:
            parse_version(entry.version)
            if entry.version in seen:
                raise ValueError(f"duplicate version {entry.version!r} in forecast")
            seen.add(entry.version)

    def in_version_order(self) -> list[VersionIncome]:
        """Entries sorted chronologically by version."""
        return sorted(self.versions, key=lambda entry: parse_version(entry.version))

    def for_version(self, version: str) -> VersionIncome:
        """The entry for one version.

        Raises:
            KeyError: if no entry exists for the version.
        """
        for entry in self.versions:
            if entry.version == version:
                return entry
        raise KeyError(f"no income recorded for version {version!r}")

    def cumulative_through(self, version: str, scenario: str = "expected") -> int:
        """Total income under `scenario` for all versions up to and including
        `version`.

        This is the aggregate income a simulation will be able to add
        between banners (§11, §16).
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
        limit = parse_version(version)
        return sum(
            entry.aggregate.scenario(scenario)
            for entry in self.versions
            if parse_version(entry.version) <= limit
        )
