"""Planner input model (Design Document §18 Phase 3).

The planner is told where the user is and what they assume; it never
mutates account or roadmap state. `current_version` / `current_phase`
describe the user's position in the banner schedule and are deliberately
planner input rather than `Account` fields (§4.1 defers version/phase to
the eventual persistent model).

Income interpretation (Phase 3 assumption, stated to prevent
double-counting later):

    IncomeForecast values represent wishes that become available AFTER
    the current account state - not total income associated with a
    version. Income forecast for the current version is treated as
    future income that still arrives; income forecast for versions
    before the current version is presumed already reflected in
    `account.wishes` and is never credited.
"""

from dataclasses import dataclass

from domain import (
    CHARACTER_EVENT_BANNER,
    SCENARIOS,
    Account,
    IncomeForecast,
    Roadmap,
    WishMechanics,
    parse_version,
)


@dataclass(frozen=True)
class PlannerContext:
    """Everything the Phase 3 planner is told (§18).

    Attributes:
        account: current account state (§4.1) - possessions and wishes.
        roadmap: goals and expected banners (§8) - user assumptions.
        current_version: version the user is currently in, e.g. "7.0".
        current_phase: 1-based phase within `current_version`.
        income: expected future income (§16), or None when untracked.
        income_scenario: which scenario ("low" / "expected" / "high")
            the planner evaluates (§16).
        confidence: required confidence threshold for protected goals
            (§1, §13 step 5).
        mechanics: mechanics data for the banner type (§17).

    Raises:
        ValueError: on a confidence outside (0, 1], an unknown income
            scenario, a malformed current version, a phase below 1, or
            account pity at or beyond hard pity.
    """

    account: Account
    roadmap: Roadmap
    current_version: str
    current_phase: int = 1
    income: IncomeForecast | None = None
    income_scenario: str = "expected"
    confidence: float = 0.9
    mechanics: WishMechanics = CHARACTER_EVENT_BANNER

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(f"confidence must be in (0, 1], got {self.confidence}")
        if self.income_scenario not in SCENARIOS:
            raise ValueError(
                f"unknown income scenario {self.income_scenario!r}; "
                f"expected one of {SCENARIOS}"
            )
        parse_version(self.current_version)
        if self.current_phase < 1:
            raise ValueError(f"current_phase must be >= 1, got {self.current_phase}")
        if not 0 <= self.account.current_pity < self.mechanics.hard_pity:
            raise ValueError(
                f"account.current_pity must satisfy 0 <= pity < hard_pity "
                f"({self.mechanics.hard_pity}), got {self.account.current_pity}"
            )

    def income_credit(self, up_to_version: str) -> int:
        """Future income credited from the current version through
        `up_to_version`, under the configured scenario (§16).

        Sums forecast versions `v` with current_version <= v <=
        up_to_version. Per the future-income interpretation in the
        module docstring, the current version's forecast is included
        (it still arrives) and earlier versions are excluded (already
        reflected in `account.wishes`). Returns 0 when no forecast is
        configured.

        Raises:
            ValueError: if `up_to_version` is malformed or
                chronologically earlier than the current version
                (negative credit is meaningless).
        """
        if self.income is None:
            return 0
        current = parse_version(self.current_version)
        limit = parse_version(up_to_version)
        if limit < current:
            raise ValueError(
                f"up_to_version {up_to_version!r} is earlier than the "
                f"current version {self.current_version!r}"
            )
        return sum(
            entry.aggregate.scenario(self.income_scenario)
            for entry in self.income.versions
            if current <= parse_version(entry.version) <= limit
        )