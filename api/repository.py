"""Stored account records and their repository (Design Document §18 Phase 6).

An `AccountRecord` is the persistent aggregate the planner needs: account
state (§4.1) plus the roadmap sub-resources the user maintains - goals (§5),
banners (§7), preferences (§15), income (§16) - plus the planner input that
is neither account state nor roadmap (current version/phase, confidence,
income scenario, mechanics).

`Roadmap` (§8) and `PlannerContext` (Phase 3) are *derived* from a record,
never stored: they are views assembled per request, so a stored record can
never drift out of agreement with them.

Validation is deliberately delegated: `AccountRecord.context()` builds a
real `PlannerContext`, so every value rule the domain owns (pity below hard
pity, unique goal priorities, confidence in (0, 1], known income scenario,
well-formed versions) is enforced by the domain and reported with the
domain's own message. The repository validates a record by constructing its
context once at write time.

A record may legitimately reference a version with no banner scheduled: an
account exists before its roadmap does. That mismatch is a planner-time
error (`planner.current_banner`), not a storage error.

Storage is in-memory behind `AccountRepository`; a database implementation
drops in without touching routers (§18).
"""

import threading
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from domain import (
    CHARACTER_EVENT_BANNER,
    Account,
    Banner,
    Goal,
    IncomeForecast,
    Preference,
    Roadmap,
    WishMechanics,
)
from planner import PlannerContext


class AccountNotFound(LookupError):
    """Raised when a record does not exist; routers map this to 404."""


@dataclass(frozen=True)
class PlannerSettings:
    """Planner input that is neither account state nor roadmap (§18 Phase 3).

    `PlannerContext` documents why these are planner input rather than
    `Account` fields; storing them here is the persistence side of the same
    decision (§4.1: the eventual persistent model also holds version/phase).

    Attributes:
        current_version: version the user is currently in, e.g. "7.0".
        current_phase: 1-based phase within `current_version`.
        confidence: required confidence threshold for protected goals (§1).
        income_scenario: "low" / "expected" / "high" (§16).
        mechanics: mechanics data for the banner type (§17).
    """

    current_version: str
    current_phase: int = 1
    confidence: float = 0.9
    income_scenario: str = "expected"
    mechanics: WishMechanics = CHARACTER_EVENT_BANNER


@dataclass(frozen=True)
class AccountRecord:
    """One stored account and everything the planner is told about it.

    Attributes:
        id: server-assigned identifier.
        label: free-form user label for the account.
        account: current account state (§4.1).
        settings: planner input that is not account state (see above).
        goals: roadmap objectives (§5), in the order supplied.
        banners: expected banner schedule (§7).
        preferences: preference chains (§15) across all characters.
        income: expected future income (§16), or None when untracked.
    """

    id: str
    label: str
    account: Account
    settings: PlannerSettings
    goals: tuple[Goal, ...] = ()
    banners: tuple[Banner, ...] = ()
    preferences: tuple[Preference, ...] = ()
    income: IncomeForecast | None = None

    def roadmap(self) -> Roadmap:
        """The record's roadmap (§8).

        Raises:
            ValueError: if goal priorities are not unique (§8).
        """
        return Roadmap(goals=list(self.goals), banners=list(self.banners))

    def context(
        self,
        *,
        confidence: float | None = None,
        income_scenario: str | None = None,
    ) -> PlannerContext:
        """The planner context for this record (Phase 3).

        Per-request overrides replace the stored setting for this call only;
        nothing is written. Both are validated by `PlannerContext`.

        Raises:
            ValueError: via `Roadmap` or `PlannerContext` for any value the
                domain rejects.
        """
        return PlannerContext(
            account=self.account,
            roadmap=self.roadmap(),
            current_version=self.settings.current_version,
            current_phase=self.settings.current_phase,
            income=self.income,
            income_scenario=(
                self.settings.income_scenario
                if income_scenario is None
                else income_scenario
            ),
            confidence=(
                self.settings.confidence if confidence is None else confidence
            ),
            mechanics=self.settings.mechanics,
        )

    def preferences_for(self, character: str) -> tuple[Preference, ...]:
        """The stored preference chain for one character (§15)."""
        return tuple(
            preference
            for preference in self.preferences
            if preference.character == character
        )


def new_account_id() -> str:
    """A fresh opaque account identifier."""
    return uuid4().hex


class AccountRepository(Protocol):
    """Storage for account records (§18: infrastructure around the domain)."""

    def create(self, record: AccountRecord) -> AccountRecord:
        """Store a new record. Raises ValueError if the id is already used."""

    def list(self) -> list[AccountRecord]:
        """Every stored record, in creation order."""

    def get(self, account_id: str) -> AccountRecord:
        """One record. Raises AccountNotFound if it does not exist."""

    def save(self, record: AccountRecord) -> AccountRecord:
        """Replace an existing record. Raises AccountNotFound if absent."""

    def delete(self, account_id: str) -> None:
        """Remove a record. Raises AccountNotFound if absent."""


@dataclass
class InMemoryAccountRepository:
    """Process-local `AccountRepository` (§18 Phase 6).

    Records are validated on write by building their `PlannerContext`, so a
    stored record is always one the planner can be asked about. The lock
    keeps concurrent requests and background simulation jobs from observing
    a half-written map.
    """

    _records: dict[str, AccountRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _validate(record: AccountRecord) -> None:
        record.context()  # domain validation; ValueError propagates as 422

    def create(self, record: AccountRecord) -> AccountRecord:
        self._validate(record)
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"account {record.id!r} already exists")
            self._records[record.id] = record
        return record

    def list(self) -> list[AccountRecord]:
        with self._lock:
            return list(self._records.values())

    def get(self, account_id: str) -> AccountRecord:
        with self._lock:
            record = self._records.get(account_id)
        if record is None:
            raise AccountNotFound(f"no account with id {account_id!r}")
        return record

    def save(self, record: AccountRecord) -> AccountRecord:
        self._validate(record)
        with self._lock:
            if record.id not in self._records:
                raise AccountNotFound(f"no account with id {record.id!r}")
            self._records[record.id] = record
        return record

    def delete(self, account_id: str) -> None:
        with self._lock:
            if account_id not in self._records:
                raise AccountNotFound(f"no account with id {account_id!r}")
            del self._records[account_id]
