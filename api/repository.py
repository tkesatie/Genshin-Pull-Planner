"""Stored account records and their repository."""

import threading
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from domain import CHARACTER_EVENT_BANNER, Account, Banner, Goal, IncomeForecast, Preference, Roadmap, WishMechanics
from planner import PlannerContext


class AccountNotFound(LookupError):
    """Raised when a record does not exist or is not owned by the caller."""


@dataclass(frozen=True)
class PlannerSettings:
    current_version: str
    current_phase: int = 1
    confidence: float = 0.9
    income_scenario: str = "expected"
    mechanics: WishMechanics = CHARACTER_EVENT_BANNER


@dataclass(frozen=True)
class AccountRecord:
    id: str
    label: str
    account: Account
    settings: PlannerSettings
    goals: tuple[Goal, ...] = ()
    banners: tuple[Banner, ...] = ()
    preferences: tuple[Preference, ...] = ()
    income: IncomeForecast | None = None
    owner_id: str | None = None

    def roadmap(self) -> Roadmap:
        return Roadmap(goals=list(self.goals), banners=list(self.banners))

    def context(self, *, confidence: float | None = None, income_scenario: str | None = None) -> PlannerContext:
        return PlannerContext(
            account=self.account,
            roadmap=self.roadmap(),
            current_version=self.settings.current_version,
            current_phase=self.settings.current_phase,
            income=self.income,
            income_scenario=self.settings.income_scenario if income_scenario is None else income_scenario,
            confidence=self.settings.confidence if confidence is None else confidence,
            mechanics=self.settings.mechanics,
        )

    def preferences_for(self, character: str) -> tuple[Preference, ...]:
        return tuple(p for p in self.preferences if p.character == character)


def new_account_id() -> str:
    return uuid4().hex


class AccountRepository(Protocol):
    def create(self, record: AccountRecord) -> AccountRecord: ...
    def list(self, owner_id: str | None = None) -> list[AccountRecord]: ...
    def get(self, account_id: str) -> AccountRecord: ...
    def save(self, record: AccountRecord) -> AccountRecord: ...
    def delete(self, account_id: str) -> None: ...


@dataclass
class InMemoryAccountRepository:
    _records: dict[str, AccountRecord] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def _validate(record: AccountRecord) -> None:
        record.context()

    def create(self, record: AccountRecord) -> AccountRecord:
        self._validate(record)
        with self._lock:
            if record.id in self._records:
                raise ValueError(f"account {record.id!r} already exists")
            self._records[record.id] = record
        return record

    def list(self, owner_id: str | None = None) -> list[AccountRecord]:
        with self._lock:
            records = list(self._records.values())
        return records if owner_id is None else [r for r in records if r.owner_id == owner_id]

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
