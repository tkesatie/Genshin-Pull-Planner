"""SQLite-backed persistence for account records.

The repository stores the complete AccountRecord as JSON in SQLite. Domain
objects remain the source of validation; this layer is only responsible for
serialization and durable storage.
"""

import json
import sqlite3
import threading
from pathlib import Path

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    IncomeSource,
    Ownership,
    Preference,
    VersionIncome,
    WishMechanics,
)

from api.repository import AccountNotFound, AccountRecord, AccountRepository, PlannerSettings


def _estimate_to_dict(value: IncomeEstimate) -> dict:
    return {"low": value.low, "expected": value.expected, "high": value.high}


def _estimate_from_dict(value: dict) -> IncomeEstimate:
    return IncomeEstimate(value["low"], value["expected"], value["high"])


def _record_to_dict(record: AccountRecord) -> dict:
    settings = record.settings
    return {
        "id": record.id,
        "owner_id": record.owner_id,
        "label": record.label,
        "account": {
            "current_pity": record.account.current_pity,
            "character_guarantee": record.account.character_guarantee,
            "wishes": record.account.wishes,
            "capturing_radiance_counter": record.account.capturing_radiance_counter,
            "owned_characters": dict(record.account.owned_characters.characters),
        },
        "settings": {
            "current_version": settings.current_version,
            "current_phase": settings.current_phase,
            "confidence": settings.confidence,
            "income_scenario": settings.income_scenario,
            "mechanics": {
                "banner_type": settings.mechanics.banner_type,
                "hard_pity": settings.mechanics.hard_pity,
                "soft_pity_start": settings.mechanics.soft_pity_start,
                "base_rate": settings.mechanics.base_rate,
                "soft_pity_increment": settings.mechanics.soft_pity_increment,
                "featured_rate": settings.mechanics.featured_rate,
            },
        },
        "goals": [
            {"character": g.character, "constellation": g.constellation, "priority": g.priority}
            for g in record.goals
        ],
        "banners": [
            {"character": b.character, "version": b.version, "phase": b.phase}
            for b in record.banners
        ],
        "preferences": [
            {
                "character": p.character,
                "rank": p.rank,
                "constellation": p.constellation,
                "weapon_refinement": p.weapon_refinement,
                "notes": p.notes,
            }
            for p in record.preferences
        ],
        "income": None
        if record.income is None
        else {
            "versions": [
                {
                    "version": v.version,
                    "estimate": None if v.estimate is None else _estimate_to_dict(v.estimate),
                    "sources": [
                        {"name": s.name, "estimate": _estimate_to_dict(s.estimate)}
                        for s in v.sources
                    ],
                }
                for v in record.income.versions
            ]
        },
    }


def _record_from_dict(value: dict) -> AccountRecord:
    account = value["account"]
    raw_settings = value["settings"]
    raw_mechanics = raw_settings["mechanics"]
    mechanics = WishMechanics(**raw_mechanics)
    settings = PlannerSettings(
        current_version=raw_settings["current_version"],
        current_phase=raw_settings["current_phase"],
        confidence=raw_settings["confidence"],
        income_scenario=raw_settings["income_scenario"],
        mechanics=mechanics,
    )
    income_value = value.get("income")
    income = None
    if income_value is not None:
        income = IncomeForecast(
            versions=tuple(
                VersionIncome(
                    version=entry["version"],
                    estimate=(
                        None
                        if entry["estimate"] is None
                        else _estimate_from_dict(entry["estimate"])
                    ),
                    sources=[
                        IncomeSource(s["name"], _estimate_from_dict(s["estimate"]))
                        for s in entry["sources"]
                    ],
                )
                for entry in income_value["versions"]
            )
        )

    return AccountRecord(
        id=value["id"],
        owner_id=value.get("owner_id"),
        label=value["label"],
        account=Account(
            current_pity=account["current_pity"],
            character_guarantee=account["character_guarantee"],
            owned_characters=Ownership(dict(account["owned_characters"])),
            wishes=account["wishes"],
            capturing_radiance_counter=account.get("capturing_radiance_counter", 0),
        ),
        settings=settings,
        goals=tuple(
            Goal(g["character"], g["constellation"], g["priority"])
            for g in value["goals"]
        ),
        banners=tuple(
            Banner(b["character"], b["version"], b["phase"])
            for b in value["banners"]
        ),
        preferences=tuple(
            Preference(
                p["character"],
                p["rank"],
                p["constellation"],
                weapon_ref=p["weapon_refinement"],
                notes=p["notes"],
            )
            for p in value["preferences"]
        ),
        income=income,
    )


class SQLiteAccountRepository:
    """Durable AccountRepository backed by a local SQLite database."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT,
                    label TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(accounts)")}
            if "owner_id" not in columns:
                connection.execute("ALTER TABLE accounts ADD COLUMN owner_id TEXT")

    @staticmethod
    def _validate(record: AccountRecord) -> None:
        record.context()

    def create(self, record: AccountRecord) -> AccountRecord:
        self._validate(record)
        payload = json.dumps(_record_to_dict(record), separators=(",", ":"))
        with self._lock, self._connect() as connection:
            try:
                connection.execute(
                    "INSERT INTO accounts (id, owner_id, label, payload) VALUES (?, ?, ?, ?)",
                    (record.id, record.owner_id, record.label, payload),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"account {record.id!r} already exists") from exc
        return record

    def list(self) -> list[AccountRecord]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT payload FROM accounts ORDER BY rowid"
            ).fetchall()
        return [_record_from_dict(json.loads(row["payload"])) for row in rows]

    def get(self, account_id: str) -> AccountRecord:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM accounts WHERE id = ?", (account_id,)
            ).fetchone()
        if row is None:
            raise AccountNotFound(f"no account with id {account_id!r}")
        return _record_from_dict(json.loads(row["payload"]))

    def save(self, record: AccountRecord) -> AccountRecord:
        self._validate(record)
        payload = json.dumps(_record_to_dict(record), separators=(",", ":"))
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE accounts SET owner_id = ?, label = ?, payload = ? WHERE id = ?",
                (record.owner_id, record.label, payload, record.id),
            )
            if cursor.rowcount == 0:
                raise AccountNotFound(f"no account with id {record.id!r}")
        return record

    def delete(self, account_id: str) -> None:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM accounts WHERE id = ?", (account_id,)
            )
            if cursor.rowcount == 0:
                raise AccountNotFound(f"no account with id {account_id!r}")
            