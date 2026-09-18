"""Tests for durable account persistence."""

from api.demo_data import create_demo_account
from api.sqlite_repository import SQLiteAccountRepository


def test_sqlite_repository_round_trips_account(tmp_path):
    path = tmp_path / "planner.sqlite3"
    first = SQLiteAccountRepository(path)
    record = create_demo_account()
    first.create(record)

    second = SQLiteAccountRepository(path)
    loaded = second.get(record.id)

    assert loaded == record
    assert second.list() == [record]


def test_sqlite_repository_round_trips_owner_id(tmp_path):
    repository = SQLiteAccountRepository(tmp_path / "planner.sqlite3")
    record = create_demo_account().__class__(
        id=create_demo_account().id,
        label=create_demo_account().label,
        account=create_demo_account().account,
        settings=create_demo_account().settings,
        goals=create_demo_account().goals,
        banners=create_demo_account().banners,
        preferences=create_demo_account().preferences,
        income=create_demo_account().income,
        owner_id="user-123",
    )
    repository.create(record)
    assert repository.get(record.id).owner_id == "user-123"
    assert repository.list("user-123") == [record]
    assert repository.list("other-user") == []


def test_sqlite_repository_save_and_delete(tmp_path):
    repository = SQLiteAccountRepository(tmp_path / "planner.sqlite3")
    record = create_demo_account()
    repository.create(record)

    updated = record.__class__(
        id=record.id,
        label="Updated account",
        account=record.account,
        settings=record.settings,
        goals=record.goals,
        banners=record.banners,
        preferences=record.preferences,
        income=record.income,
    )
    repository.save(updated)

    assert repository.get(record.id).label == "Updated account"

    repository.delete(record.id)
    assert repository.list() == []
