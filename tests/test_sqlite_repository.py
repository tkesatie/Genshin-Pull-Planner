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
    base = create_demo_account()
    record = base.__class__(
        id=base.id,
        label=base.label,
        account=base.account,
        settings=base.settings,
        goals=base.goals,
        banners=base.banners,
        preferences=base.preferences,
        income=base.income,
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

def test_sqlite_api_can_create_authenticated_account_with_full_payload(tmp_path, doc_payload):
    from fastapi.testclient import TestClient

    from api import create_app

    with TestClient(create_app(database_path=tmp_path / "planner.sqlite3")) as client:
        registered = client.post(
            "/auth/register",
            json={"username": "sqlite-user", "password": "password123"},
        )
        assert registered.status_code == 201, registered.text

        login = client.post(
            "/auth/login",
            json={"username": "sqlite-user", "password": "password123"},
        )
        assert login.status_code == 200, login.text

        created = client.post("/accounts", json=doc_payload)
        assert created.status_code == 201, created.text

        account_id = created.json()["id"]
        loaded = client.get(f"/accounts/{account_id}")
        assert loaded.status_code == 200, loaded.text
        assert loaded.json()["label"] == "doc example"
        assert loaded.json()["preferences"][0]["constellation"] == 2

