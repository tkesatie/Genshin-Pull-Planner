"""Phase 5 acceptance tests for unified account and goal management."""

from api.sqlite_repository import SQLiteAccountRepository
from api.demo_data import create_demo_account


def test_account_api_accepts_and_returns_weapon_state_and_ownership(api_client):
    payload = {
        "label": "weapon account",
        "account": {
            "current_pity": 12,
            "character_guarantee": True,
            "wishes": 250,
            "capturing_radiance_counter": 2,
            "owned_characters": {"Skirk": 0},
            "owned_weapons": {"Existing Weapon": 1},
            "weapon_state": {
                "pity": 17,
                "guarantee": True,
                "fate_points": 1,
            },
        },
        "settings": {"current_version": "7.1", "current_phase": 1},
    }
    response = api_client.post("/accounts", json=payload)
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["account"]["owned_weapons"] == {"Existing Weapon": 1}
    assert body["account"]["weapon_state"] == {
        "pity": 17,
        "guarantee": True,
        "fate_points": 1,
    }

    updated = api_client.put(
        f"/accounts/{body['id']}",
        json={
            "label": "edited",
            "account": {
                "current_pity": 22,
                "character_guarantee": False,
                "wishes": 180,
                "capturing_radiance_counter": 3,
                "owned_characters": {"Skirk": 1},
                "owned_weapons": {"Existing Weapon": 2, "New Weapon": -1},
                "weapon_state": {"pity": 31, "guarantee": False, "fate_points": 0},
            },
            "settings": {"current_version": "7.1", "current_phase": 1},
        },
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["account"]["weapon_state"]["pity"] == 31
    assert updated.json()["account"]["owned_weapons"]["Existing Weapon"] == 2


def test_unified_goals_accept_character_and_weapon_targets_and_status(api_client):
    payload = {
        "label": "goals",
        "account": {
            "owned_characters": {"Skirk": 0},
            "owned_weapons": {"Existing Weapon": 0},
            "wishes": 100,
        },
        "settings": {"current_version": "7.1", "current_phase": 1},
        "goals": [
            {"character": "Skirk", "constellation": 2, "priority": 1},
            {
                "target_kind": "weapon",
                "target_name": "Signature Weapon",
                "level": 1,
                "priority": 2,
            },
        ],
        "banners": [
            {"character": "Skirk", "version": "7.1", "phase": 1},
            {
                "target_kind": "weapon",
                "target_name": "Signature Weapon",
                "version": "7.1",
                "phase": 1,
            },
        ],
    }
    created = api_client.post("/accounts", json=payload)
    assert created.status_code == 201, created.text
    account_id = created.json()["id"]

    goals = api_client.get(f"/accounts/{account_id}/goals").json()["goals"]
    assert goals[0]["target_kind"] == "character"
    assert goals[0]["target_name"] == "Skirk"
    assert goals[0]["level"] == 2
    assert goals[0]["constellation"] == 2
    assert goals[1]["target_kind"] == "weapon"
    assert goals[1]["target_name"] == "Signature Weapon"
    assert goals[1]["level"] == 1
    assert goals[1]["refinement"] == 1

    status = api_client.get(f"/accounts/{account_id}/goals/status")
    assert status.status_code == 200, status.text
    statuses = status.json()["goals"]
    assert statuses[0]["status"] == "active"
    assert statuses[0]["copies_needed"] == 2
    assert statuses[1]["status"] == "active"
    assert statuses[1]["copies_needed"] == 1

    completed = api_client.put(
        f"/accounts/{account_id}",
        json={
            "label": "goals",
            "account": {
                "owned_characters": {"Skirk": 2},
                "owned_weapons": {"Existing Weapon": 0},
                "wishes": 100,
            },
            "settings": {"current_version": "7.1", "current_phase": 1},
        },
    )
    assert completed.status_code == 200, completed.text
    completed_status = api_client.get(f"/accounts/{account_id}/goals/status").json()["goals"]
    assert completed_status[0]["status"] == "satisfied"
    assert completed_status[0]["copies_needed"] == 0


def test_unowned_weapon_r1_requires_one_copy():
    from domain import Account, Goal, Ownership, WeaponTarget, copies_needed_for

    account = Account(owned_characters=Ownership(weapons={"Signature Weapon": -1}))
    assert copies_needed_for(
        account, Goal(target=WeaponTarget("Signature Weapon"), level=1, priority=1)
    ) == 1


def test_goal_priority_edit_and_deletion_are_persisted(api_client, doc_account_id):
    response = api_client.put(
        f"/accounts/{doc_account_id}/goals",
        json={
            "goals": [
                {"character": "Vesna", "constellation": 0, "priority": 3},
                {
                    "target_kind": "weapon",
                    "target_name": "Test Weapon",
                    "level": 1,
                    "priority": 1,
                },
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert [goal["priority"] for goal in response.json()["goals"]] == [1, 3]

    deleted = api_client.delete(f"/accounts/{doc_account_id}/goals/1")
    assert deleted.status_code == 204
    remaining = api_client.get(f"/accounts/{doc_account_id}/goals").json()["goals"]
    assert len(remaining) == 1
    assert remaining[0]["target_name"] == "Vesna"


def test_sqlite_round_trips_weapon_account_and_unified_roadmap(tmp_path):
    repository = SQLiteAccountRepository(tmp_path / "planner.sqlite3")
    base = create_demo_account()
    from dataclasses import replace
    from domain import Goal, WeaponTarget, WeaponWishState, Ownership, Banner

    record = replace(
        base,
        account=replace(
            base.account,
            owned_characters=Ownership(
                {"Vodynista": 0},
                {"Signature Weapon": 1},
            ),
            weapon_state=WeaponWishState(pity=23, guarantee=True, fate_points=1),
        ),
        goals=(
            Goal("Vesna", 0, 1),
            Goal(target=WeaponTarget("Signature Weapon"), level=1, priority=2),
        ),
        banners=(
            Banner("Vesna", "7.1", 1),
            Banner(target=WeaponTarget("Signature Weapon"), version="7.1", phase=1),
        ),
    )
    repository.create(record)
    assert repository.get(record.id) == record
