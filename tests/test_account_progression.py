from fastapi.testclient import TestClient

from api.main import create_app


def _client_with_account():
    app = create_app()
    client = TestClient(app)
    assert client.post(
        "/auth/register",
        json={"username": "progress-user", "password": "password123"},
    ).status_code == 201
    assert client.post(
        "/auth/login",
        json={"username": "progress-user", "password": "password123"},
    ).status_code == 200
    created = client.post(
        "/accounts",
        json={
            "label": "Progression",
            "account": {
                "wishes": 450,
                "current_pity": 27,
                "character_guarantee": False,
                "owned_characters": {},
                "capturing_radiance_counter": 0,
            },
            "settings": {"current_version": "7.1", "current_phase": 1},
        },
    )
    assert created.status_code == 201, created.text
    return client, created.json()["id"]


def test_record_featured_pull_updates_account():
    client, account_id = _client_with_account()
    response = client.post(
        f"/accounts/{account_id}/pull-result",
        json={"outcome": "featured", "wishes_used": 63, "character": "Vesna"},
    )

    assert response.status_code == 200, response.text
    account = response.json()["account"]
    assert account["wishes"] == 387
    assert account["current_pity"] == 0
    assert account["character_guarantee"] is False
    assert account["owned_characters"]["Vesna"] == 0


def test_record_lost_50_50_updates_guarantee_and_radiance():
    client, account_id = _client_with_account()
    response = client.post(
        f"/accounts/{account_id}/pull-result",
        json={"outcome": "lost_50_50", "wishes_used": 74},
    )

    assert response.status_code == 200, response.text
    account = response.json()["account"]
    assert account["wishes"] == 376
    assert account["current_pity"] == 0
    assert account["character_guarantee"] is True
    assert account["capturing_radiance_counter"] == 1


def test_record_stopped_pull_advances_pity_without_changing_guarantee():
    client, account_id = _client_with_account()
    response = client.post(
        f"/accounts/{account_id}/pull-result",
        json={"outcome": "stopped", "wishes_used": 10},
    )

    assert response.status_code == 200, response.text
    account = response.json()["account"]
    assert account["wishes"] == 440
    assert account["current_pity"] == 37
    assert account["character_guarantee"] is False
