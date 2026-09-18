"""Authentication and account ownership behavior."""

from fastapi.testclient import TestClient

from api import create_app


def test_register_login_me_and_logout():
    with TestClient(create_app()) as client:
        response = client.post("/auth/register", json={"username": "alice", "password": "password123"})
        assert response.status_code == 201
        assert response.json()["username"] == "alice"

        login = client.post("/auth/login", json={"username": "alice", "password": "password123"})
        assert login.status_code == 200

        me = client.get("/auth/me")
        assert me.status_code == 200
        assert me.json()["username"] == "alice"

        logout = client.post("/auth/logout")
        assert logout.status_code == 204
        assert client.get("/auth/me").status_code == 401


def test_duplicate_username_and_bad_login_have_auth_status_codes():
    with TestClient(create_app()) as client:
        assert client.post("/auth/register", json={"username": "alice", "password": "password123"}).status_code == 201
        assert client.post("/auth/register", json={"username": "alice", "password": "password123"}).status_code == 409
        assert client.post("/auth/login", json={"username": "alice", "password": "wrongpass"}).status_code == 401


def test_accounts_are_private_to_the_authenticated_user(doc_payload):
    app = create_app()
    with TestClient(app) as alice, TestClient(app) as bob:
        alice.post("/auth/register", json={"username": "alice", "password": "password123"})
        alice.post("/auth/login", json={"username": "alice", "password": "password123"})
        created = alice.post("/accounts", json=doc_payload)
        assert created.status_code == 201
        account_id = created.json()["id"]

        bob.post("/auth/register", json={"username": "bob", "password": "password123"})
        bob.post("/auth/login", json={"username": "bob", "password": "password123"})

        assert len(alice.get("/accounts").json()) == 1
        assert bob.get("/accounts").json() == []
        assert bob.get(f"/accounts/{account_id}").status_code == 404
        assert bob.delete(f"/accounts/{account_id}").status_code == 404
        assert alice.get(f"/accounts/{account_id}").status_code == 200

def test_account_endpoints_require_authentication(doc_payload):
    with TestClient(create_app()) as client:
        assert client.get("/accounts").status_code == 401
        assert client.post("/accounts", json=doc_payload).status_code == 401
        assert client.get("/accounts/missing").status_code == 401
