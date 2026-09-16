"""The account resource over HTTP (§18 Phase 6).

The account is the aggregate the planner is told about (§4.1, §8), so these
tests care about two things: the aggregate survives a round trip intact, and
value rules are still the domain's - a request that violates one comes back
as 422 carrying the domain's own message, not a rule restated in the API.
"""


class TestCreateAndRead:
    def test_round_trips_the_whole_aggregate(self, api_client, doc_payload):
        created = api_client.post("/accounts", json=doc_payload)
        assert created.status_code == 201
        body = created.json()

        assert body["label"] == "doc example"
        assert body["account"]["wishes"] == 40
        assert body["account"]["owned_characters"] == {"Vodynista": 0}
        assert len(body["goals"]) == 4
        assert len(body["banners"]) == 3
        assert len(body["preferences"]) == 5
        assert body["income"] is None

        read = api_client.get(f"/accounts/{body['id']}")
        assert read.status_code == 200
        assert read.json() == body

    def test_settings_default_to_the_character_event_banner(
        self, api_client, doc_payload
    ):
        """Mechanics are data (§17) and default to the character banner."""
        body = api_client.post("/accounts", json=doc_payload).json()
        mechanics = body["settings"]["mechanics"]
        assert mechanics["banner_type"] == "character_event"
        assert mechanics["hard_pity"] == 90
        assert mechanics["featured_rate"] == 0.5
        assert body["settings"]["confidence"] == 0.9
        assert body["settings"]["income_scenario"] == "expected"

    def test_preferences_carry_derived_labels(self, api_client, doc_payload):
        """Labels are derived for display, never stored input (§15)."""
        body = api_client.post("/accounts", json=doc_payload).json()
        assert [preference["label"] for preference in body["preferences"]] == [
            "C2R1",
            "C1R1",
            "C2",
            "C1",
            "C0",
        ]

    def test_an_account_can_exist_before_its_roadmap(self, api_client):
        """Account state is not goals (§4.1): an empty roadmap is valid."""
        response = api_client.post(
            "/accounts",
            json={"settings": {"current_version": "7.0"}},
        )
        assert response.status_code == 201
        assert response.json()["goals"] == []


class TestListAndDelete:
    def test_lists_summaries_in_creation_order(self, api_client, doc_payload):
        first = api_client.post("/accounts", json=doc_payload).json()
        second_payload = dict(doc_payload, label="second")
        second = api_client.post("/accounts", json=second_payload).json()

        listed = api_client.get("/accounts").json()
        assert [entry["id"] for entry in listed] == [first["id"], second["id"]]
        assert listed[0]["goal_count"] == 4
        assert listed[0]["banner_count"] == 3
        assert listed[1]["label"] == "second"

    def test_delete_removes_the_account(self, api_client, doc_account_id):
        assert api_client.delete(f"/accounts/{doc_account_id}").status_code == 204
        assert api_client.get(f"/accounts/{doc_account_id}").status_code == 404
        assert api_client.get("/accounts").json() == []


class TestUpdate:
    def test_replaces_state_and_settings_but_not_the_roadmap(
        self, api_client, doc_account_id, doc_payload
    ):
        """The update the planner is re-run after (§2)."""
        response = api_client.put(
            f"/accounts/{doc_account_id}",
            json={
                "label": "after the banner",
                "account": {
                    "current_pity": 62,
                    "character_guarantee": True,
                    "wishes": 12,
                    "owned_characters": {"Vodynista": 0, "Vesna": 0},
                },
                "settings": {
                    "current_version": "7.1",
                    "current_phase": 1,
                    "confidence": 0.8,
                },
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["account"]["current_pity"] == 62
        assert body["account"]["character_guarantee"] is True
        assert body["settings"]["current_version"] == "7.1"
        assert body["settings"]["confidence"] == 0.8
        # The roadmap has its own endpoints and is left alone.
        assert len(body["goals"]) == 4
        assert len(body["preferences"]) == 5


class TestDomainValidationSurfaces:
    """Value rules stay in the domain; the API reports them (invariant 2)."""

    def test_duplicate_goal_priorities_are_rejected(self, api_client, doc_payload):
        payload = dict(
            doc_payload,
            goals=[
                {"character": "Vesna", "constellation": 0, "priority": 1},
                {"character": "Tsaritsa", "constellation": 0, "priority": 1},
            ],
        )
        response = api_client.post("/accounts", json=payload)
        assert response.status_code == 422
        assert "priorities must be unique" in response.json()["detail"]

    def test_pity_at_hard_pity_is_rejected(self, api_client, doc_payload):
        account = dict(doc_payload["account"], current_pity=90)
        payload = dict(doc_payload, account=account)
        response = api_client.post("/accounts", json=payload)
        assert response.status_code == 422
        assert "current_pity" in response.json()["detail"]

    def test_unknown_income_scenario_is_rejected(self, api_client, doc_payload):
        payload = dict(
            doc_payload,
            settings={"current_version": "7.0", "income_scenario": "optimistic"},
        )
        response = api_client.post("/accounts", json=payload)
        assert response.status_code == 422
        assert "income scenario" in response.json()["detail"]

    def test_malformed_version_is_rejected(self, api_client, doc_payload):
        payload = dict(doc_payload, settings={"current_version": "seven"})
        response = api_client.post("/accounts", json=payload)
        assert response.status_code == 422
        assert "version" in response.json()["detail"]

    def test_unknown_fields_are_rejected(self, api_client, doc_payload):
        response = api_client.post(
            "/accounts", json=dict(doc_payload, wishes_spent=10)
        )
        assert response.status_code == 422

    def test_unknown_account_is_404(self, api_client):
        response = api_client.get("/accounts/does-not-exist")
        assert response.status_code == 404
        assert "does-not-exist" in response.json()["detail"]
