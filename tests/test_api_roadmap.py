"""Roadmap sub-resources over HTTP (§5, §7, §15, §16).

Collections are replaced whole and read back in the roadmap's own orders,
so a client sees goals and banners the way the planner does without
re-sorting. Income has the extra state the others do not: "no forecast" is
a different thing from "a forecast of zero".
"""


class TestGoals:
    def test_read_returns_priority_order(self, api_client, doc_account_id):
        goals = api_client.get(f"/accounts/{doc_account_id}/goals").json()["goals"]
        assert [goal["priority"] for goal in goals] == [1, 2, 3, 4]
        assert [goal["character"] for goal in goals] == [
            "Vesna",
            "Vodynista",
            "Vesna",
            "Tsaritsa",
        ]

    def test_replacing_goals_keeps_the_rest_of_the_account(
        self, api_client, doc_account_id
    ):
        response = api_client.put(
            f"/accounts/{doc_account_id}/goals",
            json={
                "goals": [
                    {"character": "Tsaritsa", "constellation": 0, "priority": 2},
                    {"character": "Vesna", "constellation": 0, "priority": 1},
                ]
            },
        )
        assert response.status_code == 200
        assert [goal["character"] for goal in response.json()["goals"]] == [
            "Vesna",
            "Tsaritsa",
        ]

        account = api_client.get(f"/accounts/{doc_account_id}").json()
        assert len(account["preferences"]) == 5
        assert len(account["banners"]) == 3

    def test_duplicate_priorities_are_rejected_and_nothing_is_written(
        self, api_client, doc_account_id
    ):
        response = api_client.put(
            f"/accounts/{doc_account_id}/goals",
            json={
                "goals": [
                    {"character": "Vesna", "constellation": 0, "priority": 1},
                    {"character": "Tsaritsa", "constellation": 0, "priority": 1},
                ]
            },
        )
        assert response.status_code == 422
        stored = api_client.get(f"/accounts/{doc_account_id}/goals").json()
        assert len(stored["goals"]) == 4


class TestBanners:
    def test_read_returns_chronological_order(self, api_client, doc_account_id):
        api_client.put(
            f"/accounts/{doc_account_id}/banners",
            json={
                "banners": [
                    {"character": "Late", "version": "7.10", "phase": 1},
                    {"character": "Early", "version": "7.9", "phase": 2},
                    {"character": "Earliest", "version": "7.9", "phase": 1},
                ]
            },
        )
        banners = api_client.get(f"/accounts/{doc_account_id}/banners").json()
        # Numeric version ordering: 7.9 precedes 7.10 (§7, domain.versions).
        assert [banner["character"] for banner in banners["banners"]] == [
            "Earliest",
            "Early",
            "Late",
        ]


class TestPreferences:
    def test_read_returns_rank_order_with_labels(self, api_client, doc_account_id):
        body = api_client.get(f"/accounts/{doc_account_id}/preferences").json()
        assert [preference["rank"] for preference in body["preferences"]] == [
            1,
            2,
            3,
            4,
            5,
        ]
        assert body["preferences"][0]["label"] == "C2R1"

    def test_replacing_preferences_replaces_every_character(
        self, api_client, doc_account_id
    ):
        response = api_client.put(
            f"/accounts/{doc_account_id}/preferences",
            json={
                "preferences": [
                    {"character": "Tsaritsa", "rank": 1, "constellation": 0}
                ]
            },
        )
        assert response.status_code == 200
        assert response.json()["preferences"] == [
            {
                "character": "Tsaritsa",
                "rank": 1,
                "constellation": 0,
                "weapon_refinement": 0,
                "notes": "",
                "label": "C0",
            }
        ]


class TestIncome:
    def test_absent_forecast_reads_as_null(self, api_client, doc_account_id):
        assert api_client.get(f"/accounts/{doc_account_id}/income").json() is None

    def test_replace_returns_aggregates(self, api_client, doc_account_id):
        response = api_client.put(
            f"/accounts/{doc_account_id}/income",
            json={
                "versions": [
                    {
                        "version": "7.1",
                        "sources": [
                            {
                                "name": "Commissions",
                                "estimate": {"low": 10, "expected": 12, "high": 14},
                            },
                            {
                                "name": "Events",
                                "estimate": {"low": 10, "expected": 18, "high": 26},
                            },
                        ],
                    },
                    {
                        "version": "7.0",
                        "estimate": {"low": 20, "expected": 30, "high": 40},
                    },
                ]
            },
        )
        assert response.status_code == 200
        versions = response.json()["versions"]
        # Read back in version order (§16), with the derived aggregate.
        assert [entry["version"] for entry in versions] == ["7.0", "7.1"]
        assert versions[1]["aggregate"] == {"low": 20, "expected": 30, "high": 40}

    def test_inconsistent_estimate_and_sources_are_rejected(
        self, api_client, doc_account_id
    ):
        response = api_client.put(
            f"/accounts/{doc_account_id}/income",
            json={
                "versions": [
                    {
                        "version": "7.0",
                        "estimate": {"low": 1, "expected": 2, "high": 3},
                        "sources": [
                            {
                                "name": "Commissions",
                                "estimate": {"low": 9, "expected": 9, "high": 9},
                            }
                        ],
                    }
                ]
            },
        )
        assert response.status_code == 422
        assert "does not match" in response.json()["detail"]

    def test_delete_clears_the_forecast(self, api_client, doc_account_id):
        api_client.put(
            f"/accounts/{doc_account_id}/income",
            json={
                "versions": [
                    {"version": "7.0", "estimate": {"low": 1, "expected": 2, "high": 3}}
                ]
            },
        )
        assert (
            api_client.delete(f"/accounts/{doc_account_id}/income").status_code == 204
        )
        assert api_client.get(f"/accounts/{doc_account_id}/income").json() is None
