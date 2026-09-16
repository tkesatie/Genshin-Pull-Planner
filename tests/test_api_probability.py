"""Probability endpoints over HTTP (§10).

The engine is independent of roadmap logic (§10), so the API's job is to
supply defaults and pass questions through unchanged. Each test compares the
HTTP answer with the engine called directly: a difference would mean the API
had grown probability logic of its own.
"""

from domain import CHARACTER_EVENT_BANNER
from probability import cumulative_probability, wishes_for_confidence


class TestCharacterProbability:
    def test_matches_the_engine_and_defaults_to_account_state(
        self, api_client, doc_account_id
    ):
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/character",
            params={"wishes": 40},
        )
        assert response.status_code == 200
        body = response.json()

        expected = cumulative_probability(40, 0, False, CHARACTER_EVENT_BANNER)
        assert body["starting_pity"] == 0
        assert body["guaranteed"] is False
        assert body["probability"] == float(expected[40])
        assert len(body["curve"]) == 41
        assert body["curve"][0] == 0.0

    def test_state_can_be_overridden_without_touching_the_account(
        self, api_client, doc_account_id
    ):
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/character",
            params={
                "wishes": 20,
                "starting_pity": 75,
                "guaranteed": True,
                "include_curve": False,
            },
        )
        body = response.json()
        expected = cumulative_probability(20, 75, True, CHARACTER_EVENT_BANNER)
        assert body["probability"] == float(expected[20])
        assert body["curve"] is None

        stored = api_client.get(f"/accounts/{doc_account_id}").json()
        assert stored["account"]["current_pity"] == 0

    def test_hard_pity_with_a_guarantee_is_certain(self, api_client, doc_account_id):
        body = api_client.get(
            f"/accounts/{doc_account_id}/probability/character",
            params={"wishes": 90, "starting_pity": 0, "guaranteed": True},
        ).json()
        assert body["probability"] == 1.0

    def test_pity_beyond_hard_pity_is_rejected(self, api_client, doc_account_id):
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/character",
            params={"wishes": 10, "starting_pity": 90},
        )
        assert response.status_code == 422
        assert "starting_pity" in response.json()["detail"]

    def test_lookahead_is_bounded(self, api_client, doc_account_id):
        """An API-level bound on response size and compute, not a domain rule."""
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/character",
            params={"wishes": 10_001},
        )
        assert response.status_code == 422
        assert "10000" in response.json()["detail"].replace(",", "")


class TestWishesNeeded:
    def test_matches_the_engine_and_defaults_to_the_stored_threshold(
        self, api_client, doc_account_id
    ):
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/wishes-needed"
        )
        assert response.status_code == 200
        body = response.json()

        expected = wishes_for_confidence(0.9, 0, False, CHARACTER_EVENT_BANNER)
        assert body["confidence"] == 0.9
        assert body["wishes_needed"] == expected
        assert body["probability_at_wishes_needed"] >= 0.9

    def test_confidence_can_be_overridden(self, api_client, doc_account_id):
        lower = api_client.get(
            f"/accounts/{doc_account_id}/probability/wishes-needed",
            params={"confidence": 0.5},
        ).json()
        higher = api_client.get(
            f"/accounts/{doc_account_id}/probability/wishes-needed",
            params={"confidence": 0.99},
        ).json()
        assert lower["wishes_needed"] < higher["wishes_needed"]

    def test_confidence_outside_the_open_unit_interval_is_rejected(
        self, api_client, doc_account_id
    ):
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/wishes-needed",
            params={"confidence": 1.5},
        )
        assert response.status_code == 422
        assert "confidence" in response.json()["detail"]


class TestWeaponProbability:
    def test_is_not_implemented_rather_than_guessed(self, api_client, doc_account_id):
        """Weapon mechanics are absent until verified (§17, §19)."""
        response = api_client.get(
            f"/accounts/{doc_account_id}/probability/weapon"
        )
        assert response.status_code == 501
        assert "not implemented" in response.json()["detail"]
