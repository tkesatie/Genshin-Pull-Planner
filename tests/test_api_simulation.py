"""Simulation endpoints over HTTP (§11, §12).

The submit/poll pair has three jobs: reject a plan the simulator could never
execute *before* sampling, reproduce `simulate()` exactly for the same runs
and seed, and keep a job visible only under the account that submitted it.
"""

from domain import Account, Banner, Goal, Ownership, Roadmap
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate

VESNA_ENTRY = {
    "banner": {"character": "Vesna", "version": "7.0", "phase": 1},
    "target_constellation": 0,
    "budget": 40,
}


def _doc_context() -> PlannerContext:
    return PlannerContext(
        account=Account(
            current_pity=0,
            character_guarantee=False,
            owned_characters=Ownership({"Vodynista": 0}),
            wishes=40,
        ),
        roadmap=Roadmap(
            goals=[
                Goal("Vesna", 0, 1),
                Goal("Vodynista", 0, 2),
                Goal("Vesna", 2, 3),
                Goal("Tsaritsa", 0, 4),
            ],
            banners=[
                Banner("Vesna", "7.0", 1),
                Banner("Tsaritsa", "7.1", 1),
                Banner("Vodynista", "7.2", 1),
            ],
        ),
        current_version="7.0",
    )


class TestSubmitAndPoll:
    def test_submission_is_accepted_as_a_queued_job(self, api_client, doc_account_id):
        response = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={"plan": {"entries": [VESNA_ENTRY]}, "runs": 100, "seed": 3},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "queued"
        assert body["runs"] == 100
        assert body["seed"] == 3
        assert body["result"] is None

    def test_the_finished_job_reproduces_the_simulator(
        self, api_client, doc_account_id
    ):
        job_id = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={"plan": {"entries": [VESNA_ENTRY]}, "runs": 100, "seed": 3},
        ).json()["job_id"]

        polled = api_client.get(
            f"/accounts/{doc_account_id}/simulation/{job_id}"
        ).json()
        assert polled["status"] == "succeeded"
        assert polled["error"] is None

        expected = simulate(
            _doc_context(),
            SpendPlan(
                entries=(
                    PlannedSpend(Banner("Vesna", "7.0", 1), 0, 40),
                )
            ),
            runs=100,
            seed=3,
        )
        result = polled["result"]
        assert result["runs"] == 100
        assert result["seed"] == 3
        assert result["all_goals_probability"] == expected.all_goals_probability
        assert [goal["probability"] for goal in result["goals"]] == [
            goal.probability for goal in expected.goals
        ]
        assert [banner["banner"]["character"] for banner in result["banners"]] == [
            "Vesna",
            "Tsaritsa",
            "Vodynista",
        ]

    def test_skipped_banners_still_appear_in_the_history(
        self, api_client, doc_account_id
    ):
        """An absent entry means skip, not stop (§11 invariant 8)."""
        job_id = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={"plan": {"entries": [VESNA_ENTRY]}, "runs": 50, "seed": 1},
        ).json()["job_id"]
        result = api_client.get(
            f"/accounts/{doc_account_id}/simulation/{job_id}"
        ).json()["result"]

        tsaritsa = result["banners"][1]
        assert tsaritsa["target_constellation"] is None
        assert tsaritsa["mean_wishes_spent"] == 0.0


class TestValidation:
    def test_a_banner_outside_the_roadmap_fails_at_submission(
        self, api_client, doc_account_id
    ):
        response = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={
                "plan": {
                    "entries": [
                        {
                            "banner": {
                                "character": "Nobody",
                                "version": "7.5",
                                "phase": 1,
                            },
                            "target_constellation": 0,
                            "budget": 10,
                        }
                    ]
                },
                "runs": 10,
            },
        )
        assert response.status_code == 422
        assert "not a roadmap banner" in response.json()["detail"]

    def test_a_past_banner_fails_at_submission(self, api_client, doc_account_id):
        api_client.put(
            f"/accounts/{doc_account_id}",
            json={
                "account": {"wishes": 40, "owned_characters": {"Vodynista": 0}},
                "settings": {"current_version": "7.2"},
            },
        )
        response = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={"plan": {"entries": [VESNA_ENTRY]}, "runs": 10},
        )
        assert response.status_code == 422
        assert "before the current banner" in response.json()["detail"]

    def test_zero_runs_is_rejected(self, api_client, doc_account_id):
        response = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={"plan": {"entries": [VESNA_ENTRY]}, "runs": 0},
        )
        assert response.status_code == 422
        assert "runs" in response.json()["detail"]


class TestJobVisibility:
    def test_an_unknown_job_is_404(self, api_client, doc_account_id):
        response = api_client.get(
            f"/accounts/{doc_account_id}/simulation/nope"
        )
        assert response.status_code == 404

    def test_a_job_is_not_visible_under_another_account(
        self, api_client, doc_account_id, doc_payload
    ):
        job_id = api_client.post(
            f"/accounts/{doc_account_id}/simulation/run",
            json={"plan": {"entries": [VESNA_ENTRY]}, "runs": 10, "seed": 1},
        ).json()["job_id"]
        other_id = api_client.post("/accounts", json=doc_payload).json()["id"]

        response = api_client.get(f"/accounts/{other_id}/simulation/{job_id}")
        assert response.status_code == 404
