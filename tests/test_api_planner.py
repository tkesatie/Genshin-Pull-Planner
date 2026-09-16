"""Planner endpoints over HTTP (§9, §13, §14).

These are parity tests: every endpoint's answer is compared with the planner
or optimizer called directly on the same context. The API is a shell around
that logic (invariant 1), so any drift shows up here rather than in a
frontend months later.
"""

from domain import Account, Banner, Goal, Ownership, Preference, Roadmap
from planner import PlannerContext, safe_spend
from optimizer import recommend


def _doc_context(wishes: int = 40, confidence: float = 0.9) -> PlannerContext:
    """The stored doc example, rebuilt for direct comparison (§8)."""
    return PlannerContext(
        account=Account(
            current_pity=0,
            character_guarantee=False,
            owned_characters=Ownership({"Vodynista": 0}),
            wishes=wishes,
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
        confidence=confidence,
    )


class TestPlannerGoals:
    def test_reports_states_dependencies_and_actionability(
        self, api_client, doc_account_id
    ):
        body = api_client.get(f"/accounts/{doc_account_id}/planner/goals").json()
        assert body["current_banner"]["character"] == "Vesna"

        by_key = {
            (goal["goal"]["character"], goal["goal"]["constellation"]): goal
            for goal in body["goals"]
        }
        # Vesna C0: relevant and actionable (§9).
        assert by_key[("Vesna", 0)]["state"] == "active"
        assert by_key[("Vesna", 0)]["actionable"] is True
        # Vesna C2: relevant but blocked behind C0 (§9).
        assert by_key[("Vesna", 2)]["state"] == "blocked"
        assert by_key[("Vesna", 2)]["relevant"] is True
        assert by_key[("Vesna", 2)]["actionable"] is False
        assert by_key[("Vesna", 2)]["blocked_by"]["constellation"] == 0
        # Vodynista C0: already owned at C0, nothing remains (§6).
        assert by_key[("Vodynista", 0)]["state"] == "satisfied"
        assert by_key[("Vodynista", 0)]["copies_needed"] == 0
        # Tsaritsa C0: unsatisfied, but not this banner's business.
        assert by_key[("Tsaritsa", 0)]["actionable"] is False
        assert by_key[("Tsaritsa", 0)]["next_banner"]["version"] == "7.1"

    def test_a_version_with_no_banner_is_a_planner_error(
        self, api_client, doc_account_id
    ):
        """An account may sit outside its schedule; the planner says so."""
        api_client.put(
            f"/accounts/{doc_account_id}",
            json={
                "account": {"wishes": 40},
                "settings": {"current_version": "9.9"},
            },
        )
        response = api_client.get(f"/accounts/{doc_account_id}/planner/goals")
        assert response.status_code == 422
        assert "no roadmap banner" in response.json()["detail"]


class TestSafeSpend:
    def test_matches_the_planner(self, api_client, doc_account_id):
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/safe-spend"
        ).json()
        assert body["safe_spend"] == safe_spend(_doc_context())
        assert body["account_wishes"] == 40
        assert body["confidence"] == 0.9

    def test_the_doc_example_cannot_protect_its_roadmap_at_90_percent(
        self, api_client, doc_account_id
    ):
        """"Do not spend" is a legitimate answer (§1, §14)."""
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/safe-spend"
        ).json()
        assert body["safe_spend"] == 0
        # Vodynista C0 is already owned and Vesna's goals belong to the
        # current banner, so Tsaritsa 7.1 is the only protected goal (§13).
        assert [outcome["goal"]["character"] for outcome in body["protected"]] == [
            "Tsaritsa"
        ]
        protected = body["protected"][0]
        assert protected["banner"]["version"] == "7.1"
        assert protected["budget_at_banner"] < protected["required_wishes"]
        assert protected["meets_threshold"] is False

    def test_confidence_override_is_not_persisted(self, api_client, doc_account_id):
        relaxed = api_client.get(
            f"/accounts/{doc_account_id}/planner/safe-spend",
            params={"confidence": 0.5},
        ).json()
        assert relaxed["confidence"] == 0.5
        assert relaxed["safe_spend"] == safe_spend(_doc_context(confidence=0.5))

        stored = api_client.get(f"/accounts/{doc_account_id}").json()
        assert stored["settings"]["confidence"] == 0.9


class TestSpendTable:
    def test_rows_cover_every_spend_and_report_both_sides(
        self, api_client, doc_account_id
    ):
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/spend-table",
            params={"step": 10},
        ).json()
        assert body["goal"]["character"] == "Vesna"
        assert body["copies_needed"] == 1
        assert [row["wishes_spent"] for row in body["rows"]] == [0, 10, 20, 30, 40]
        # Spending more raises the goal's probability (§10.2 is monotone)...
        confidences = [row["goal_confidence"] for row in body["rows"]]
        assert confidences == sorted(confidences)
        # ...and is an explicit tradeoff against the protected roadmap (§1).
        assert body["rows"][0]["protected"][0]["budget_at_banner"] > (
            body["rows"][-1]["protected"][0]["budget_at_banner"]
        )

    def test_a_multi_copy_goal_is_rejected_not_approximated(
        self, api_client, doc_account_id
    ):
        """Multi-copy targets are simulation territory (§10.4, §18)."""
        api_client.put(
            f"/accounts/{doc_account_id}/goals",
            json={
                "goals": [
                    {"character": "Vesna", "constellation": 2, "priority": 1}
                ]
            },
        )
        response = api_client.get(
            f"/accounts/{doc_account_id}/planner/spend-table"
        )
        assert response.status_code == 422
        assert "single-copy" in response.json()["detail"]


class TestRecommendation:
    def test_matches_the_optimizer_for_the_same_runs_and_seed(
        self, api_client, doc_account_id
    ):
        params = {"runs": 200, "seed": 5, "budgets": [40, 20, 0]}
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/recommendation", params=params
        ).json()

        expected = recommend(
            _doc_context(),
            [
                Preference("Vesna", 1, 2, weapon_refinement=1),
                Preference("Vesna", 2, 1, weapon_refinement=1),
                Preference("Vesna", 3, 2),
                Preference("Vesna", 4, 1),
                Preference("Vesna", 5, 0),
            ],
            runs=200,
            seed=5,
            budgets=[40, 20, 0],
        )
        assert body["action"] == expected.action
        assert body["budget"] == expected.budget
        assert body["outcome_probability"] == expected.outcome_probability
        assert body["runs"] == 200
        assert body["seed"] == 5

    def test_the_doc_example_skips_with_per_outcome_rejections(
        self, api_client, doc_account_id
    ):
        """Nothing is pursueable at 90% here, and the API says why (§1, §13)."""
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/recommendation",
            params={"runs": 200, "seed": 5, "budgets": [40, 20, 0]},
        ).json()
        assert body["action"] == "skip"
        assert body["outcome"] is None
        assert body["plan"] is None
        assert body["skip_reason"] is not None
        assert [rejected["outcome"]["label"] for rejected in body["rejected"]] == [
            "C2R1",
            "C1R1",
            "C0",
        ]
        assert body["rejected"][0]["shortfalls"]

    def test_a_pursue_recommendation_carries_its_plan_and_stops(
        self, api_client, doc_account_id
    ):
        """With no protected goals left, spending is unconstrained (§14)."""
        api_client.put(
            f"/accounts/{doc_account_id}/goals",
            json={
                "goals": [
                    {"character": "Vesna", "constellation": 0, "priority": 1}
                ]
            },
        )
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/recommendation",
            params={"runs": 200, "seed": 5},
        ).json()
        assert body["action"] == "pursue"
        # Nothing future constrains the spend, so the whole pool is the cap
        # (§14) - a cap, not a commitment (§12).
        assert body["budget"] == 40
        assert body["protected"] == []
        # Which outcome wins depends on sampling; that it comes from the
        # user's chain, and that the plan pursues exactly it, does not (§2).
        assert body["outcome"]["label"] in {"C2R1", "C1R1", "C2", "C1", "C0"}
        entry = body["plan"]["entries"][0]
        assert entry["banner"]["character"] == "Vesna"
        assert entry["target_constellation"] == body["outcome"]["constellation"]
        assert entry["budget"] == 40
        assert body["stops"]["action"] == "pursue"
        assert body["stops"]["outcome_label"] == body["outcome"]["label"]
        assert body["stops"]["spend_cap"] == 40


class TestStopConditions:
    def test_returns_the_recommendation_s_own_stops(self, api_client, doc_account_id):
        params = {"runs": 200, "seed": 5, "budgets": [40, 0]}
        stops = api_client.get(
            f"/accounts/{doc_account_id}/planner/stop-conditions", params=params
        ).json()
        recommendation = api_client.get(
            f"/accounts/{doc_account_id}/planner/recommendation", params=params
        ).json()
        assert stops == recommendation["stops"]
        assert stops["rules"][-1].startswith("Re-run")
