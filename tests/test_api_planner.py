"""Planner endpoints over HTTP (§9, §13, §14).

These are parity tests: every endpoint's answer is compared with the planner
or optimizer called directly on the same context. The API is a shell around
that logic (invariant 1), so any drift shows up here rather than in a
frontend months later.
"""

from domain import Account, Banner, Goal, Ownership, Preference, Roadmap
from planner import PlannerContext, safe_spend
from optimizer import recommend

import pytest


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
    def test_multi_copy_outcome_shows_spending_tradeoff(
        self, api_client, doc_account_id
    ):
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/spend-table",
            params={"step": 10, "runs": 10_000, "seed": 5},
        ).json()

        assert [outcome["label"] for outcome in body["outcomes"]] == ["C0", "C2"]
        assert body["runs"] == 10_000
        assert body["seed"] == 5
        assert [row["wishes_spent"] for row in body["rows"]] == [0, 10, 20, 30, 40]

        # The table shows cumulative constellation milestones, not mutually
        # exclusive outcomes: reaching C2 necessarily also reaches C0.
        probabilities = {
            outcome["label"]: [
                row["outcomes"][index]["probability"]
                for row in body["rows"]
            ]
            for index, outcome in enumerate(body["outcomes"])
        }
        assert probabilities["C0"] == sorted(probabilities["C0"])
        assert probabilities["C2"] == sorted(probabilities["C2"])
        assert probabilities["C0"][0] == 0.0
        assert probabilities["C2"][0] == 0.0
        assert probabilities["C0"][-1] >= probabilities["C2"][-1]
        # C2 is very unlikely in 40 wishes, but not mechanically impossible:
        # three 5-stars can occur before hard pity.  A nonzero endpoint verifies
        # that the multi-copy simulation is actually modeling the milestone
        # rather than treating C2 as a single copy.
        assert probabilities["C2"][-1] > 0.0

        # Future protection is evaluated against the same simulated spend.
        assert body["rows"][0]["protected"]
        assert body["rows"][0]["protected"][0]["goal"]["character"] == "Tsaritsa"
        # Tsaritsa is Priority 4 while the reference C0 outcome is Priority 1,
        # so it is protected and reported but does not constrain this spend.
        assert body["rows"][0]["protected"][0]["constraining"] is False
        assert body["rows"][-1]["protected"][0]["probability"] <= body["rows"][0]["protected"][0]["probability"]

    def test_spend_analysis_uses_selected_outcome_even_when_recommendation_is_discretionary(
        self, api_client, doc_account_id
    ):
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/spend-table",
            params={"step": 20, "runs": 100, "seed": 7},
        ).json()

        assert body["outcomes"]
        assert [outcome["label"] for outcome in body["outcomes"]] == ["C0", "C2"]


class TestMultipleCurrentBanners:
    def test_recommendation_selects_highest_priority_current_banner(self, api_client):
        response = api_client.post(
            "/accounts",
            json={
                "label": "simultaneous banners",
                "account": {
                    "wishes": 450,
                    "current_pity": 27,
                    "character_guarantee": False,
                    "owned_characters": {"Skirk": 0},
                },
                "settings": {
                    "current_version": "7.1",
                    "current_phase": 1,
                },
                "banners": [
                    {"character": "Vodynista", "version": "7.1", "phase": 1},
                    {"character": "Vesna", "version": "7.1", "phase": 1},
                    {"character": "Skirk", "version": "7.1", "phase": 2},
                ],
                "goals": [
                    {"character": "Vodynista", "constellation": 0, "priority": 1},
                    {"character": "Vesna", "constellation": 0, "priority": 2},
                    {"character": "Skirk", "constellation": 2, "priority": 3},
                    {"character": "Vesna", "constellation": 2, "priority": 4},
                ],
            },
        )
        assert response.status_code == 201, response.text
        account_id = response.json()["id"]

        goals = api_client.get(f"/accounts/{account_id}/planner/goals").json()
        assert goals["current_banner"] is None
        assert {
            banner["character"] for banner in goals["available_banners"]
        } == {"Vodynista", "Vesna"}
        assert {
            (goal["goal"]["character"], goal["goal"]["constellation"])
            for goal in goals["goals"]
            if goal["actionable"]
        } == {("Vodynista", 0), ("Vesna", 0)}

        recommendation = api_client.get(
            f"/accounts/{account_id}/planner/recommendation",
            params={"runs": 100, "seed": 7, "budgets": [450, 0]},
        )
        assert recommendation.status_code == 200, recommendation.text
        body = recommendation.json()
        assert body["banner"]["character"] == "Vodynista"
        assert body["outcome"]["character"] == "Vodynista"
        assert body["outcome"]["label"] == "C0"


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

    def test_the_doc_example_pursues_because_its_future_goal_lower_priority(
        self, api_client, doc_account_id
    ):
        """The doc example's Vesna C0 is Priority 1 and current; Tsaritsa C0
        is Priority 4. §2 only lets *higher-priority* future goals constrain a
        decision, so the API pursues Vesna's top preference at the whole pool
        and reports Tsaritsa as protected but non-gating (§13)."""
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/recommendation",
            params={"runs": 200, "seed": 5, "budgets": [40, 20, 0]},
        ).json()
        # Issue 2 (minimum_outcome_probability, §14): the winning outcome on
        # 40 wishes from fresh pity is well under the 25% minimum for an
        # ordinary recommendation, so the API reports the disclosed-gamble
        # "discretionary" action here. Tsaritsa's non-gating status - this
        # test's actual subject - is unaffected.
        assert body["action"] == "discretionary"
        assert body["outcome_probability"] < 0.25
        assert body["outcome"]["character"] == "Vesna"
        assert body["budget"] == 40
        assert body["skip_reason"] is None
        # Any fallthrough is the empirical-pursuit criterion, never Tsaritsa:
        # a non-constraining goal is reported but never named as a blocker.
        assert all(not rejected["shortfalls"] for rejected in body["rejected"])

        (protected,) = body["protected"]
        assert protected["goal"] == {
            "character": "Tsaritsa",
            "constellation": 0,
            "priority": 4,
        }
        assert protected["constraining"] is False
        assert protected["meets_threshold"] is False

    def test_a_higher_priority_future_goal_blocks_and_the_api_says_why(
        self, api_client, doc_account_id
    ):
        """The inverse: make Tsaritsa Priority 1 and Vesna C0 Priority 2 and
        the same account skips again, with the blocker named in the shortfall
        list (§1, §13 step 7)."""
        api_client.put(
            f"/accounts/{doc_account_id}/goals",
            json={
                "goals": [
                    {"character": "Tsaritsa", "constellation": 0, "priority": 1},
                    {"character": "Vesna", "constellation": 0, "priority": 2},
                    {"character": "Vesna", "constellation": 2, "priority": 3},
                    {"character": "Vodynista", "constellation": 0, "priority": 4},
                ]
            },
        )
        body = api_client.get(
            f"/accounts/{doc_account_id}/planner/recommendation",
            params={"runs": 200, "seed": 5, "budgets": [40, 20, 0]},
        ).json()
        assert body["action"] == "skip"
        assert body["outcome"] is None
        assert body["plan"] is None
        assert "higher-priority" in body["skip_reason"]
        assert [rejected["outcome"]["label"] for rejected in body["rejected"]] == [
            "C2R1",
            "C1R1",
            "C0",
        ]
        for rejected in body["rejected"]:
            assert rejected["shortfalls"], "the 'why' must name the blocker"
            (shortfall,) = rejected["shortfalls"]
            assert shortfall["goal"]["character"] == "Tsaritsa"
            assert shortfall["constraining"] is True
            assert shortfall["meets_threshold"] is False

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
        # Issue 2 (minimum_outcome_probability, §14): with no protected
        # goals left, the deepest reachable rung of the chain wins outright
        # (unaffected by this test's own subject), but that rung is a long
        # shot on 40 wishes - under the 25% minimum - so the API reports the
        # disclosed-gamble "discretionary" action rather than "pursue".
        assert body["action"] == "discretionary"
        assert body["outcome_probability"] < 0.25
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
        assert body["stops"]["action"] == "discretionary"
        assert body["stops"]["outcome_label"] == body["outcome"]["label"]
        assert body["stops"]["spend_cap"] == 40


class TestDiscretionaryRecommendation:
    """Issue 2 over HTTP (§14): minimum_outcome_probability is a request
    parameter (like runs/seed/budgets), and a feasible-but-unlikely
    recommendation carries a discretionary_reason. Sparse mechanics with a
    tuned featured_rate give an exact, low-variance probability (see
    test_optimizer_recommend.TestDiscretionaryGamble for why)."""

    def _sparse_account(self, api_client, featured_rate: float) -> str:
        response = api_client.post(
            "/accounts",
            json={
                "label": "discretionary gamble",
                "account": {"wishes": 3},
                "settings": {
                    "current_version": "7.0",
                    "current_phase": 1,
                    "mechanics": {
                        "banner_type": "sparse",
                        "hard_pity": 3,
                        "soft_pity_start": 2,
                        "base_rate": 1e-9,
                        "soft_pity_increment": 1.0,
                        "featured_rate": featured_rate,
                    },
                },
                "banners": [{"character": "Navia", "version": "7.0", "phase": 1}],
                "preferences": [
                    {"character": "Navia", "rank": 1, "constellation": 0}
                ],
            },
        )
        assert response.status_code == 201, response.text
        return response.json()["id"]

    def test_low_probability_is_discretionary_with_a_reason(self, api_client):
        account_id = self._sparse_account(api_client, featured_rate=0.2)
        body = api_client.get(
            f"/accounts/{account_id}/planner/recommendation",
            params={"runs": 6_000, "seed": 11},
        ).json()

        assert body["action"] == "discretionary"
        assert body["outcome"]["label"] == "C0"
        assert body["budget"] == 3
        assert body["outcome_probability"] == pytest.approx(0.2, abs=0.03)
        assert body["minimum_outcome_probability"] == pytest.approx(0.25)
        assert body["skip_reason"] is None
        assert body["discretionary_reason"] is not None
        assert "C0" in body["discretionary_reason"]
        assert body["stops"]["action"] == "discretionary"
        # Existing fields are otherwise untouched by the new field.
        assert body["plan"]["entries"][0]["budget"] == 3
        assert body["protected"] == []
        assert body["rejected"] == []

    def test_a_normal_recommendation_carries_no_discretionary_reason(
        self, api_client
    ):
        account_id = self._sparse_account(api_client, featured_rate=0.3)
        body = api_client.get(
            f"/accounts/{account_id}/planner/recommendation",
            params={"runs": 6_000, "seed": 11},
        ).json()

        assert body["action"] == "pursue"
        assert body["outcome_probability"] == pytest.approx(0.3, abs=0.03)
        assert body["discretionary_reason"] is None
        assert body["skip_reason"] is None

    def test_minimum_outcome_probability_is_accepted_and_passed_through(
        self, api_client
    ):
        """Lowering the query parameter for the same ~20% scenario turns it
        into an ordinary recommendation - proof the parameter reaches the
        optimizer rather than being ignored - and the reported minimum
        echoes back whatever was requested."""
        account_id = self._sparse_account(api_client, featured_rate=0.2)
        body = api_client.get(
            f"/accounts/{account_id}/planner/recommendation",
            params={"runs": 6_000, "seed": 11, "minimum_outcome_probability": 0.1},
        ).json()

        assert body["action"] == "pursue"
        assert body["discretionary_reason"] is None
        assert body["minimum_outcome_probability"] == pytest.approx(0.1)

        expected = recommend(
            PlannerContext(
                account=Account(wishes=3),
                roadmap=Roadmap(
                    goals=[], banners=[Banner("Navia", "7.0", 1)]
                ),
                current_version="7.0",
                mechanics=self._mechanics(0.2),
            ),
            (Preference("Navia", 1, 0),),
            runs=6_000,
            seed=11,
            minimum_outcome_probability=0.1,
        )
        assert body["action"] == expected.action
        assert body["outcome_probability"] == expected.outcome_probability

    def test_stop_conditions_endpoint_reflects_discretionary_action(
        self, api_client
    ):
        account_id = self._sparse_account(api_client, featured_rate=0.2)
        stops = api_client.get(
            f"/accounts/{account_id}/planner/stop-conditions",
            params={"runs": 6_000, "seed": 11},
        ).json()
        assert stops["action"] == "discretionary"
        assert stops["spend_cap"] == 3

    def _mechanics(self, featured_rate: float):
        from domain import WishMechanics

        return WishMechanics(
            banner_type="sparse",
            hard_pity=3,
            soft_pity_start=2,
            base_rate=1e-9,
            soft_pity_increment=1.0,
            featured_rate=featured_rate,
        )


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
