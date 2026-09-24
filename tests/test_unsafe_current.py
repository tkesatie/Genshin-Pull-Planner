"""unsafe_current: conditional diagnostics for current-banner goals (§13).

The regression this pins: a currently available roadmap goal that sits
BELOW the confidence threshold under the recommendation's own constrained
spending decision must surface in `Recommendation.unsafe_current` so the
Next Goals roadmap can render it "NOT SAFE NOW". The old implementation
lost it twice over: the goal was evaluated standalone at the full winner
budget (Vesna greedily spending the whole shared pool, no reservation for
higher-priority pursuits or protected reserves) and the resulting inflated
number was compared against `minimum_outcome_probability` (0.25) instead
of `context.confidence` (0.9).

The scenario mirrors the persisted demo account: 473 wishes, 27 character
pity, no guarantee, Skirk C0 owned, priorities Vodynista C0 > Vesna C0 >
Beyond the Chrysalis R1 > Skirk C2 > Vesna C2 at 90% confidence, with a
weapon banner sharing the current slot and Skirk's banner one phase later.

Caps use a coarse list (the winner shortcut still takes the largest cap),
which keeps phase one's scan short; the conditional diagnostic scans every
cap from the winner budget down, so its answer does not depend on the list.
"""

import pytest

from api.schemas.planner import RecommendationView
from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
)
from domain.targets import WeaponTarget
from optimizer import recommend
from optimizer.evaluation import evaluate_candidate
from planner import PlannerContext

RUNS = 600
SEED = 0
BUDGETS = [340, 0]


def _demo_context() -> PlannerContext:
    """The demo account's state, rebuilt in code (§8-style fixtures)."""
    return PlannerContext(
        account=Account(
            current_pity=27,
            character_guarantee=False,
            owned_characters=Ownership({"Skirk": 0}),
            wishes=473,
        ),
        roadmap=Roadmap(
            goals=[
                Goal("Vodynista", 0, 1),
                Goal("Vesna", 0, 2),
                Goal(target=WeaponTarget("Beyond the Chrysalis"), level=1, priority=3),
                Goal("Skirk", 2, 4),
                Goal("Vesna", 2, 5),
            ],
            banners=[
                Banner(target=WeaponTarget("Beyond the Chrysalis"), version="7.1", phase=1),
                Banner("Hymn of the Maelstrom", "7.1", 1),
                Banner("Vesna", "7.1", 1),
                Banner("Vodynista", "7.1", 1),
                Banner("Skirk", "7.1", 2),
            ],
        ),
        current_version="7.1",
        current_phase=1,
        income=IncomeForecast(
            versions=[VersionIncome("7.1", estimate=IncomeEstimate(90, 114, 140))]
        ),
        confidence=0.9,
    )


@pytest.fixture(scope="module")
def demo_decision():
    context = _demo_context()
    rec = recommend(context, (), runs=RUNS, seed=SEED, budgets=BUDGETS)
    return context, rec


class TestSelectionUnchanged:
    """The diagnostic must not touch recommendation semantics (§13)."""

    def test_vodynista_c0_still_wins_at_the_largest_cap(self, demo_decision):
        context, rec = demo_decision
        assert rec.action == "pursue"
        assert rec.outcome.character == "Vodynista"
        assert rec.outcome.constellation == 0
        assert rec.budget == max(BUDGETS)
        assert rec.skip_reason is None

    def test_plan_allocation_is_the_winner_shape(self, demo_decision):
        context, rec = demo_decision
        entries = [entry.banner.target.name for entry in rec.plan.entries]
        # Winner outcome first, same-slot Vesna group second (residual of
        # the shared cap), Skirk's future protection last.
        assert entries == ["Vodynista", "Vesna", "Skirk"]
        assert rec.plan.shared_current_phase_budget == rec.budget


class TestVesnaC2Diagnostic:
    def test_vesna_c2_is_the_flagged_goal_below_confidence(self, demo_decision):
        context, rec = demo_decision
        (flagged,) = rec.unsafe_current
        assert (flagged.goal.target.name, flagged.goal.level) == ("Vesna", 2)
        assert flagged.goal.priority == 5
        assert flagged.banner.target.name == "Vesna"

        # Below the configured 90% safety threshold, and deliberately ABOVE
        # the old 0.25 comparison - under minimum_outcome_probability this
        # diagnostic used to vanish entirely.
        assert context.confidence == 0.9
        assert 0.25 < flagged.outcome_probability < context.confidence

        # Measured at the conditional cap the commitments allow - below the
        # winner budget, never at the full pool.
        assert 0 < flagged.budget < rec.budget

    def test_the_flagged_number_is_not_the_unconstrained_plan_number(
        self, demo_decision
    ):
        context, rec = demo_decision
        (flagged,) = rec.unsafe_current
        unconditional = evaluate_candidate(
            context,
            rec.outcome,
            context.account.wishes,  # the FULL pool: unconstrained allocation
            banner=rec.banner,
            runs=RUNS,
            seed=SEED,
        )
        unconstrained = next(
            item.probability
            for item in unconditional.result.goals
            if item.goal.target.name == "Vesna" and item.goal.level == 2
        )
        # The winner plan at the full budget legitimately reports a high
        # probability for Vesna C2; unsafe_current must report the
        # conditional one instead.
        assert unconstrained > context.confidence
        assert flagged.outcome_probability < unconstrained

    def test_honored_goals_are_not_flagged(self, demo_decision):
        context, rec = demo_decision
        flagged_keys = {
            (item.goal.target.name, item.goal.level) for item in rec.unsafe_current
        }
        assert flagged_keys == {("Vesna", 2)}
        # Higher-priority current goals and the weapon goal hold at 90%
        # under their own conditional evaluations.
        assert ("Vodynista", 0) not in flagged_keys  # the winner itself
        assert ("Vesna", 0) not in flagged_keys
        assert ("Beyond the Chrysalis", 1) not in flagged_keys
        # The future protected goal is not a current-banner diagnostic.
        assert ("Skirk", 2) not in flagged_keys


class TestApiExposure:
    def test_recommendation_view_serializes_unsafe_current(self, demo_decision):
        context, rec = demo_decision
        view = RecommendationView.from_domain(
            rec,
            confidence=context.confidence,
            minimum_outcome_probability=0.25,
        )
        payload = view.model_dump()
        (item,) = payload["unsafe_current"]
        assert item["goal"]["character"] == "Vesna"
        assert item["goal"]["constellation"] == 2
        assert item["banner"]["target_name"] == "Vesna"
        assert item["outcome_probability"] < payload["confidence"]
        (flagged,) = rec.unsafe_current
        assert item["outcome_probability"] == pytest.approx(
            flagged.outcome_probability
        )
        assert item["budget"] == flagged.budget

    def test_endpoint_exposes_unsafe_current(self, api_client):
        created = api_client.post(
            "/accounts",
            json={
                "label": "unsafe demo",
                "account": {
                    "current_pity": 27,
                    "character_guarantee": False,
                    "wishes": 473,
                    "owned_characters": {"Skirk": 0},
                },
                "settings": {
                    "current_version": "7.1",
                    "current_phase": 1,
                    "confidence": 0.9,
                },
                "goals": [
                    {"character": "Vodynista", "constellation": 0, "priority": 1},
                    {"character": "Vesna", "constellation": 0, "priority": 2},
                    {"weapon": "Beyond the Chrysalis", "refinement": 1, "priority": 3},
                    {"character": "Skirk", "constellation": 2, "priority": 4},
                    {"character": "Vesna", "constellation": 2, "priority": 5},
                ],
                "banners": [
                    {"weapon": "Beyond the Chrysalis", "version": "7.1", "phase": 1},
                    {"character": "Hymn of the Maelstrom", "version": "7.1", "phase": 1},
                    {"character": "Vesna", "version": "7.1", "phase": 1},
                    {"character": "Vodynista", "version": "7.1", "phase": 1},
                    {"character": "Skirk", "version": "7.1", "phase": 2},
                ],
                "income": {
                    "versions": [
                        {
                            "version": "7.1",
                            "estimate": {"low": 90, "expected": 114, "high": 140},
                        }
                    ]
                },
                "preferences": [],
            },
        )
        assert created.status_code == 201, created.text
        account_id = created.json()["id"]

        body = api_client.get(
            f"/accounts/{account_id}/planner/recommendation",
            params={"runs": RUNS, "seed": SEED, "budgets": BUDGETS},
        ).json()
        assert body["action"] == "pursue"
        assert body["outcome"]["character"] == "Vodynista"
        (item,) = body["unsafe_current"]
        assert item["goal"]["character"] == "Vesna"
        assert item["goal"]["constellation"] == 2
        assert item["banner"]["target_name"] == "Vesna"
        # The 90% confidence threshold, not minimum_outcome_probability.
        assert 0.25 < item["outcome_probability"] < body["confidence"]
