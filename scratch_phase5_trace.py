"""Scratch trace for the Phase 5 priority bug (deleted after investigation)."""

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Preference,
    Roadmap,
    VersionIncome,
    WishMechanics,
)
from optimizer import (
    available_outcomes,
    protected_groups,
    recommend,
)
from optimizer.evaluation import evaluate_candidate
from planner import PlannerContext, evaluate_goals

MECHANICS = WishMechanics(
    banner_type="character",
    hard_pity=90,
    soft_pity_start=74,
    base_rate=0.006,
    soft_pity_increment=0.06,
    featured_rate=0.5,
)


def build(navia_priority: int, arlecchino_priority: int) -> PlannerContext:
    account = Account(current_pity=70, character_guarantee=False, wishes=80)
    roadmap = Roadmap(
        goals=[
            Goal("Navia", 0, navia_priority),
            Goal("Arlecchino", 0, arlecchino_priority),
        ],
        banners=[Banner("Navia", "6.1", 1), Banner("Arlecchino", "6.3", 2)],
    )
    income = IncomeForecast(
        versions=[
            VersionIncome("6.1", estimate=IncomeEstimate(20, 30, 40)),
            VersionIncome("6.3", estimate=IncomeEstimate(20, 30, 40)),
        ]
    )
    return PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version="6.1",
        current_phase=1,
        income=income,
        income_scenario="expected",
        confidence=0.9,
        mechanics=MECHANICS,
    )


def trace(label: str, navia_priority: int, arlecchino_priority: int) -> None:
    print("=" * 72)
    print(label)
    context = build(navia_priority, arlecchino_priority)
    for evaluation in evaluate_goals(context):
        print(
            f"  goal {evaluation.goal} state={evaluation.state.value} "
            f"next_banner={evaluation.next_banner}"
        )
    print(f"  protected_groups: {protected_groups(context)}")
    print(f"  outcomes: {available_outcomes(context, (Preference('Navia', 1, 0),))}")
    for cap in (80, 60, 40, 20, 0):
        candidate = evaluate_candidate(
            context,
            available_outcomes(context, (Preference("Navia", 1, 0),))[0],
            cap,
            runs=2_000,
            seed=1,
        )
        standings = ", ".join(
            f"{s.goal.character} C{s.goal.constellation}@{s.banner.version}="
            f"{s.probability:.3f}{'OK' if s.meets_threshold else 'LOW'}"
            for s in candidate.protected
        )
        print(
            f"  cap={cap:>2} outcome_p={candidate.outcome_probability:.3f} "
            f"feasible={candidate.feasible} [{standings}]"
        )
    rec = recommend(context, (Preference("Navia", 1, 0),), runs=2_000, seed=1)
    print(
        f"  RECOMMENDATION action={rec.action} budget={rec.budget} "
        f"outcome={rec.outcome.label if rec.outcome else None} "
        f"skip_reason={rec.skip_reason}"
    )


trace("CASE A: Navia P1 (current) vs Arlecchino P2 (future)", 1, 2)
trace("CASE B: Navia P2 (current) vs Arlecchino P1 (future)", 2, 1)


def default_seed_trace() -> None:
    from simulation import DEFAULT_SEED

    print("=" * 72)
    print(f"defaults: runs=DEFAULT_RUNS seed={DEFAULT_SEED}")
    context = build(1, 2)
    rec = recommend(context, (Preference("Navia", 1, 0),))
    print(
        f"  RECOMMENDATION action={rec.action} budget={rec.budget} "
        f"outcome={rec.outcome.label if rec.outcome else None} "
        f"skip_reason={rec.skip_reason}"
    )
    for rejection in rec.rejected:
        print(f"  rejected {rejection.outcome.label} best cap={rejection.best.budget}")


default_seed_trace()


import json

from api import create_app
from fastapi.testclient import TestClient

SCENARIO = json.loads(
    r"""
{
  "label": "Protected Lower-Priority Goal vs. Current Higher-Priority Goal",
  "account": {"current_pity": 70, "character_guarantee": false, "wishes": 80, "owned_characters": {}},
  "settings": {
    "current_version": "6.1",
    "current_phase": 1,
    "confidence": 0.9,
    "income_scenario": "expected",
    "mechanics": {"banner_type": "character", "hard_pity": 90, "soft_pity_start": 74, "base_rate": 0.006, "soft_pity_increment": 0.06, "featured_rate": 0.5}
  },
  "goals": [
    {"character": "Navia", "constellation": 0, "priority": 1},
    {"character": "Arlecchino", "constellation": 0, "priority": 2}
  ],
  "banners": [
    {"character": "Navia", "version": "6.1", "phase": 1},
    {"character": "Arlecchino", "version": "6.3", "phase": 2}
  ],
  "preferences": [
    {"character": "Navia", "rank": 1, "constellation": 0, "weapon_refinement": 0, "notes": ""},
    {"character": "Arlecchino", "rank": 2, "constellation": 0, "weapon_refinement": 0, "notes": ""}
  ],
  "income": {
    "versions": [
      {"version": "6.1", "estimate": {"low": 20, "expected": 30, "high": 40}, "sources": [{"name": "Events", "estimate": {"low": 20, "expected": 30, "high": 40}}]},
      {"version": "6.3", "estimate": {"low": 20, "expected": 30, "high": 40}, "sources": [{"name": "Events", "estimate": {"low": 20, "expected": 30, "high": 40}}]}
    ]
  }
}
"""
)


def api_trace() -> None:
    print("=" * 72)
    print("API reproduction of the reported scenario")
    with TestClient(create_app()) as client:
        created = client.post("/accounts", json=SCENARIO)
        print(f"  create status={created.status_code}")
        if created.status_code != 201:
            print(created.text)
            return
        account_id = created.json()["id"]
        response = client.get(f"/accounts/{account_id}/planner/recommendation")
        print(f"  recommendation status={response.status_code}")
        print(json.dumps(response.json(), indent=2)[:3000])


api_trace()
