"""Shared fixtures built from the design document's running example."""

import pytest

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Preference,
    Roadmap,
    VersionIncome,
)
from planner import PlannerContext


@pytest.fixture
def doc_account() -> Account:
    """Account state for the design-document example (§4.1, §8).

    Vesna and Tsaritsa are not owned; Vodynista is owned at C0.
    """
    return Account(
        current_pity=0,
        character_guarantee=False,
        owned_characters=Ownership({"Vodynista": 0}),
        wishes=40,
    )


@pytest.fixture
def doc_roadmap() -> Roadmap:
    """The roadmap from §8 (Vesna appears in two separate goals)."""
    return Roadmap(
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
    )


@pytest.fixture
def doc_income() -> IncomeForecast:
    """Future income for the doc example (§16): ~30 expected per version.

    IncomeForecast values are future income from the current account
    state (planner.context): they arrive after the account's existing 40
    wishes. Low/expected/high brackets at ±10.
    """
    return IncomeForecast(
        versions=[
            VersionIncome("7.0", estimate=IncomeEstimate(20, 30, 40)),
            VersionIncome("7.1", estimate=IncomeEstimate(20, 30, 40)),
            VersionIncome("7.2", estimate=IncomeEstimate(30, 40, 50)),
        ]
    )


@pytest.fixture
def doc_context(doc_account, doc_roadmap) -> PlannerContext:
    """The design-document example, planner at Vesna 7.0 phase 1.

    Default 90% threshold and no income forecast: the planner must be
    able to answer "do not spend".
    """
    return PlannerContext(
        account=doc_account,
        roadmap=doc_roadmap,
        current_version="7.0",
        current_phase=1,
    )


@pytest.fixture
def doc_plan(doc_roadmap) -> "SpendPlan":
    """The Phase 4 running-example plan: pursue Vesna C0 with everything.

    All 40 wishes are capped onto the current Vesna banner (target C0 - a
    desired resulting constellation, not a copy count, §4.2/§12); the
    Tsaritsa and Vodynista banners have no entries and are therefore
    skipped, not stopped.
    """
    from simulation import PlannedSpend, SpendPlan

    return SpendPlan(
        entries=(
            PlannedSpend(
                Banner("Vesna", "7.0", 1), target_constellation=0, budget=40
            ),
        )
    )


@pytest.fixture
def api_client():
    """A TestClient over a freshly built application (§18 Phase 6).

    Each test gets its own repository and job store: `create_app` owns its
    storage, so nothing leaks between tests.
    """
    from fastapi.testclient import TestClient

    from api import create_app

    with TestClient(create_app()) as client:
        registered = client.post("/auth/register", json={"username": "test-user", "password": "test-password"})
        assert registered.status_code == 201, registered.text
        logged_in = client.post("/auth/login", json={"username": "test-user", "password": "test-password"})
        assert logged_in.status_code == 200, logged_in.text
        yield client


@pytest.fixture
def doc_payload() -> dict:
    """The design-document example as an account-creation request body.

    The same account, roadmap, preferences and income the other fixtures
    build directly - at Vesna 7.0 phase 1 with the default 90% threshold.
    """
    return {
        "label": "doc example",
        "account": {
            "current_pity": 0,
            "character_guarantee": False,
            "wishes": 40,
            "owned_characters": {"Vodynista": 0},
        },
        "settings": {"current_version": "7.0", "current_phase": 1},
        "goals": [
            {"character": "Vesna", "constellation": 0, "priority": 1},
            {"character": "Vodynista", "constellation": 0, "priority": 2},
            {"character": "Vesna", "constellation": 2, "priority": 3},
            {"character": "Tsaritsa", "constellation": 0, "priority": 4},
        ],
        "banners": [
            {"character": "Vesna", "version": "7.0", "phase": 1},
            {"character": "Tsaritsa", "version": "7.1", "phase": 1},
            {"character": "Vodynista", "version": "7.2", "phase": 1},
        ],
        "preferences": [
            {
                "character": "Vesna",
                "rank": 1,
                "constellation": 2,
                "weapon_refinement": 1,
            },
            {
                "character": "Vesna",
                "rank": 2,
                "constellation": 1,
                "weapon_refinement": 1,
            },
            {"character": "Vesna", "rank": 3, "constellation": 2},
            {"character": "Vesna", "rank": 4, "constellation": 1},
            {"character": "Vesna", "rank": 5, "constellation": 0},
        ],
    }


@pytest.fixture
def doc_account_id(api_client, doc_payload) -> str:
    """A stored design-document account, ready to be planned against."""
    response = api_client.post("/accounts", json=doc_payload)
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def doc_preferences() -> tuple[Preference, ...]:
    """Vesna's preference chain from the design document (§15).

    C2R1 > C1R1 > C2 > C1 > C0. The chain deliberately contains duplicate
    constellations (C2 twice, C1 twice) to exercise outcome collapsing.
    """
    return (
        Preference("Vesna", 1, 2, weapon_refinement=1),
        Preference("Vesna", 2, 1, weapon_refinement=1),
        Preference("Vesna", 3, 2),
        Preference("Vesna", 4, 1),
        Preference("Vesna", 5, 0),
    )

