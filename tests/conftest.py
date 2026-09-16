"""Shared fixtures built from the design document's running example."""

import pytest

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

