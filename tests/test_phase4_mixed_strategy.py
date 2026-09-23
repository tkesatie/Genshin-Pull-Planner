"""Focused Phase 4 acceptance tests for mixed character/weapon strategy."""

from domain import (
    Account,
    Banner,
    Goal,
    Ownership,
    Roadmap,
    WeaponTarget,
)
from optimizer import recommend
from planner import PlannerContext
from planner.protection import protected_goal_outcomes


def weapon_banner(name: str, version: str = "7.1", phase: int = 1) -> Banner:
    return Banner(target=WeaponTarget(name), version=version, phase=phase)


def test_weapon_only_current_goal_reaches_unified_recommendation():
    """A weapon goal can be the sole current actionable recommendation."""
    weapon = weapon_banner("Astra")
    context = PlannerContext(
        account=Account(wishes=80),
        roadmap=Roadmap(
            goals=[Goal(target=WeaponTarget("Astra"), level=0, priority=1)],
            banners=[weapon],
        ),
        current_version="7.1",
        current_phase=1,
        confidence=0.90,
    )

    result = recommend(
        context,
        budgets=[80],
        runs=200,
        seed=7,
        minimum_outcome_probability=0.0,
    )

    assert result.action == "pursue"
    assert result.outcome is not None
    assert result.outcome.target == WeaponTarget("Astra")
    assert result.target_kind == "weapon"
    assert result.target_name == "Astra"
    assert result.outcome.label == "R0"


def test_higher_priority_weapon_beats_lower_priority_character():
    """Character and weapon banners compete in one ordered opportunity set."""
    weapon = weapon_banner("Astra")
    character = Banner("Vesna", "7.1", 1)
    context = PlannerContext(
        account=Account(wishes=80),
        roadmap=Roadmap(
            goals=[
                Goal(target=WeaponTarget("Astra"), level=0, priority=1),
                Goal("Vesna", 0, 2),
            ],
            banners=[character, weapon],
        ),
        current_version="7.1",
        current_phase=1,
        confidence=0.90,
    )

    result = recommend(
        context,
        budgets=[80],
        runs=200,
        seed=11,
        minimum_outcome_probability=0.0,
    )

    assert result.action == "pursue"
    assert result.target_kind == "weapon"
    assert result.target_name == "Astra"


def test_higher_priority_character_beats_lower_priority_weapon():
    """Reversing roadmap priority reverses the selected opportunity."""
    weapon = weapon_banner("Astra")
    character = Banner("Vesna", "7.1", 1)
    context = PlannerContext(
        account=Account(wishes=80),
        roadmap=Roadmap(
            goals=[
                Goal("Vesna", 0, 1),
                Goal(target=WeaponTarget("Astra"), level=0, priority=2),
            ],
            banners=[character, weapon],
        ),
        current_version="7.1",
        current_phase=1,
        confidence=0.90,
    )

    result = recommend(
        context,
        budgets=[80],
        runs=200,
        seed=11,
        minimum_outcome_probability=0.0,
    )

    assert result.action == "pursue"
    assert result.target_kind == "character"
    assert result.target_name == "Vesna"


def test_future_character_and_weapon_goals_share_one_protected_pool():
    """Future reserves are consumed from the same account wish pool."""
    character = Banner("Vesna", "7.1", 1)
    weapon = weapon_banner("Astra", "7.2", 1)
    context = PlannerContext(
        account=Account(
            wishes=500,
            owned_characters=Ownership({"Vesna": -1}),
        ),
        roadmap=Roadmap(
            goals=[
                Goal("Vesna", 0, 1),
                Goal(target=WeaponTarget("Astra"), level=0, priority=2),
            ],
            banners=[
                Banner("Filler", "7.0", 1),
                character,
                weapon,
            ],
        ),
        current_version="7.0",
        current_phase=1,
        confidence=0.90,
    )

    outcomes = protected_goal_outcomes(context, spent=0)

    assert [outcome.goal for outcome in outcomes] == [
        Goal("Vesna", 0, 1),
        Goal(target=WeaponTarget("Astra"), level=0, priority=2),
    ]
    first, second = outcomes
    assert first.required_wishes > 0
    assert second.required_wishes > 0
    assert second.budget_at_banner == 500 - first.required_wishes


def test_recommendation_exposes_phase4_fields_for_weapon():
    """The recommendation object exposes the UI-facing unified fields."""
    weapon = weapon_banner("Astra")
    context = PlannerContext(
        account=Account(wishes=80),
        roadmap=Roadmap(
            goals=[Goal(target=WeaponTarget("Astra"), level=0, priority=1)],
            banners=[weapon],
        ),
        current_version="7.1",
        current_phase=1,
        confidence=0.90,
    )

    result = recommend(
        context,
        budgets=[20],
        runs=100,
        seed=3,
        minimum_outcome_probability=0.0,
    )

    assert result.target_kind == "weapon"
    assert result.target_name == "Astra"
    assert result.spend_limit == 20
    assert result.spend_down_to == 60
    assert result.protected_reserve == 0
    assert result.protected_goal is None
    assert result.reasons
