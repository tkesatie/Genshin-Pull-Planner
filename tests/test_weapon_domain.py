"""Phase 3 weapon-domain foundation tests.

These verify the existing unified Phase 2 domain model covers the weapon
side: ownership/refinement, weapon goals, weapon banners, weapon account
state, and weapon pull-state transitions. No probability or simulation.
"""

import pytest

from domain import (
    NOT_OWNED,
    Account,
    Banner,
    CharacterTarget,
    Goal,
    GoalStatus,
    Ownership,
    Roadmap,
    TargetKind,
    WeaponTarget,
    WeaponWishState,
    copies_needed_for,
    goal_status,
    goal_statuses,
    sorted_chronologically,
)


# ---------------------------------------------------------------------------
# Ownership / refinement
# ---------------------------------------------------------------------------


def test_weapon_unowned_defaults_to_not_owned():
    ownership = Ownership()

    assert ownership.owned_refinement("Staff of Homa") == NOT_OWNED
    assert not ownership.owns_weapon("Staff of Homa")


def test_weapon_owned_at_r0():
    ownership = Ownership(weapons={"Staff of Homa": 0})

    assert ownership.owned_refinement("Staff of Homa") == 0
    assert ownership.owns_weapon("Staff of Homa")


def test_higher_refinement_is_preserved():
    ownership = Ownership(weapons={"Staff of Homa": 3})

    assert ownership.owned_refinement("Staff of Homa") == 3


def test_refinement_update_returns_new_ownership():
    base = Ownership(weapons={"Staff of Homa": 1})

    updated = base.with_refinement("Staff of Homa", 2)

    assert updated.owned_refinement("Staff of Homa") == 2
    # frozen: the original is unchanged
    assert base.owned_refinement("Staff of Homa") == 1


def test_refinement_update_keeps_character_ownership():
    base = Ownership(characters={"Skirk": 1}, weapons={"Staff of Homa": 0})

    updated = base.with_refinement("Staff of Homa", 5)

    assert updated.owned_constellation("Skirk") == 1
    assert updated.owned_refinement("Staff of Homa") == 5


def test_invalid_refinement_is_rejected():
    with pytest.raises(ValueError):
        Ownership(weapons={"Staff of Homa": -2})
    with pytest.raises(ValueError):
        Ownership(weapons={"Staff of Homa": 0}).with_refinement("Staff of Homa", -2)


def test_character_ownership_regression():
    ownership = Ownership(characters={"Skirk": 2})

    assert ownership.owned_constellation("Skirk") == 2
    assert ownership.owned_constellation("Vesna") == NOT_OWNED
    assert ownership.owns("Skirk")
    assert not ownership.owns("Vesna")

    updated = ownership.with_constellation("Vesna", 0)
    assert updated.owned_constellation("Vesna") == 0
    assert updated.owned_constellation("Skirk") == 2


# ---------------------------------------------------------------------------
# Goals
# ---------------------------------------------------------------------------


def test_weapon_goal_construction_uses_unified_goal():
    goal = Goal(target=WeaponTarget("Staff of Homa"), level=1, priority=2)

    assert isinstance(goal.target, WeaponTarget)
    assert goal.target.kind is TargetKind.WEAPON
    assert goal.weapon == "Staff of Homa"
    assert goal.refinement == 1
    assert goal.level == 1
    assert goal.priority == 2


def test_weapon_goal_copies_needed_is_refinements():
    account = Account(
        owned_characters=Ownership(weapons={"Staff of Homa": 0})
    )
    goal = Goal(target=WeaponTarget("Staff of Homa"), level=1)

    assert copies_needed_for(account, goal) == 1
    status = goal_status(account, goal)
    assert isinstance(status, GoalStatus)
    assert status.copies_needed == 1


def test_completed_weapon_goal_needs_nothing():
    account = Account(
        owned_characters=Ownership(weapons={"Staff of Homa": 2})
    )
    goal = Goal(target=WeaponTarget("Staff of Homa"), level=2)

    status = goal_status(account, goal)
    assert status.copies_needed == 0


def test_weapon_goal_at_refinement_five_is_complete():
    account = Account(
        owned_characters=Ownership(weapons={"Staff of Homa": 5})
    )
    goal = Goal(target=WeaponTarget("Staff of Homa"), level=5)

    assert goal_status(account, goal).copies_needed == 0


def test_mixed_character_and_weapon_goal_statuses():
    account = Account(
        owned_characters=Ownership(
            characters={"Skirk": 1}, weapons={"Staff of Homa": 0}
        )
    )
    goals = [
        Goal("Skirk", 2, 1),
        Goal(target=WeaponTarget("Staff of Homa"), level=1, priority=2),
    ]
    statuses = goal_statuses(account, goals)

    assert [s.copies_needed for s in statuses] == [1, 1]


def test_cross_type_priority_ordering_in_one_roadmap():
    roadmap = Roadmap(
        goals=[
            Goal("Skirk", 2, 3),
            Goal(target=WeaponTarget("Staff of Homa"), level=1, priority=1),
        ]
    )

    ordered = roadmap.goals_in_priority_order()
    assert [g.priority for g in ordered] == [1, 3]
    assert [g.target.kind for g in ordered] == [
        TargetKind.WEAPON,
        TargetKind.CHARACTER,
    ]


def test_duplicate_priorities_rejected_across_types():
    with pytest.raises(ValueError, match="duplicates"):
        Roadmap(
            goals=[
                Goal("Skirk", 0, 1),
                Goal(target=WeaponTarget("Staff of Homa"), level=1, priority=1),
            ]
        )


def test_completed_goal_keeps_its_protected_priority_slot():
    roadmap = Roadmap(
        goals=[
            Goal(target=WeaponTarget("Staff of Homa"), level=1, priority=1),
            Goal("Skirk", 1, 2),
        ]
    )
    account = Account(
        owned_characters=Ownership(weapons={"Staff of Homa": 1})
    )

    # priority order is a strict protection order: the completed weapon goal
    # still occupies priority 1 and does not free that priority for others
    ordered = roadmap.goals_in_priority_order()
    assert ordered[0].target == WeaponTarget("Staff of Homa")
    assert goal_status(account, ordered[0]).copies_needed == 0
    assert goal_status(account, ordered[1]).copies_needed == 2  # unowned C1


def test_cross_type_accessor_errors_are_consistent():
    weapon_goal = Goal(target=WeaponTarget("Staff of Homa"), level=1)
    character_goal = Goal("Skirk", 1)

    with pytest.raises(AttributeError):
        _ = weapon_goal.character
    with pytest.raises(AttributeError):
        _ = weapon_goal.constellation
    with pytest.raises(AttributeError):
        _ = character_goal.weapon
    with pytest.raises(AttributeError):
        _ = character_goal.refinement


def test_invalid_weapon_goal_input_rejected():
    with pytest.raises(ValueError):
        Goal(target=WeaponTarget("Staff of Homa"), level=-1)
    with pytest.raises(ValueError):
        WeaponTarget("  ")


# ---------------------------------------------------------------------------
# Banners
# ---------------------------------------------------------------------------


def test_weapon_banner_construction():
    banner = Banner(target=WeaponTarget("Staff of Homa"), version="7.1", phase=2)

    assert banner.target == WeaponTarget("Staff of Homa")
    assert banner.target.kind is TargetKind.WEAPON
    assert banner.weapon == "Staff of Homa"
    assert banner.version == "7.1"
    assert banner.phase == 2


def test_weapon_banner_rejects_character_accessor():
    banner = Banner(target=WeaponTarget("Staff of Homa"), version="7.1")

    with pytest.raises(AttributeError):
        _ = banner.character


def test_character_banner_rejects_weapon_accessor():
    banner = Banner(character="Skirk", version="7.1")

    with pytest.raises(AttributeError):
        _ = banner.weapon


def test_mixed_banners_sort_chronologically():
    weapon = Banner(target=WeaponTarget("Staff of Homa"), version="7.1", phase=2)
    character = Banner(character="Skirk", version="7.0", phase=1)
    character2 = Banner(character="Vesna", version="7.1", phase=1)

    ordered = sorted_chronologically([weapon, character, character2])

    assert ordered == [character, character2, weapon]
    assert [b.target.kind for b in ordered] == [
        TargetKind.CHARACTER,
        TargetKind.CHARACTER,
        TargetKind.WEAPON,
    ]


def test_mixed_banners_coexist_in_one_roadmap():
    roadmap = Roadmap(
        banners=[
            Banner(target=WeaponTarget("Staff of Homa"), version="7.1", phase=1),
            Banner(character="Skirk", version="7.1", phase=1),
        ]
    )

    assert len(roadmap.banners_in_chronological_order()) == 2
    kinds = {b.target.kind for b in roadmap.banners}
    assert kinds == {TargetKind.CHARACTER, TargetKind.WEAPON}


def test_invalid_banner_target_rejected():
    with pytest.raises(TypeError):
        Banner(version="7.1")  # neither target nor character
    with pytest.raises(TypeError):
        Banner(
            character="Skirk",
            target=WeaponTarget("Staff of Homa"),
            version="7.1",
        )


# ---------------------------------------------------------------------------
# Account state
# ---------------------------------------------------------------------------


def test_weapon_pity_in_account_state():
    state = WeaponWishState(pity=42)

    assert state.pity == 42
    assert state.guarantee is False
    assert state.fate_points == 0


def test_weapon_guarantee_state():
    state = WeaponWishState(pity=10, guarantee=True)

    assert state.guarantee is True


def test_weapon_fate_points():
    state = WeaponWishState(pity=10, guarantee=False, fate_points=2)

    assert state.fate_points == 2
    with pytest.raises(ValueError):
        WeaponWishState(fate_points=-1)
    with pytest.raises(ValueError):
        WeaponWishState(pity=-1)


def test_account_holds_both_character_and_weapon_state():
    account = Account(
        current_pity=77,
        character_guarantee=True,
        capturing_radiance_counter=1,
        weapon_state=WeaponWishState(pity=30, guarantee=True, fate_points=1),
    )

    assert account.character_state.pity == 77
    assert account.character_state.guarantee is True
    assert account.weapon_state.pity == 30
    assert account.weapon_state.guarantee is True
    assert account.weapon_state.fate_points == 1


# ---------------------------------------------------------------------------
# Pull-state transitions
# ---------------------------------------------------------------------------


def test_weapon_five_star_resets_pity():
    state = WeaponWishState(pity=64, guarantee=False, fate_points=0)

    updated = state.after_five_star(next_guarantee=False, next_fate_points=0)

    assert updated.pity == 0
    assert updated.guarantee is False
    assert updated.fate_points == 0


def test_weapon_five_star_guarantee_transition():
    state = WeaponWishState(pity=10, guarantee=False, fate_points=1)

    # non-featured five-star: next five-star is guaranteed
    lost = state.after_five_star(next_guarantee=True, next_fate_points=2)
    assert lost.guarantee is True

    # featured five-star on a guarantee: guarantee clears
    won = WeaponWishState(pity=5, guarantee=True, fate_points=2).after_five_star(
        next_guarantee=False, next_fate_points=0
    )
    assert won.guarantee is False
    assert won.fate_points == 0


def test_weapon_five_star_fate_point_transition():
    state = WeaponWishState(pity=20, guarantee=False, fate_points=0)

    # off-banner five-star accumulates a fate point
    off_banner = state.after_five_star(next_guarantee=True, next_fate_points=1)
    assert off_banner.fate_points == 1
    assert off_banner.pity == 0

    # featured five-star clears fate points
    featured = WeaponWishState(
        pity=20, guarantee=False, fate_points=1
    ).after_five_star(next_guarantee=False, next_fate_points=0)
    assert featured.fate_points == 0


def test_weapon_transition_is_immutable():
    state = WeaponWishState(pity=64, guarantee=False, fate_points=1)

    state.after_five_star(next_guarantee=True, next_fate_points=2)

    assert state.pity == 64
    assert state.guarantee is False
    assert state.fate_points == 1
