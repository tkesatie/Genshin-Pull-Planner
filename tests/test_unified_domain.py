"""Phase 2 unified domain model tests."""

import pytest

from domain import (
    Account,
    Banner,
    CharacterTarget,
    Goal,
    Ownership,
    TargetKind,
    WeaponTarget,
    WeaponWishState,
    copies_needed_for,
)


def test_character_and_weapon_targets_share_one_target_model():
    character = CharacterTarget("Skirk")
    weapon = WeaponTarget("Example Weapon")

    assert isinstance(character, type(weapon))
    assert character.kind is TargetKind.CHARACTER
    assert weapon.kind is TargetKind.WEAPON
    assert character.name == "Skirk"
    assert weapon.name == "Example Weapon"


def test_character_goal_compatibility_constructor_still_works():
    goal = Goal("Skirk", 2, 3)

    assert goal.target == CharacterTarget("Skirk")
    assert goal.level == 2
    assert goal.constellation == 2
    assert goal.priority == 3


def test_weapon_goal_uses_same_goal_type():
    goal = Goal(target=WeaponTarget("Example Weapon"), level=1, priority=2)

    assert goal.target.kind is TargetKind.WEAPON
    assert goal.weapon == "Example Weapon"
    assert goal.refinement == 1


def test_weapon_and_character_goals_can_share_one_roadmap():
    from domain import Roadmap

    roadmap = Roadmap(
        goals=[
            Goal(target=WeaponTarget("Example Weapon"), level=1, priority=2),
            Goal("Skirk", 2, 3),
        ]
    )

    assert [g.priority for g in roadmap.goals_in_priority_order()] == [2, 3]
    assert [g.target.kind for g in roadmap.goals_in_priority_order()] == [
        TargetKind.WEAPON,
        TargetKind.CHARACTER,
    ]


def test_weapon_goal_completion_uses_refinement():
    account = Account(
        owned_characters=Ownership(weapons={"Example Weapon": 1})
    )
    goal = Goal(target=WeaponTarget("Example Weapon"), level=2, priority=1)

    assert copies_needed_for(account, goal) == 1


def test_weapon_account_state_is_part_of_account():
    account = Account(
        weapon_state=WeaponWishState(pity=17, guarantee=True, fate_points=1)
    )

    assert account.weapon_state.pity == 17
    assert account.weapon_state.guarantee is True
    assert account.weapon_state.fate_points == 1


def test_weapon_five_star_transition_resets_pity():
    state = WeaponWishState(pity=64, guarantee=False, fate_points=1)
    updated = state.after_five_star(
        featured=True,
        next_guarantee=False,
        next_fate_points=0,
    )

    assert updated.pity == 0
    assert updated.guarantee is False
    assert updated.fate_points == 0


def test_weapon_banner_uses_same_banner_type():
    banner = Banner(
        target=WeaponTarget("Example Weapon"),
        version="7.1",
        phase=1,
    )

    assert banner.target.kind is TargetKind.WEAPON
    assert banner.weapon == "Example Weapon"


def test_weapon_banner_does_not_expose_character_accessor():
    banner = Banner(
        target=WeaponTarget("Example Weapon"),
        version="7.1",
    )

    with pytest.raises(AttributeError):
        _ = banner.character


def test_global_priority_is_still_unique():
    from domain import Roadmap

    with pytest.raises(ValueError, match="duplicates"):
        Roadmap(
            goals=[
                Goal("Skirk", 0, 1),
                Goal(target=WeaponTarget("Example Weapon"), level=1, priority=1),
            ]
        )
