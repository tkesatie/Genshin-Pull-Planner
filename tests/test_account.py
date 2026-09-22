"""Account model (Design Document §4.1)."""

import dataclasses

import pytest

from domain import Account, Ownership, WeaponWishState


def test_account_contains_character_and_weapon_state():
    """Phase 2: account contains both banner-state domains plus wishes/ownership."""
    fields = [f.name for f in dataclasses.fields(Account)]
    assert fields == [
        "current_pity",
        "character_guarantee",
        "owned_characters",
        "wishes",
        "capturing_radiance_counter",
        "weapon_state",
    ]


def test_defaults_represent_a_fresh_account():
    account = Account()
    assert account.current_pity == 0
    assert account.character_guarantee is False
    assert account.owned_characters == Ownership()
    assert account.wishes == 0
    assert account.capturing_radiance_counter == 0
    assert account.weapon_state == WeaponWishState()


def test_delegates_ownership_lookups():
    account = Account(owned_characters=Ownership({"Vesna": 1}))
    assert account.owned_constellation("Vesna") == 1
    assert account.owned_constellation("Tsaritsa") == -1
    assert account.owns("Vesna")
    assert not account.owns("Tsaritsa")


def test_negative_pity_is_rejected():
    with pytest.raises(ValueError, match="current_pity"):
        Account(current_pity=-1)


def test_negative_wishes_are_rejected():
    with pytest.raises(ValueError, match="wishes"):
        Account(wishes=-5)


def test_wishes_is_a_possession_not_a_decision():
    account = Account(wishes=40)
    assert account.wishes == 40
    assert account.capturing_radiance_counter == 0
