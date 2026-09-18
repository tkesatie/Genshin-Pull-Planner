"""Account model (Design Document §4.1)."""

import dataclasses

import pytest

from domain import Account, Ownership


def test_account_holds_only_ownership_state_not_desires():
    """§4.1: the account describes what the user owns, not what they want.

    Goals live in the roadmap (§5), never on the account.
    """
    assert [f.name for f in dataclasses.fields(Account)] == [
        "current_pity",
        "character_guarantee",
        "owned_characters",
        "wishes",
        "capturing_radiance_counter",
    ]


def test_defaults_represent_a_fresh_account():
    account = Account()
    assert account.current_pity == 0
    assert account.character_guarantee is False
    assert account.owned_characters == Ownership()
    assert account.wishes == 0
    assert account.capturing_radiance_counter == 0


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
    """`wishes` records wishes currently owned.

    A spending decision (`wishes_to_spend`) belongs to the strategy layer
    (§12-§14) and must never be stored as account state.
    """
    account = Account(wishes=40)
    assert account.wishes == 40
    assert account.capturing_radiance_counter == 0
