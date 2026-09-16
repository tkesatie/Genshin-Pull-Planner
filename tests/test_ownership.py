"""Ownership semantics (Design Document §4.2)."""

import dataclasses

import pytest

from domain import NOT_OWNED, Ownership


class TestOwnedConstellationLookup:
    def test_known_character_returns_its_constellation(self):
        ownership = Ownership({"Vesna": 2})
        assert ownership.owned_constellation("Vesna") == 2

    def test_unknown_character_returns_not_owned_sentinel(self):
        ownership = Ownership({"Vesna": 0})
        assert ownership.owned_constellation("Tsaritsa") == NOT_OWNED
        assert ownership.owned_constellation("Tsaritsa") == -1

    def test_empty_ownership_returns_not_owned_for_everything(self):
        assert Ownership().owned_constellation("Vesna") == NOT_OWNED

    def test_dict_style_get_mirrors_design_document_usage(self):
        """§4.2 conceptual usage: account.owned_characters.get("Vesna", -1)."""
        ownership = Ownership({"Vesna": 0})
        assert ownership.get("Vesna", -1) == 0
        assert ownership.get("Tsaritsa", -1) == -1

    def test_owns_is_true_for_any_owned_constellation(self):
        ownership = Ownership({"Vesna": 0, "Vodynista": 3})
        assert ownership.owns("Vesna")
        assert ownership.owns("Vodynista")

    def test_owns_is_false_for_missing_character(self):
        assert not Ownership({}).owns("Vesna")


class TestValidation:
    def test_constellation_below_not_owned_is_rejected(self):
        with pytest.raises(ValueError, match="Tsaritsa"):
            Ownership({"Tsaritsa": -2})

    def test_not_owned_is_the_only_negative_value(self):
        Ownership({"Tsaritsa": -1})  # does not raise


class TestImmutability:
    def test_with_constellation_returns_new_instance(self):
        original = Ownership({"Vesna": 0})
        updated = original.with_constellation("Vesna", 1)

        assert updated.owned_constellation("Vesna") == 1
        assert original.owned_constellation("Vesna") == 0  # unchanged

    def test_with_constellation_adds_new_character(self):
        updated = Ownership({}).with_constellation("Vesna", 0)
        assert updated.owned_constellation("Vesna") == 0

    def test_with_constellation_validates(self):
        with pytest.raises(ValueError):
            Ownership({}).with_constellation("Vesna", -2)

    def test_ownership_is_frozen(self):
        ownership = Ownership({"Vesna": 0})
        with pytest.raises(dataclasses.FrozenInstanceError):
            ownership.characters = {}
