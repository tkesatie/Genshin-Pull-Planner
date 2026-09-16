"""Preferences (Design Document §15)."""

import dataclasses

import pytest

from domain import Preference, sort_by_rank


def test_fields_match_phase_1_model():
    """§15 lists character, rank, constellation, weapon_refinement, tier, notes.

    `tier` is intentionally omitted in Phase 1 until its meaning is decided.
    """
    assert [f.name for f in dataclasses.fields(Preference)] == [
        "character",
        "rank",
        "constellation",
        "weapon_refinement",
        "notes",
    ]


class TestLabels:
    def test_constellation_only_label(self):
        assert Preference("Vesna", 5, 0).label == "C0"
        assert Preference("Vesna", 3, 2).label == "C2"

    def test_refinement_appends_r(self):
        assert Preference("Vesna", 1, 2, weapon_refinement=1).label == "C2R1"
        assert Preference("Vesna", 2, 1, weapon_refinement=1).label == "C1R1"

    def test_zero_refinement_means_no_weapon_in_label(self):
        """Refinement 0 is "no weapon wanted", consistent with C-numbering."""
        assert Preference("Vesna", 4, 2, weapon_refinement=0).label == "C2"


class TestValidation:
    def test_rank_starts_at_one(self):
        with pytest.raises(ValueError, match="rank"):
            Preference("Vesna", 0, 0)

    def test_negative_constellation_rejected(self):
        with pytest.raises(ValueError, match="constellation"):
            Preference("Vesna", 1, -1)

    def test_negative_refinement_rejected(self):
        with pytest.raises(ValueError, match="weapon_refinement"):
            Preference("Vesna", 1, 2, weapon_refinement=-1)


def test_sort_by_rank_orders_the_preference_chain():
    """The §15 example chain, supplied unordered."""
    chain = [
        Preference("Vesna", 3, 2),
        Preference("Vesna", 1, 2, weapon_refinement=1),
        Preference("Vesna", 5, 0),
        Preference("Vesna", 2, 1, weapon_refinement=1),
        Preference("Vesna", 4, 2),
    ]

    ordered = sort_by_rank(chain)

    assert ordered == [
        Preference("Vesna", 1, 2, weapon_refinement=1),  # C2R1
        Preference("Vesna", 2, 1, weapon_refinement=1),  # C1R1
        Preference("Vesna", 3, 2),  # C2
        Preference("Vesna", 4, 2),  # (duplicate of C2 in the doc example)
        Preference("Vesna", 5, 0),  # C0
    ]
