"""Roadmap (Design Document §8)."""

import pytest

from domain import Banner, Goal, Roadmap


def test_goals_in_priority_order():
    roadmap = Roadmap(
        goals=[
            Goal("Tsaritsa", 0, 4),
            Goal("Vesna", 0, 1),
            Goal("Vodynista", 0, 2),
            Goal("Vesna", 2, 3),
        ]
    )

    assert [g.character for g in roadmap.goals_in_priority_order()] == [
        "Vesna",
        "Vodynista",
        "Vesna",
        "Tsaritsa",
    ]


def test_priority_order_and_chronological_order_are_independent():
    """§7: banner order and priority order may differ deliberately."""
    roadmap = Roadmap(
        goals=[Goal("Vodynista", 0, 1)],
        banners=[Banner("Vodynista", "7.2", 1)],
    )

    assert roadmap.goals_in_priority_order()[0].character == "Vodynista"
    assert roadmap.banners_in_chronological_order()[0].version == "7.2"


def test_banners_in_chronological_order():
    roadmap = Roadmap(
        banners=[
            Banner("Vodynista", "7.2", 1),
            Banner("Vesna", "7.0", 1),
            Banner("Tsaritsa", "7.1", 1),
        ]
    )

    assert [b.character for b in roadmap.banners_in_chronological_order()] == [
        "Vesna",
        "Tsaritsa",
        "Vodynista",
    ]


def test_banners_for_character_in_chronological_order():
    roadmap = Roadmap(
        banners=[
            Banner("Vodynista", "7.2", 1),
            Banner("Vesna", "7.2", 2),  # rerun
            Banner("Vesna", "7.0", 1),
        ]
    )

    vesna = roadmap.banners_for("Vesna")
    assert [b.version for b in vesna] == ["7.0", "7.2"]
    assert vesna[1].phase == 2

    assert roadmap.banners_for("Tsaritsa") == []


def test_duplicate_goal_priorities_are_rejected():
    with pytest.raises(ValueError, match="duplicates"):
        Roadmap(goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 1)])


def test_same_character_may_hold_several_priorities():
    """§5: Vesna C0 and Vesna C2 are separate objectives."""
    roadmap = Roadmap(goals=[Goal("Vesna", 0, 1), Goal("Vesna", 2, 3)])
    assert len(roadmap.goals) == 2


def test_empty_roadmap_is_allowed():
    roadmap = Roadmap()
    assert roadmap.goals == []
    assert roadmap.banners == []
