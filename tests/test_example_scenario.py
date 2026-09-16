"""Integration: the design document's running example (§8, §9).

Phase 1 asserts only raw domain data: the roadmap is preserved as given and
`goal_statuses()` correctly computes each goal's raw copies needed from the
account. Whether Vesna C2 is *actionable* while Vesna C0 is a prerequisite
is planner logic (§9, Phase 3) and is deliberately not asserted here.
"""

from domain import Banner, Goal, goal_statuses


def test_roadmap_preserves_all_four_goals(doc_roadmap):
    assert doc_roadmap.goals_in_priority_order() == [
        Goal("Vesna", 0, 1),
        Goal("Vodynista", 0, 2),
        Goal("Vesna", 2, 3),
        Goal("Tsaritsa", 0, 4),
    ]


def test_roadmap_preserves_banner_schedule(doc_roadmap):
    assert doc_roadmap.banners_in_chronological_order() == [
        Banner("Vesna", "7.0", 1),
        Banner("Tsaritsa", "7.1", 1),
        Banner("Vodynista", "7.2", 1),
    ]


def test_goal_statuses_compute_raw_copies_needed(doc_account, doc_roadmap):
    statuses = goal_statuses(doc_account, doc_roadmap.goals)

    assert [
        (status.goal.character, status.goal.constellation, status.copies_needed)
        for status in statuses
    ] == [
        ("Vesna", 0, 1),  # not owned -> 1 copy
        ("Vodynista", 0, 0),  # owned at C0 -> nothing remains
        ("Vesna", 2, 3),  # not owned -> 2 - (-1) = 3 copies
        ("Tsaritsa", 0, 1),  # not owned -> 1 copy
    ]


def test_account_knows_nothing_about_the_roadmap(doc_account):
    """The account only knows ownership; it knows nothing about goals."""
    assert doc_account.owns("Vodynista")
    assert not doc_account.owns("Vesna")
    assert doc_account.wishes == 40
