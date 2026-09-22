"""Goal and GoalStatus (Design Document §5, §6)."""

import pytest

from domain import (
    Account,
    Goal,
    Ownership,
    copies_needed_for,
    goal_status,
    goal_statuses,
)


@pytest.mark.parametrize(
    ("owned", "goal_constellation", "expected_copies"),
    [
        (-1, 0, 1),  # not owned -> C0 needs 1 copy
        (0, 0, 0),  # C0 -> C0 complete
        (0, 2, 2),  # C0 -> C2 needs 2 copies
        (1, 2, 1),  # C1 -> C2 needs 1 copy
        (2, 2, 0),  # C2 -> C2 complete
        (2, 0, 0),  # already beyond the goal
        (0, 3, 3),  # C0 -> C3 needs 3 copies
    ],
    ids=[
        "not-owned-c0",
        "c0-to-c0",
        "c0-to-c2",
        "c1-to-c2",
        "c2-to-c2",
        "beyond-goal",
        "c0-to-c3",
    ],
)
def test_copies_needed_table_from_design_document(
    owned, goal_constellation, expected_copies
):
    """The §6 table: max(goal_constellation - owned_constellation, 0)."""
    account = Account(owned_characters=Ownership({"Vesna": owned}))
    goal = Goal("Vesna", goal_constellation, 1)

    status = goal_status(account, goal)

    assert status.copies_needed == expected_copies


def test_goal_status_keeps_the_goal_and_the_raw_count():
    goal = Goal("Vesna", 2, 3)
    account = Account(owned_characters=Ownership({"Vesna": 0}))

    status = goal_status(account, goal)

    assert status.goal is goal
    assert status.copies_needed == 2


def test_copies_needed_for_matches_goal_status():
    goal = Goal("Vodynista", 2, 1)
    account = Account(owned_characters=Ownership({"Vodynista": 1}))
    assert copies_needed_for(account, goal) == 1
    assert goal_status(account, goal).copies_needed == 1


def test_goal_statuses_preserve_input_order():
    goals = [Goal("Tsaritsa", 0, 4), Goal("Vesna", 0, 1)]
    account = Account()
    statuses = goal_statuses(account, goals)
    assert [s.goal.character for s in statuses] == ["Tsaritsa", "Vesna"]


def test_status_is_raw_data_not_actionability():
    """§6: status says how much remains, nothing more.

    Vesna C2 while Vesna is not owned is raw data (3 copies). Deciding that
    it is blocked by the C0 prerequisite is planner logic (§9, Phase 3) and
    is deliberately not computed here.
    """
    goal = Goal("Vesna", 2, 3)
    status = goal_status(Account(), goal)
    assert status.copies_needed == 3


class TestGoalValidation:
    def test_negative_constellation_rejected(self):
        with pytest.raises(ValueError, match="level"):
            Goal("Vesna", -1, 1)

    def test_priority_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="priority"):
            Goal("Vesna", 0, 0)
