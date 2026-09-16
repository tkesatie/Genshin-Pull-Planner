"""Goal matching and dependencies (Design Document §9)."""

from domain import Account, Banner, Goal, Ownership, Roadmap
from planner import (
    GoalState,
    PlannerContext,
    actionable_goals,
    evaluate_goal,
    evaluate_goals,
    relevant_goal_evaluations,
)


def context_for(account: Account, roadmap: Roadmap) -> PlannerContext:
    return PlannerContext(account=account, roadmap=roadmap, current_version="7.0")


def test_doc_example_states(doc_context):
    """The §9 story: Vesna C0 active, Vesna C2 blocked by Vesna C0,
    Vodynista C0 satisfied, Tsaritsa C0 active on a future banner."""
    evaluations = evaluate_goals(doc_context)
    assert [
        (e.goal, e.copies_needed, e.state, e.blocked_by) for e in evaluations
    ] == [
        (Goal("Vesna", 0, 1), 1, GoalState.ACTIVE, None),
        (Goal("Vodynista", 0, 2), 0, GoalState.SATISFIED, None),
        (Goal("Vesna", 2, 3), 3, GoalState.BLOCKED, Goal("Vesna", 0, 1)),
        (Goal("Tsaritsa", 0, 4), 1, GoalState.ACTIVE, None),
    ]


def test_next_banner_is_at_or_after_current(doc_context):
    by_character = {
        e.goal.character: e.next_banner for e in evaluate_goals(doc_context)
    }
    assert by_character["Vesna"] == Banner("Vesna", "7.0", 1)
    assert by_character["Tsaritsa"] == Banner("Tsaritsa", "7.1", 1)
    assert by_character["Vodynista"] == Banner("Vodynista", "7.2", 1)


def test_relevant_goals_are_the_banner_character_only(doc_context):
    """Both Vesna goals are relevant to the Vesna banner (§9), in
    priority order."""
    relevant = relevant_goal_evaluations(doc_context)
    assert [e.goal.priority for e in relevant] == [1, 3]


def test_actionable_excludes_blocked_and_other_characters(doc_context):
    """Vesna C2 is relevant but blocked; Tsaritsa C0 is unsatisfied but
    belongs to a future banner - neither is actionable (§9)."""
    active = actionable_goals(doc_context)
    assert [e.goal for e in active] == [Goal("Vesna", 0, 1)]


def test_dependency_chain_blocks_through_each_level():
    account = Account()
    roadmap = Roadmap(
        goals=[Goal("X", 0, 1), Goal("X", 1, 2), Goal("X", 2, 3)],
        banners=[Banner("X", "7.0", 1)],
    )
    states = {
        e.goal.constellation: e for e in evaluate_goals(context_for(account, roadmap))
    }
    assert states[0].state is GoalState.ACTIVE
    assert states[1].state is GoalState.BLOCKED
    assert states[1].blocked_by == Goal("X", 0, 1)
    assert states[2].state is GoalState.BLOCKED
    assert states[2].blocked_by == Goal("X", 1, 2)


def test_satisfied_blocker_frees_the_higher_goal():
    """Owning X at C1 satisfies the C0 goal; C2 becomes active with the
    §10.4 single-copy count (C2 from C1 -> 1 copy)."""
    account = Account(owned_characters=Ownership({"X": 1}))
    roadmap = Roadmap(
        goals=[Goal("X", 0, 1), Goal("X", 2, 2)],
        banners=[Banner("X", "7.0", 1)],
    )
    states = {
        e.goal.constellation: e for e in evaluate_goals(context_for(account, roadmap))
    }
    assert states[0].state is GoalState.SATISFIED
    assert states[2].state is GoalState.ACTIVE
    assert states[2].copies_needed == 1


def test_dependency_ignores_priorities():
    """Priorities protect roadmap objectives (§2); they do not change
    pull order on a banner (§9): C2 still waits for C0 even when the C2
    goal has the higher priority."""
    account = Account()
    roadmap = Roadmap(
        goals=[Goal("X", 2, 1), Goal("X", 0, 3)],
        banners=[Banner("X", "7.0", 1)],
    )
    states = {
        e.goal.constellation: e for e in evaluate_goals(context_for(account, roadmap))
    }
    assert states[0].state is GoalState.ACTIVE
    assert states[2].state is GoalState.BLOCKED
    assert states[2].blocked_by == Goal("X", 0, 3)


def test_no_lower_goal_leaves_multi_copy_goal_active():
    account = Account()
    roadmap = Roadmap(
        goals=[Goal("X", 2, 1)],
        banners=[Banner("X", "7.0", 1)],
    )
    states = {
        e.goal.constellation: e for e in evaluate_goals(context_for(account, roadmap))
    }
    assert states[2].state is GoalState.ACTIVE
    assert states[2].copies_needed == 3  # C2 from not-owned (§4.2)


def test_goal_without_upcoming_banner_stays_visible():
    """A goal whose character has no banner at-or-after the current one
    is surfaced with next_banner None - not silently dropped (§8)."""
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
        banners=[Banner("Tsaritsa", "7.0", 1), Banner("Vesna", "6.9", 1)],
    )
    evaluations = {
        e.goal.character: e
        for e in evaluate_goals(context_for(Account(), roadmap))
    }
    assert evaluations["Vesna"].next_banner is None
    assert evaluations["Vesna"].state is GoalState.ACTIVE
    assert evaluations["Tsaritsa"].next_banner == Banner("Tsaritsa", "7.0", 1)


def test_evaluate_goal_matches_evaluate_goals(doc_context):
    solo = evaluate_goal(doc_context, Goal("Vesna", 2, 3))
    assert solo in evaluate_goals(doc_context)