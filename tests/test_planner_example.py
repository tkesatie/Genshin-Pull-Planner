"""Integration: the Phase 3 planner on the design document's running
example (§8, §9, §13, §14, §18 Phase 3)."""

import pytest

from domain import Banner, Goal, CHARACTER_EVENT_BANNER
from planner import (
    GoalState,
    PlannerContext,
    actionable_goals,
    current_banner,
    evaluate_goals,
    protected_goal_outcomes,
    relevant_goal_evaluations,
    safe_spend,
    spend_table,
)
from probability import cumulative_probability


def test_doc_example_at_90_percent(doc_context):
    """The honest headline: with 40 wishes and no income, a 90%
    protected Tsaritsa C0 cannot be maintained - "do not spend"."""
    assert current_banner(doc_context) == Banner("Vesna", "7.0", 1)

    # Relevant: both Vesna goals; only C0 actionable; C2 blocked by C0 (§9).
    relevant = relevant_goal_evaluations(doc_context)
    assert [e.goal.priority for e in relevant] == [1, 3]
    assert [e.goal for e in actionable_goals(doc_context)] == [Goal("Vesna", 0, 1)]
    assert relevant[1].state is GoalState.BLOCKED
    assert relevant[1].blocked_by == Goal("Vesna", 0, 1)

    # Protection: Tsaritsa C0 at 7.1 is the only protected goal (§13).
    protected = protected_goal_outcomes(doc_context)
    assert [o.goal.character for o in protected] == ["Tsaritsa"]
    assert protected[0].required_wishes == 155

    # The answer (§1, §14): spending is not safe at any level.
    assert safe_spend(doc_context) == 0
    assert all(
        not row.all_protected_meet_threshold for row in spend_table(doc_context)
    )


def test_doc_example_with_income_at_50_percent(doc_context, doc_income):
    """With income and a 50% threshold the planner quantifies a real
    safe spend - and proves it tracks WHEN income arrives (§14)."""
    context = PlannerContext(
        account=doc_context.account,
        roadmap=doc_context.roadmap,
        current_version="7.0",
        income=doc_income,
        confidence=0.5,
    )
    # Naive `wishes - required` would floor at 0 (40 - 80); the 60
    # expected income credited through 7.1 makes 20 wishes safe now.
    assert safe_spend(context) == 20

    rows = {row.wishes_spent: row for row in spend_table(context)}
    boundary = rows[20]
    assert boundary.goal_confidence == pytest.approx(
        float(cumulative_probability(20, 0, False, CHARACTER_EVENT_BANNER)[20])
    )
    tsaritsa = boundary.protected[0]
    assert tsaritsa.budget_at_banner == 80  # 40 - 20 spent + 60 income
    assert tsaritsa.meets_threshold is True
    assert rows[21].protected[0].meets_threshold is False


def test_rerun_from_a_later_banner(doc_account, doc_roadmap):
    """The planner is re-run after every account update (§2): from 7.1
    the Vesna goals become past business and Tsaritsa becomes current."""
    context = PlannerContext(
        account=doc_account, roadmap=doc_roadmap, current_version="7.1"
    )
    assert current_banner(context) == Banner("Tsaritsa", "7.1", 1)
    assert [e.goal for e in actionable_goals(context)] == [Goal("Tsaritsa", 0, 4)]
    # Vesna C0/C2 have no banner at-or-after 7.1: surfaced, not protected.
    evaluations = {e.goal.character: e for e in evaluate_goals(context)}
    assert evaluations["Vesna"].next_banner is None
    assert protected_goal_outcomes(context) == []
    assert safe_spend(context) == doc_account.wishes