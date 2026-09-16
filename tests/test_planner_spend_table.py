"""Spending table (Design Document §18 Phase 3)."""

import pytest

from domain import (
    CHARACTER_EVENT_BANNER,
    Account,
    Banner,
    Goal,
    Ownership,
    Roadmap,
)
from planner import PlannerContext, protected_goal_outcomes, safe_spend, spend_table
from probability import cumulative_probability


def make_boundary_context() -> PlannerContext:
    """One active single-copy goal (Vesna C0) and one protectable future
    goal (Tsaritsa C0), 100 wishes, 50% threshold: the safe boundary
    falls inside the spend range instead of at its edge."""
    account = Account(wishes=100)
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)],
    )
    return PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0", confidence=0.5
    )


def test_rows_cover_zero_to_wishes(doc_context):
    rows = spend_table(doc_context)
    assert len(rows) == 41  # spends 0..40 inclusive
    assert rows[0].wishes_spent == 0
    assert rows[-1].wishes_spent == 40


def test_row_zero_confidence_is_zero(doc_context):
    assert spend_table(doc_context)[0].goal_confidence == 0.0


def test_goal_confidence_uses_the_account_state(doc_context):
    """The goal column follows the account's real pity/guarantee state
    (§10.2) - unlike protection, which is conservative (§14)."""
    rows = spend_table(doc_context)
    curve = cumulative_probability(40, 0, False, CHARACTER_EVENT_BANNER)
    for row in rows:
        assert row.goal_confidence == pytest.approx(
            float(curve[row.wishes_spent])
        )
    assert rows[-1].goal_confidence == pytest.approx(0.119, abs=1e-3)


def test_goal_confidence_is_monotone(doc_context):
    confidences = [row.goal_confidence for row in spend_table(doc_context)]
    assert confidences == sorted(confidences)


def test_each_row_evaluates_protection_at_that_spend(doc_context):
    for row in spend_table(doc_context):
        assert row.protected == tuple(
            protected_goal_outcomes(doc_context, spent=row.wishes_spent)
        )


def test_doc_example_every_row_is_unsafe(doc_context):
    """40 wishes cannot protect Tsaritsa C0 at 90%: the table says so for
    every spend, including 0 (§1: honest probabilities)."""
    assert all(
        not row.all_protected_meet_threshold for row in spend_table(doc_context)
    )


def test_safe_boundary_flips_exactly_at_safe_spend():
    context = make_boundary_context()
    rows = spend_table(context)
    bound = safe_spend(context)
    assert bound == 20
    assert [row.all_protected_meet_threshold for row in rows] == [
        spend <= bound for spend in range(101)
    ]
    # At the boundary the budget equals the reserve exactly (§10.3).
    assert rows[20].protected[0].budget_at_banner == 80
    assert rows[20].protected[0].required_wishes == 80
    assert rows[20].protected[0].meets_threshold is True
    assert rows[21].protected[0].meets_threshold is False


def test_step_controls_resolution(doc_context):
    rows = spend_table(doc_context, step=10)
    assert [row.wishes_spent for row in rows] == [0, 10, 20, 30, 40]


def test_zero_step_rejected(doc_context):
    with pytest.raises(ValueError, match="step"):
        spend_table(doc_context, step=0)


def test_multi_copy_active_goal_rejected():
    """Phase 3 recognizes the multi-copy target and refuses to
    approximate it (§18, §10.4): Vesna C2 from C0 needs 2 copies."""
    account = Account(wishes=40, owned_characters=Ownership({"Vesna": 0}))
    roadmap = Roadmap(
        goals=[Goal("Vesna", 2, 1)], banners=[Banner("Vesna", "7.0", 1)]
    )
    context = PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0"
    )
    with pytest.raises(ValueError, match="single-copy"):
        spend_table(context)


def test_zero_active_goals_rejected():
    """A satisfied current-banner goal is not something to spend toward."""
    account = Account(wishes=40, owned_characters=Ownership({"Vesna": 0}))
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1)], banners=[Banner("Vesna", "7.0", 1)]
    )
    context = PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0"
    )
    with pytest.raises(ValueError, match="no active goal"):
        spend_table(context)


def test_multiple_active_goals_rejected():
    """Degenerate duplicate goals must not be silently resolved (§9)."""
    account = Account(wishes=40)
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1), Goal("Vesna", 0, 2)],
        banners=[Banner("Vesna", "7.0", 1)],
    )
    context = PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0"
    )
    with pytest.raises(ValueError, match="multiple active goals"):
        spend_table(context)