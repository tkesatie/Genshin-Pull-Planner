"""Basic protected-goal calculation: the sequential independent-reserve
approximation (Design Document §13 step 5, §14)."""

import pytest

from domain import (
    CHARACTER_EVENT_BANNER,
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Roadmap,
    VersionIncome,
    WishMechanics,
)
from planner import PlannerContext, evaluate_goals, protected_goal_outcomes
from probability import cumulative_probability, wishes_for_confidence


def test_doc_example_protects_only_tsaritsa(doc_context):
    outcomes = protected_goal_outcomes(doc_context)
    assert [(o.goal, o.banner) for o in outcomes] == [
        (Goal("Tsaritsa", 0, 4), Banner("Tsaritsa", "7.1", 1)),
    ]


def test_satisfied_and_current_banner_goals_are_not_protected(doc_context):
    """Vodynista C0 needs nothing; Vesna C2 belongs to the current banner
    (and is blocked anyway) - protection is strictly future (§13)."""
    characters = [o.goal.character for o in protected_goal_outcomes(doc_context)]
    assert "Vodynista" not in characters
    assert "Vesna" not in characters


def test_required_wishes_is_the_conservative_full_reserve(doc_context):
    """Reserves are computed at pity 0 with no guarantee, regardless of
    the account's actual state (Phase 3 carries nothing forward, §14)."""
    outcome = protected_goal_outcomes(doc_context)[0]
    assert outcome.required_wishes == wishes_for_confidence(
        0.9, 0, False, CHARACTER_EVENT_BANNER
    )
    assert outcome.required_wishes == 155  # pinned Phase 2 anchor


def test_budget_starts_at_account_wishes(doc_context):
    assert protected_goal_outcomes(doc_context)[0].budget_at_banner == 40


def test_spending_now_shrinks_the_budget(doc_context):
    outcomes = protected_goal_outcomes(doc_context, spent=10)
    assert outcomes[0].budget_at_banner == 30


def test_confidence_is_the_single_copy_curve(doc_context):
    outcome = protected_goal_outcomes(doc_context)[0]
    expected = float(cumulative_probability(40, 0, False, CHARACTER_EVENT_BANNER)[40])
    assert outcome.confidence == pytest.approx(expected)
    assert outcome.confidence == pytest.approx(0.119, abs=1e-3)


def test_doc_example_cannot_meet_the_threshold(doc_context):
    """40 wishes cannot hold a 90% protected goal from fresh pity - the
    honest answer is that nothing is safe to spend (§1)."""
    outcome = protected_goal_outcomes(doc_context)[0]
    assert outcome.meets_threshold is False


def test_income_and_lower_threshold_can_meet_it(doc_account, doc_roadmap, doc_income):
    context = PlannerContext(
        account=doc_account,
        roadmap=doc_roadmap,
        current_version="7.0",
        income=doc_income,
        confidence=0.5,
    )
    outcome = protected_goal_outcomes(context)[0]
    # 40 wishes + 60 expected income through 7.1 >= the 50% reserve (80).
    assert outcome.budget_at_banner == 100
    assert outcome.required_wishes == 80
    assert outcome.meets_threshold is True


def test_income_arrives_between_banners_not_all_upfront(doc_account):
    """Goal B's budget contains only the income credited after goal A's
    banner: the model tracks WHEN income arrives (§14)."""
    roadmap = Roadmap(
        goals=[Goal("A", 0, 1), Goal("B", 0, 2)],
        banners=[
            Banner("Z", "7.0", 1),
            Banner("A", "7.1", 1),
            Banner("B", "7.2", 1),
        ],
    )
    income = IncomeForecast(
        versions=[
            VersionIncome("7.0", estimate=IncomeEstimate(30, 30, 30)),
            VersionIncome("7.1", estimate=IncomeEstimate(30, 30, 30)),
            VersionIncome("7.2", estimate=IncomeEstimate(40, 40, 40)),
        ]
    )
    context = PlannerContext(
        account=doc_account,
        roadmap=roadmap,
        current_version="7.0",
        income=income,
        confidence=0.5,
    )
    first, second = protected_goal_outcomes(context)
    assert first.goal == Goal("A", 0, 1)
    assert first.budget_at_banner == 40 + 60  # wishes + income through 7.1
    assert first.meets_threshold is True
    # A's worst-case reserve is consumed in full, then 7.2 income arrives.
    assert second.budget_at_banner == (40 + 60 - 80) + 40
    assert second.meets_threshold is False


def test_current_version_income_counts_but_past_versions_do_not(doc_account):
    roadmap = Roadmap(
        goals=[Goal("A", 0, 1)],
        banners=[Banner("Z", "7.0", 1), Banner("A", "7.1", 1)],
    )
    income = IncomeForecast(
        versions=[
            VersionIncome("6.9", estimate=IncomeEstimate(100, 100, 100)),
            VersionIncome("7.0", estimate=IncomeEstimate(5, 5, 5)),
            VersionIncome("7.1", estimate=IncomeEstimate(25, 25, 25)),
        ]
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0", income=income
    )
    outcome = protected_goal_outcomes(context)[0]
    # 7.0 + 7.1 are future income; the 6.9 entry is presumed banked.
    assert outcome.budget_at_banner == 40 + 30


def test_goal_without_future_banner_is_excluded_but_still_evaluated():
    """'Not protectable' must not become 'does not exist' (§8): the goal
    keeps its evaluation with next_banner None but is not protected."""
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
        banners=[Banner("Tsaritsa", "7.0", 1), Banner("Vesna", "6.9", 1)],
    )
    context = PlannerContext(
        account=Account(wishes=40), roadmap=roadmap, current_version="7.0"
    )
    assert protected_goal_outcomes(context) == []
    evaluations = {e.goal.character: e for e in evaluate_goals(context)}
    assert evaluations["Vesna"].next_banner is None
    assert evaluations["Vesna"].copies_needed == 1


def test_chronological_order_regardless_of_priority(doc_account):
    """Banner order and priority order may differ (§7); protection walks
    the calendar."""
    roadmap = Roadmap(
        goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
        banners=[
            Banner("Z", "7.0", 1),
            Banner("A", "7.2", 1),
            Banner("B", "7.1", 1),
        ],
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    outcomes = protected_goal_outcomes(context)
    assert [o.goal.character for o in outcomes] == ["B", "A"]


def test_spent_bounds_are_enforced(doc_context):
    with pytest.raises(ValueError, match="cannot exceed"):
        protected_goal_outcomes(doc_context, spent=41)
    with pytest.raises(ValueError, match="non-negative"):
        protected_goal_outcomes(doc_context, spent=-1)


def test_mechanics_data_flows_through(doc_account):
    """Mechanics are data (§17): tiny mechanics shrink every reserve."""
    tiny = WishMechanics(
        banner_type="tiny",
        hard_pity=3,
        soft_pity_start=2,
        base_rate=0.5,
        soft_pity_increment=0.25,
        featured_rate=0.5,
    )
    roadmap = Roadmap(
        goals=[Goal("A", 0, 1)],
        banners=[Banner("Z", "7.0", 1), Banner("A", "7.1", 1)],
    )
    context = PlannerContext(
        account=Account(wishes=4),
        roadmap=roadmap,
        current_version="7.0",
        confidence=1.0,
        mechanics=tiny,
    )
    outcome = protected_goal_outcomes(context)[0]
    assert outcome.required_wishes == 6  # wishes_for_confidence(1.0, 0, False, tiny)
    assert outcome.budget_at_banner == 4
    assert outcome.meets_threshold is False