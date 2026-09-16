"""safe_spend: the safe spending bound (Design Document §14, §18)."""

import pytest

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
)
from planner import PlannerContext, protected_goal_outcomes, safe_spend


def funded(doc_account, doc_roadmap, doc_income, **overrides) -> PlannerContext:
    """The doc example with income and a 50% threshold: fundable."""
    values = dict(
        account=doc_account,
        roadmap=doc_roadmap,
        current_version="7.0",
        income=doc_income,
        confidence=0.5,
    )
    values.update(overrides)
    return PlannerContext(**values)


def test_doc_example_safe_spend_is_zero(doc_context):
    """40 wishes cannot protect a 90% Tsaritsa C0 from fresh pity: the
    planner is allowed to answer "do not spend" (§1)."""
    assert safe_spend(doc_context) == 0


def test_safe_spend_accounts_for_income_timing(doc_context, doc_income):
    """Not `wishes - required` (which would floor at 0): 60 expected
    income arrives before the 7.1 banner, so 20 wishes are safe now."""
    context = funded(doc_context.account, doc_context.roadmap, doc_income)
    naive = max(0, context.account.wishes - 80)
    assert naive == 0
    assert safe_spend(context) == 20  # 40 - (80 - 60 credited through 7.1)


def test_underfunded_roadmap_floors_at_zero(doc_context, doc_income):
    """Even 40 wishes + 60 income < the 90% reserve of 155."""
    context = funded(doc_context.account, doc_context.roadmap, doc_income, confidence=0.9)
    assert safe_spend(context) == 0


def test_boundary_is_exact(doc_context, doc_income):
    context = funded(doc_context.account, doc_context.roadmap, doc_income)
    assert protected_goal_outcomes(context, spent=20)[0].meets_threshold is True
    assert protected_goal_outcomes(context, spent=21)[0].meets_threshold is False


def test_safety_property_matches_protection_everywhere(
    doc_context, doc_income
):
    """The Phase 3 invariant: when the roadmap is fundable at zero
    spend, protection(spent=N) meets the threshold everywhere if and only
    if N <= safe_spend - the closed form equals the walk under the
    independent-reserve assumptions (§14). When even zero spend is
    unsafe, safe_spend floors at 0 and no spend is safe."""
    contexts = [
        doc_context,
        funded(doc_context.account, doc_context.roadmap, doc_income),
    ]
    for context in contexts:
        bound = safe_spend(context)
        fundable = all(
            o.meets_threshold
            for o in protected_goal_outcomes(context, spent=0)
        )
        for spent in range(context.account.wishes + 1):
            meets = all(
                o.meets_threshold
                for o in protected_goal_outcomes(context, spent=spent)
            )
            if fundable:
                assert meets == (spent <= bound), (context, spent)
            else:
                assert not meets and bound == 0, (context, spent)


def test_multiple_goals_accumulate_reserves():
    """Each protected goal gets its own full reserve; income timing
    decides how much of the pool each consumes (§14)."""
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
        account=Account(wishes=100),
        roadmap=roadmap,
        current_version="7.0",
        income=income,
        confidence=0.5,
    )
    # required = 80 each; credit(7.1) = 60, credit(7.2) = 100
    # worst = max(80 - 60, 160 - 100) = 60 -> safe_spend = 100 - 60 = 40
    assert safe_spend(context) == 40
    assert all(
        o.meets_threshold for o in protected_goal_outcomes(context, spent=40)
    )
    assert not all(
        o.meets_threshold for o in protected_goal_outcomes(context, spent=41)
    )


def test_no_protected_goals_leaves_the_whole_pool():
    account = Account(wishes=40, owned_characters=Ownership({"X": 0}))
    roadmap = Roadmap(goals=[Goal("X", 0, 1)], banners=[Banner("X", "7.0", 1)])
    context = PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0"
    )
    assert safe_spend(context) == 40


def test_never_exceeds_wishes_even_with_huge_income():
    roadmap = Roadmap(
        goals=[Goal("A", 0, 1)],
        banners=[Banner("Z", "7.0", 1), Banner("A", "7.1", 1)],
    )
    income = IncomeForecast(
        versions=[VersionIncome("7.1", estimate=IncomeEstimate(200, 200, 200))]
    )
    context = PlannerContext(
        account=Account(wishes=40),
        roadmap=roadmap,
        current_version="7.0",
        income=income,
    )
    # A credit surplus cannot buy wishes that do not exist.
    assert safe_spend(context) == 40