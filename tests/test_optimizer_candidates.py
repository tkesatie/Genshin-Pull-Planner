"""Candidate strategies as executable spend plans (Design Document §13 step 3)."""

import pytest

from domain import Account, Banner, Goal, Roadmap
from optimizer import OutcomeOption, candidate_plan, protected_groups
from planner import PlannerContext
from simulation import PlannedSpend

VESNA = Banner("Vesna", "7.0", 1)
TSARITSA = Banner("Tsaritsa", "7.1", 1)


def vesna_outcome(constellation: int = 0) -> OutcomeOption:
    return OutcomeOption(character="Vesna", constellation=constellation, rank=1)


def test_current_entry_pursues_the_outcome(doc_context):
    plan = candidate_plan(doc_context, vesna_outcome(2), 15)
    assert plan.entries[0] == PlannedSpend(VESNA, 2, 15)


def test_future_entries_carry_the_protected_groups(doc_context, doc_income):
    """Every protected group appears exactly once, with its target
    constellation and strategic budget (§13 step 3)."""
    context = PlannerContext(
        account=doc_context.account,
        roadmap=doc_context.roadmap,
        current_version="7.0",
        income=doc_income,
    )
    plan = candidate_plan(context, vesna_outcome(0), 40)
    groups = protected_groups(context)
    assert len(plan.entries) == 1 + len(groups)
    tsaritsa = next(e for e in plan.entries if e.banner == TSARITSA)
    assert tsaritsa.target_constellation == 0
    assert tsaritsa.budget == groups[0].uncapped_budget == 100


def test_group_target_is_the_max_constellation(doc_account):
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 1, 2), Goal("Tsaritsa", 3, 3)],
        banners=[VESNA, TSARITSA],
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    plan = candidate_plan(context, vesna_outcome(0), 10)
    tsaritsa = next(e for e in plan.entries if e.banner == TSARITSA)
    assert tsaritsa.target_constellation == 3


def test_entries_are_chronological_after_the_current(doc_account):
    roadmap = Roadmap(
        goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
        banners=[VESNA, Banner("A", "7.2", 1), Banner("B", "7.1", 1)],
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    plan = candidate_plan(context, vesna_outcome(0), 10)
    assert [entry.banner for entry in plan.entries] == [
        VESNA,
        Banner("B", "7.1", 1),
        Banner("A", "7.2", 1),
    ]


def test_plan_is_executable_by_the_simulator(doc_context):
    plan = candidate_plan(doc_context, vesna_outcome(0), 40)
    plan.require_valid_for(doc_context)  # no raise


def test_negative_budget_is_rejected(doc_context):
    with pytest.raises(ValueError, match="budget must satisfy"):
        candidate_plan(doc_context, vesna_outcome(0), -1)


def test_budget_beyond_wishes_is_rejected(doc_context):
    with pytest.raises(ValueError, match="budget must satisfy"):
        candidate_plan(doc_context, vesna_outcome(0), 41)
