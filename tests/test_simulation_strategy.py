"""Spending plans (Design Document §12).

Strategy is deliberately minimal in Phase 4: per-banner spending decisions
the simulator can execute faithfully. The exact class structure is to be
designed with the Phase 5 optimizer (§12: "we should not over-design
`Strategy` before the decision logic exists").
"""

import pytest

from domain import Account, Banner, Goal, Roadmap
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan

VESNA = Banner("Vesna", "7.0", 1)
TSARITSA = Banner("Tsaritsa", "7.1", 1)


class TestPlannedSpendValidation:
    def test_negative_target_is_rejected(self):
        with pytest.raises(ValueError, match="target_constellation"):
            PlannedSpend(VESNA, -1, 10)

    def test_negative_budget_is_rejected(self):
        with pytest.raises(ValueError, match="budget"):
            PlannedSpend(VESNA, 0, -5)

    def test_zero_budget_is_a_valid_cap(self):
        entry = PlannedSpend(VESNA, 0, 0)
        assert entry.budget == 0

    def test_zero_target_is_pursuing_c0(self):
        entry = PlannedSpend(VESNA, 0, 10)
        assert entry.target_constellation == 0


class TestSpendPlanStructure:
    def test_empty_plan_is_valid_and_skips_everything(self):
        assert SpendPlan().entries == ()

    def test_duplicate_banner_entries_are_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            SpendPlan(
                entries=(
                    PlannedSpend(VESNA, 0, 10),
                    PlannedSpend(VESNA, 2, 10),
                )
            )

    def test_entry_for_finds_entries_and_none_otherwise(self):
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 10),))
        assert plan.entry_for(VESNA) == PlannedSpend(VESNA, 0, 10)
        assert plan.entry_for(TSARITSA) is None
        # Same character, different slot: not the planned banner.
        assert plan.entry_for(Banner("Vesna", "7.0", 2)) is None


def _context(
    banners: list[Banner] | None = None, goals: list[Goal] | None = None
) -> PlannerContext:
    return PlannerContext(
        account=Account(wishes=40),
        roadmap=Roadmap(
            goals=goals if goals is not None else [Goal("Vesna", 0, 1)],
            banners=banners
            if banners is not None
            else [VESNA, TSARITSA],
        ),
        current_version="7.0",
    )


class TestRequireValidFor:
    def test_current_and_future_banners_are_accepted(self):
        plan = SpendPlan(
            entries=(
                PlannedSpend(VESNA, 0, 10),
                PlannedSpend(TSARITSA, 0, 10),
            )
        )
        plan.require_valid_for(_context())  # no raise

    def test_empty_plan_is_accepted(self):
        SpendPlan().require_valid_for(_context())  # no raise

    def test_unknown_banner_is_rejected(self):
        plan = SpendPlan(entries=(PlannedSpend(Banner("Mavuika", "7.5", 1), 0, 10),))
        with pytest.raises(ValueError, match="not a roadmap banner"):
            plan.require_valid_for(_context())

    def test_banner_before_the_current_banner_is_rejected(self):
        past = Banner("Old", "6.9", 1)
        plan = SpendPlan(entries=(PlannedSpend(past, 0, 10),))
        with pytest.raises(ValueError, match="before the current banner"):
            plan.require_valid_for(_context(banners=[past, VESNA]))
