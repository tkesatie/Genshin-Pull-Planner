"""Strategy frontier tests for multi-goal current-banner allocation.

The important Phase 5 property is that a current progression budget is a
TOTAL cap on that banner. When Vesna C2 is the constrained decision, the
simulated plan must pursue Vesna C0 first without assuming how many wishes
that C0 will consume. The remaining budget can then continue toward C2 while
protecting the higher-priority future Skirk C2 goal.
"""

from types import SimpleNamespace

from domain import Account, Banner, Goal, Roadmap
from optimizer import strategy as strategy_module
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan
from simulation.results import BannerAggregate, GoalProbability


def make_context() -> PlannerContext:
    return PlannerContext(
        account=Account(wishes=450),
        roadmap=Roadmap(
            goals=[
                Goal("Vodynista", 0, 1),
                Goal("Vesna", 0, 2),
                Goal("Skirk", 2, 3),
                Goal("Vesna", 2, 4),
            ],
            banners=[
                Banner("Vesna", "7.1", 1),
                Banner("Vodynista", "7.1", 1),
                Banner("Skirk", "7.1", 2),
            ],
        ),
        current_version="7.1",
        current_phase=1,
        confidence=0.90,
    )


def test_frontier_table_models_unknown_c0_cost(monkeypatch):
    """The frontier must budget C0 + progression together, not sequentially.

    The table below is deliberately a mocked probability curve: it isolates
    strategy allocation logic from Monte Carlo variance.

        Vesna banner budget | Skirk C2 probability | decision
        --------------------+----------------------+---------
        450                  | 0.89                 | unsafe
        300                  | 0.91                 | safe
    """
    context = make_context()
    captured: dict[int, SpendPlan] = {}

    def fake_simulate(context, plan, *, runs, seed):
        vesna_entry = plan.entry_for(Banner("Vesna", "7.1", 1))
        assert vesna_entry is not None
        captured[vesna_entry.budget] = plan

        skirk = Goal("Skirk", 2, 3)
        probability = 0.91 if vesna_entry.budget <= 300 else 0.89
        return SimpleNamespace(
            goals=(GoalProbability(skirk, probability),),
            banners=(
                BannerAggregate(
                    banner=Banner("Vodynista", "7.1", 1),
                    target_constellation=0,
                    planned_budget=450,
                    mean_income_credited=0.0,
                    mean_wishes_spent=0.0,
                    target_met_probability=1.0,
                    mean_copies_obtained=0.0,
                ),
                BannerAggregate(
                    banner=Banner("Vesna", "7.1", 1),
                    target_constellation=2,
                    planned_budget=vesna_entry.budget,
                    mean_income_credited=0.0,
                    mean_wishes_spent=0.0,
                    target_met_probability=0.42,
                    mean_copies_obtained=0.0,
                ),
            ),
        )

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result = strategy_module.build_strategy(context, runs=1, seed=0)

    assert result.reserve_wishes == 150
    assert result.reserve_goal == Goal("Skirk", 2, 3)

    vesna_step = next(
        step
        for step in result.steps
        if step.goal == Goal("Vesna", 2, 4)
        and step.action == "pursue_until_reserve"
    )
    assert vesna_step.safe_spend == 300
    assert vesna_step.reserve_wishes == 150
    assert vesna_step.outcome_probability == 0.42

    plan = captured[300]
    assert plan.entries == (
        PlannedSpend(
            banner=Banner("Vodynista", "7.1", 1),
            target_constellation=0,
            budget=450,
        ),
        PlannedSpend(
            banner=Banner("Vesna", "7.1", 1),
            target_constellation=2,
            budget=300,
        ),
        PlannedSpend(
            banner=Banner("Skirk", "7.1", 2),
            target_constellation=2,
            budget=450,
        ),
    )


def test_safe_spend_uses_simulation_for_non_monotonic_frontier(monkeypatch):
    """A feasible cap can sit below an unsafe cap; binary search must not skip it."""
    context = make_context()
    probabilities = {
        450: 0.89,
        400: 0.89,
        350: 0.89,
        300: 0.89,
        275: 0.91,
    }

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        vesna_entry = plan.entry_for(Banner("Vesna", "7.1", 1))
        skirk = Goal("Skirk", 2, 3)
        probability = probabilities.get(vesna_entry.budget, 0.91)
        return SimpleNamespace(
            goals=(GoalProbability(skirk, probability),),
            banners=(
                BannerAggregate(
                    banner=Banner("Vodynista", "7.1", 1),
                    target_constellation=0,
                    planned_budget=450,
                    mean_income_credited=0.0,
                    mean_wishes_spent=0.0,
                    target_met_probability=1.0,
                    mean_copies_obtained=0.0,
                ),
                BannerAggregate(
                    banner=Banner("Vesna", "7.1", 1),
                    target_constellation=2,
                    planned_budget=vesna_entry.budget,
                    mean_income_credited=0.0,
                    mean_wishes_spent=0.0,
                    target_met_probability=0.42,
                    mean_copies_obtained=0.0,
                ),
            ),
        )

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result = strategy_module.build_strategy(context, runs=1, seed=0)

    vesna_step = next(
        step
        for step in result.steps
        if step.goal == Goal("Vesna", 2, 4)
        and step.action == "pursue_until_reserve"
    )
    assert vesna_step.safe_spend == 275
    assert vesna_step.reserve_wishes == 175
