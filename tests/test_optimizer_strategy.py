"""Strategy frontier tests for reserve-first allocation."""

from types import SimpleNamespace

from domain import Account, Banner, Goal, IncomeEstimate, IncomeForecast, Roadmap, VersionIncome
from optimizer import strategy as strategy_module
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan
from simulation.results import BannerAggregate, GoalJointProbability, GoalProbability


VESNA = Banner("Vesna", "7.1", 1)
VODYNISTA = Banner("Vodynista", "7.1", 1)
SKIRK = Banner("Skirk", "7.1", 2)


def make_context(*, income: int = 90) -> PlannerContext:
    return PlannerContext(
        account=Account(wishes=450),
        roadmap=Roadmap(
            goals=[
                Goal("Vodynista", 0, 1),
                Goal("Vesna", 0, 2),
                Goal("Skirk", 2, 3),
                Goal("Vesna", 2, 4),
            ],
            banners=[VESNA, VODYNISTA, SKIRK],
        ),
        current_version="7.1",
        current_phase=1,
        income=IncomeForecast(
            versions=[
                VersionIncome(
                    "7.1",
                    estimate=IncomeEstimate(income, income, income),
                )
            ]
        ),
        confidence=0.90,
    )


def fake_result(plan: SpendPlan, skirk_probability: float) -> SimpleNamespace:
    vesna = plan.entry_for(VESNA)
    return SimpleNamespace(
        goals=(GoalProbability(Goal("Skirk", 2, 3), skirk_probability),),
        joint_goal_probability=GoalJointProbability(
            goals=(Goal("Vodynista", 0, 1), Goal("Vesna", 2, 4)),
            probability=0.42,
        ),
        banners=(
            BannerAggregate(
                banner=VODYNISTA,
                target_constellation=0,
                planned_budget=450,
                mean_income_credited=0.0,
                mean_wishes_spent=0.0,
                target_met_probability=1.0,
                mean_copies_obtained=0.0,
            ),
            BannerAggregate(
                banner=VESNA,
                target_constellation=2,
                planned_budget=vesna.budget if vesna is not None else 0,
                mean_income_credited=0.0,
                mean_wishes_spent=0.0,
                target_met_probability=0.42,
                mean_copies_obtained=0.0,
            ),
        ),
    )


def test_frontier_is_derived_from_protected_requirement(monkeypatch):
    """450 + 90 future income - 265 reserve = 275 current spend."""
    context = make_context()

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        skirk = plan.entry_for(SKIRK)
        if skirk is not None:
            probability = 0.91 if skirk.budget >= 265 else 0.89
        else:
            vesna = plan.entry_for(VESNA)
            assert vesna is not None
            probability = 0.91
        return SimpleNamespace(
            goals=(GoalProbability(Goal("Skirk", 2, 3), probability),),
            joint_goal_probability=GoalJointProbability(
                goals=(Goal("Skirk", 2, 3),) if skirk is not None else (
                    Goal("Vodynista", 0, 1), Goal("Vesna", 2, 4)
                ),
                probability=probability,
            ),
            banners=fake_result(plan, probability).banners,
        )

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result = strategy_module.build_strategy(context, runs=1, seed=0)

    assert result.reserve_goal == Goal("Skirk", 2, 3)
    assert result.reserve_wishes == 175

    vesna_step = next(
        step for step in result.steps
        if step.goal == Goal("Vesna", 2, 4)
        and step.action == "pursue_until_reserve"
    )
    assert vesna_step.safe_spend == 275
    assert vesna_step.reserve_wishes == 175
    assert vesna_step.future_income == 90
    assert vesna_step.protected_total_wishes == 265


def test_frontier_is_monotonic_and_does_not_scan_every_spend(monkeypatch):
    """The reserve-first frontier uses logarithmic simulation, not 451 scans."""
    context = make_context()
    calls = 0

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        nonlocal calls
        calls += 1
        skirk = plan.entry_for(SKIRK)
        if skirk is not None:
            probability = min(1.0, skirk.budget / 265)
        else:
            probability = 1.0
        return fake_result(plan, probability)

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result = strategy_module.build_strategy(context, runs=1, seed=0)

    vesna_step = next(
        step for step in result.steps
        if step.goal == Goal("Vesna", 2, 4)
        and step.action == "pursue_until_reserve"
    )
    assert vesna_step.safe_spend == 275
    assert calls <= 20
