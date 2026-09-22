"""Acceptance tests for the corrected Phase 5 strategy semantics.

These tests deliberately separate:
- Phase 4 execution semantics (shared caps, target stopping, state carryover)
- Phase 5 decision semantics (reserve-first frontier, future income, path probability)

Some Phase 5 tests are expected to fail until the strategy refactor is made.
They define the contract we agreed on before implementation.
"""

from types import SimpleNamespace

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
    WishMechanics,
)
from optimizer import strategy as strategy_module
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate_history
from simulation.results import BannerAggregate, GoalJointProbability, GoalProbability


VESNA = Banner("Vesna", "7.1", 1)
VODYNISTA = Banner("Vodynista", "7.1", 1)
SKIRK = Banner("Skirk", "7.1", 2)


def forced_mechanics(featured_rate: float = 1.0) -> WishMechanics:
    """Every pull is a 5-star; featured_rate controls the 50/50 result."""
    return WishMechanics(
        banner_type="forced",
        hard_pity=2,
        soft_pity_start=1,
        base_rate=0.5,
        soft_pity_increment=0.5,
        featured_rate=featured_rate,
    )


def strategy_context(
    *,
    wishes: int = 450,
    income: int = 0,
    owned: dict[str, int] | None = None,
) -> PlannerContext:
    return PlannerContext(
        account=Account(
            wishes=wishes,
            owned_characters=Ownership(owned or {}),
        ),
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
        income=(
            IncomeForecast(
                versions=[
                    VersionIncome(
                        "7.1",
                        estimate=IncomeEstimate(income, income, income),
                    )
                ]
            )
            if income
            else None
        ),
        confidence=0.90,
    )


def fake_strategy_result(
    plan: SpendPlan,
    skirk_probability: float,
    *,
    joint_probability: float = 0.42,
    joint_goals: tuple[Goal, ...] = (
        Goal("Vodynista", 0, 1),
        Goal("Vesna", 2, 4),
    ),
):
    """Minimal result shape needed by build_strategy/_safe_spend."""
    vesna_entry = plan.entry_for(VESNA)
    return SimpleNamespace(
        goals=(GoalProbability(Goal("Skirk", 2, 3), skirk_probability),),
        all_goals_probability=joint_probability,
        joint_goal_probability=GoalJointProbability(
            goals=joint_goals,
            probability=joint_probability,
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
                planned_budget=vesna_entry.budget if vesna_entry is not None else 0,
                mean_income_credited=0.0,
                mean_wishes_spent=0.0,
                target_met_probability=0.42,
                mean_copies_obtained=0.0,
            ),
        ),
    )


def test_shared_current_phase_budget_is_one_total_cap():
    """P1/P2/current progression consume one shared current-phase pool."""
    context = PlannerContext(
        account=Account(wishes=5),
        roadmap=Roadmap(
            goals=[Goal("Vodynista", 0, 1), Goal("Vesna", 0, 2)],
            banners=[VESNA, VODYNISTA],
        ),
        current_version="7.1",
        mechanics=forced_mechanics(),
    )
    plan = SpendPlan(
        entries=(
            PlannedSpend(VODYNISTA, 0, 5),
            PlannedSpend(VESNA, 0, 5),
        ),
        shared_current_phase_budget=2,
    )

    run = simulate_history(context, plan, __import__("numpy").random.default_rng(0))

    assert sum(result.wishes_spent for result in run.banner_results) == 2
    assert run.banner_results[0].wishes_spent + run.banner_results[1].wishes_spent == 2


def test_luck_on_higher_priority_goal_frees_shared_budget_for_lower_priority():
    """If P1 costs fewer wishes, the unused cap remains available to P2."""
    context = PlannerContext(
        account=Account(wishes=5),
        roadmap=Roadmap(
            goals=[Goal("Vodynista", 0, 1), Goal("Vesna", 0, 2)],
            banners=[VESNA, VODYNISTA],
        ),
        current_version="7.1",
        mechanics=forced_mechanics(),
    )
    plan = SpendPlan(
        entries=(
            PlannedSpend(VODYNISTA, 0, 5),
            PlannedSpend(VESNA, 0, 5),
        ),
        shared_current_phase_budget=3,
    )

    run = simulate_history(context, plan, __import__("numpy").random.default_rng(0))

    vodynista, vesna = sorted(
        run.banner_results, key=lambda result: result.banner.character
    )
    assert vodynista.wishes_spent == 1
    assert vesna.wishes_spent == 1
    assert sum(result.wishes_spent for result in run.banner_results) == 2


def test_target_stops_immediately_after_character_is_obtained():
    """A target completion ends spending on that banner; remaining cap is reusable."""
    context = PlannerContext(
        account=Account(wishes=5),
        roadmap=Roadmap(
            goals=[Goal("Vesna", 0, 1)],
            banners=[VESNA],
        ),
        current_version="7.1",
        mechanics=forced_mechanics(),
    )
    plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 5),))

    run = simulate_history(context, plan, __import__("numpy").random.default_rng(0))

    result = run.banner_results[0]
    assert result.copies_obtained == 1
    assert result.wishes_spent == 1
    assert result.account_after.current_pity == 0


def test_reserve_first_with_future_income_gives_275_current_spend_ceiling(monkeypatch):
    """450 now + 90 later - 265 protected reserve = 275 spendable now."""
    context = strategy_context(income=90)

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        entry = plan.entry_for(SKIRK)
        if entry is not None:
            probability = 0.91 if entry.budget >= 265 else 0.89
        else:
            vesna = plan.entry_for(VESNA)
            assert vesna is not None
            probability = 0.91 if vesna.budget <= 275 else 0.89
        return fake_strategy_result(
            plan,
            probability,
            joint_probability=probability if entry is not None else 0.42,
            joint_goals=(Goal("Skirk", 2, 3),) if entry is not None else (
                Goal("Vodynista", 0, 1),
                Goal("Vesna", 2, 4),
            ),
        )

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result = strategy_module.build_strategy(context, runs=1, seed=0)

    step = next(
        step
        for step in result.steps
        if step.goal == Goal("Vesna", 2, 4)
        and step.action == "pursue_until_reserve"
    )

    assert step.safe_spend == 275
    assert step.reserve_wishes == 175
    assert step.future_income == 90
    assert step.protected_total_wishes == 265


def test_future_income_increases_current_spend_ceiling(monkeypatch):
    """More guaranteed future income permits more spending today."""
    ceilings = {}

    for income, expected in [(90, 275), (180, 365)]:
        context = strategy_context(income=income)

        def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
            entry = plan.entry_for(SKIRK)
            if entry is not None:
                probability = 0.91 if entry.budget >= 265 else 0.89
            else:
                vesna = plan.entry_for(VESNA)
                assert vesna is not None
                probability = 0.91
            return fake_strategy_result(
                plan,
                probability,
                joint_probability=probability if entry is not None else 0.42,
                joint_goals=(Goal("Skirk", 2, 3),) if entry is not None else (
                    Goal("Vodynista", 0, 1),
                    Goal("Vesna", 2, 4),
                ),
            )

        monkeypatch.setattr(strategy_module, "simulate", fake_simulate)
        result = strategy_module.build_strategy(context, runs=1, seed=0)
        step = next(
            step
            for step in result.steps
            if step.goal == Goal("Vesna", 2, 4)
            and step.action == "pursue_until_reserve"
        )
        ceilings[income] = step.safe_spend

    assert ceilings == {90: 275, 180: 365}


def test_current_spend_cannot_exceed_current_wishes(monkeypatch):
    """Future income is usable for protection, but cannot be spent today."""
    context = strategy_context(wishes=100, income=300)

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        return fake_strategy_result(plan, 0.91)

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result = strategy_module.build_strategy(context, runs=1, seed=0)

    step = next(
        step
        for step in result.steps
        if step.goal == Goal("Vesna", 2, 4)
        and step.action == "pursue_until_reserve"
    )
    assert step.safe_spend <= 100


def test_protected_probability_decreases_as_current_spend_increases(monkeypatch):
    """The protected frontier is monotonic: spending more now cannot help P3."""
    context = strategy_context()

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        entry = plan.entry_for(VESNA)
        assert entry is not None
        probability = max(0.0, 1.0 - entry.budget / 1000)
        return fake_strategy_result(
            plan,
            probability,
            joint_probability=probability if entry is not None else 0.42,
            joint_goals=(Goal("Skirk", 2, 3),) if entry is not None else (
                Goal("Vodynista", 0, 1),
                Goal("Vesna", 2, 4),
            ),
        )

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    probabilities = []
    for spend in (100, 200, 300, 400):
        plan = strategy_module._evaluate_spend(
            context,
            Goal("Vesna", 2, 4),
            VESNA,
            spend,
            runs=1,
            seed=0,
        )
        result = plan[0]
        probabilities.append(
            next(
                goal.probability
                for goal in result.goals
                if goal.goal == Goal("Skirk", 2, 3)
            )
        )

    assert probabilities == sorted(probabilities, reverse=True)


def test_outcome_probability_is_strategy_path_probability(monkeypatch):
    """Vesna C2 probability includes the higher-priority Vodynista C0 path."""
    context = strategy_context()

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        assert joint_goals == (
            Goal("Vodynista", 0, 1),
            Goal("Vesna", 2, 4),
        )
        result = fake_strategy_result(plan, 0.91)
        return result

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    result, _ = strategy_module._evaluate_spend(
        context,
        Goal("Vesna", 2, 4),
        VESNA,
        275,
        runs=1,
        seed=0,
    )

    assert result.joint_goal_probability is not None
    assert result.joint_goal_probability.probability == 0.42


def test_owned_skirk_c0_requires_two_copies_for_c2():
    """The protected target is Skirk C2, but ownership already supplies C0."""
    context = strategy_context(owned={"Skirk": 0})
    plan = SpendPlan(entries=(PlannedSpend(SKIRK, 2, 100),))

    # Use a real engine run so copies_needed comes from simulated ownership.
    context = PlannerContext(
        account=context.account,
        roadmap=Roadmap(
            goals=[Goal("Skirk", 2, 1)],
            banners=[SKIRK],
        ),
        current_version="7.1",
        current_phase=2,
        mechanics=forced_mechanics(),
    )
    run = simulate_history(context, plan, __import__("numpy").random.default_rng(0))

    assert run.banner_results[0].copies_needed == 2


def test_protected_requirement_has_a_finite_search_bound(monkeypatch):
    """A failing simulation cannot make the exponential search run forever."""
    context = strategy_context()

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        entry = plan.entry_for(SKIRK)
        probability = 0.0
        return fake_strategy_result(
            plan,
            probability,
            joint_probability=probability,
            joint_goals=(Goal("Skirk", 2, 3),),
        )

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    assert (
        strategy_module._protected_requirement(
            context,
            (Goal("Skirk", 2, 3),),
            runs=1,
            seed=0,
        )
        is None
    )


def test_strategy_frontier_does_not_simulate_every_possible_spend(monkeypatch):
    """The optimized frontier search must not regress to 451 full simulations."""
    context = strategy_context()
    calls = 0

    def fake_simulate(context, plan, *, runs, seed, joint_goals=()):
        nonlocal calls
        calls += 1
        return fake_strategy_result(plan, 0.91)

    monkeypatch.setattr(strategy_module, "simulate", fake_simulate)

    strategy_module.build_strategy(context, runs=1, seed=0)

    assert calls <= 20
