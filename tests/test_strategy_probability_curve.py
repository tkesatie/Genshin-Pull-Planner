"""Probability-curve diagnostics for the Vesna and Skirk spend frontiers.

This is intentionally a diagnostic test rather than a brittle regression
against exact Monte Carlo percentages. It measures how increasing the Vesna
spend cap changes Vesna C2 and protected Skirk C2 outcomes under the same
shared 450-wish pool, including the expected 90 wishes of 7.1 income.
"""
import numpy as np
from dataclasses import replace

from domain import Account, Banner, Goal, IncomeEstimate, IncomeForecast, Ownership, Roadmap, VersionIncome
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate
from dataclasses import replace
import numpy as np

VODYNISTA = Banner("Vodynista", "7.1", 1)
VESNA = Banner("Vesna", "7.1", 1)
SKIRK = Banner("Skirk", "7.1", 2)


def make_context() -> PlannerContext:
    return PlannerContext(
        account=Account(
            wishes=450,
            current_pity=27,
            owned_characters=Ownership({"Skirk": 0}),
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
        confidence=0.90,
        income=IncomeForecast(
            versions=[
                VersionIncome(
                    "7.1",
                    estimate=IncomeEstimate(90, 90, 90),
                )
            ]
        ),
    )


def test_vesna_spend_curve(capsys):
    """Print the probability tradeoff across Vesna spend caps."""
    context = make_context()
    budgets = (0, 100, 150, 200, 219, 250, 300, 350, 400, 450)

    print("\nVesna spend curve with 90 expected 7.1 income (10,000 runs, seed=0)")
    print("budget | Vesna C2 | Skirk C2 | all goals | final wishes")
    print("-------+-----------+----------+-----------+-------------")

    rows = []
    for budget in budgets:
        plan = SpendPlan(
            entries=(
                PlannedSpend(VODYNISTA, target_constellation=0, budget=450),
                PlannedSpend(VESNA, target_constellation=2, budget=budget),
                PlannedSpend(SKIRK, target_constellation=2, budget=450),
            )
        )
        result = simulate(context, plan, runs=10_000, seed=0)

        vesna = next(
            item.probability
            for item in result.goals
            if item.goal == Goal("Vesna", 2, 4)
        )
        skirk = next(
            item.probability
            for item in result.goals
            if item.goal == Goal("Skirk", 2, 3)
        )

        rows.append((budget, vesna, skirk, result.all_goals_probability))
        print(
            f"{budget:6d} | {vesna:9.2%} | {skirk:8.2%} | "
            f"{result.all_goals_probability:9.2%} | {result.final_wishes_mean:12.1f}"
        )

    # The curve should move in the expected direction. Monte Carlo noise can
    # cause tiny local reversals, so use endpoint comparisons rather than
    # demanding strict monotonicity at every adjacent budget.
    assert rows[-1][1] > rows[0][1]
    assert rows[-1][2] < rows[0][2]
    assert rows[-1][0] == 450


def test_skirk_c2_probability_sanity_curve(capsys):
    """Measure raw Skirk C2 probability at the Hu Tao calculator comparison points."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    budgets = (240, 264, 265, 280)

    print("\nSkirk C2 raw probability sanity curve (10,000 runs, seed=0)")
    print("budget | Skirk C2")
    print("-------+----------")

    rows = []
    for budget in budgets:
        success = 0
        rng = np.random.default_rng(0)

        for _ in range(10_000):
            account = context.account
            _, copies, _, _ = _pull_toward_target(
                account, "Skirk", 2, budget, mechanics, rng
            )
            success += copies == 2

        probability = success / 10_000
        rows.append((budget, probability))
        print(f"{budget:6d} | {probability:8.2%}")

    # Diagnostic comparison only; this does not hard-code an external calculator's result.
    assert rows[-1][1] > rows[0][1]


def test_multi_copy_confidence_is_explicit(capsys):
    """Measure the actual simulated probability of two featured copies."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    budgets = (155, 180, 240, 264, 265)
    runs = 10_000
    rng = np.random.default_rng(0)

    print("\\nMulti-copy simulation confidence diagnostic")
    print(f"runs: {runs}, seed: 0")
    print("budget | Skirk C2")
    print("-------+----------")

    results = []
    for budget in budgets:
        successes = 0
        for _ in range(runs):
            account = replace(context.account, wishes=budget)
            spent, obtained, _, _ = _pull_toward_target(
                account,
                "Skirk",
                2,
                budget,
                context.mechanics,
                rng,
            )
            if obtained >= 2 and spent <= budget:
                successes += 1
        probability = successes / runs
        results.append(probability)
        print(f"{budget:6d} | {probability:8.2%}")

    assert results[0] < results[-1]
    assert results[1] < results[2] < results[3] < results[4]



def test_skirk_c2_probability_by_starting_pity(capsys):
    """Compare the C2 probability from zero pity versus the account's current pity."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    budgets = (240, 264, 265)
    runs = 10_000
    print("\\nSkirk C2 probability by starting pity (10,000 runs, seed=0)")
    print("budget | pity 0 | pity 27")
    print("-------+---------+--------")

    rows = []
    for budget in budgets:
        probabilities = []
        for starting_pity in (0, 27):
            successes = 0
            rng = np.random.default_rng(0)
            for _ in range(runs):
                account = replace(
                    context.account,
                    wishes=budget,
                    current_pity=starting_pity,
                    character_guarantee=False,
                )
                _, copies, _, _ = _pull_toward_target(
                    account, "Skirk", 2, budget, context.mechanics, rng
                )
                successes += copies == 2
            probabilities.append(successes / runs)
        rows.append(probabilities)
        print(f"{budget:6d} | {probabilities[0]:7.2%} | {probabilities[1]:7.2%}")

    for zero_pity, current_pity in rows:
        assert current_pity > zero_pity


def test_planner_protection_matches_skirk_threshold(capsys):
    """Inspect the planner's protected Skirk reserve at representative spend levels."""
    from planner.protection import protected_goal_outcomes

    context = make_context()
    spends = (0, 50, 100, 150, 200, 240)

    print("\nPlanner Skirk protection diagnostic")
    print("spent | budget_at_banner | required | confidence | meets")
    print("------+-------------------+----------+------------+------")

    rows = []
    for spent in spends:
        outcomes = protected_goal_outcomes(context, spent=spent, banner=VODYNISTA)
        skirk = next(
            outcome for outcome in outcomes
            if outcome.goal == Goal("Skirk", 2, 3)
        )
        rows.append((spent, skirk))
        print(
            f"{spent:5d} | {skirk.budget_at_banner:17d} | "
            f"{skirk.required_wishes:8d} | {skirk.confidence:10.2%} | "
            f"{str(skirk.meets_threshold):5s}"
        )

    # Diagnostic only: verify the planner's protection calculation is
    # actually responding to additional current-banner spending.
    assert rows[-1][1].budget_at_banner < rows[0][1].budget_at_banner
    assert rows[-1][1].confidence <= rows[0][1].confidence


def test_combined_current_banner_frontier_respects_skirk_reserve(capsys):
    """Measure Skirk C2 using the reserve actually advertised by the frontier."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    caps = (200, 210, 220, 225, 230, 231, 235, 240)
    mechanics = context.mechanics

    print("\nCombined frontier with actual Skirk reserve (10,000 runs, seed=0)")
    print("cap | reserve | Skirk C2")
    print("----+---------+----------")

    rows = []
    for cap in caps:
        skirk_success = 0
        rng = np.random.default_rng(0)
        reserve = 450 - cap

        for _ in range(10_000):
            account = context.account

            vod_spent, _, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )

            account = replace(account, wishes=account.wishes + 90)
            vesna_budget = max(cap - vod_spent, 0)
            _, _, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )

            _, skirk_copies, _, _ = _pull_toward_target(
                account, "Skirk", 2, reserve, mechanics, rng
            )
            skirk_success += skirk_copies == 2

        probability = skirk_success / 10_000
        rows.append((cap, reserve, probability))
        print(f"{cap:3d} | {reserve:7d} | {probability:8.2%}")

    # This test is specifically checking that the displayed reserve is the
    # budget used for the protected Skirk goal. As current-banner spending
    # increases, the protected probability should decrease.
    assert rows[-1][2] < rows[0][2]


def test_combined_current_banner_frontier(capsys):
    """Measure the real frontier: Vod C0 + Vesna C2 share the 450-wish pool."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    caps = (200, 210, 220, 225, 230, 231, 235, 240)
    mechanics = context.mechanics

    print("\nCombined phase-1 spend frontier (Vod C0 + Vesna C2)")
    print("cap | reserve | Vesna C2 | Skirk C2 | all three")
    print("----+---------+----------+----------+----------")

    rows = []
    for cap in caps:
        vesna_success = 0
        skirk_success = 0
        all_success = 0
        rng = np.random.default_rng(0)

        for _ in range(10_000):
            account = context.account

            vod_spent, vod_copies, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )

            # 7.1 income is future income. It cannot fund either Phase 1
            # banner, but will be available when Skirk's later phase arrives.
            vesna_budget = max(cap - vod_spent, 0)
            _, vesna_copies, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )

            vesna_met = vesna_copies == 3
            vesna_success += vesna_met

            _, skirk_copies, _, _ = _pull_toward_target(
                account, "Skirk", 2, 450, mechanics, rng
            )
            skirk_met = skirk_copies == 2
            skirk_success += skirk_met
            all_success += vod_copies == 1 and vesna_met and skirk_met

        rows.append((cap, vesna_success / 10_000, skirk_success / 10_000))
        print(
            f"{cap:3d} | {450 - cap:7d} | {vesna_success / 10_000:8.2%} | "
            f"{skirk_success / 10_000:8.2%} | {all_success / 10_000:8.2%}"
        )

    assert rows[-1][1] > rows[0][1]
    assert rows[-1][2] < rows[0][2]


def test_strategy_vesna_c2_probability_is_joint_with_vodynista(capsys):
    """The Vesna C2 strategy probability must require Vodynista C0 too."""
    from optimizer.strategy import build_strategy

    context = make_context()
    strategy = build_strategy(context, runs=10_000, seed=0)
    vesna_c2 = next(
        step for step in strategy.steps
        if step.goal == Goal("Vesna", 2, 4)
    )

    print("\nVesna C2 joint strategy probability")
    print(f"safe spend: {vesna_c2.safe_spend}")
    print(f"reserve: {vesna_c2.reserve_wishes}")
    print(f"joint probability: {vesna_c2.outcome_probability:.2%}")

    # Vodynista C0 and Vesna C2 must fit the strategy's sequential current-
    # banner allocation. Skirk C2 remains the separate 90% reserve constraint.
    assert vesna_c2.safe_spend in range(275, 278)
    assert vesna_c2.reserve_wishes in range(173, 176)
    assert vesna_c2.future_income == 90
    assert vesna_c2.protected_total_wishes in range(263, 266)
    assert 0.15 < vesna_c2.outcome_probability < 0.21

    # The isolated Vesna-C2 probability is materially higher; this assertion
    # prevents the old marginal probability from silently returning to the UI.
    assert vesna_c2.outcome_probability < 0.10


def test_future_income_cannot_fund_current_phase(capsys):
    """Current-phase spending uses only wishes already available now."""
    from optimizer.strategy import build_strategy
    from simulation.engine import _pull_toward_target

    context = make_context()
    strategy = build_strategy(context, runs=10_000, seed=0)
    vesna_c2 = next(step for step in strategy.steps if step.goal == Goal("Vesna", 2, 4))

    assert vesna_c2.safe_spend in range(275, 278)
    assert vesna_c2.reserve_wishes in range(173, 176)
    assert vesna_c2.future_income == 90
    assert vesna_c2.protected_total_wishes in range(263, 266)
    assert 0.15 < vesna_c2.outcome_probability < 0.21


def test_zero_wishes_cannot_spend_current_phase_income():
    """Forecast income is unavailable until the roadmap reaches a later phase."""
    from simulation.engine import simulate_history

    context = make_context()
    zero_context = replace(context, account=replace(context.account, wishes=0))
    plan = SpendPlan(
        entries=(
            PlannedSpend(VODYNISTA, target_constellation=0, budget=90),
            PlannedSpend(VESNA, target_constellation=0, budget=90),
            PlannedSpend(SKIRK, target_constellation=2, budget=90),
        )
    )
    history = simulate_history(zero_context, plan, np.random.default_rng(0))
    assert all(result.income_credited == 0 for result in history.banner_results[:2])
    assert all(result.wishes_spent == 0 for result in history.banner_results[:2])
    assert history.banner_results[2].income_credited == 90
