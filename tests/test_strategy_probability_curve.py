"""Probability-curve diagnostic for the Vesna C2 spend frontier.

This is intentionally a diagnostic test rather than a brittle regression
against exact Monte Carlo percentages. It measures how increasing the Vesna
spend cap changes Vesna C2 and protected Skirk C2 outcomes under the same
shared 450-wish pool, including the expected 90 wishes of 7.1 income.
"""
from domain import Account, Banner, Goal, IncomeEstimate, IncomeForecast, Ownership, Roadmap, VersionIncome
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate


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

            # 7.1 income arrives at the second processed banner. It is
            # available to Vesna, but cannot increase the original 450-wish
            # phase-1 spending cap.
            account = replace(account, wishes=account.wishes + 90)
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
