"""Probability-curve diagnostic for the Vesna C2 spend frontier.

This is intentionally a diagnostic test rather than a brittle regression
against exact Monte Carlo percentages. It measures how increasing the Vesna
spend cap changes Vesna C2 and protected Skirk C2 outcomes under the same
shared 450-wish pool.
"""
from domain import Account, Banner, Goal, Ownership, Roadmap
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
    )


def test_vesna_spend_curve(capsys):
    """Print the probability tradeoff across Vesna spend caps."""
    context = make_context()
    budgets = (0, 100, 150, 200, 219, 250, 300, 350, 400, 450)

    print("\nVesna spend curve (10,000 runs, seed=0)")
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
