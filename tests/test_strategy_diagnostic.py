"""Diagnostic test for the Vesna C2 / Skirk C2 strategy frontier.

Run with:
    pytest -s tests/test_strategy_diagnostic.py

This intentionally uses the real demo account and real simulator. It answers
whether Skirk C2 is already below the 90% protection threshold after the
required Vodynista C0 allocation, even when Vesna receives zero wishes.
"""

from api.demo_data import create_demo_account
from optimizer.strategy import _evaluate_spend
from planner import PlannerContext


def test_report_vesna_c2_frontier_diagnostic():
    record = create_demo_account()
    context: PlannerContext = record.context()
    vesna = next(
        banner for banner in context.roadmap.banners
        if banner.character == "Vesna"
    )
    vesna_c2 = next(
        goal for goal in context.roadmap.goals
        if goal.character == "Vesna" and goal.constellation == 2
    )
    skirk_c2 = next(
        goal for goal in context.roadmap.goals
        if goal.character == "Skirk" and goal.constellation == 2
    )

    print("\n=== Vesna C2 / Skirk C2 diagnostic ===")
    print(f"Initial wishes: {context.account.wishes}")
    print(f"Initial character pity: {context.account.current_pity}")
    print(f"Confidence threshold: {context.confidence:.0%}")
    print("")

    for spend in (0, 100, 200, 300, 400, 450):
        result, protected = _evaluate_spend(
            context,
            vesna_c2,
            vesna,
            spend,
            runs=10_000,
            seed=0,
        )
        skirk = next(item for item in result.goals if item.goal == skirk_c2)
        vesna_result = next(
            item for item in result.banners
            if item.banner == vesna
        )

        print(
            f"Vesna budget {spend:3d}: "
            f"Skirk C2 = {skirk.probability:6.2%}, "
            f"Vesna C2 = {vesna_result.target_met_probability:6.2%}, "
            f"protected = {', '.join(g.character + ' C' + str(g.constellation) for g in protected) or 'none'}"
        )

    print("")
