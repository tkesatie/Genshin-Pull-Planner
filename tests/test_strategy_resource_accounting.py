"""Resource-accounting diagnostics for the multi-goal pull strategy.

This test uses the real Monte Carlo simulator, rather than mocking
probabilities. It verifies the concrete 450-wish scenario that the strategy
frontier is supposed to model: Vodynista C0 -> Vesna C2 -> Skirk C2.

The assertions intentionally focus on invariants rather than expected
probability percentages. The purpose is to establish that the simulator is
using one shared wish pool and deriving multi-copy targets from ownership.
"""
import numpy as np

from domain import Account, Banner, Goal, Ownership, Roadmap
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate, simulate_history


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


def test_450_wish_strategy_uses_one_shared_pool_and_correct_copy_targets():
    """The exact 450-wish scenario must have shared resource accounting.

    The plan deliberately gives each banner a 450-wish cap. Those caps are
    independent limits, not independent pools: the simulator must consume
    wishes from the same Account as it walks Vodynista -> Vesna -> Skirk.

    No income forecast is supplied so that the only resource entering the
    history is the account's initial 450 wishes.
    """
    context = make_context()
    plan = SpendPlan(
        entries=(
            PlannedSpend(VODYNISTA, target_constellation=0, budget=450),
            PlannedSpend(VESNA, target_constellation=2, budget=219),
            PlannedSpend(SKIRK, target_constellation=2, budget=450),
        )
    )

    result = simulate(context, plan, runs=10_000, seed=0)

    assert result.runs == 10_000
    assert result.final_wishes_min >= 0
    assert result.final_wishes_max <= 450

    # Inspect individual histories too. This verifies the shared-pool
    # invariant directly rather than only through aggregate statistics.
    rng = np.random.default_rng(0)
    for _ in range(100):
        run = simulate_history(context, plan, rng)
        results = {item.banner: item for item in run.banner_results}

        vodynista = results[VODYNISTA]
        vesna = results[VESNA]
        skirk = results[SKIRK]

        # Vesna starts unowned, so C2 requires C0 plus two additional copies.
        assert vesna.copies_needed == 3
        # Skirk starts at C0, so C2 requires two additional copies.
        assert skirk.copies_needed == 2

        # Every banner consumes from the same evolving account.
        assert vodynista.account_after.wishes == 450 - vodynista.wishes_spent
        assert vesna.account_after.wishes == (
            vodynista.account_after.wishes - vesna.wishes_spent
        )
        assert skirk.account_after.wishes == (
            vesna.account_after.wishes - skirk.wishes_spent
        )

        assert run.account_after.wishes >= 0
        total_spent = (
            vodynista.wishes_spent
            + vesna.wishes_spent
            + skirk.wishes_spent
        )
        assert total_spent <= 450

    vodynista = next(b for b in result.banners if b.banner == VODYNISTA)
    vesna = next(b for b in result.banners if b.banner == VESNA)
    skirk = next(b for b in result.banners if b.banner == SKIRK)

    assert vodynista.mean_wishes_spent <= 450
    assert vesna.mean_wishes_spent <= 219
    assert skirk.mean_wishes_spent <= 450

    mean_total_spent = (
        vodynista.mean_wishes_spent
        + vesna.mean_wishes_spent
        + skirk.mean_wishes_spent
    )
    assert mean_total_spent <= 450
    assert result.final_wishes_mean >= 0
    assert result.final_wishes_mean <= 450
