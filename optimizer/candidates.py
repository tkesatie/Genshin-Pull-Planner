"""Candidate strategies as executable spend plans (Design Document §13 step 3).

A candidate strategy is (preferred outcome, spend cap) rendered as the
Phase 4 `SpendPlan` the simulator executes faithfully (§12):

    current banner:  pursue the outcome's constellation, capped at the
                     candidate's spend budget
    future banners:  every protected group pursued with its strategic
                     budget (optimizer.protection) - the confidence
                     question is "can the roadmap still be completed if I
                     spend this much now?" (§2)

Construction only: anything involving simulation lives in
optimizer.evaluation.
"""

from planner import PlannerContext
from planner.banners import current_banner
from simulation import PlannedSpend, SpendPlan

from optimizer.outcomes import OutcomeOption
from optimizer.protection import protected_groups


def candidate_plan(
    context: PlannerContext, outcome: OutcomeOption, budget: int
) -> SpendPlan:
    """The executable plan for pursuing `outcome` with `budget` wishes
    capped on the current banner (§13 step 3).

    Raises:
        ValueError: if `budget` is negative or exceeds the account's
            wishes - a cap the account could never approach is planner
            input error, not a strategy.
    """
    if not 0 <= budget <= context.account.wishes:
        raise ValueError(
            f"budget must satisfy 0 <= budget <= account wishes "
            f"({context.account.wishes}), got {budget}"
        )
    entries = [
        PlannedSpend(
            banner=current_banner(context),
            target_constellation=outcome.constellation,
            budget=budget,
        )
    ]
    entries.extend(
        PlannedSpend(
            banner=group.banner,
            target_constellation=group.target_constellation,
            budget=group.uncapped_budget,
        )
        for group in protected_groups(context)
    )
    return SpendPlan(entries=tuple(entries))
