"""Candidate strategies as executable spend plans (Design Document §13 step 3).

A candidate strategy is (preferred outcome, spend cap) rendered as the
Phase 4 `SpendPlan` the simulator executes faithfully (§12):

    current banner:  pursue the outcome's constellation, capped at the
                     candidate's spend budget
    current-phase alternatives: share that same cap, so spending on one
                     current-banner character reduces what remains for the
                     others
    future banners:  every protected group pursued with its strategic
                     budget (optimizer.protection) - the confidence
                     question is "can the roadmap still be completed if I
                     spend this much now?" (§2)

Construction only: anything involving simulation lives in
optimizer.evaluation.
"""

from domain import TargetKind
from planner import PlannerContext
from planner.banners import current_banner
from simulation import PlannedSpend, SpendPlan

from optimizer.outcomes import OutcomeOption
from optimizer.protection import protected_groups


def candidate_plan(
    context: PlannerContext,
    outcome: OutcomeOption,
    budget: int,
    banner=None,
) -> SpendPlan:
    """The executable plan for pursuing `outcome` with `budget` wishes
    capped across the current phase (§13 step 3).

    If multiple roadmap banners share the current version/phase, they are
    alternatives within the same spend pool. The candidate budget is
    therefore a shared current-phase cap, not an independent budget for
    every same-slot banner. This is required for the roadmap-wide result to
    describe the same resource allocation as the recommendation.

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
    selected_banner = banner if banner is not None else current_banner(context)
    entries = [
        PlannedSpend(
            banner=selected_banner,
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
        for group in protected_groups(context, banner=selected_banner)
        # Weapon banners are never character-simulated (Phase 4): the
        # character Monte Carlo models character pity/guarantee/ownership
        # only. Protected weapon goals are evaluated against the shared
        # wish pool through the exact weapon probability engine
        # (optimizer.evaluation), not by plan entries here.
        if group.banner.target.kind is TargetKind.CHARACTER
    )
    return SpendPlan(
        entries=tuple(entries),
        shared_current_phase_budget=budget,
    )
