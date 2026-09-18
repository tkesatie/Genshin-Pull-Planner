"""Roadmap outcome aggregation (Design Document §11).

The layering this package keeps separate:

    Phase 2       "What is the probability?"                 probability/
    Phase 4       "What does one possible future look like?"  simulation.engine
    Aggregation   "Across N futures, how often?"              this module
    Phase 5       "Which strategy should we run?"             not here (§13)

Aggregation is plain counting over auditable per-run records: per-goal
satisfaction fractions, per-banner spending behavior, and end-of-history
wish-pool statistics. It never re-samples and never inspects the rng.
"""

from collections.abc import Sequence

from simulation.results import (
    BannerAggregate,
    GoalJointProbability,
    GoalProbability,
    RunResult,
    SimulationResult,
)
from simulation.strategy import SpendPlan


def aggregate_runs(
    runs: Sequence[RunResult],
    plan: SpendPlan,
    seed: int | None,
    joint_goals: Sequence = (),
) -> SimulationResult:
    """Aggregate simulated histories into roadmap outcome probabilities (§11).

    All runs must come from the same (context, plan): the engine produces
    identical goal and banner sequences per run, so columns are taken by
    position. Goal probabilities are ordered by goal priority (§5), banner
    aggregates chronologically (§7).

    Raises:
        ValueError: if `runs` is empty.
    """
    if not runs:
        raise ValueError("aggregate_runs requires at least one run")

    count = len(runs)
    first = runs[0]

    goal_probabilities = tuple(
        GoalProbability(
            goal=first.goal_outcomes[index].goal,
            probability=sum(
                1 for run in runs if run.goal_outcomes[index].satisfied
            )
            / count,
        )
        for index in range(len(first.goal_outcomes))
    )

    banner_aggregates: list[BannerAggregate] = []
    for index, banner_result in enumerate(first.banner_results):
        column = [run.banner_results[index] for run in runs]
        entry = plan.entry_for(banner_result.banner)
        banner_aggregates.append(
            BannerAggregate(
                banner=banner_result.banner,
                target_constellation=(
                    entry.target_constellation if entry is not None else None
                ),
                planned_budget=entry.budget if entry is not None else 0,
                mean_income_credited=sum(
                    result.income_credited for result in column
                )
                / count,
                mean_wishes_spent=sum(
                    result.wishes_spent for result in column
                )
                / count,
                target_met_probability=sum(
                    1 for result in column if result.target_met
                )
                / count,
                mean_copies_obtained=sum(
                    result.copies_obtained for result in column
                )
                / count,
            )
        )

    final_wishes = [run.account_after.wishes for run in runs]
    joint = tuple(joint_goals)
    joint_probability = None
    if joint:
        joint_probability = GoalJointProbability(
            goals=joint,
            probability=sum(
                1 for run in runs
                if all(
                    next(outcome for outcome in run.goal_outcomes if outcome.goal == goal).satisfied
                    for goal in joint
                )
            ) / count,
        )

    all_goals_probability = sum(
        1
        for run in runs
        if all(outcome.satisfied for outcome in run.goal_outcomes)
    ) / count

    return SimulationResult(
        runs=count,
        seed=seed,
        plan=plan,
        goals=goal_probabilities,
        banners=tuple(banner_aggregates),
        all_goals_probability=all_goals_probability,
        final_wishes_mean=sum(final_wishes) / count,
        final_wishes_min=min(final_wishes),
        final_wishes_max=max(final_wishes),
        joint_goal_probability=joint_probability,
    )
