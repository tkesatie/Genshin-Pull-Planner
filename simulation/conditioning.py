"""Condition retained Monte Carlo evidence on an observed pull outcome.

Conditioning filters already-generated histories; it never samples new
histories. An empty match set means the observation was not represented in the
cached sample and the caller should fall back to a fresh simulation.
"""

from simulation.outcomes import aggregate_runs
from simulation.results import RunResult, SimulationResult


def condition_on_pull(
    result: SimulationResult,
    *,
    character: str,
    outcome: str,
    wishes_used: int,
) -> SimulationResult | None:
    """Return evidence consistent with one recorded character-banner outcome.

    wishes_used is the 1-based within-banner position of the observed 5-star.
    featured matches a featured 5-star; lost_50_50 matches an off-banner 5-star.
    Only the first processed banner for the character is considered, because
    the observation belongs to the current banner.

    Returns None when no retained history contains the observation.
    """
    if outcome not in {"featured", "lost_50_50"}:
        raise ValueError("outcome must be featured or lost_50_50")
    if wishes_used < 1:
        raise ValueError("wishes_used must be >= 1")

    featured = outcome == "featured"
    matched: list[RunResult] = []
    for history in result.histories:
        banner_result = next(
            (
                item
                for item in history.banner_results
                if item.banner.character == character
            ),
            None,
        )
        if banner_result is None:
            continue
        if (wishes_used, featured) in banner_result.five_star_outcomes:
            matched.append(history)

    if not matched:
        return None

    joint_goals = (
        result.joint_goal_probability.goals
        if result.joint_goal_probability is not None
        else ()
    )
    return aggregate_runs(matched, result.plan, result.seed, joint_goals=joint_goals)
