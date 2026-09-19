"""Condition retained Monte Carlo evidence on observed pull outcomes.

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
    prior_observations: tuple[tuple[str, str, int], ...] = (),
) -> SimulationResult | None:
    """Return evidence consistent with the recorded character-banner events.

    wishes_used is measured since the previous observed 5-star on that
    banner. The first observation is therefore an absolute within-banner
    wish index; later observations on the same banner are relative deltas.
    """
    if outcome not in {"featured", "lost_50_50"}:
        raise ValueError("outcome must be featured or lost_50_50")
    if wishes_used < 1:
        raise ValueError("wishes_used must be >= 1")

    observations = (*prior_observations, (character, outcome, wishes_used))
    matched: list[RunResult] = []

    for history in result.histories:
        cursor = 0
        previous_wish_by_banner: dict[int, int] = {}
        matched_history = True

        for observed_character, observed_outcome, observed_wishes in observations:
            banner_index = next(
                (
                    index
                    for index in range(cursor, len(history.banner_results))
                    if history.banner_results[index].banner.character == observed_character
                ),
                None,
            )
            if banner_index is None:
                matched_history = False
                break

            banner_result = history.banner_results[banner_index]
            previous_wish = previous_wish_by_banner.get(banner_index, 0)
            featured = observed_outcome == "featured"
            expected_wish = previous_wish + observed_wishes
            if (expected_wish, featured) not in banner_result.five_star_outcomes:
                matched_history = False
                break

            previous_wish_by_banner[banner_index] = expected_wish
            cursor = banner_index

        if matched_history:
            matched.append(history)

    if not matched:
        return None

    joint_goals = (
        result.joint_goal_probability.goals
        if result.joint_goal_probability is not None
        else ()
    )
    return aggregate_runs(matched, result.plan, result.seed, joint_goals=joint_goals)
