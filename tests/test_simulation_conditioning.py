"""Tests for deterministic conditioning of retained simulation evidence."""

from domain import Account, Banner, Goal, Ownership
from simulation import (
    BannerResult,
    GoalOutcome,
    PlannedSpend,
    RunResult,
    SimulationResult,
    SpendPlan,
    condition_on_pull,
)


BANNER = Banner("Vesna", "7.0", 1)
GOAL = Goal("Vesna", 0, 1)
PLAN = SpendPlan((PlannedSpend(BANNER, 0, 100),))


def _history(five_star_outcomes):
    account = Account(wishes=350, owned_characters=Ownership({"Vesna": -1}))
    banner = BannerResult(
        banner=BANNER,
        target_constellation=0,
        budget=100,
        copies_needed=1,
        income_credited=0,
        wishes_spent=100,
        copies_obtained=1,
        copy_wishes=tuple(w for w, featured in five_star_outcomes if featured),
        target_met=True,
        account_after=account,
        five_star_outcomes=five_star_outcomes,
    )
    return RunResult(
        banner_results=(banner,),
        account_after=account,
        goal_outcomes=(GoalOutcome(GOAL, True, BANNER),),
    )


def _result(*outcomes) -> SimulationResult:
    from simulation.outcomes import aggregate_runs

    return aggregate_runs(
        [_history(outcome) for outcome in outcomes],
        PLAN,
        seed=7,
    )


def test_condition_featured_filters_histories_without_resampling():
    result = _result(((83, True),), ((90, False),), ((83, False),))

    conditioned = condition_on_pull(
        result,
        character="Vesna",
        outcome="featured",
        wishes_used=83,
    )

    assert conditioned is not None
    assert conditioned.runs == 1
    assert conditioned.histories[0].banner_results[0].five_star_outcomes == ((83, True),)


def test_condition_lost_50_50_filters_histories():
    result = _result(((83, True),), ((83, False),), ((90, False),))

    conditioned = condition_on_pull(
        result,
        character="Vesna",
        outcome="lost_50_50",
        wishes_used=83,
    )

    assert conditioned is not None
    assert conditioned.runs == 1
    assert conditioned.histories[0].banner_results[0].five_star_outcomes == ((83, False),)


def test_condition_returns_none_when_observation_is_not_in_sample():
    result = _result(((80, True),), ((90, False),))

    assert condition_on_pull(
        result,
        character="Vesna",
        outcome="featured",
        wishes_used=83,
    ) is None


def test_condition_supports_multiple_observations_on_one_banner():
    result = _result(
        ((60, True), (140, False)),
        ((60, True), (150, False)),
    )

    conditioned = condition_on_pull(
        result,
        character="Vesna",
        outcome="lost_50_50",
        wishes_used=80,
        prior_observations=(("Vesna", "featured", 60),),
    )

    assert conditioned is not None
    assert conditioned.runs == 1
    assert conditioned.histories[0].banner_results[0].five_star_outcomes == (
        (60, True),
        (140, False),
    )
