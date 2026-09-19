"""Tests for the process-local planner evidence cache."""

from domain import Goal
from simulation import SimulationResult

from api.planner_cache import PlannerEvidenceCache


def _result() -> SimulationResult:
    return SimulationResult(
        runs=1,
        seed=7,
        plan=None,
        goals=(),
        banners=(),
        all_goals_probability=0.0,
        final_wishes_mean=0.0,
        final_wishes_min=0,
        final_wishes_max=0,
        histories=(),
    )


def test_cache_round_trip():
    cache = PlannerEvidenceCache()
    goal = Goal("Vesna", 2, 3)
    result = _result()

    cache.put(
        "account-1",
        goal,
        result,
        runs=1,
        seed=7,
        confidence=0.9,
        income_scenario="expected",
    )

    evidence = cache.get("account-1", goal)
    assert evidence is not None
    assert evidence.result is result
    assert evidence.runs == 1
    assert evidence.seed == 7


def test_cache_is_scoped_to_account_and_goal():
    cache = PlannerEvidenceCache()
    goal = Goal("Vesna", 2, 3)
    result = _result()
    cache.put(
        "account-1",
        goal,
        result,
        runs=1,
        seed=7,
        confidence=0.9,
        income_scenario="expected",
    )

    assert cache.get("account-2", goal) is None
    assert cache.get("account-1", Goal("Vesna", 0, 1)) is None


def test_clear_account_does_not_clear_other_accounts():
    cache = PlannerEvidenceCache()
    goal = Goal("Vesna", 2, 3)
    result = _result()
    for account_id in ("account-1", "account-2"):
        cache.put(
            account_id,
            goal,
            result,
            runs=1,
            seed=7,
            confidence=0.9,
            income_scenario="expected",
        )

    cache.clear_account("account-1")

    assert cache.get("account-1", goal) is None
    assert cache.get("account-2", goal) is not None


def test_cache_is_bounded():
    cache = PlannerEvidenceCache(max_entries=2)
    result = _result()
    for index in range(3):
        cache.put(
            f"account-{index}",
            Goal("Vesna", 2, index + 1),
            result,
            runs=1,
            seed=7,
            confidence=0.9,
            income_scenario="expected",
        )

    assert cache.get("account-0", Goal("Vesna", 2, 1)) is None
    assert cache.get("account-1", Goal("Vesna", 2, 2)) is not None
    assert cache.get("account-2", Goal("Vesna", 2, 3)) is not None
