"""Tests for the process-local planner evidence cache."""

from domain import Account, Banner, Goal, Roadmap
from planner.context import PlannerContext
from simulation import SimulationResult

from api.planner_cache import PlannerEvidenceCache


def _context() -> PlannerContext:
    return PlannerContext(
        account=Account(wishes=350),
        roadmap=Roadmap(
            goals=[Goal("Vesna", 2, 3)],
            banners=[Banner("Vesna", "7.0", 1)],
        ),
        current_version="7.0",
        current_phase=1,
    )


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
    context = _context()

    cache.put(
        "account-1",
        goal,
        result,
        runs=1,
        seed=7,
        confidence=0.9,
        income_scenario="expected",
        context=context,
    )

    evidence = cache.get("account-1", goal)
    assert evidence is not None
    assert evidence.result is result
    assert evidence.runs == 1
    assert evidence.seed == 7


def test_cache_is_scoped_to_account_and_goal():
    cache = PlannerEvidenceCache()
    context = _context()
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
    context=context,
    )

    assert cache.get("account-2", goal) is None
    assert cache.get("account-1", Goal("Vesna", 0, 1)) is None


def test_clear_account_does_not_clear_other_accounts():
    cache = PlannerEvidenceCache()
    goal = Goal("Vesna", 2, 3)
    result = _result()
    context = _context()
    for account_id in ("account-1", "account-2"):
        cache.put(
            account_id,
            goal,
            result,
            runs=1,
            seed=7,
            confidence=0.9,
            income_scenario="expected",
        context=context,
        )

    cache.clear_account("account-1")

    assert cache.get("account-1", goal) is None
    assert cache.get("account-2", goal) is not None


def test_cache_is_bounded():
    cache = PlannerEvidenceCache(max_entries=2)
    result = _result()
    context = _context()
    for index in range(3):
        cache.put(
            f"account-{index}",
            Goal("Vesna", 2, index + 1),
            result,
            runs=1,
            seed=7,
            confidence=0.9,
            income_scenario="expected",
        context=context,
        )

    assert cache.get("account-0", Goal("Vesna", 2, 1)) is None
    assert cache.get("account-1", Goal("Vesna", 2, 2)) is not None
    assert cache.get("account-2", Goal("Vesna", 2, 3)) is not None


def test_condition_account_reuses_matching_histories():
    from simulation import BannerResult, GoalOutcome, PlannedSpend, RunResult
    from domain import Account, Banner, Ownership
    from simulation.outcomes import aggregate_runs

    banner = Banner("Vesna", "7.0", 1)
    goal = Goal("Vesna", 0, 1)
    plan = PlannedSpend(banner, 0, 100)

    def history(outcomes):
        account = __import__("domain").Account(
            wishes=350,
            owned_characters=Ownership({"Vesna": -1}),
        )
        result = BannerResult(
            banner=banner,
            target_constellation=0,
            budget=100,
            copies_needed=1,
            income_credited=0,
            wishes_spent=100,
            copies_obtained=1,
            copy_wishes=tuple(w for w, featured in outcomes if featured),
            target_met=True,
            account_after=account,
            five_star_outcomes=outcomes,
        )
        return RunResult(
            banner_results=(result,),
            account_after=account,
            goal_outcomes=(GoalOutcome(goal, True, banner),),
        )

    simulation = aggregate_runs(
        [history(((83, True),)), history(((90, False),))],
        __import__("simulation").SpendPlan((plan,)),
        seed=7,
        joint_goals=(goal,),
    )
    cache = PlannerEvidenceCache()
    context = _context()
    cache.put(
        "account-1",
        goal,
        simulation,
        runs=2,
        seed=7,
        confidence=0.9,
        income_scenario="expected",
    context=context,
    )

    conditioned = cache.condition_account(
        "account-1",
        character="Vesna",
        outcome="featured",
        wishes_used=83,
    )

    assert conditioned[goal].result.runs == 1
    assert conditioned[goal].result.joint_goal_probability is not None
    assert conditioned[goal].result.joint_goal_probability.probability == 1.0


def test_condition_candidate_reuses_recommendation_evidence():
    from domain import Account, Banner, Ownership
    from optimizer.evaluation import CandidateStrategy
    from optimizer.outcomes import OutcomeOption
    from simulation import BannerResult, GoalOutcome, PlannedSpend, RunResult, SpendPlan
    from simulation.outcomes import aggregate_runs

    banner = Banner("Vesna", "7.0", 1)
    goal = Goal("Vesna", 2, 1)
    outcome = OutcomeOption("Vesna", 2, 1)
    plan = SpendPlan((PlannedSpend(banner, 2, 100),))

    def history(outcomes, target_met):
        account = Account(wishes=350, owned_characters=Ownership({"Vesna": -1}))
        result = BannerResult(
            banner=banner,
            target_constellation=2,
            budget=100,
            copies_needed=3,
            income_credited=0,
            wishes_spent=100,
            copies_obtained=3 if target_met else 1,
            copy_wishes=tuple(w for w, featured in outcomes if featured),
            target_met=target_met,
            account_after=account,
            five_star_outcomes=outcomes,
        )
        return RunResult(
            banner_results=(result,),
            account_after=account,
            goal_outcomes=(GoalOutcome(goal, target_met, banner),),
        )

    simulation = aggregate_runs(
        [history(((83, True), (100, True)), True), history(((90, False),), False)],
        plan,
        seed=7,
        joint_goals=(goal,),
    )
    candidate = CandidateStrategy(
        outcome=outcome,
        budget=100,
        plan=plan,
        result=simulation,
        outcome_probability=simulation.banners[0].target_met_probability,
        protected=(),
        min_protected_probability=None,
        feasible=True,
    )

    cache = PlannerEvidenceCache()
    cache.put_candidate("account-1", candidate, runs=2, seed=7, context=_context())

    conditioned = cache.condition_candidate(
        "account-1",
        character="Vesna",
        outcome="featured",
        constellation=2,
        banner_version="7.0",
        banner_phase=1,
        new_budget=17,
        wishes_used=83,
    )

    assert conditioned is not None
    assert conditioned.budget == 17
    assert conditioned.candidate.outcome_probability == 1.0
    assert conditioned.candidate.result.runs == 1
