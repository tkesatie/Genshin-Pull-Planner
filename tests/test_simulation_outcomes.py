"""Roadmap outcome aggregation (Design Document §11).

The exact-math tests build RunResults by hand (no engine), so the
aggregation arithmetic is verified independently of the simulator; the
integration tests run the doc example end to end.
"""

import pytest

from domain import Account, Banner, Goal, Roadmap
from planner import PlannerContext
from simulation import (
    GoalOutcome,
    PlannedSpend,
    RunResult,
    SpendPlan,
    BannerResult,
    aggregate_runs,
    simulate,
)
from probability import cumulative_probability

VESNA = Banner("Vesna", "7.0", 1)
TSARITSA = Banner("Tsaritsa", "7.1", 1)


def _run(
    spend: tuple[int, int],
    final_wishes: int,
    vesna: bool,
    tsaritsa: bool,
) -> RunResult:
    """One hand-built history over two banners and two goals."""
    first = BannerResult(
        banner=VESNA,
        target_constellation=0,
        budget=10,
        copies_needed=1,
        income_credited=0,
        wishes_spent=spend[0],
        copies_obtained=int(vesna),
        copy_wishes=(1,) if vesna else (),
        target_met=vesna,
        account_after=Account(wishes=final_wishes),
    )
    second = BannerResult(
        banner=TSARITSA,
        target_constellation=0,
        budget=10,
        copies_needed=1,
        income_credited=10,
        wishes_spent=spend[1],
        copies_obtained=int(tsaritsa),
        copy_wishes=(1,) if tsaritsa else (),
        target_met=tsaritsa,
        account_after=Account(wishes=final_wishes),
    )
    goals = (
        GoalOutcome(Goal("Vesna", 0, 1), vesna, VESNA if vesna else None),
        GoalOutcome(Goal("Tsaritsa", 0, 2), tsaritsa, TSARITSA if tsaritsa else None),
    )
    return RunResult((first, second), Account(wishes=final_wishes), goals)


def _plan() -> SpendPlan:
    return SpendPlan(
        entries=(
            PlannedSpend(VESNA, 0, 10),
            PlannedSpend(TSARITSA, 0, 10),
        )
    )


class TestAggregateRuns:
    def test_empty_runs_are_rejected(self):
        with pytest.raises(ValueError, match="at least one run"):
            aggregate_runs([], _plan(), seed=0)

    def test_counts_fractions_and_means_exactly(self):
        runs = [
            _run((5, 3), 20, True, True),
            _run((8, 2), 18, True, False),
            _run((10, 0), 15, False, True),
            _run((7, 4), 12, False, False),
        ]
        result = aggregate_runs(runs, _plan(), seed=5)

        assert result.runs == 4
        assert result.seed == 5

        # Goal satisfaction fractions, priority order (§5).
        assert [(g.goal, g.probability) for g in result.goals] == [
            (Goal("Vesna", 0, 1), 0.5),
            (Goal("Tsaritsa", 0, 2), 0.5),
        ]

        # Roadmap-wide: exactly one run satisfied everything.
        assert result.all_goals_probability == 0.25

        # Banner behavior, chronological order (§7).
        first, second = result.banners
        assert first.banner == VESNA
        assert first.planned_budget == 10
        assert first.target_constellation == 0
        assert first.mean_income_credited == 0.0
        assert first.mean_wishes_spent == 7.5
        assert first.target_met_probability == 0.5
        assert first.mean_copies_obtained == 0.5
        assert second.banner == TSARITSA
        assert second.mean_income_credited == 10.0
        assert second.mean_wishes_spent == 2.25
        assert second.target_met_probability == 0.5

        # End-of-history wish pool.
        assert result.final_wishes_mean == 16.25
        assert result.final_wishes_min == 12
        assert result.final_wishes_max == 20

    def test_plan_fields_reflect_skipped_banners(self):
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 10),))
        result = aggregate_runs([_run((5, 0), 20, True, False)], plan, seed=0)
        first, second = result.banners
        assert first.planned_budget == 10
        assert second.planned_budget == 0
        assert second.target_constellation is None


class TestDocExampleIntegration:
    def test_vesna_c0_probability_matches_the_phase2_curve(self, doc_context, doc_plan):
        """Phase 2 <-> Phase 4 consistency on the doc example: the empirical
        P(Vesna C0 within the 40-wish pool) must reproduce the analytical
        curve within sampling error (probability invariant 10)."""
        result = simulate(doc_context, doc_plan, runs=5_000, seed=20260916)
        expected = float(
            cumulative_probability(40, 0, False, doc_context.mechanics)[40]
        )
        assert expected == pytest.approx(0.119, abs=1e-3)  # pinned Phase 2 anchor
        assert result.goals[0].goal == Goal("Vesna", 0, 1)
        assert result.goals[0].probability == pytest.approx(expected, abs=0.02)

        # Exact, not sampled: Vodynista C0 is pre-satisfied; a C0-target
        # plan can never satisfy Vesna C2, and no entry means no Tsaritsa
        # pulls at all (§10.4, §12).
        assert result.goals[1].probability == 1.0
        assert result.goals[2].probability == 0.0
        assert result.goals[3].probability == 0.0
        assert result.all_goals_probability == 0.0

        first, second, third = result.banners
        assert first.planned_budget == 40
        assert second.planned_budget == 0
        assert third.planned_budget == 0
        assert 0.0 < first.mean_wishes_spent < 40.0

    def test_do_not_spend_plan(self, doc_context):
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 0),))
        result = simulate(doc_context, plan, runs=200, seed=1)
        assert result.goals[0].probability == 0.0
        assert result.goals[1].probability == 1.0
        assert result.all_goals_probability == 0.0
        assert result.final_wishes_min == 40
        assert result.final_wishes_max == 40
        assert result.banners[0].mean_wishes_spent == 0.0

    def test_pursuing_the_next_banner_propagates_state_forward(
        self, doc_account, doc_roadmap, doc_income
    ):
        """A two-goal plan exercises pity/guarantee propagation (§11): the
        Tsaritsa probability must beat the fresh-pity bound by a wide,
        seed-stable margin (~0.36 across seeds at 5k runs), because carried
        pity and guarantee only help."""
        context = PlannerContext(
            account=doc_account,
            roadmap=doc_roadmap,
            current_version="7.0",
            income=doc_income,
        )
        plan = SpendPlan(
            entries=(
                PlannedSpend(VESNA, 0, 40),
                PlannedSpend(TSARITSA, 0, 60),
            )
        )
        result = simulate(context, plan, runs=5_000, seed=42)

        vesna, vodynista, vesna_c2, tsaritsa = result.goals
        fresh = float(cumulative_probability(60, 0, False, context.mechanics)[60])
        assert vesna.probability == pytest.approx(
            float(cumulative_probability(40, 0, False, context.mechanics)[40]),
            abs=0.02,
        )
        assert tsaritsa.probability > fresh
        # Vesna C2 (priority 3) can never be satisfied by a C0-target plan,
        # so "every roadmap goal" is exactly impossible even though each
        # individually pursued goal has positive probability (§10.4, §12).
        assert result.all_goals_probability == 0.0
        assert vodynista.probability == 1.0
        assert vesna_c2.probability == 0.0  # the plan pursued C0, not C2

    def test_c2_plan_satisfies_the_blocked_goal_only_with_three_copies(
        self, doc_account
    ):
        """Vesna C2 is blocked behind Vesna C0 (§9); the simulator walks the
        same dependency in copies (§10.4): 200 wishes always reach C0, but
        C2 = 3 copies from unowned is far from certain."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Vesna", 2, 3)],
            banners=[VESNA],
        )
        context = PlannerContext(
            account=Account(wishes=200), roadmap=roadmap, current_version="7.0"
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 2, 200),))
        result = simulate(context, plan, runs=2_000, seed=8)

        vesna_c0, vesna_c2 = result.goals
        assert vesna_c0.probability == 1.0  # one copy within 200 is certain
        assert 0.0 < vesna_c2.probability < 1.0
        assert vesna_c2.probability <= vesna_c0.probability
        # With only these two goals, both are satisfied exactly when C2 is:
        # C2 satisfied implies C0 satisfied (§9 dependency, in copies).
        assert result.all_goals_probability == vesna_c2.probability

