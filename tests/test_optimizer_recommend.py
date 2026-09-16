"""Strategy optimization on the design-document example (§13, §14, §18 Phase 5).

The headline decisions:

* 90% / no income: nothing is pursueable - "do not spend" with per-outcome
  rejections (§1, matching Phase 3's safe_spend == 0 story honestly).
* 50% / income: the simulation-based cap can exceed the Phase 3
  approximation, because carried pity and guarantee count (§14).
* Sparse-mechanics fallthrough: a top preference that is empirically
  unpursueable is rejected; the next rank wins at its largest feasible
  cap - and the feasibility set is not an interval, so the scan (not a
  binary search) finds it.
"""

import pytest

from domain import (
    Account,
    Banner,
    Goal,
    Ownership,
    Preference,
    Roadmap,
    WishMechanics,
)
from optimizer import evaluate_skip_baseline, recommend
from planner import PlannerContext, safe_spend

VESNA = Banner("Vesna", "7.0", 1)
TSARITSA = Banner("Tsaritsa", "7.1", 1)


def sparse_mechanics() -> WishMechanics:
    return WishMechanics(
        banner_type="sparse",
        hard_pity=3,
        soft_pity_start=2,
        base_rate=1e-9,
        soft_pity_increment=1.0,
        featured_rate=0.5,
    )


class TestDocExampleAt90Percent:
    def test_nothing_is_pursueable(self, doc_context, doc_preferences):
        rec = recommend(
            doc_context, doc_preferences, runs=400, seed=5, budgets=[40, 20, 0]
        )
        assert rec.action == "skip"
        assert rec.outcome is None
        assert rec.budget == 0
        assert rec.plan is None
        assert rec.skip_reason is not None
        assert "threshold" in rec.skip_reason

        # Every preference outcome is rejected, most preferred first,
        # each blocked by the same protected goal (§13 step 7 "why").
        assert [r.outcome.label for r in rec.rejected] == ["C2R1", "C1R1", "C0"]
        for rejection in rec.rejected:
            assert rejection.best.min_protected_probability < 0.9
            assert rejection.shortfalls, "the 'why' must name the blocker"
            standing = rejection.shortfalls[0]
            assert standing.goal == Goal("Tsaritsa", 0, 4)
            assert standing.banner == TSARITSA
            assert standing.meets_threshold is False

        # The skip baseline: doing nothing leaves Tsaritsa C0 at the fresh
        # 40-wish probability (§1 honest probabilities).
        tsaritsa = rec.protected[0]
        assert tsaritsa.goal == Goal("Tsaritsa", 0, 4)
        assert tsaritsa.probability == pytest.approx(0.119, abs=0.03)
        assert tsaritsa.meets_threshold is False

        assert rec.stops.action == "skip"
        assert rec.stops.spend_cap == 0


class TestIncomeAtHalfThreshold:
    def test_simulation_raises_the_cap_above_the_phase3_approximation(
        self, doc_account, doc_roadmap, doc_income
    ):
        """§14 in action: the Phase 3 independent-reserve model credits
        nothing for carried pity and demands the full fresh-pity reserve;
        the simulator counts the pity carried through current-banner
        wishes, so the feasible cap reaches the analytic boundary or
        beyond. Chain is C0 only - the borderline multi-copy outcomes are
        covered by the sparse-mechanics tests where exact arithmetic is
        possible."""
        chain = (Preference("Vesna", 1, 0),)
        context = PlannerContext(
            account=doc_account,
            roadmap=doc_roadmap,
            current_version="7.0",
            income=doc_income,
            confidence=0.5,
        )
        rec = recommend(
            context, chain, runs=5_000, seed=11, budgets=[40, 20, 10, 0]
        )
        assert rec.action == "pursue"
        assert rec.outcome.label == "C0"
        # The analytic safe spend for this context (pinned Phase 3).
        assert safe_spend(context) == 20
        # The simulation-based cap is at least the analytic one - the
        # carried pity/guarantee protect Tsaritsa too.
        assert rec.budget >= 20
        tsaritsa = rec.protected[0]
        assert tsaritsa.goal == Goal("Tsaritsa", 0, 4)
        assert tsaritsa.meets_threshold is True
        assert tsaritsa.probability >= 0.5
        assert rec.outcome_probability > 0.0
        assert rec.rejected == ()


class TestLexicographicSelection:
    def _context(self, wishes: int = 120) -> PlannerContext:
        roadmap = Roadmap(goals=[Goal("Vesna", 0, 1)], banners=[VESNA])
        return PlannerContext(
            account=Account(wishes=wishes), roadmap=roadmap, current_version="7.0"
        )

    def test_fallback_pursues_the_goal_at_the_full_cap(self):
        """No protected goals: nothing future constrains spending, so the
        cap is everything and the (fallback) goal is pursued with it."""
        rec = recommend(self._context(), runs=3_000, seed=13)
        assert rec.action == "pursue"
        assert rec.outcome.label == "C0"
        assert rec.budget == 120
        assert rec.rejected == ()

    def test_top_preference_wins_even_with_lower_probability(self):
        """The §15 chain: C2R1 (3 copies, far lower success probability) is
        recommended over C0 because it ranks first - the preference system
        is a preference system, not a probability score (§13 step 7)."""
        chain = (
            Preference("Vesna", 1, 2, weapon_refinement=1),
            Preference("Vesna", 2, 0),
        )
        rec = recommend(self._context(), chain, runs=3_000, seed=13)
        assert rec.action == "pursue"
        assert rec.outcome.label == "C2R1"
        assert rec.budget == 120
        assert 0.0 < rec.outcome_probability < 1.0
        assert rec.plan.entries[0].target_constellation == 2

    def test_outcome_probability_matches_phase2_for_the_single_copy(self):
        """Optimizer -> simulator -> Phase 2 (review point 13): the
        fallback C0 pursuit at the full cap reproduces the analytical
        curve (§10.2)."""
        from probability import cumulative_probability

        context = self._context()
        rec = recommend(context, runs=6_000, seed=17)
        expected = float(
            cumulative_probability(120, 0, False, context.mechanics)[120]
        )
        assert rec.outcome_probability == pytest.approx(expected, abs=0.02)

        assert rec.stops.action == "pursue"
        assert rec.stops.outcome_label == "C0"
        assert str(rec.budget) in rec.stops.rules[1]
        # Provenance (§2, review point 12): the recommendation carries the
        # simulation parameters so its probabilities can be reproduced.
        assert rec.runs == 6_000
        assert rec.seed == 17
        assert rec.plan is not None
        assert rec.plan.entries[0].target_constellation == 0
        assert rec.plan.entries[0].budget == rec.budget


class TestFallthroughAndDiagnostics:
    """A rejected top preference falls through to the next rank, and the
    rejection record explains the closest-to-feasible candidate (§13 step 7,
    review point 10)."""

    def _context(self, confidence: float = 0.6) -> PlannerContext:
        """Sparse mechanics, pity 1, 3 wishes (§14 carry in full view): A
        needs a certain 5-star per wish, so two copies (C1) fixed-cost
        3 wishes - and spending all 3 leaves B only 0.25. The C1 outcome is
        therefore feasible at no cap; C0 is feasible at cap 2."""
        roadmap = Roadmap(
            goals=[Goal("A", 0, 1), Goal("B", 0, 2)],
            banners=[Banner("A", "7.0", 1), Banner("B", "7.1", 1)],
        )
        return PlannerContext(
            account=Account(current_pity=1, wishes=3),
            roadmap=roadmap,
            current_version="7.0",
            confidence=confidence,
            mechanics=sparse_mechanics(),
        )

    def test_next_rank_wins_and_the_rejection_is_recorded(self):
        chain = (Preference("A", 1, 1), Preference("A", 2, 0))
        rec = recommend(self._context(), chain, runs=4_000, seed=7)
        assert rec.action == "pursue"
        assert rec.outcome.label == "C0"
        assert rec.budget == 2

        assert [r.outcome.label for r in rec.rejected] == ["C1"]
        rejection = rec.rejected[0]
        # C1's closest-to-feasible candidate is cap 0: nothing spent, so
        # every protected goal is maximally safe (floor 1.0) - yet C1
        # achieves nothing, and `outcome_probability > 0` is what rules it
        # out. The shortfall list is empty because protection is not the
        # blocker here: the pursuit criterion is (review point 3).
        assert rejection.best.budget == 0
        assert rejection.best.min_protected_probability == pytest.approx(1.0)
        assert rejection.best.outcome_probability == 0.0
        assert rejection.shortfalls == ()

    def test_rejection_floor_is_the_minimum_protected_probability(self):
        """Review point 10: `best` is defined by the highest MINIMUM
        protected probability, so the field always equals the weakest
        protected standing of that candidate."""
        chain = (Preference("A", 1, 1), Preference("A", 2, 0))
        rec = recommend(self._context(), chain, runs=4_000, seed=7)
        best = rec.rejected[0].best
        assert best.min_protected_probability == pytest.approx(
            min(standing.probability for standing in best.protected)
        )
        assert best.protected  # a protection-constrained outcome does show


class TestDeterminismAndIsolation:
    def test_same_seed_reproduces_the_recommendation(
        self, doc_context, doc_preferences
    ):
        first = recommend(
            doc_context, doc_preferences, runs=400, seed=5, budgets=[40, 0]
        )
        second = recommend(
            doc_context, doc_preferences, runs=400, seed=5, budgets=[40, 0]
        )
        assert first == second

    def test_domain_state_is_not_mutated(self, doc_context, doc_preferences):
        """The optimizer reads the account/roadmap and never writes (§4.1):
        the planner is re-run after every account update (§2)."""
        import copy

        snapshot = copy.deepcopy(doc_context)
        recommend(doc_context, doc_preferences, runs=200, seed=5, budgets=[40, 0])
        assert doc_context == snapshot


class TestNothingToPursue:
    def test_zero_wishes_cannot_pursue_anything(self):
        roadmap = Roadmap(goals=[Goal("A", 0, 1)], banners=[Banner("A", "7.0", 1)])
        context = PlannerContext(
            account=Account(wishes=0), roadmap=roadmap, current_version="7.0"
        )
        rec = recommend(context, runs=500, seed=1)
        assert rec.action == "skip"
        assert rec.budget == 0
        assert rec.rejected[0].best.outcome_probability == 0.0
        assert "threshold" in rec.skip_reason

    def test_satisfied_goal_and_no_preferences_gives_a_dedicated_skip(self):
        """No preference chain and no ACTIVE goal for the banner character:
        there is nothing to declare a preference about (§15, §13 step 2)."""
        roadmap = Roadmap(goals=[Goal("A", 0, 1)], banners=[Banner("A", "7.0", 1)])
        context = PlannerContext(
            account=Account(wishes=40, owned_characters=Ownership({"A": 0})),
            roadmap=roadmap,
            current_version="7.0",
        )
        rec = recommend(context, runs=500, seed=1)
        assert rec.action == "skip"
        assert rec.outcome is None
        assert rec.rejected == ()
        assert "no active goal" in rec.skip_reason

    def test_skip_carries_the_do_nothing_baseline(self):
        """A skip still reports what the roadmap looks like with no spending
        (§1, §2): the do-nothing standings for the protected future goals -
        here B, unreachable at 90% with a single wish."""
        roadmap = Roadmap(
            goals=[Goal("A", 0, 1), Goal("B", 0, 2)],
            banners=[Banner("A", "7.0", 1), Banner("B", "7.1", 1)],
        )
        context = PlannerContext(
            account=Account(wishes=1), roadmap=roadmap, current_version="7.0"
        )
        rec = recommend(context, runs=1_000, seed=3)
        assert rec.action == "skip"
        assert rec.plan is None
        assert rec.protected == evaluate_skip_baseline(context, runs=1_000, seed=3)
        (standing,) = rec.protected
        assert standing.goal == Goal("B", 0, 2)
        assert standing.banner == Banner("B", "7.1", 1)
        assert standing.meets_threshold is False
        assert standing.probability < 0.9

