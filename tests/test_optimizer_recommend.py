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
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Preference,
    Roadmap,
    VersionIncome,
    WishMechanics,
)
from optimizer import constraining_goals, evaluate_skip_baseline, recommend
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
    """§2's priority gate in the design document's own roadmap.

    Vesna C0 is Priority 1 and is on the current banner; Tsaritsa C0 is
    Priority 4 and arrives at 7.1. §2 asks what spending now does to the
    *higher-priority* future objectives, and Tsaritsa is not one of them: a
    Priority 4 goal two banners out does not get to veto the user's
    Priority 1 goal. Tsaritsa is still protected, still pursued by the plan
    and still reported - its probability simply does not gate the decision.
    """

    def test_a_lower_priority_future_goal_does_not_block_the_current_goal(
        self, doc_context, doc_preferences
    ):
        rec = recommend(
            doc_context, doc_preferences, runs=2_000, seed=5, budgets=[40, 20, 0]
        )
        assert rec.action == "pursue"
        assert rec.outcome.character == "Vesna"
        # Nothing future outranks Vesna C0, so the whole pool is the cap
        # (§14: a cap, not a commitment) instead of the single digits a
        # Priority 4 future goal used to leave.
        assert rec.budget == 40
        assert rec.skip_reason is None
        assert rec.stops.action == "pursue"
        assert rec.stops.spend_cap == 40
        # Ranks can still fall through on the empirical-pursuit criterion
        # (three copies inside 40 wishes is a long shot), but Tsaritsa is
        # never the blocker: no rejection names a shortfall.
        assert all(rejection.shortfalls == () for rejection in rec.rejected)

        # Tsaritsa is still protected and still reported - honestly, even
        # when the decision knowingly spends past it.
        (tsaritsa,) = rec.protected
        assert tsaritsa.goal == Goal("Tsaritsa", 0, 4)
        assert tsaritsa.banner == TSARITSA
        assert tsaritsa.constraining is False
        assert tsaritsa.probability < 0.9

    def test_a_higher_priority_future_goal_still_blocks_spending(self):
        """The inverse, and the reason the gate is a comparison rather than
        an exemption: when the future goal outranks the current one, the
        scan is back to (and the answer may be) "do not spend" (§1, §13)."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 2), Goal("Tsaritsa", 0, 1)],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=Account(wishes=40), roadmap=roadmap, current_version="7.0"
        )
        rec = recommend(
            context, (Preference("Vesna", 1, 0),), runs=400, seed=5, budgets=[40, 0]
        )
        assert rec.action == "skip"
        assert rec.outcome is None
        assert rec.budget == 0
        assert "higher-priority" in rec.skip_reason

        (rejection,) = rec.rejected
        assert rejection.outcome.label == "C0"
        (shortfall,) = rejection.shortfalls
        assert shortfall.goal == Goal("Tsaritsa", 0, 1)
        assert shortfall.constraining is True
        assert shortfall.meets_threshold is False

        # The do-nothing baseline still reports the protected goal (§1, §2).
        (baseline,) = rec.protected
        assert baseline.goal == Goal("Tsaritsa", 0, 1)
        assert baseline.constraining is True


class TestReportedPriorityRegression:
    """The reported bug: a lower-priority future goal must not veto spending
    on the current banner's higher-priority goal (§2).

    Navia is Priority 1 and is on the current banner (6.1 phase 1);
    Arlecchino is Priority 2 and arrives at 6.3 phase 2. With pity 70, 80
    wishes and no realistic 90% path to Arlecchino from fresh pity,
    Arlecchino's protection used to cap the Navia spend at single digits -
    the optimizer effectively answered "do not spend" on the user's own
    Priority 1 objective.
    """

    def _context(
        self, navia_priority: int, arlecchino_priority: int
    ) -> PlannerContext:
        """The reported account, with the two goals' priorities as arguments."""
        return PlannerContext(
            account=Account(current_pity=70, wishes=80),
            roadmap=Roadmap(
                goals=[
                    Goal("Navia", 0, navia_priority),
                    Goal("Arlecchino", 0, arlecchino_priority),
                ],
                banners=[
                    Banner("Navia", "6.1", 1),
                    Banner("Arlecchino", "6.3", 2),
                ],
            ),
            current_version="6.1",
            current_phase=1,
            income=IncomeForecast(
                versions=[
                    VersionIncome("6.1", estimate=IncomeEstimate(20, 30, 40)),
                    VersionIncome("6.3", estimate=IncomeEstimate(20, 30, 40)),
                ]
            ),
            confidence=0.9,
            mechanics=WishMechanics(
                banner_type="character",
                hard_pity=90,
                soft_pity_start=74,
                base_rate=0.006,
                soft_pity_increment=0.06,
                featured_rate=0.5,
            ),
        )

    def test_the_current_priority_one_goal_is_pursued(self):
        context = self._context(navia_priority=1, arlecchino_priority=2)
        assert constraining_goals(context) == frozenset()

        rec = recommend(context, (Preference("Navia", 1, 0),), runs=400, seed=0)
        assert rec.action == "pursue"
        assert rec.outcome.character == "Navia"
        assert rec.outcome.label == "C0"
        # Nothing future outranks Navia, so the whole pool is the cap - a cap,
        # not a commitment (§12): the pursuit stops when C0 is achieved.
        assert rec.budget == 80
        assert rec.plan.entries[0].banner == Banner("Navia", "6.1", 1)
        assert rec.plan.entries[0].budget == 80
        assert rec.stops.action == "pursue"
        assert rec.stops.spend_cap == 80
        assert rec.rejected == ()

        # Arlecchino is still protected, still planned, still reported - it
        # just no longer gates the decision (§2).
        (arlecchino,) = rec.protected
        assert arlecchino.goal == Goal("Arlecchino", 0, 2)
        assert arlecchino.banner == Banner("Arlecchino", "6.3", 2)
        assert arlecchino.constraining is False
        assert [entry.banner for entry in rec.plan.entries] == [
            Banner("Navia", "6.1", 1),
            Banner("Arlecchino", "6.3", 2),
        ]

    def test_a_higher_priority_future_goal_still_constrains(self):
        """The inverse: the gate is a comparison, not an exemption. With
        Arlecchino at Priority 1 and Navia at Priority 2, spending on Navia is
        blocked again - exactly as it was before the fix."""
        context = self._context(navia_priority=2, arlecchino_priority=1)
        assert constraining_goals(context) == frozenset({Goal("Arlecchino", 0, 1)})

        rec = recommend(
            context,
            (Preference("Navia", 1, 0),),
            runs=400,
            seed=0,
            budgets=[80, 40],
        )
        assert rec.action == "skip"
        assert rec.budget == 0
        assert rec.plan is None
        assert "higher-priority" in rec.skip_reason

        # Arlecchino is the blocker, and the reason is named (§13 step 7).
        (rejection,) = rec.rejected
        assert rejection.best.budget == 40
        (shortfall,) = rejection.shortfalls
        assert shortfall.goal == Goal("Arlecchino", 0, 1)
        assert shortfall.constraining is True
        assert shortfall.meets_threshold is False

        # The do-nothing baseline: with no Navia spending Arlecchino is
        # certain, which is exactly what the current banner cannot have.
        (baseline,) = rec.protected
        assert baseline.goal == Goal("Arlecchino", 0, 1)
        assert baseline.constraining is True
        assert baseline.probability == pytest.approx(1.0, abs=0.005)


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
        therefore feasible at no cap; C0 is feasible at cap 2.

        B is Priority 1 and A (whose banner is current) is Priority 2, so B
        outranks the decision and gates it (§2); see
        optimizer.protection.constraining_goals."""
        roadmap = Roadmap(
            goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
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
        here B, unreachable at 90% with a single wish. B is Priority 1 and A
        is Priority 2, so B legitimately constrains the spend (§2)."""
        roadmap = Roadmap(
            goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
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
        assert standing.goal == Goal("B", 0, 1)
        assert standing.banner == Banner("B", "7.1", 1)
        assert standing.constraining is True
        assert standing.meets_threshold is False
        assert standing.probability < 0.9

