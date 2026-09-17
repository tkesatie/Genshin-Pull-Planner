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
from optimizer import (
    available_outcomes,
    constraining_goals,
    evaluate_skip_baseline,
    recommend,
)
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
        # Issue 2 (minimum_outcome_probability, §14): the winning outcome
        # here (see below) is only reachable a small fraction of the time
        # on 40 wishes from fresh pity, well under the 25% minimum for an
        # ordinary recommendation, so this is the disclosed-gamble
        # "discretionary" action. Tsaritsa's non-gating status - this
        # test's actual subject - is unaffected: it is still reported,
        # still non-constraining, and never named as a blocker below.
        assert rec.action == "discretionary"
        assert rec.outcome_probability < 0.25
        assert rec.outcome.character == "Vesna"
        # Nothing future outranks Vesna C0, so the whole pool is the cap
        # (§14: a cap, not a commitment) instead of the single digits a
        # Priority 4 future goal used to leave.
        assert rec.budget == 40
        assert rec.skip_reason is None
        assert rec.stops.action == "discretionary"
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
        # Issue 2 (minimum_outcome_probability, §14): 40 wishes from fresh
        # pity on this banner's mechanics gives this C0 only ~11% empirical
        # success - feasible (Tsaritsa stays protected) but below the 25%
        # minimum for an ordinary recommendation, so this is now the
        # disclosed-gamble "discretionary" action rather than "pursue". The
        # cap itself - this test's actual subject - is unaffected.
        assert rec.action == "discretionary"
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
        won over C0 because it ranks first - the preference system is a
        preference system, not a probability score (§13 step 7). This is
        also a textbook case for Issue 2's minimum_outcome_probability
        (§14): the win itself does not change, but reaching for 3 copies on
        120 wishes is only ~1-2% likely, so it is reported as the
        disclosed-gamble "discretionary" action rather than an ordinary
        "pursue" - the 25% floor decides how the winner is presented, never
        which outcome wins (optimizer.recommend module docstring)."""
        chain = (
            Preference("Vesna", 1, 2, weapon_refinement=1),
            Preference("Vesna", 2, 0),
        )
        rec = recommend(self._context(), chain, runs=3_000, seed=13)
        assert rec.action == "discretionary"
        assert rec.outcome.label == "C2R1"
        assert rec.budget == 120
        assert 0.0 < rec.outcome_probability < 0.25
        assert rec.plan.entries[0].target_constellation == 2
        assert rec.discretionary_reason is not None
        assert "C2R1" in rec.discretionary_reason

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


class TestSameCharacterProgression:
    """Reported bug: a same-character chain is a progression, not a set of
    mutually exclusive alternatives. Navia C0 (rank 1) and Navia C2 (rank
    3) both name the current banner's character; since reaching C2
    necessarily reaches C0 (§4.2, §12), pursuing C2 is never worse than
    pursuing C0 alone, and the optimizer must not stop at the first
    rank-ordered feasible outcome (optimizer.outcomes module docstring).

    Arlecchino C0 sits at rank 2, between the two Navia targets, purely to
    show that an unrelated character's place in the chain has no bearing
    on this ordering: `available_outcomes` already restricts outcomes to
    the current banner's character (Navia here), so Arlecchino never
    becomes a competing outcome regardless of its rank.
    """

    def _context(self, wishes: int) -> PlannerContext:
        roadmap = Roadmap(goals=[], banners=[Banner("Navia", "7.0", 1)])
        return PlannerContext(
            account=Account(wishes=wishes), roadmap=roadmap, current_version="7.0"
        )

    def _chain(self) -> tuple[Preference, ...]:
        return (
            Preference("Navia", 1, 0),  # P1: Navia C0
            Preference("Arlecchino", 2, 0),  # P2: Arlecchino C0 (other banner)
            Preference("Navia", 3, 2),  # P3: Navia C2
        )

    def test_the_higher_constellation_is_recommended_when_feasible(self):
        """600 wishes: C2 is well within reach, so it is recommended over
        C0 even though C0 outranks it in the stated preference order."""
        rec = recommend(
            self._context(wishes=600), self._chain(), runs=2_000, seed=7
        )
        assert rec.action == "pursue"
        assert rec.outcome.character == "Navia"
        assert rec.outcome.label == "C2"
        assert rec.outcome_probability > 0.0
        assert rec.plan.entries[0].target_constellation == 2
        # C0 was never even the closest rejection - it wasn't tried, since
        # the more-inclusive C2 outcome won outright.
        assert rec.rejected == ()

    def test_falls_back_to_the_lower_constellation_when_constrained(self):
        """Too few wishes to safely reach C2: the optimizer falls back to
        C0 rather than skipping outright (the required behavior to
        preserve). With only 1 wish, even C0 is a long shot - well under
        Issue 2's 25% minimum_outcome_probability (§14) - so the winning
        outcome is reported as the disclosed-gamble "discretionary" action;
        the fallback to C0 itself, this test's actual subject, is
        unaffected."""
        rec = recommend(
            self._context(wishes=1), self._chain(), runs=2_000, seed=7
        )
        assert rec.action == "discretionary"
        assert rec.outcome.label == "C0"
        assert rec.plan.entries[0].target_constellation == 0
        assert rec.outcome_probability < 0.25
        # C2 was tried first (per the new ordering) and rejected before C0
        # was reached.
        assert [r.outcome.label for r in rec.rejected] == ["C2"]

    def test_skips_when_neither_constellation_is_reachable(self):
        """Zero wishes: neither Navia target can be pursued at all."""
        rec = recommend(
            self._context(wishes=0), self._chain(), runs=500, seed=7
        )
        assert rec.action == "skip"
        assert [r.outcome.label for r in rec.rejected] == ["C2", "C0"]

    def test_outcomes_are_ordered_by_descending_constellation(self):
        """The ordering itself, independent of feasibility: C2 (rank 3)
        precedes C0 (rank 1) because it is the more-inclusive target, and
        Arlecchino (a different character) never appears at all."""
        outcomes = available_outcomes(
            self._context(wishes=600), self._chain()
        )
        assert [(o.character, o.label, o.rank) for o in outcomes] == [
            ("Navia", "C2", 3),
            ("Navia", "C0", 1),
        ]

    def test_a_preference_alone_never_protects_anything(self):
        """Arlecchino here is only ever a Preference entry (rank 2, no
        roadmap Goal) - exactly the original bug report's shape. Priority
        protection is a Goals-only concept (§2, §15): with no Goal, there
        is nothing to gate the reach to C2, at any wish count. This is
        the reason `TestPriorityGatedProgression` below models Arlecchino
        as an actual roadmap Goal instead. 150 wishes for 2 copies is only
        a ~4% empirical shot, below Issue 2's 25% minimum_outcome_probability
        (§14), so the winning C2 outcome - unchanged by that check - is
        reported as "discretionary" rather than "pursue"."""
        rec = recommend(
            self._context(wishes=150), self._chain(), runs=1_500, seed=3
        )
        assert rec.action == "discretionary"
        assert rec.outcome.label == "C2"
        assert rec.budget == 150
        assert rec.outcome_probability < 0.25
        assert rec.rejected == ()


class TestPriorityGatedProgression:
    """The design document's own worked example, applied to this bug:

        Priority 1 -> Navia C0
        Priority 2 -> Arlecchino C0
        Priority 3 -> Navia C2

    Unlike `TestSameCharacterProgression`, priority here comes from actual
    roadmap Goals (the only place priority lives, §2/§15) - not from
    preference rank. The Navia preference chain (C0, then C2) is what
    makes C2 an available outcome at all; the Goals are what let
    Arlecchino's Priority 2 gate the reach to Navia's Priority 3 target
    without gating Navia's own Priority 1 target.
    """

    MECHANICS = WishMechanics(
        banner_type="character",
        hard_pity=90,
        soft_pity_start=74,
        base_rate=0.006,
        soft_pity_increment=0.06,
        featured_rate=0.5,
    )

    def _context(self, wishes: int) -> PlannerContext:
        roadmap = Roadmap(
            goals=[
                Goal("Navia", 0, 1),
                Goal("Arlecchino", 0, 2),
                Goal("Navia", 2, 3),
            ],
            banners=[Banner("Navia", "6.1", 1), Banner("Arlecchino", "6.3", 2)],
        )
        return PlannerContext(
            account=Account(wishes=wishes),
            roadmap=roadmap,
            current_version="6.1",
            current_phase=1,
            confidence=0.9,
            mechanics=self.MECHANICS,
        )

    def _chain(self) -> tuple[Preference, ...]:
        return (Preference("Navia", 1, 0), Preference("Navia", 2, 2))

    def test_c2_is_pursued_up_to_the_cap_that_still_protects_arlecchino(self):
        """Plenty of wishes: C2 is reachable, and the recommended budget is
        the largest one that still keeps Arlecchino (Priority 2, gating
        Navia's Priority 3 target) at or above the confidence threshold -
        not necessarily every wish in the account (§14: a cap, not a
        commitment)."""
        rec = recommend(self._context(wishes=600), self._chain(), runs=2_000, seed=3)
        assert rec.action == "pursue"
        assert rec.outcome.label == "C2"
        assert rec.outcome_probability > 0.0
        (arlecchino,) = [s for s in rec.protected if s.goal.character == "Arlecchino"]
        assert arlecchino.constraining is True
        assert arlecchino.meets_threshold is True

    def test_falls_back_to_c0_when_c2_would_endanger_arlecchino(self):
        """Tighter budget: pushing for C2 would leave Arlecchino below the
        confidence threshold, so the optimizer falls back to Navia's own
        Priority 1 target - which is not gated by Arlecchino at all."""
        rec = recommend(self._context(wishes=90), self._chain(), runs=2_000, seed=3)
        assert rec.action == "pursue"
        assert rec.outcome.label == "C0"
        assert [r.outcome.label for r in rec.rejected] == ["C2"]
        (shortfall,) = rec.rejected[0].shortfalls
        assert shortfall.goal == Goal("Arlecchino", 0, 2)
        assert shortfall.constraining is True
        assert shortfall.meets_threshold is False


class TestDiscretionaryGamble:
    """Issue 2 (Phase 5 correction): minimum_outcome_probability (§14).

    The confidence threshold protects higher-priority future goals; this is
    the separate question of whether the winning current-banner outcome, at
    its largest feasible cap, is itself likely enough to present as an
    ordinary "pursue" recommendation. Sparse mechanics (hard_pity=3,
    featured_rate tuned to the exact probability wanted) make the numbers
    exact rather than approximate: with base_rate effectively 0 and a hard
    pity of 3, a fresh account is guaranteed its first 5-star at exactly 3
    wishes, and featured_rate is then, deterministically, the chance that
    5-star is the target character - so `featured_rate=0.2` and
    `featured_rate=0.3` give exact 20% and 30% outcome probabilities with
    only ordinary Monte Carlo sampling noise around them (well clear of the
    25% line at a few thousand runs), never the coarse 0%/50%/100% jumps a
    50/50 featured rate would produce.
    """

    def _mechanics(self, featured_rate: float) -> WishMechanics:
        return WishMechanics(
            banner_type="sparse",
            hard_pity=3,
            soft_pity_start=2,
            base_rate=1e-9,
            soft_pity_increment=1.0,
            featured_rate=featured_rate,
        )

    def _context(self, featured_rate: float, wishes: int = 3) -> PlannerContext:
        """No protected goals: the cap is the whole pool, unconstrained, so
        the outcome's own probability is the only thing in question."""
        roadmap = Roadmap(goals=[], banners=[Banner("Navia", "7.0", 1)])
        return PlannerContext(
            account=Account(wishes=wishes),
            roadmap=roadmap,
            current_version="7.0",
            mechanics=self._mechanics(featured_rate),
        )

    def test_probability_just_below_25_percent_is_discretionary(self):
        """A guaranteed 5-star at exactly 20% featured rate: the outcome is
        feasible (nonzero, and nothing to protect) but the empirical
        probability sits at ~20%, below the 25% minimum - a disclosed
        gamble, not an ordinary recommendation."""
        rec = recommend(
            self._context(featured_rate=0.2), (Preference("Navia", 1, 0),),
            runs=6_000, seed=11,
        )
        assert rec.action == "discretionary"
        assert rec.outcome.label == "C0"
        assert rec.budget == 3
        assert rec.outcome_probability == pytest.approx(0.2, abs=0.03)
        assert rec.outcome_probability < 0.25
        assert rec.skip_reason is None
        assert rec.discretionary_reason is not None
        assert "C0" in rec.discretionary_reason
        assert "20%" in rec.discretionary_reason
        assert rec.stops.action == "discretionary"
        assert rec.stops.spend_cap == 3
        # The plan still executes the pursuit - a discretionary action is
        # still an executable strategy, just not a recommended one (§14).
        assert rec.plan is not None
        assert rec.plan.entries[0].target_constellation == 0
        assert rec.plan.entries[0].budget == 3

    def test_probability_at_or_above_25_percent_is_pursue(self):
        """The identical setup, 10 percentage points higher: now an ordinary
        recommendation, at the same budget and the same winning outcome -
        only the empirical probability changed."""
        rec = recommend(
            self._context(featured_rate=0.3), (Preference("Navia", 1, 0),),
            runs=6_000, seed=11,
        )
        assert rec.action == "pursue"
        assert rec.outcome.label == "C0"
        assert rec.budget == 3
        assert rec.outcome_probability == pytest.approx(0.3, abs=0.03)
        assert rec.outcome_probability >= 0.25
        assert rec.discretionary_reason is None
        assert rec.stops.action == "pursue"

    def test_custom_minimum_does_not_change_the_winning_outcome_or_cap(self):
        """Lowering the minimum for the same ~20% scenario turns it back
        into a "pursue" - proof the threshold only relabels the decision
        already made, it never re-derives the outcome or the cap (module
        docstring: not an outcome-selection rule)."""
        rec = recommend(
            self._context(featured_rate=0.2), (Preference("Navia", 1, 0),),
            runs=6_000, seed=11, minimum_outcome_probability=0.1,
        )
        assert rec.action == "pursue"
        assert rec.outcome.label == "C0"
        assert rec.budget == 3
        assert rec.outcome_probability == pytest.approx(0.2, abs=0.03)

    def test_rejects_an_out_of_range_minimum(self):
        with pytest.raises(ValueError):
            recommend(
                self._context(featured_rate=0.2),
                (Preference("Navia", 1, 0),),
                minimum_outcome_probability=1.5,
            )


class TestDiscretionaryUnderPriorityProtection:
    """Issue 2's own worked example, made exact with sparse mechanics: a
    higher-priority future goal (B) reserves enough of the pool that the
    current banner's own objective (A) is left with a small, feasible-but-
    unlikely budget. Confirms the 25% check operates *after* priority
    protection has already picked the cap - it never substitutes for it,
    and protection (§2) keeps working exactly as it does under a "pursue".
    """

    def _mechanics(self, featured_rate: float) -> WishMechanics:
        return WishMechanics(
            banner_type="sparse",
            hard_pity=3,
            soft_pity_start=2,
            base_rate=1e-9,
            soft_pity_increment=1.0,
            featured_rate=featured_rate,
        )

    def _context(self, featured_rate: float, wishes: int) -> PlannerContext:
        """A (current, Priority 2) vs B (future, Priority 1, gating). B
        needs 4 wishes to reach certainty at this mechanics (0.2 -> 1.0 via
        the character-guarantee carry), so protecting B to 90% at 6 total
        wishes leaves A a cap of exactly 3 - a guaranteed single 5-star
        pull, whose featured_rate is A's entire outcome probability."""
        roadmap = Roadmap(
            goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
            banners=[Banner("A", "7.0", 1), Banner("B", "7.1", 1)],
        )
        return PlannerContext(
            account=Account(wishes=wishes),
            roadmap=roadmap,
            current_version="7.0",
            confidence=0.9,
            mechanics=self._mechanics(featured_rate),
        )

    def test_low_probability_after_protection_is_discretionary(self):
        rec = recommend(
            self._context(featured_rate=0.2, wishes=6),
            (Preference("A", 1, 0),),
            runs=6_000,
            seed=11,
        )
        assert rec.action == "discretionary"
        assert rec.outcome.character == "A"
        assert rec.outcome.label == "C0"
        # The cap is what protection allows - not the whole pool (§14).
        assert rec.budget == 3
        assert rec.outcome_probability == pytest.approx(0.2, abs=0.03)
        assert rec.outcome_probability < 0.25

        (b,) = [s for s in rec.protected if s.goal.character == "B"]
        assert b.constraining is True
        assert b.meets_threshold is True
        assert b.probability == pytest.approx(1.0, abs=0.01)
        assert rec.rejected == ()
        assert rec.discretionary_reason is not None

    def test_higher_probability_at_the_same_protected_cap_is_pursue(self):
        """Same protection story, same resulting cap (3) - only A's own
        featured rate changed, so only the label changes."""
        rec = recommend(
            self._context(featured_rate=0.3, wishes=6),
            (Preference("A", 1, 0),),
            runs=6_000,
            seed=11,
        )
        assert rec.action == "pursue"
        assert rec.outcome.character == "A"
        assert rec.budget == 3
        assert rec.outcome_probability == pytest.approx(0.3, abs=0.03)

        (b,) = [s for s in rec.protected if s.goal.character == "B"]
        assert b.constraining is True
        assert b.meets_threshold is True

    def test_too_few_wishes_to_protect_b_at_all_is_still_skip(self):
        """Below B's own reserve, the decision is still "do not spend" -
        the 25% check never turns an infeasible (protection-violating)
        candidate into a discretionary one; feasibility is unchanged."""
        rec = recommend(
            self._context(featured_rate=0.2, wishes=4),
            (Preference("A", 1, 0),),
            runs=2_000,
            seed=11,
        )
        assert rec.action == "skip"
        assert rec.outcome is None
        assert rec.discretionary_reason is None


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

