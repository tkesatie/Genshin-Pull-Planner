"""Candidate evaluation through the Phase 4 simulator (§13 steps 4-5, §14).

Deterministic-mechanics helpers keep protection math exact:

    sparse_mechanics(): no 5-star before soft pity; pull 2 (soft) and
    pull 3 (hard pity) are certain 5-stars, and featured_rate=0.5 makes
    each one a coin flip. Cycles of lost-50/50-then-guaranteed are
    therefore exactly reproducible, so probabilities are clean fractions
    and the guarantee-carry arithmetic below can be asserted within
    sampling noise of exact values.
"""

import pytest

from domain import (
    CHARACTER_EVENT_BANNER,
    Account,
    Banner,
    Goal,
    Ownership,
    Roadmap,
    WishMechanics,
)
from optimizer import OutcomeOption, evaluate_candidate, evaluate_skip_baseline
from planner import PlannerContext
from probability import cumulative_probability

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


class TestOutcomeProbability:
    def test_matches_the_phase2_curve(self):
        """No protected goals: the outcome probability is exactly the
        Phase 2 single-copy curve at the account's real state (§10.2,
        §14: the optimizer inherits simulation's Phase 2 validation)."""
        roadmap = Roadmap(goals=[Goal("Vesna", 0, 1)], banners=[VESNA])
        account = Account(current_pity=37, character_guarantee=False, wishes=60)
        context = PlannerContext(
            account=account, roadmap=roadmap, current_version="7.0"
        )
        candidate = evaluate_candidate(
            context, OutcomeOption("Vesna", 0, 1), budget=60, runs=6_000, seed=1234
        )
        expected = float(
            cumulative_probability(60, 37, False, CHARACTER_EVENT_BANNER)[60]
        )
        assert candidate.outcome_probability == pytest.approx(expected, abs=0.03)
        # 6000 trials: SE <= sqrt(0.25/6000) ~ 0.0065, so 0.03 is a
        # generous ~4.6-sigma band - never flaky, still catches bugs.
        assert candidate.feasible is True  # nothing to protect

    def test_zero_budget_pursues_nothing(self):
        roadmap = Roadmap(goals=[Goal("Vesna", 0, 1)], banners=[VESNA])
        context = PlannerContext(
            account=Account(wishes=40), roadmap=roadmap, current_version="7.0"
        )
        candidate = evaluate_candidate(
            context, OutcomeOption("Vesna", 0, 1), budget=0, runs=100, seed=0
        )
        assert candidate.outcome_probability == 0.0
        assert candidate.feasible is False  # protection holds, pursuit does not


class TestFeasibility:
    def test_protection_met_but_unpursueable_is_infeasible(self):
        """The empirical-nonzero criterion (§13 step 5, documented Monte
        Carlo semantics): a met threshold cannot rescue a pursuit that
        spends nothing."""
        tiny = WishMechanics(
            banner_type="tiny",
            hard_pity=5,
            soft_pity_start=3,
            base_rate=0.25,
            soft_pity_increment=0.2,
            featured_rate=0.5,
        )
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=Account(wishes=5),
            roadmap=roadmap,
            current_version="7.0",
            confidence=0.1,
            mechanics=tiny,
        )
        candidate = evaluate_candidate(
            context, OutcomeOption("Vesna", 0, 1), budget=0, runs=400, seed=2
        )
        assert candidate.protected[0].meets_threshold is True
        assert candidate.outcome_probability == 0.0
        assert candidate.feasible is False

    def test_budget_bounds_are_enforced(self, doc_context):
        outcome = OutcomeOption("Vesna", 0, 1)
        with pytest.raises(ValueError, match="budget must satisfy"):
            evaluate_candidate(doc_context, outcome, -1, runs=10, seed=0)
        with pytest.raises(ValueError, match="budget must satisfy"):
            evaluate_candidate(doc_context, outcome, 41, runs=10, seed=0)


class TestGroupingNeverMergesGoals:
    def test_each_roadmap_goal_keeps_its_own_standing(self):
        """Review point 2: grouping is execution only. Two Vesna goals on
        one future banner share a plan entry (target C2) yet are evaluated
        independently: 6 wishes can reach C0 but never C2 (3 copies)."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Vesna", 2, 2)],
            # Vesna is a strictly-future banner: the current banner (§7) is
            # Z 7.0 phase 1, so the two Vesna goals are protected (§13).
            banners=[Banner("Z", "7.0", 1), Banner("Vesna", "7.1", 1)],
        )
        context = PlannerContext(
            account=Account(wishes=6),
            roadmap=roadmap,
            current_version="7.0",
            mechanics=sparse_mechanics(),
        )
        candidate = evaluate_candidate(
            context, OutcomeOption("Z", 0, 1), budget=6, runs=2_000, seed=3
        )
        standings = {s.goal: s for s in candidate.protected}
        assert set(standings) == {Goal("Vesna", 0, 1), Goal("Vesna", 2, 2)}
        assert standings[Goal("Vesna", 0, 1)].probability > 0.0
        assert standings[Goal("Vesna", 2, 2)].probability == 0.0


class TestSkipBaseline:
    def test_matches_a_zero_budget_candidate(self, doc_context, doc_income):
        context = PlannerContext(
            account=doc_context.account,
            roadmap=doc_context.roadmap,
            current_version="7.0",
            income=doc_income,
        )
        baseline = evaluate_skip_baseline(context, runs=500, seed=4)
        candidate = evaluate_candidate(
            context, OutcomeOption("Vesna", 0, 1), budget=0, runs=500, seed=4
        )
        assert baseline == candidate.protected


class TestGuaranteeCarry:
    """The pity/guarantee carry §14 demands be modelled (§14, review point 9),
    in its honest form.

    Sparse mechanics make the state fully transparent: pity is 1 and with
    soft_pity_start=2 the very next wish is a *certain* 5-star (rate 1.0),
    as is every wish after it; featured_rate=0.5 makes each 5-star a coin
    flip. A lost 50/50 therefore carries a guarantee plus a pity head start
    into the next banner - exactly what the Phase 3 independent-reserve
    approximation cannot represent.

    Three facts are pinned here:

    1. At EQUAL remaining wishes the carried guarantee dominates:
       P(B | 2 wishes, guarantee on) = 1.0 while
       P(B | 2 wishes, guarantee off) = 0.5.
    2. Feasibility is NOT monotone in the cap: caps 0..3 give
       {False, True, True, False}. Spending the third wish costs B a
       quarter of its probability (0.75 -> 0.25), so the largest feasible
       cap is 2, not 3 - a descending *scan* finds it, and "spend
       everything" does not (§13, §14).
    3. Spending on this banner never *improves* raw protection: the
       do-nothing baseline keeps both its wishes and its pity head start,
       so it is weakly best (1.0 here). Spending creates the carried
       guarantee only by consuming wishes and resetting pity, which is why
       the carry matters between states of EQUAL wishes and as the
       non-interval feasibility of fact 2 - not as a free lunch.

    B is Priority 1 and A (the current banner's goal) is Priority 2: only a
    future goal that *outranks* the current decision constrains it (§2, see
    optimizer.protection.constraining_goals). The protection story below is
    therefore the higher-priority-future-goal story.
    """

    def _scenario(self) -> PlannerContext:
        roadmap = Roadmap(
            goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
            banners=[Banner("A", "7.0", 1), Banner("B", "7.1", 1)],
        )
        account = Account(current_pity=1, character_guarantee=False, wishes=3)
        return PlannerContext(
            account=account,
            roadmap=roadmap,
            current_version="7.0",
            confidence=0.6,
            mechanics=sparse_mechanics(),
        )

    def test_equal_wishes_guarantee_domination(self):
        """Fact 1, exactly: B needs one featured copy and the next wish is a
        certain 5-star at pity 1, so guarantee on -> featured (1.0) and
        guarantee off -> 50/50 (0.5). The guarantee branch consumes no
        randomness at all (simulation.engine short-circuits it), so 1.0 is
        an exact value, not a sampled one."""
        roadmap = Roadmap(goals=[Goal("B", 0, 1)], banners=[Banner("B", "7.1", 1)])
        probabilities: dict[bool, float] = {}
        for guarantee in (True, False):
            context = PlannerContext(
                account=Account(
                    current_pity=1, character_guarantee=guarantee, wishes=2
                ),
                roadmap=roadmap,
                current_version="7.1",
                confidence=0.6,
                mechanics=sparse_mechanics(),
            )
            probabilities[guarantee] = evaluate_candidate(
                context, OutcomeOption("B", 0, 1), 2, runs=4_000, seed=1
            ).outcome_probability
        assert probabilities[True] == 1.0
        assert probabilities[False] == pytest.approx(0.5, abs=0.03)

    def test_caps_one_and_two_carry_the_guarantee(self):
        """cap 1: half the runs win the copy and stop 1 wish in (2 wishes
        left, no guarantee -> B = 0.5); half lose the 50/50 and cap out
        (2 wishes left, guarantee on -> B = 1.0). P(B) = 0.75.
        cap 2: the lost branch spends its second wish on pity and still
        leaves B a certain guaranteed 5-star -> P(B) = 0.75 again, because
        the extra cap is only spent when the target is still unmet."""
        context = self._scenario()
        for cap in (1, 2):
            candidate = evaluate_candidate(
                context, OutcomeOption("A", 0, 1), cap, runs=4_000, seed=7
            )
            assert candidate.protected[0].probability == pytest.approx(
                0.75, abs=0.03
            )
            assert candidate.feasible is True

    def test_a_third_wish_costs_protection(self):
        """cap 3: the lost branch now spends all three wishes on A, leaving
        B nothing to pull with: P(B) = 0.5 * 0.5 + 0.5 * 0.0 = 0.25, below
        the 0.6 threshold. One more wish made the strategy infeasible
        (fact 2)."""
        context = self._scenario()
        candidate = evaluate_candidate(
            context, OutcomeOption("A", 0, 1), 3, runs=4_000, seed=7
        )
        assert candidate.protected[0].probability == pytest.approx(0.25, abs=0.03)
        assert candidate.feasible is False

    def test_skip_baseline_keeps_the_pity_head_start(self):
        """Fact 3: doing nothing keeps 3 wishes AND pity 1, so B's pulls are
        certain 5-stars until the goal is met - 1.0, the ceiling no spending
        plan can beat."""
        context = self._scenario()
        baseline = evaluate_skip_baseline(context, runs=4_000, seed=7)
        assert baseline[0].probability == pytest.approx(1.0, abs=0.005)
        assert baseline[0].meets_threshold is True

    def test_feasibility_is_not_monotone_in_the_cap(self):
        context = self._scenario()
        feasibilities = [
            evaluate_candidate(
                context, OutcomeOption("A", 0, 1), cap, runs=4_000, seed=7
            ).feasible
            for cap in (0, 1, 2, 3)
        ]
        assert feasibilities == [False, True, True, False]

    def test_recommendation_takes_the_largest_feasible_cap(self):
        """The exhaustive scan beats "spend everything": cap 3 is
        infeasible while cap 2 is feasible, so the recommendation caps at 2
        (§14: the cap is a decision, not the whole balance)."""
        from optimizer import recommend

        recommendation = recommend(self._scenario(), runs=4_000, seed=7)
        assert recommendation.action == "pursue"
        assert recommendation.outcome.label == "C0"
        assert recommendation.budget == 2
        assert recommendation.protected[0].meets_threshold is True
        assert recommendation.stops.spend_cap == 2

