"""Protected future goals, grouped per banner (Design Document §13, §14)."""

import pytest

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
)
from optimizer import constraining_goals, current_goal_priority, protected_groups
from planner import PlannerContext, evaluate_goals, protected_goal_outcomes

VESNA = Banner("Vesna", "7.0", 1)
TSARITSA = Banner("Tsaritsa", "7.1", 1)
VODYNISTA = Banner("Vodynista", "7.2", 1)


def test_doc_example_protects_only_tsaritsa(doc_context):
    groups = protected_groups(doc_context)
    assert [group.banner for group in groups] == [TSARITSA]
    assert groups[0].goals == (Goal("Tsaritsa", 0, 4),)
    assert groups[0].target_constellation == 0


def test_satisfied_and_current_banner_goals_are_not_protected(doc_context):
    """Vodynista C0 needs nothing; Vesna C2 belongs to the current banner
    (and is blocked anyway) - protection is strictly future (§13)."""
    characters = [
        goal.character
        for group in protected_groups(doc_context)
        for goal in group.goals
    ]
    assert "Vodynista" not in characters
    assert "Vesna" not in characters


def test_strategic_budget_is_wishes_plus_income_credit(doc_account, doc_roadmap):
    """The strategic budget is large enough that the entry's budget is
    never the limiting factor - it is NOT claimed to be the exact future
    balance (optimizer.protection docstring); min(budget, available) in
    the simulator is the real bound (§12)."""
    income = IncomeForecast(
        versions=[VersionIncome("7.1", estimate=IncomeEstimate(20, 60, 80))]
    )
    context = PlannerContext(
        account=doc_account, roadmap=doc_roadmap, current_version="7.0", income=income
    )
    group = protected_groups(context)[0]
    assert group.uncapped_budget == 40 + 60


def test_future_entry_cannot_drive_wishes_negative(doc_account, doc_roadmap):
    """The safety mechanism is min(budget, available) (§12): even with a
    strategic budget beyond any plausible balance, simulated wishes never
    go negative and no banner ever spends more than its cap (§4.1).

    Review point 1: the strategic budget is NOT claimed to be an exact
    future balance - it is 40 (current wishes) + 5 (7.1 income) while the
    current banner may already spend 40 of that. That is harmless because
    the simulator's min(budget, available) is what actually bounds
    spending, which is exactly what this test pins.
    """
    import numpy as np

    from optimizer import available_outcomes, evaluate_candidate
    from simulation import simulate_history

    income = IncomeForecast(
        versions=[
            VersionIncome("7.1", estimate=IncomeEstimate(0, 5, 10)),
            VersionIncome("7.2", estimate=IncomeEstimate(0, 5, 10)),
        ]
    )
    context = PlannerContext(
        account=doc_account, roadmap=doc_roadmap, current_version="7.0", income=income
    )
    group = protected_groups(context)[0]
    assert group.uncapped_budget == doc_account.wishes + context.income_credit(
        group.banner.version
    )

    candidate = evaluate_candidate(
        context, available_outcomes(context)[0], budget=40, runs=200, seed=1
    )
    assert candidate.result.final_wishes_min >= 0
    rng = np.random.default_rng(1)
    for _ in range(50):
        history = simulate_history(context, candidate.plan, rng)
        for banner_result in history.banner_results:
            assert banner_result.wishes_spent <= banner_result.budget
            assert banner_result.account_after.wishes >= 0


def test_multiple_goals_on_one_banner_group_with_max_constellation(doc_account):
    """Two Tsaritsa goals (C1 and C3) share a banner: one entry with target
    C3 - sufficient for both because constellation goals of a character
    are strictly ordered (§9)."""
    roadmap = Roadmap(
        goals=[
            Goal("Z", 0, 1),
            Goal("Tsaritsa", 1, 2),
            Goal("Tsaritsa", 3, 3),
        ],
        banners=[VESNA, TSARITSA, VODYNISTA],
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    groups = protected_groups(context)
    assert len(groups) == 1
    assert groups[0].banner == TSARITSA
    assert groups[0].goals == (Goal("Tsaritsa", 1, 2), Goal("Tsaritsa", 3, 3))
    assert groups[0].target_constellation == 3


def test_goals_on_different_banners_stay_separate_groups(doc_account):
    roadmap = Roadmap(
        goals=[Goal("Tsaritsa", 0, 1), Goal("Vodynista", 1, 2)],
        banners=[VESNA, TSARITSA, VODYNISTA],
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    groups = protected_groups(context)
    assert [group.banner for group in groups] == [TSARITSA, VODYNISTA]


def test_chronological_order_regardless_of_priority(doc_account):
    """Banner order and priority order may differ (§7); protection walks
    the calendar."""
    roadmap = Roadmap(
        goals=[Goal("A", 0, 2), Goal("B", 0, 1)],
        banners=[VESNA, Banner("A", "7.2", 1), Banner("B", "7.1", 1)],
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    assert [group.banner.character for group in protected_groups(context)] == [
        "B",
        "A",
    ]


def test_goals_without_future_banner_are_excluded_but_still_evaluated():
    """'Not protectable' must not become 'does not exist' (§8)."""
    roadmap = Roadmap(
        goals=[Goal("Vesna", 0, 1)],
        # Vesna's only banner is in the past (6.9): the goal has no upcoming
        # banner, so it cannot be scheduled even though it is unsatisfied.
        banners=[Banner("Z", "7.0", 1), Banner("Vesna", "6.9", 1)],
    )
    context = PlannerContext(
        account=Account(wishes=40), roadmap=roadmap, current_version="7.0"
    )
    assert protected_groups(context) == ()
    evaluations = {e.goal.character: e for e in evaluate_goals(context)}
    assert evaluations["Vesna"].next_banner is None
    assert evaluations["Vesna"].copies_needed == 1  # still evaluated (§8)


def _group_goal_banner_pairs(context):
    return {
        (goal, group.banner)
        for group in protected_groups(context)
        for goal in group.goals
    }


def _planner_goal_banner_pairs(context):
    return {
        (outcome.goal, outcome.banner)
        for outcome in protected_goal_outcomes(context)
    }


class TestPriorityGate:
    """Classification (what is protected) stays priority-independent; the
    *constraint* (what may veto this decision) does not (§2)."""

    def _context(self, navia_priority: int, arlecchino_priority: int):
        roadmap = Roadmap(
            goals=[
                Goal("Navia", 0, navia_priority),
                Goal("Arlecchino", 0, arlecchino_priority),
            ],
            banners=[Banner("Navia", "6.1", 1), Banner("Arlecchino", "6.3", 2)],
        )
        return PlannerContext(
            account=Account(wishes=80), roadmap=roadmap, current_version="6.1"
        )

    def test_the_anchor_is_the_current_banners_best_active_goal(self):
        assert current_goal_priority(self._context(1, 2)) == 1
        assert current_goal_priority(self._context(2, 1)) == 2

    def test_the_anchor_is_none_without_an_active_goal_on_this_banner(self):
        """A decision made from a preference chain alone is not anchored to a
        priority, so nothing is filtered out (the conservative reading)."""
        roadmap = Roadmap(
            goals=[Goal("Arlecchino", 0, 1)],
            banners=[Banner("Navia", "6.1", 1), Banner("Arlecchino", "6.3", 2)],
        )
        context = PlannerContext(
            account=Account(wishes=80), roadmap=roadmap, current_version="6.1"
        )
        assert current_goal_priority(context) is None
        assert constraining_goals(context) == frozenset({Goal("Arlecchino", 0, 1)})

    def test_a_lower_priority_future_goal_does_not_constrain(self):
        context = self._context(1, 2)
        assert protected_groups(context)  # still protected ...
        assert constraining_goals(context) == frozenset()  # ... but not gating

    def test_a_higher_priority_future_goal_does_constrain(self):
        context = self._context(2, 1)
        assert constraining_goals(context) == frozenset({Goal("Arlecchino", 0, 1)})

    def test_the_doc_example_tsaritsa_is_protected_but_not_gating(self, doc_context):
        """The classification still lists Tsaritsa - the Phase 3 agreement
        test keeps holding - while the §2 gate excludes it, because the
        current banner's Vesna C0 is Priority 1 and Tsaritsa C0 is Priority 4."""
        assert _group_goal_banner_pairs(doc_context) == _planner_goal_banner_pairs(
            doc_context
        )
        assert constraining_goals(doc_context) == frozenset()
        assert current_goal_priority(doc_context) == 1  # Vesna C0, Priority 1


class TestGroupingNeverMergesGoalsForEvaluation:
    """Review point 2: grouping goals onto one banner is a strategy-execution
    optimization only. The simulator still reports satisfaction per original
    Goal (§11) and feasibility scores every goal separately."""

    def _context(self, doc_account) -> PlannerContext:
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 1, 2), Goal("Tsaritsa", 3, 3)],
            banners=[VESNA, TSARITSA],
        )
        return PlannerContext(
            account=doc_account, roadmap=roadmap, current_version="7.0"
        )

    def test_each_grouped_goal_keeps_its_own_standing(self, doc_account):
        from optimizer import OutcomeOption, evaluate_candidate

        context = self._context(doc_account)
        (group,) = protected_groups(context)
        assert group.target_constellation == 3  # one execution target
        assert [goal.constellation for goal in group.goals] == [1, 3]

        candidate = evaluate_candidate(
            context, OutcomeOption("Vesna", 0, 1), budget=10, runs=500, seed=2
        )
        assert [standing.goal for standing in candidate.protected] == [
            Goal("Tsaritsa", 1, 2),
            Goal("Tsaritsa", 3, 3),
        ]
        # Standings are the simulator's per-goal probabilities (§11), not one
        # merged figure, and the C1 goal (a subset of C3) is never discarded.
        by_goal = {
            probability.goal: probability.probability
            for probability in candidate.result.goals
        }
        for standing in candidate.protected:
            assert standing.banner == TSARITSA
            assert standing.probability == by_goal[standing.goal]
            assert standing.meets_threshold == (
                standing.probability >= context.confidence
            )

    def test_uncapped_budget_never_limits_the_entry(self, doc_account):
        """The strategic budget is sized so the entry is never the limiting
        factor (review point 1); actual spending is bounded by the simulated
        available wishes via min(budget, available) (§12)."""
        context = self._context(doc_account)
        (group,) = protected_groups(context)
        assert group.uncapped_budget == doc_account.wishes + context.income_credit(
            group.banner.version
        )

        # With income before the protected banner the budget strictly exceeds
        # the current balance - which is precisely why the simulator's
        # min(budget, available), not the budget, must bound spending.
        with_income = PlannerContext(
            account=doc_account,
            roadmap=context.roadmap,
            current_version="7.0",
            income=IncomeForecast(
                versions=[VersionIncome("7.1", estimate=IncomeEstimate(0, 5, 10))]
            ),
        )
        (funded,) = protected_groups(with_income)
        assert funded.uncapped_budget == doc_account.wishes + 5
        assert funded.uncapped_budget > doc_account.wishes


class TestClassificationAgreement:
    """One definition of 'protected' across phases (review point 8): the
    optimizer's grouping must agree with planner.protection's outcomes."""

    @pytest.mark.parametrize(
        "wishes,confidence",
        [(40, 0.9), (100, 0.5), (0, 0.9)],
    )
    def test_group_goals_match_planner_protection(
        self, doc_account, doc_roadmap, wishes, confidence
    ):
        account = Account(
            current_pity=doc_account.current_pity,
            character_guarantee=doc_account.character_guarantee,
            owned_characters=doc_account.owned_characters,
            wishes=wishes,
        )
        context = PlannerContext(
            account=account,
            roadmap=doc_roadmap,
            current_version="7.0",
            confidence=confidence,
        )
        assert _group_goal_banner_pairs(context) == _planner_goal_banner_pairs(
            context
        )

    def test_agreement_on_a_larger_roadmap(self, doc_account):
        roadmap = Roadmap(
            goals=[
                Goal("A", 0, 2),
                Goal("B", 1, 1),
                Goal("B", 0, 3),
                Goal("Vesna", 0, 4),
            ],
            banners=[
                VESNA,
                Banner("A", "7.2", 1),
                Banner("B", "7.1", 1),
            ],
        )
        context = PlannerContext(
            account=doc_account, roadmap=roadmap, current_version="7.0"
        )
        assert _group_goal_banner_pairs(context) == _planner_goal_banner_pairs(
            context
        )


class TestCurrentGoalPriority:
    """The priority the current decision is anchored to (optimizer.protection
    §2, §9).

    `current_goal_priority` is the best-ranked ACTIVE goal on the current
    banner: the priority any future goal must beat to constrain the spend. It
    is None when the current banner has no ACTIVE goal - an unanchored decision
    is conservative and every protected goal constrains it.
    """

    def test_anchors_to_the_current_banner_s_active_goal(self, doc_context):
        assert current_goal_priority(doc_context) == 1

    def test_is_none_when_the_current_banner_has_no_active_goal(self, doc_account):
        roadmap = Roadmap(
            goals=[Goal("Tsaritsa", 0, 1)],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        assert current_goal_priority(context) is None

    def test_ignores_future_and_satisfied_goals(self, doc_account):
        roadmap = Roadmap(
            goals=[
                Goal("Vesna", 0, 2),
                Goal("Tsaritsa", 0, 1),
            ],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        assert current_goal_priority(context) == 2

    def test_returns_the_lowest_priority_when_multiple_active(self, doc_account):
        """When multiple goals on the current banner are actionable (e.g. C0 and
        C1 both needed, both actionable because C0 is already satisfied), the
        anchor is the best-ranked one - the lowest priority number."""
        roadmap = Roadmap(
            goals=[
                Goal("Vesna", 0, 2),  # priority 2, C0 actionable
                Goal("Vesna", 1, 1),  # priority 1, C1 actionable once C0 met
            ],
            banners=[VESNA, TSARITSA],
        )
        account = Account(
            current_pity=0,
            character_guarantee=False,
            owned_characters=Ownership({"Vesna": 0}),  # C0 already owned
            wishes=40,
        )
        context = PlannerContext(
            account=account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        # Both C0 (priority 2) and C1 (priority 1) are actionable now that
        # C0 is owned; the anchor is the best-ranked: priority 1.
        assert current_goal_priority(context) == 1


class TestConstrainingGoals:
    """Which protected goals may constrain the current decision (§2).

    A protected goal constrains only when its priority is higher (a lower
    number) than the current decision's own priority. A protected goal ranked
    below the current objective keeps its place in the strategy - it is still
    pursued and reported - but its probability can no longer force the current
    spend to zero.
    """

    def test_lower_priority_future_goal_does_not_constrain(
        self, doc_account, doc_roadmap
    ):
        """The reported bug scenario: current Priority 1 goal, future
        Priority 2 goal. The future goal is protected but does not gate the
        current decision."""
        roadmap = Roadmap(
            goals=[
                Goal("Vesna", 0, 1),
                Goal("Tsaritsa", 0, 2),
            ],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        constraining = constraining_goals(context)
        assert Goal("Tsaritsa", 0, 2) not in constraining

    def test_higher_priority_future_goal_does_constrain(
        self, doc_account, doc_roadmap
    ):
        """Inverse: future Priority 1 goal constrains a current Priority 2
        goal."""
        roadmap = Roadmap(
            goals=[
                Goal("Vesna", 0, 2),
                Goal("Tsaritsa", 0, 1),
            ],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        constraining = constraining_goals(context)
        assert Goal("Tsaritsa", 0, 1) in constraining

    def test_unanchored_decision_is_conservative(
        self, doc_account, doc_roadmap
    ):
        """When the current banner has no ACTIVE goal, every protected goal
        constrains (the conservative reading, §2)."""
        roadmap = Roadmap(
            goals=[Goal("Tsaritsa", 0, 1)],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        constraining = constraining_goals(context)
        assert Goal("Tsaritsa", 0, 1) in constraining

    def test_no_protected_goals_yields_empty(
        self, doc_account, doc_roadmap
    ):
        """With nothing left to protect, constraining is empty."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1)],
            banners=[VESNA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
        )
        assert constraining_goals(context) == frozenset()

