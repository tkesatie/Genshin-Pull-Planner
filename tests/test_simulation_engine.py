"""The Monte Carlo engine: account transitions, propagation, income timing,
spend caps and skipped banners (Design Document §11, §16, §18 Phase 4).

Deterministic-mechanics helpers make propagation exact:

    forced_mechanics(): every pull is a certain 5-star. With
    soft_pity_start=1 the first pull already carries a full soft-pity
    increment, so its rate is 1.0, and hard_pity=2 makes the next pull the
    hard-pity pull. Determinism comes from the guarantee chain:
    featured_rate=1.0 features every 5-star; featured_rate ~= 0 loses the
    first 50/50 and then wins the guaranteed one - which is exactly the
    pity/guarantee propagation under test (§11).

    slow_mechanics(): base rate ~= 0 up to hard pity, so "no 5-star ->
    pity +1" is deterministic until the hard-pity pull closes the cycle.
"""

import numpy as np
import pytest

from domain import (
    CHARACTER_EVENT_BANNER,
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
    WishMechanics,
)
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate, simulate_history
from simulation.engine import _simulate_scalar

VESNA = Banner("Vesna", "7.0", 1)
TSARITSA = Banner("Tsaritsa", "7.1", 1)
VODYNISTA = Banner("Vodynista", "7.2", 1)
ARIA = Banner("Aria", "7.0", 1)


def forced_mechanics(featured_rate: float = 1.0) -> WishMechanics:
    """Every pull is a certain 5-star; featured_rate drives the 50/50."""
    return WishMechanics(
        banner_type="forced",
        hard_pity=2,
        soft_pity_start=1,
        base_rate=0.5,
        soft_pity_increment=0.5,
        featured_rate=featured_rate,
    )


def slow_mechanics() -> WishMechanics:
    """No 5-star until hard pity: rate ~= 0 before the hard-pity pull."""
    return WishMechanics(
        banner_type="slow",
        hard_pity=3,
        soft_pity_start=2,
        base_rate=1e-9,
        soft_pity_increment=0.5,
        featured_rate=1.0,
    )


def single_banner_context(
    account: Account,
    mechanics: WishMechanics = CHARACTER_EVENT_BANNER,
    character: str = "Vesna",
    version: str = "7.0",
) -> PlannerContext:
    roadmap = Roadmap(
        goals=[Goal(character, 0, 1)],
        banners=[Banner(character, version, 1)],
    )
    return PlannerContext(
        account=account, roadmap=roadmap, current_version=version, mechanics=mechanics
    )




class TestVectorizedSimulation:
    def test_vectorized_simulate_matches_scalar_for_deterministic_banners(
        self, doc_account
    ):
        """The integrated vectorized path preserves the scalar state machine.

        The roadmap uses unique goal priorities (a Roadmap invariant) and
        covers two sequential banners, so pity/guarantee/ownership/wishes
        propagation between banners is exercised exactly.
        """
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            mechanics=forced_mechanics(),
        )
        plan = SpendPlan(
            entries=(
                PlannedSpend(VESNA, 0, 2),
                PlannedSpend(TSARITSA, 0, 2),
            )
        )

        vectorized = simulate(context, plan, runs=20, seed=17)
        scalar = _simulate_scalar(context, plan, runs=20, seed=17)

        assert vectorized == scalar


class TestBannerWalk:
    def test_processes_every_banner_from_the_current_one_on(self, doc_context):
        run = simulate_history(doc_context, SpendPlan(), np.random.default_rng(0))
        assert [b.banner for b in run.banner_results] == [VESNA, TSARITSA, VODYNISTA]
        assert run.account_after is run.banner_results[-1].account_after

    def test_banners_before_the_current_are_never_processed(self, doc_account):
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1)],
            banners=[Banner("Old", "6.9", 1), VESNA],
        )
        context = PlannerContext(
            account=doc_account, roadmap=roadmap, current_version="7.0"
        )
        run = simulate_history(context, SpendPlan(), np.random.default_rng(0))
        assert [b.banner for b in run.banner_results] == [VESNA]

    def test_skipped_banner_between_planned_banners_is_processed_not_stopped(self):
        """Absent plan entry = skip, not stop (§11): the skipped banner is
        processed chronologically, spends nothing, credits its version's
        income, carries pity/guarantee unchanged, and stays in the history."""
        mechanics = forced_mechanics(featured_rate=1e-9)
        roadmap = Roadmap(
            goals=[Goal("Vodynista", 0, 1)],
            banners=[VESNA, TSARITSA, VODYNISTA],
        )
        income = IncomeForecast(
            versions=[VersionIncome("7.1", estimate=IncomeEstimate(10, 10, 10))]
        )
        context = PlannerContext(
            account=Account(current_pity=1, wishes=10),
            roadmap=roadmap,
            current_version="7.0",
            income=income,
            mechanics=mechanics,
        )
        plan = SpendPlan(
            entries=(
                # One pull: loses the 50/50, leaving the guarantee on.
                PlannedSpend(VESNA, 0, 1),
                # Tsaritsa 7.1 deliberately absent: skipped.
                PlannedSpend(VODYNISTA, 0, 1),
            )
        )
        run = simulate_history(context, plan, np.random.default_rng(0))

        vesna, tsaritsa, vodynista = run.banner_results
        assert [b.banner for b in run.banner_results] == [VESNA, TSARITSA, VODYNISTA]

        assert vesna.wishes_spent == 1
        assert vesna.copies_obtained == 0
        assert vesna.account_after.character_guarantee is True
        assert vesna.account_after.current_pity == 0

        assert tsaritsa.target_constellation is None
        assert tsaritsa.wishes_spent == 0
        assert tsaritsa.income_credited == 10
        assert tsaritsa.account_after.character_guarantee is True  # carried
        assert tsaritsa.account_after.current_pity == 0  # carried

        # The carried guarantee wins despite the ~0 featured rate (§11).
        assert vodynista.copies_obtained == 1
        assert vodynista.copy_wishes == (1,)
        assert vodynista.account_after.character_guarantee is False

        assert run.account_after.wishes == 10 - 1 + 10 - 1
        (vodynista_outcome,) = run.goal_outcomes
        assert vodynista_outcome.satisfied is True
        assert vodynista_outcome.satisfied_after == VODYNISTA




class TestSpendCapsAndTargets:
    def test_budget_is_a_cap_not_a_commitment(self):
        """Target C2 from unowned = 3 copies; the budget cuts it at 2."""
        context = single_banner_context(
            Account(wishes=40), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 2, 2),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_needed == 3
        assert first.wishes_spent == 2
        assert first.copies_obtained == 2
        assert first.copy_wishes == (1, 2)
        assert first.target_met is False
        assert first.account_after.owned_constellation("Vesna") == 1
        assert first.account_after.wishes == 38

    def test_budget_beyond_available_wishes_spends_only_what_exists(self):
        context = single_banner_context(
            Account(wishes=5), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 4, 100),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.wishes_spent == 5
        assert first.copies_obtained == 5
        assert first.target_met is True
        assert first.account_after.wishes == 0

    def test_target_already_met_spends_nothing(self):
        """Owned C2, plan C1: a resulting constellation, never a copy count
        (§4.2) - copies_needed is 0, nothing is spent, the target is met."""
        context = single_banner_context(
            Account(owned_characters=Ownership({"Vesna": 2}), wishes=40),
            mechanics=forced_mechanics(),
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 1, 10),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_needed == 0
        assert first.wishes_spent == 0
        assert first.target_met is True
        assert first.account_after.owned_constellation("Vesna") == 2
        assert first.account_after.wishes == 40

    def test_multi_copy_target_pulls_copies_not_constellations(self):
        """C2 from NOT_OWNED is 3 copies through one code path (§10.4)."""
        context = single_banner_context(
            Account(wishes=40), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 2, 40),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_needed == 3
        assert first.copies_obtained == 3
        assert first.copy_wishes == (1, 2, 3)
        assert first.target_met is True
        assert first.account_after.owned_constellation("Vesna") == 2
        assert first.account_after.wishes == 37

    def test_current_banner_spends_only_account_wishes(self):
        """Current-version income (30) must not be available to the current
        banner: the 40-wish pool is the only spendable source (§16)."""
        income = IncomeForecast(
            versions=[VersionIncome("7.0", estimate=IncomeEstimate(20, 30, 40))]
        )
        context = PlannerContext(
            account=Account(wishes=40),
            roadmap=Roadmap(goals=[Goal("Vesna", 0, 1)], banners=[VESNA]),
            current_version="7.0",
            income=income,
            # Target C40 needs 41 copies, so no 40-wish pool can finish it;
            # the cap, rather than current-version income, must stop spending.
            mechanics=forced_mechanics(featured_rate=1e-9),
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 40, 40),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.income_credited == 0
        assert first.wishes_spent == 40
        assert first.account_after.wishes == 0
        assert first.target_met is False


class TestPityAndGuaranteePropagation:
    def test_featured_copy_resets_pity_and_guarantee(self):
        context = single_banner_context(
            Account(current_pity=1, character_guarantee=True, wishes=5),
            mechanics=forced_mechanics(),
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 5),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_obtained == 1
        assert first.copy_wishes == (1,)
        assert first.account_after.current_pity == 0
        assert first.account_after.character_guarantee is False

    def test_lost_fifty_fifty_sets_the_guarantee(self):
        """The 5-star happens (forced) but is not featured (rate ~ 0):
        pity resets, the guarantee switches on, no copy lands."""
        context = single_banner_context(
            Account(current_pity=1, wishes=1),
            mechanics=forced_mechanics(featured_rate=1e-9),
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 1),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_obtained == 0
        assert first.copy_wishes == ()
        assert first.target_met is False
        assert first.account_after.current_pity == 0
        assert first.account_after.character_guarantee is True

    def test_guarantee_carries_across_banners_and_overrides_the_fifty_fifty(self):
        """A lost 50/50 on banner 1 must make banner 2's 5-star featured
        even when its own featured_rate is ~0 (§11 propagation)."""
        mechanics = forced_mechanics(featured_rate=1e-9)
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
            banners=[VESNA, TSARITSA],
        )
        context = PlannerContext(
            account=Account(current_pity=1, wishes=4),
            roadmap=roadmap,
            current_version="7.0",
            mechanics=mechanics,
        )
        plan = SpendPlan(
            entries=(
                PlannedSpend(VESNA, 0, 1),      # the pull loses the 50/50
                PlannedSpend(TSARITSA, 0, 1),   # the guarantee wins it
            )
        )
        run = simulate_history(context, plan, np.random.default_rng(0))
        first, second = run.banner_results
        assert first.copies_obtained == 0
        assert first.account_after.character_guarantee is True
        assert second.copies_obtained == 1
        assert second.copy_wishes == (1,)
        assert second.account_after.character_guarantee is False
        assert second.account_after.owned_constellation("Tsaritsa") == 0

    def test_no_five_star_increments_pity(self):
        """No 5-star -> pity +1, guarantee unchanged (§11)."""
        context = single_banner_context(
            Account(current_pity=0, character_guarantee=True, wishes=5),
            mechanics=slow_mechanics(),
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 1),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_obtained == 0
        assert first.account_after.current_pity == 1
        assert first.account_after.character_guarantee is True

    def test_pity_carry_closes_the_cycle_at_hard_pity(self):
        """Pity 2 with hard_pity 3: the next pull is the hard-pity pull."""
        context = single_banner_context(
            Account(current_pity=2, wishes=5), mechanics=slow_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 1),))
        run = simulate_history(context, plan, np.random.default_rng(0))
        first = run.banner_results[0]
        assert first.copies_obtained == 1
        assert first.copy_wishes == (1,)
        assert first.account_after.current_pity == 0
        assert first.account_after.character_guarantee is False


class TestIncomeTiming:
    def test_income_is_credited_between_banners_not_before_the_current(
        self, doc_account, doc_roadmap, doc_income
    ):
        """The Phase 3/4 version-level approximation (§16): the current
        banner spends only account.wishes; all income for versions through
        V is credited by the first processed banner in V."""
        context = PlannerContext(
            account=doc_account,
            roadmap=doc_roadmap,
            current_version="7.0",
            income=doc_income,
        )
        run = simulate_history(context, SpendPlan(), np.random.default_rng(0))
        # Vesna (current): nothing. Tsaritsa 7.1: 7.0's 30 + 7.1's 30.
        # Vodynista 7.2: 7.2's 40.
        assert [b.income_credited for b in run.banner_results] == [0, 60, 40]
        assert [b.account_after.wishes for b in run.banner_results] == [40, 100, 140]
        assert run.account_after.wishes == 140

    def test_two_banners_in_one_version_credit_income_once(self):
        """Income for version V lands at the first processed banner in V;
        a second banner in V must not credit it again (§16)."""
        roadmap = Roadmap(
            goals=[],
            banners=[
                Banner("A", "7.0", 1),
                Banner("B", "7.0", 2),
                Banner("C", "7.1", 1),
            ],
        )
        income = IncomeForecast(
            versions=[
                VersionIncome("7.0", estimate=IncomeEstimate(30, 30, 30)),
                VersionIncome("7.1", estimate=IncomeEstimate(40, 40, 40)),
            ]
        )
        context = PlannerContext(
            account=Account(wishes=40),
            roadmap=roadmap,
            current_version="7.0",
            income=income,
        )
        run = simulate_history(context, SpendPlan(), np.random.default_rng(0))
        assert [b.income_credited for b in run.banner_results] == [0, 30, 40]
        assert run.account_after.wishes == 110

    def test_income_for_versions_before_the_current_is_never_credited(self):
        """Pre-current-version income is presumed banked in account.wishes
        (planner.context); crediting it again would double-count."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1)],
            banners=[VESNA, TSARITSA],
        )
        income = IncomeForecast(
            versions=[
                VersionIncome("6.9", estimate=IncomeEstimate(100, 100, 100)),
                VersionIncome("7.1", estimate=IncomeEstimate(30, 30, 30)),
            ]
        )
        context = PlannerContext(
            account=Account(wishes=40),
            roadmap=roadmap,
            current_version="7.0",
            income=income,
        )
        run = simulate_history(context, SpendPlan(), np.random.default_rng(0))
        assert [b.income_credited for b in run.banner_results] == [0, 30]
        assert run.account_after.wishes == 70


class TestRoadmapOutcomeTracking:
    def test_goal_outcomes_are_in_priority_order_with_pre_satisfied_goal(
        self, doc_context, doc_plan
    ):
        """Vodynista C0 is satisfied from the start (satisfied_after None);
        Vesna C2 can never be satisfied by a C0-target plan (§10.4)."""
        run = simulate_history(doc_context, doc_plan, np.random.default_rng(5))
        outcomes = run.goal_outcomes
        assert [o.goal for o in outcomes] == [
            Goal("Vesna", 0, 1),
            Goal("Vodynista", 0, 2),
            Goal("Vesna", 2, 3),
            Goal("Tsaritsa", 0, 4),
        ]
        assert outcomes[1].satisfied is True
        assert outcomes[1].satisfied_after is None
        assert outcomes[2].satisfied is False

        vesna_c0, first_result = outcomes[0], run.banner_results[0]
        assert vesna_c0.satisfied == (first_result.copies_obtained >= 1)
        if vesna_c0.satisfied:
            assert vesna_c0.satisfied_after == VESNA


class TestIsolationAndDeterminism:
    def test_simulation_never_mutates_domain_state(self, doc_context, doc_plan):
        account = doc_context.account
        pity = account.current_pity
        guarantee = account.character_guarantee
        wishes = account.wishes
        owned_snapshot = dict(account.owned_characters.characters)
        goals_snapshot = list(doc_context.roadmap.goals)
        banners_snapshot = list(doc_context.roadmap.banners)

        simulate(doc_context, doc_plan, runs=20, seed=3)

        assert account.current_pity == pity
        assert account.character_guarantee == guarantee
        assert account.wishes == wishes
        assert dict(account.owned_characters.characters) == owned_snapshot
        assert doc_context.roadmap.goals == goals_snapshot
        assert doc_context.roadmap.banners == banners_snapshot

    def test_repeated_simulations_are_isolated_and_reproducible(
        self, doc_context, doc_plan
    ):
        """Two simulations on the same Account/Roadmap objects produce the
        same result with the same seed, even after the first completed."""
        first = simulate(doc_context, doc_plan, runs=40, seed=11)
        second = simulate(doc_context, doc_plan, runs=40, seed=11)
        third = simulate(doc_context, doc_plan, runs=40, seed=11)
        assert first == second == third

    def test_simulate_history_is_deterministic_given_the_rng_state(
        self, doc_context, doc_plan
    ):
        a = simulate_history(doc_context, doc_plan, np.random.default_rng(7))
        b = simulate_history(doc_context, doc_plan, np.random.default_rng(7))
        assert a == b

    def test_runs_must_be_positive(self, doc_context, doc_plan):
        with pytest.raises(ValueError, match="runs must be >= 1"):
            simulate(doc_context, doc_plan, runs=0, seed=0)

    def test_seed_none_is_accepted(self, doc_context, doc_plan):
        result = simulate(doc_context, doc_plan, runs=5, seed=None)
        assert result.runs == 5
        assert result.seed is None

    def test_simulation_requires_a_current_banner_match(self):
        roadmap = Roadmap(goals=[], banners=[VESNA])
        context = PlannerContext(
            account=Account(wishes=10), roadmap=roadmap, current_version="7.5"
        )
        with pytest.raises(ValueError, match="no roadmap banner"):
            simulate(context, SpendPlan(), runs=5, seed=0)


class TestVectorizedEdgeCases:
    """The §13 edge-case battery through both engines: with deterministic
    mechanics the vectorized `simulate()` must reproduce `_simulate_scalar`
    exactly, scenario by scenario."""

    def assert_engines_agree(self, context, plan, runs=25, seed=5):
        vectorized = simulate(context, plan, runs=runs, seed=seed)
        scalar = _simulate_scalar(context, plan, runs=runs, seed=seed)
        assert vectorized == scalar
        return vectorized

    def test_zero_wishes(self):
        context = single_banner_context(
            Account(wishes=0), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 10),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.wishes_spent == 0
        assert first.copies_obtained == 0
        assert first.target_met is False

    def test_zero_budget(self):
        context = single_banner_context(
            Account(wishes=10), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 0),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.budget == 0
        assert first.wishes_spent == 0

    def test_target_already_owned(self):
        account = Account(owned_characters=Ownership({"Vesna": 0}), wishes=10)
        context = single_banner_context(account, mechanics=forced_mechanics())
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 10),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.copies_needed == 0
        assert first.wishes_spent == 0
        assert first.target_met is True

    def test_target_requiring_multiple_copies(self):
        context = single_banner_context(
            Account(wishes=10), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 2, 10),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.copies_needed == 3
        assert first.wishes_spent == 3
        assert first.copy_wishes == (1, 2, 3)
        assert first.account_after.owned_constellation("Vesna") == 2

    def test_insufficient_wishes(self):
        context = single_banner_context(
            Account(wishes=2), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 2, 10),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.wishes_spent == 2
        assert first.target_met is False

    def test_budget_greater_than_available_wishes(self):
        context = single_banner_context(
            Account(wishes=5), mechanics=forced_mechanics()
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 100),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.wishes_spent == 1

    def test_starting_pity_guarantee_and_radiance_propagate(self):
        """Pity 1 / guarantee off: the hard-pity pull loses the 50/50, the
        carried guarantee wins, and the counter (0 -> 1 -> 2) carries while
        the guarantee chain forces every featured outcome.

        Starting at radiance 0 keeps the whole chain deterministic: a loss
        from radiance 1 or the guaranteed win never leaves the 6/11
        radiance-2 state exposed to a non-guaranteed pull (engines fed
        different RNG streams legitimately diverge there - see the
        stochastic comparison)."""
        mechanics = forced_mechanics(featured_rate=1e-9)
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1)], banners=[VESNA, TSARITSA]
        )
        context = PlannerContext(
            account=Account(current_pity=1, wishes=4),
            roadmap=roadmap,
            current_version="7.0",
            mechanics=mechanics,
        )
        plan = SpendPlan(
            entries=(PlannedSpend(VESNA, 0, 2), PlannedSpend(TSARITSA, 0, 2))
        )
        result = self.assert_engines_agree(context, plan)
        vesna, tsaritsa = result.histories[0].banner_results
        # Vesna: hard-pity pull loses the 50/50, the guarantee wins pull 2.
        assert vesna.wishes_spent == 2
        assert vesna.copies_obtained == 1
        assert vesna.copy_wishes == (2,)
        assert vesna.account_after.character_guarantee is False
        assert vesna.account_after.capturing_radiance_counter == 1
        # Tsaritsa: non-guaranteed at radiance 1 loses (rate ~0), then the
        # guarantee wins pull 2; the counter carries 1 -> 2 -> 2.
        assert tsaritsa.wishes_spent == 2
        assert tsaritsa.copies_obtained == 1
        assert tsaritsa.account_after.character_guarantee is False
        assert tsaritsa.account_after.capturing_radiance_counter == 2

    def test_starting_guarantee_wins_and_preserves_radiance(self):
        """A starting guarantee forces the featured copy on the first wish
        and never touches the Capturing Radiance counter."""
        mechanics = forced_mechanics(featured_rate=1e-9)
        context = single_banner_context(
            Account(
                current_pity=1,
                character_guarantee=True,
                wishes=4,
                capturing_radiance_counter=2,
            ),
            mechanics=mechanics,
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 2),))
        result = self.assert_engines_agree(context, plan)
        first = result.histories[0].banner_results[0]
        assert first.wishes_spent == 1
        assert first.copies_obtained == 1
        assert first.account_after.character_guarantee is False
        assert first.account_after.capturing_radiance_counter == 2

    def test_pity_carrying_across_banners(self):
        """Banner 1 spends its budget on the hard-pity pull; banner 2 starts
        from the reset pity and (at rate ~0) gains exactly one pity."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1)], banners=[VESNA, TSARITSA]
        )
        context = PlannerContext(
            account=Account(current_pity=2, wishes=4),
            roadmap=roadmap,
            current_version="7.0",
            mechanics=slow_mechanics(),
        )
        plan = SpendPlan(
            entries=(PlannedSpend(VESNA, 0, 1), PlannedSpend(TSARITSA, 0, 1))
        )
        result = self.assert_engines_agree(context, plan)
        vesna, tsaritsa = result.histories[0].banner_results
        assert vesna.copies_obtained == 1
        assert vesna.account_after.current_pity == 0
        assert tsaritsa.copies_obtained == 0
        assert tsaritsa.account_after.current_pity == 1

    def test_guarantee_and_radiance_carry_through_skipped_banner(self):
        """A lost 50/50's guarantee (and the incremented counter) survive a
        skipped banner and win on the next planned one."""
        mechanics = forced_mechanics(featured_rate=1e-9)
        roadmap = Roadmap(
            goals=[Goal("Vodynista", 0, 1)],
            banners=[VESNA, TSARITSA, VODYNISTA],
        )
        context = PlannerContext(
            account=Account(current_pity=1, wishes=10),
            roadmap=roadmap,
            current_version="7.0",
            mechanics=mechanics,
        )
        plan = SpendPlan(
            entries=(PlannedSpend(VESNA, 0, 1), PlannedSpend(VODYNISTA, 0, 1))
        )
        result = self.assert_engines_agree(context, plan)
        vesna, tsaritsa, vodynista = result.histories[0].banner_results
        assert vesna.copies_obtained == 0
        assert vesna.account_after.character_guarantee is True
        assert vesna.account_after.capturing_radiance_counter == 1
        assert tsaritsa.target_constellation is None
        assert tsaritsa.wishes_spent == 0
        assert tsaritsa.account_after.character_guarantee is True
        assert vodynista.copies_obtained == 1
        assert vodynista.account_after.character_guarantee is False
        assert vodynista.account_after.capturing_radiance_counter == 1

    def test_income_between_banners_and_two_banners_in_one_version(
        self, doc_account
    ):
        """Income lands at the first processed banner beyond the current
        (version, phase): Aria (7.0 p2) credits 7.0's forecast, Tsaritsa
        (7.1 p1) credits 7.1's - and no banner is credited twice."""
        later_phase_aria = Banner("Aria", "7.0", 2)
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
            banners=[VESNA, later_phase_aria, TSARITSA],
        )
        income = IncomeForecast(
            versions=[
                VersionIncome("7.0", estimate=IncomeEstimate(10, 10, 10)),
                VersionIncome("7.1", estimate=IncomeEstimate(20, 20, 20)),
            ]
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            income=income,
            mechanics=forced_mechanics(),
        )
        plan = SpendPlan(
            entries=(PlannedSpend(VESNA, 0, 40), PlannedSpend(TSARITSA, 0, 40))
        )
        result = self.assert_engines_agree(context, plan)
        vesna, aria, tsaritsa = result.histories[0].banner_results
        assert [b.income_credited for b in (vesna, aria, tsaritsa)] == [0, 10, 20]

    def test_shared_current_phase_budget(self):
        """Two banners in the current (version, phase) share one budget;
        the vectorized per-history accounting must match the scalar."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Aria", 1, 2)],
            banners=[VESNA, ARIA],
        )
        context = PlannerContext(
            account=Account(wishes=10),
            roadmap=roadmap,
            current_version="7.0",
            current_phase=1,
            mechanics=forced_mechanics(),
        )
        plan = SpendPlan(
            entries=(PlannedSpend(VESNA, 0, 5), PlannedSpend(ARIA, 1, 5)),
            shared_current_phase_budget=3,
        )
        result = self.assert_engines_agree(context, plan)
        vesna, aria = result.histories[0].banner_results
        assert vesna.wishes_spent == 1
        assert aria.wishes_spent == 2

    def test_goals_already_satisfied_and_satisfied_later(self, doc_account):
        """Vodynista C0 is satisfied from the start (satisfied_after None);
        Vesna C0 becomes satisfied on its banner."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Vodynista", 0, 2)],
            banners=[VESNA, VODYNISTA],
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            mechanics=forced_mechanics(),
        )
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 2),))
        result = self.assert_engines_agree(context, plan)
        outcomes = result.histories[0].goal_outcomes
        assert outcomes[0].satisfied is True
        assert outcomes[0].satisfied_after == VESNA
        assert outcomes[1].satisfied is True
        assert outcomes[1].satisfied_after is None

    def test_unwon_banner_characters_are_not_added_to_ownership(
        self, doc_account
    ):
        """Ownership keys must match the scalar engine exactly (§4.2): a
        character the plan or goals mention but the run never wins is not
        auto-added with NOT_OWNED; a character the run wins is added."""
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1)], banners=[VESNA, TSARITSA]
        )
        context = PlannerContext(
            account=doc_account,
            roadmap=roadmap,
            current_version="7.0",
            mechanics=forced_mechanics(),
        )
        # Budget 0: Vesna is never pulled, so ownership stays as it started.
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 0),))
        result = self.assert_engines_agree(context, plan)
        for run in result.histories:
            assert run.account_after.owned_characters.characters == {
                "Vodynista": 0
            }

        # Budget 1 with forced mechanics: Vesna is won and added, Tsaritsa
        # (a skipped banner character) is not.
        plan = SpendPlan(entries=(PlannedSpend(VESNA, 0, 1),))
        result = self.assert_engines_agree(context, plan)
        for run in result.histories:
            assert run.account_after.owned_characters.characters == {
                "Vodynista": 0,
                "Vesna": 0,
            }


class TestVectorizedStochasticComparison:
    def test_aggregates_are_statistically_consistent_with_the_scalar_oracle(
        self, doc_account, doc_roadmap, doc_income
    ):
        """Genuinely stochastic mechanics: aggregate quantities from the
        vectorized engine must be statistically consistent with the scalar
        oracle. Seeded histories are NOT expected to match one-for-one
        (the implementations consume the RNG differently)."""
        context = PlannerContext(
            account=doc_account,
            roadmap=doc_roadmap,
            current_version="7.0",
            income=doc_income,
        )
        plan = SpendPlan(
            entries=(
                PlannedSpend(VESNA, 0, 60),
                PlannedSpend(TSARITSA, 0, 60),
            )
        )
        runs = 4_000
        vectorized = simulate(context, plan, runs=runs, seed=2024)
        scalar = _simulate_scalar(context, plan, runs=runs, seed=999)

        # Goal satisfaction probabilities.
        for v_goal, s_goal in zip(vectorized.goals, scalar.goals):
            assert v_goal.goal == s_goal.goal
            assert abs(v_goal.probability - s_goal.probability) < 0.03

        # Per-banner spending and success aggregates.
        for v_banner, s_banner in zip(vectorized.banners, scalar.banners):
            assert v_banner.banner == s_banner.banner
            assert (
                abs(v_banner.mean_wishes_spent - s_banner.mean_wishes_spent)
                < 1.5
            )
            assert (
                abs(
                    v_banner.target_met_probability
                    - s_banner.target_met_probability
                )
                < 0.03
            )
            assert (
                abs(
                    v_banner.mean_copies_obtained
                    - s_banner.mean_copies_obtained
                )
                < 0.05
            )

        # Roadmap-wide and end-of-history aggregates.
        assert (
            abs(vectorized.all_goals_probability - scalar.all_goals_probability)
            < 0.03
        )
        assert abs(vectorized.final_wishes_mean - scalar.final_wishes_mean) < 1.5

