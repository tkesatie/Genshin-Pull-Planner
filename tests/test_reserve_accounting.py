"""Reserve-accounting validation (Design Document §13 step 5, §14, §10.4).

Targets the punch-list items:

    * future income
    * 90% Skirk C2 protection
    * Vesna C2 -> remaining resources scenarios
    * post-pull pity/guarantee/CR state feeding the simulation-based
      protection layer (optimizer), as distinct from the Phase 3
      analytical approximation (planner.protection) these tests mostly
      exercise

This suite pins down a real bug found while writing it: `planner.protection
.protected_goal_outcomes` (and therefore `planner.safe_spend.safe_spend`
and `planner.spend_table.spend_table`) computed every protected goal's
reserve as a SINGLE-COPY requirement, regardless of how many copies the
goal actually needed. A goal like Skirk C2 from an unowned account (3
copies) was reported as "100% confidence" protected at a budget whose true
probability of completing all 3 copies was ~58%. See
test_multicopy_reserve_matches_exact_reference and
test_regression_previously_overstated_confidence_now_corrected below for
the direct before/after evidence.
"""

import numpy as np
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
from domain.mechanics import CHARACTER_EVENT_BANNER
from planner.context import PlannerContext
from planner.protection import protected_goal_outcomes
from planner.safe_spend import safe_spend
from probability import (
    cumulative_probability,
    multi_copy_cumulative_probability,
    multi_copy_wishes_for_confidence,
)

MECHANICS = CHARACTER_EVENT_BANNER


def _context(
    *,
    wishes: int,
    goals: list[Goal],
    banners: list[Banner],
    owned: dict | None = None,
    confidence: float = 0.9,
    current_version: str = "7.0",
    current_phase: int = 1,
    income: IncomeForecast | None = None,
) -> PlannerContext:
    account = Account(
        current_pity=0,
        character_guarantee=False,
        owned_characters=Ownership(owned or {}),
        wishes=wishes,
    )
    roadmap = Roadmap(goals=goals, banners=banners)
    return PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version=current_version,
        current_phase=current_phase,
        confidence=confidence,
        mechanics=MECHANICS,
        income=income,
    )


# ---------------------------------------------------------------------------
# Multi-copy reserve math itself: cross-check against the same exact
# convolution reference used to validate the simulator.
# ---------------------------------------------------------------------------

def _reference_convolved_cdf(copies: int, max_wishes: int) -> np.ndarray:
    cdf = cumulative_probability(max_wishes, 0, False, MECHANICS)
    pmf = np.diff(cdf, prepend=0.0)
    total = pmf.copy()
    for _ in range(copies - 1):
        total = np.convolve(total, pmf)[: max_wishes + 1]
    return np.cumsum(total)[: max_wishes + 1]


@pytest.mark.parametrize("copies,wishes", [(1, 90), (2, 180), (3, 270), (3, 450)])
def test_multicopy_cumulative_probability_matches_reference(copies, wishes):
    reference = _reference_convolved_cdf(copies, wishes)
    actual = multi_copy_cumulative_probability(wishes, copies, 0, False, MECHANICS)
    assert actual[wishes] == pytest.approx(reference[wishes], abs=1e-9)


def test_multicopy_wishes_for_confidence_round_trips():
    """The wishes count `multi_copy_wishes_for_confidence` returns must
    itself clear the requested confidence, and one fewer wish must not.
    """
    for copies in (1, 2, 3):
        needed = multi_copy_wishes_for_confidence(0.90, copies, 0, False, MECHANICS)
        curve = multi_copy_cumulative_probability(needed, copies, 0, False, MECHANICS)
        assert curve[needed] >= 0.90
        if needed > 0:
            shorter = multi_copy_cumulative_probability(needed - 1, copies, 0, False, MECHANICS)
            assert shorter[needed - 1] < 0.90


# ---------------------------------------------------------------------------
# 90% Skirk C2 protection - the exact scenario that surfaced the bug.
# ---------------------------------------------------------------------------

def test_skirk_c2_protection_matches_exact_multicopy_probability():
    """Skirk C2 from unowned (3 copies) as a protected future goal behind
    a current Vesna banner. The reported confidence/required_wishes must
    match the exact 3-copy convolution, not the 1-copy curve.
    """
    context = _context(
        wishes=300,
        goals=[Goal("Vesna", 0, 1), Goal("Skirk", 2, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Skirk", "7.1", 1)],
    )
    outcomes = protected_goal_outcomes(context, spent=0)
    assert len(outcomes) == 1
    skirk = outcomes[0]
    assert skirk.goal.character == "Skirk"

    reference = float(_reference_convolved_cdf(3, 300)[300])
    assert skirk.confidence == pytest.approx(reference, abs=1e-9)
    # At 300 wishes, true 3-copy confidence is well under 90% - must not
    # be reported as protected.
    assert reference < 0.90
    assert skirk.meets_threshold is False

    required_reference = multi_copy_wishes_for_confidence(0.90, 3, 0, False, MECHANICS)
    assert skirk.required_wishes == required_reference
    assert required_reference > 300  # confirms 300 wishes is genuinely short


def test_skirk_c2_protection_meets_threshold_once_funded():
    """The same Skirk C2 goal, given enough wishes to actually clear 90%."""
    required = multi_copy_wishes_for_confidence(0.90, 3, 0, False, MECHANICS)
    context = _context(
        wishes=required,
        goals=[Goal("Vesna", 0, 1), Goal("Skirk", 2, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Skirk", "7.1", 1)],
    )
    outcomes = protected_goal_outcomes(context, spent=0)
    skirk = outcomes[0]
    assert skirk.confidence >= 0.90
    assert skirk.meets_threshold is True


# ---------------------------------------------------------------------------
# Vesna C2 -> remaining resources at different ownership states.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "owned_constellation,expected_copies",
    [(-1, 3), (0, 2), (1, 1)],
)
def test_vesna_c2_reserve_scales_with_copies_needed(owned_constellation, expected_copies):
    """The reserve for a Vesna C2 protected goal must reflect exactly how
    many copies remain given current ownership - not a fixed value.
    """
    owned = {} if owned_constellation < 0 else {"Vesna": owned_constellation}
    context = _context(
        wishes=500,
        goals=[Goal("Tsaritsa", 0, 1), Goal("Vesna", 2, 2)],
        banners=[Banner("Tsaritsa", "7.0", 1), Banner("Vesna", "7.1", 1)],
        owned=owned,
    )
    outcomes = protected_goal_outcomes(context, spent=0)
    vesna = next(o for o in outcomes if o.goal.character == "Vesna")

    expected_required = multi_copy_wishes_for_confidence(0.90, expected_copies, 0, False, MECHANICS)
    assert vesna.required_wishes == expected_required


def test_vesna_c2_remaining_resources_after_reserve_consumed():
    """The remaining budget for a goal AFTER an earlier protected goal's
    reserve must be its own wishes minus that earlier goal's own (correct,
    multi-copy-aware) reserve - not a shared/uniform amount.
    """
    context = _context(
        wishes=600,
        goals=[Goal("Tsaritsa", 0, 1), Goal("Vesna", 2, 2)],  # Vesna unowned -> 3 copies
        # "Filler" is the current banner (matches current_version/phase, no
        # goal attached to it) so BOTH Tsaritsa and Vesna are genuinely
        # future protected goals rather than one of them being "current
        # banner business" and excluded.
        banners=[
            Banner("Filler", "7.0", 1),
            Banner("Tsaritsa", "7.1", 1),
            Banner("Vesna", "7.2", 1),
        ],
    )
    outcomes = protected_goal_outcomes(context, spent=0)
    assert len(outcomes) == 2
    tsaritsa, vesna = outcomes[0], outcomes[1]
    assert tsaritsa.goal.character == "Tsaritsa"
    assert vesna.goal.character == "Vesna"

    tsaritsa_required = multi_copy_wishes_for_confidence(0.90, 1, 0, False, MECHANICS)
    assert tsaritsa.required_wishes == tsaritsa_required
    # Vesna's budget_at_banner is whatever remains after Tsaritsa's own
    # (single-copy) reserve is fully consumed - not after some uniform
    # per-goal amount.
    assert vesna.budget_at_banner == max(0, 600 - tsaritsa_required)


# ---------------------------------------------------------------------------
# Future income folded into a multi-copy reserve.
# ---------------------------------------------------------------------------

def test_future_income_reduces_multicopy_deficit_at_the_right_banner():
    income = IncomeForecast(
        versions=[
            # Available by the time Skirk's banner (7.1) arrives, per
            # income_available_before's "later slot" rule.
            VersionIncome(version="7.1", estimate=IncomeEstimate(low=90, expected=90, high=90))
        ]
    )
    without_income = _context(
        wishes=300,
        goals=[Goal("Vesna", 0, 1), Goal("Skirk", 2, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Skirk", "7.1", 1)],
    )
    with_income = _context(
        wishes=300,
        goals=[Goal("Vesna", 0, 1), Goal("Skirk", 2, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Skirk", "7.1", 1)],
        income=income,
    )

    skirk_without = protected_goal_outcomes(without_income, spent=0)[0]
    skirk_with = protected_goal_outcomes(with_income, spent=0)[0]

    # Same required reserve (income doesn't change what's needed)...
    assert skirk_with.required_wishes == skirk_without.required_wishes
    # ...but more budget actually available when it arrives, so strictly
    # higher confidence.
    assert skirk_with.budget_at_banner == skirk_without.budget_at_banner + 90
    assert skirk_with.confidence > skirk_without.confidence

    reference_with = float(_reference_convolved_cdf(3, skirk_with.budget_at_banner)[skirk_with.budget_at_banner])
    assert skirk_with.confidence == pytest.approx(reference_with, abs=1e-9)


def test_current_version_income_not_credited_to_current_banner_reserve():
    """Income timing must follow `income_available_before` exactly: a
    forecast for the CURRENT version is not available at the current
    (version, phase) position, but IS available to a strictly later
    position - even one in the same version (matches
    `PlannerContext.income_available_before` / simulation/engine.py).
    """
    income = IncomeForecast(
        versions=[VersionIncome(version="7.0", estimate=IncomeEstimate(low=90, expected=90, high=90))]
    )
    context = _context(
        wishes=100,
        goals=[Goal("Tsaritsa", 0, 1)],
        banners=[Banner("Filler", "7.0", 1), Banner("Tsaritsa", "7.0", 2)],
        income=income,
    )
    tsaritsa = protected_goal_outcomes(context, spent=0)[0]
    # Tsaritsa's banner (7.0 phase 2) is strictly after "now" (7.0 phase
    # 1), so the 7.0 forecast HAS arrived by then - this is correct, not
    # a bug (contrast with the "same slot as current" case below).
    assert tsaritsa.budget_at_banner == 190


def test_income_never_negative_and_never_credited_before_current_position():
    """A protected goal sharing the *current* position isn't possible (it
    would be current-banner business, not protection) - but
    income_available_before itself must still return 0 exactly at the
    current position, which protected_goal_outcomes relies on to avoid
    ever crediting income the account doesn't have yet.
    """
    income = IncomeForecast(
        versions=[VersionIncome(version="7.0", estimate=IncomeEstimate(low=90, expected=90, high=90))]
    )
    context = _context(
        wishes=100,
        goals=[Goal("Tsaritsa", 0, 1)],
        banners=[Banner("Filler", "7.0", 1), Banner("Tsaritsa", "7.0", 1)],
        income=income,
    )
    assert context.income_available_before("7.0", 1) == 0


# ---------------------------------------------------------------------------
# safe_spend: reserves must SUM per-goal, not multiply a uniform value.
# ---------------------------------------------------------------------------

def test_safe_spend_sums_mixed_single_and_multicopy_reserves():
    """A single-copy goal followed by a multi-copy goal: safe_spend must
    reserve (goal1.required + goal2.required), not
    2 * (either goal's own required) - the bug this module's fix removed.
    """
    context = _context(
        wishes=1000,
        goals=[Goal("Tsaritsa", 0, 1), Goal("Skirk", 2, 2)],  # 1 copy, then 3 copies
        banners=[
            Banner("Filler", "7.0", 1),
            Banner("Tsaritsa", "7.1", 1),
            Banner("Skirk", "7.2", 1),
        ],
    )
    outcomes = protected_goal_outcomes(context, spent=0)
    assert len(outcomes) == 2
    tsaritsa_required = outcomes[0].required_wishes
    skirk_required = outcomes[1].required_wishes
    assert tsaritsa_required != skirk_required  # sanity: genuinely different

    result = safe_spend(context)
    expected = max(0, 1000 - (tsaritsa_required + skirk_required))
    assert result == expected
    # The old (buggy) formula would have computed a deficit of
    # 2 * skirk_required or 2 * tsaritsa_required instead of their sum;
    # make sure we're not accidentally matching either.
    wrong_uniform_low = max(0, 1000 - 2 * min(tsaritsa_required, skirk_required))
    wrong_uniform_high = max(0, 1000 - 2 * max(tsaritsa_required, skirk_required))
    if wrong_uniform_low != expected:
        assert result != wrong_uniform_low
    if wrong_uniform_high != expected:
        assert result != wrong_uniform_high


def test_safe_spend_zero_when_multicopy_goal_underfunded():
    """Directly pins the Skirk C2 / 300-wishes scenario at the safe_spend
    level: nothing is safe to spend on the current banner once the true
    multi-copy reserve is accounted for.
    """
    context = _context(
        wishes=300,
        goals=[Goal("Vesna", 0, 1), Goal("Skirk", 2, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Skirk", "7.1", 1)],
    )
    assert safe_spend(context) == 0


# ---------------------------------------------------------------------------
# Regression: pin the exact before/after gap that motivated this fix.
# ---------------------------------------------------------------------------

def test_regression_previously_overstated_confidence_now_corrected():
    """Before the fix, `protected_goal_outcomes` reported confidence=1.0
    and meets_threshold=True for Skirk C2 (3 copies, unowned) at 300
    wishes and 90% confidence - because it evaluated the SINGLE-copy
    curve, which is indeed ~1.0 at 300 wishes. The true 3-copy probability
    at 300 wishes is ~58%. This test locks in the corrected numbers so a
    future change can't silently reintroduce the single-copy shortcut.
    """
    context = _context(
        wishes=300,
        goals=[Goal("Vesna", 0, 1), Goal("Skirk", 2, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Skirk", "7.1", 1)],
    )
    skirk = protected_goal_outcomes(context, spent=0)[0]

    # The old, wrong answer this used to give:
    old_wrong_confidence = float(cumulative_probability(300, 0, False, MECHANICS)[300])
    assert old_wrong_confidence == pytest.approx(1.0, abs=1e-6)

    # The corrected answer:
    assert skirk.confidence == pytest.approx(0.5840518258672598, abs=1e-6)
    assert skirk.confidence < old_wrong_confidence - 0.3  # not a rounding difference
    assert skirk.meets_threshold is False


# ---------------------------------------------------------------------------
# Post-pull pity/guarantee/CR state reaching the SIMULATION-based
# protection layer (optimizer), as distinct from the Phase 3 approximation
# above. The Phase 3 approximation deliberately assumes a fresh pity-0/
# no-guarantee state for every protected goal (planner.protection module
# docstring) - that's by design, not something to validate against a
# post-pull state. It's the Phase 5 simulator that is supposed to carry
# pity/guarantee/Capturing Radiance forward, and that carry must actually
# reach a protected goal's reported probability, not just the current
# banner's own outcome (already covered in test_correctness_validation.py).
# ---------------------------------------------------------------------------

def test_post_pull_guarantee_carries_into_protected_goal_probability():
    """The pull-mechanics guarantee is a single account-wide pool shared
    across character banners (real character-event-wish mechanics, and
    how simulation/engine.py models it): a guarantee active right after a
    lost 50/50 must measurably raise a DIFFERENT character's protected
    probability on the very next banner processed, not just the current
    one.
    """
    from domain.mechanics import WishMechanics
    from optimizer.evaluation import evaluate_candidate
    from optimizer.outcomes import OutcomeOption

    # A cheap, fast-converging mechanics profile keeps this test quick
    # while still exercising the same guarantee-carry logic.
    fast_mechanics = WishMechanics(
        banner_type="test", hard_pity=20, soft_pity_start=16,
        base_rate=0.05, soft_pity_increment=0.1, featured_rate=0.5,
    )

    def build(guarantee: bool) -> PlannerContext:
        account = Account(current_pity=0, character_guarantee=guarantee, wishes=20)
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
            banners=[Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)],
        )
        return PlannerContext(
            account=account, roadmap=roadmap, current_version="7.0", current_phase=1,
            confidence=0.5, mechanics=fast_mechanics,
        )

    banner = Banner("Vesna", "7.0", 1)
    outcome = OutcomeOption("Vesna", 0, 1)

    no_guarantee = evaluate_candidate(build(False), outcome, 0, banner=banner, runs=4000, seed=1)
    with_guarantee = evaluate_candidate(build(True), outcome, 0, banner=banner, runs=4000, seed=1)

    tsaritsa_prob_no_guarantee = next(
        s.probability for s in no_guarantee.protected if s.goal.character == "Tsaritsa"
    )
    tsaritsa_prob_with_guarantee = next(
        s.probability for s in with_guarantee.protected if s.goal.character == "Tsaritsa"
    )
    # Spending 0 on Vesna means all 20 wishes carry to Tsaritsa's banner in
    # both cases; the only difference is the carried guarantee, which must
    # make Tsaritsa's protected probability measurably higher.
    assert tsaritsa_prob_with_guarantee > tsaritsa_prob_no_guarantee + 0.1


def test_post_pull_pity_reduces_wishes_needed_for_protected_goal():
    """Elevated pity carried out of the current banner (a near-miss that
    didn't produce a 5-star) must also measurably help - not just a full
    guarantee - since the very next pull starts closer to (soft) pity.
    """
    from domain.mechanics import WishMechanics
    from optimizer.evaluation import evaluate_candidate
    from optimizer.outcomes import OutcomeOption

    fast_mechanics = WishMechanics(
        banner_type="test", hard_pity=20, soft_pity_start=10,
        base_rate=0.02, soft_pity_increment=0.15, featured_rate=0.5,
    )

    def build(pity: int) -> PlannerContext:
        account = Account(current_pity=pity, character_guarantee=False, wishes=15)
        roadmap = Roadmap(
            goals=[Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)],
            banners=[Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)],
        )
        return PlannerContext(
            account=account, roadmap=roadmap, current_version="7.0", current_phase=1,
            confidence=0.5, mechanics=fast_mechanics,
        )

    banner = Banner("Vesna", "7.0", 1)
    outcome = OutcomeOption("Vesna", 0, 1)

    low_pity = evaluate_candidate(build(0), outcome, 0, banner=banner, runs=4000, seed=2)
    high_pity = evaluate_candidate(build(15), outcome, 0, banner=banner, runs=4000, seed=2)

    # NOTE: with pity=15 and wishes=15, the current banner itself can
    # reach pity 15 -> hard pity territory does not apply here (hard_pity
    # 20), but Vesna's own probability at cap 0 is irrelevant (nothing is
    # spent there); what matters is the *carried* pity feeding Tsaritsa's
    # simulated pulls a few wishes closer to its own soft pity.
    tsaritsa_low = next(s.probability for s in low_pity.protected if s.goal.character == "Tsaritsa")
    tsaritsa_high = next(s.probability for s in high_pity.protected if s.goal.character == "Tsaritsa")
    assert tsaritsa_high > tsaritsa_low
