"""Regression: one extra unused wish must not change the same candidate (§13, §14).

Manual-testing report (current banner Navia 7.0 phase 2, future banner
Arlecchino 7.1 phase 1, Navia C0 at Priority 2, Arlecchino C0 at Priority 1,
pity 0, character guarantee on, no income, confidence 90%, seed 0,
2,000 runs, normal mechanics, no preferences):

    200 wishes -> "pursue Navia C0, up to 75 wishes", Navia ~48%
    201 wishes -> "pursue Navia C0, up to 71 wishes", Navia ~36%
    202 wishes -> "up to 72";   210 wishes -> "up to 75"

The suspicion investigated here: the 75-wish current-banner candidate
itself changed when the account gained one unused wish. It did not.
Measured directly through `evaluate_candidate()` (runs=2_000, seed=0):

    (200 wishes, cap 75): Navia 0.4795, Arlecchino 0.9350, feasible
    (201 wishes, cap 75): Navia 0.4775, Arlecchino 0.9395, feasible

The same candidate is materially identical - the 0.002 difference is Monte
Carlo stream divergence (the extra wish changes the protected banner's
draw counts in the histories that exhaust the pool, shifting the shared rng
stream for later runs: statistically equivalent redraws, well under one
standard error). Arlecchino's protected probability moves slightly UP with
the extra wish, as it must: per history the extra wish can only enlarge the
protected banner's leftover pool.

The reported cap difference is the *coarse cap grid* the dashboard sends
(frontend `run()`: step = ceil(wishes / 8) unless "exact scan" is checked):

    200 wishes -> caps 200, 175, 150, 125, 100, 75, 50, 25, 0 -> 75 feasible
    201 wishes -> caps 201, 175, 149, 123, 97, 71, 45, 19, 0  -> 97 breaks
                  Arlecchino protection (0.801 < 0.9); 71 is the next grid
                  cap down. Cap 75 is never evaluated for 201 wishes.

The true feasibility boundary lies at 76/77: cap 76 keeps Arlecchino at
0.914 (feasible) while 77 and every cap through 96 fall below the 90%
threshold (0.8845 at 77 down to 0.8010 at 96). So the exact cap scan -
`recommend()`'s documented default (`budgets=None`) - selects cap 76 for
BOTH accounts: the optimizer's cap-selection logic treats the two accounts
identically, and the 201-wish 75-cap candidate sits strictly inside the
feasible region it picks.

Verdict: no production defect. The optimizer behavior is legitimate; the
visible 75 -> 71 jump is the documented coarse-grid sampling tradeoff
("a coarse list can miss a narrow feasible window", optimizer.recommend
docstring / `_caps`). These tests pin that diagnosis so a future change to
candidate evaluation, protection carry, or cap selection cannot silently
reintroduce a wish-count-dependent candidate.
"""

import pytest

from domain import Account, Banner, Goal, Ownership, Roadmap
from optimizer import available_outcomes, evaluate_candidate, recommend
from planner import PlannerContext

NAVI = Banner("Navia", "7.0", 2)
ARLECCHINO = Banner("Arlecchino", "7.1", 1)
RUNS = 2_000
SEED = 0


def _arlecchino(standings):
    (standing,) = [s for s in standings if s.goal.character == "Arlecchino"]
    return standing


class TestOneExtraUnusedWish:
    """The manual report above, pinned as automated checks.

    The scenario: Navia C0 is the current banner's own goal at Priority 2;
    Arlecchino C0 is the future goal at Priority 1 - so Arlecchino is the
    constraining protected goal (§2) and gates every Navia cap. No
    preference chain: the outcome comes from the roadmap-goal fallback
    (§13 step 2).
    """

    def _context(self, wishes: int) -> PlannerContext:
        """The manual-test scenario at `wishes` - the only differing field."""
        return PlannerContext(
            account=Account(
                current_pity=0,
                character_guarantee=True,
                owned_characters=Ownership({}),
                wishes=wishes,
            ),
            roadmap=Roadmap(
                goals=[Goal("Navia", 0, 2), Goal("Arlecchino", 0, 1)],
                banners=[NAVI, ARLECCHINO],
            ),
            current_version="7.0",
            current_phase=2,
            confidence=0.9,
            # income=None and the normal CHARACTER_EVENT_BANNER mechanics are
            # the defaults; the scenario has no income and normal mechanics.
        )

    def test_the_75_wish_candidate_is_unchanged_by_one_extra_unused_wish(self):
        """The SAME (Navia C0, cap 75) candidate at 200 vs 201 wishes.

        Records what the manual investigation asked for: the plan's
        current-banner budget, the Navia outcome probability, Arlecchino's
        protected probability and the feasibility verdict - for both
        starting-wish counts, same seed and runs.
        """
        context_200 = self._context(200)
        context_201 = self._context(201)
        (outcome_200,) = available_outcomes(context_200)
        (outcome_201,) = available_outcomes(context_201)
        assert outcome_200.character == "Navia" and outcome_200.constellation == 0

        candidate_200 = evaluate_candidate(
            context_200, outcome_200, budget=75, runs=RUNS, seed=SEED
        )
        candidate_201 = evaluate_candidate(
            context_201, outcome_201, budget=75, runs=RUNS, seed=SEED
        )

        # 1. The candidate really carries a current-banner Navia budget of
        #    exactly 75; the protected Arlecchino entry is uncapped (its
        #    budget is the account pool + income credit - never the limit,
        #    optimizer.protection.uncapped_budget).
        for candidate, wishes in ((candidate_200, 200), (candidate_201, 201)):
            assert candidate.budget == 75
            navia_entry, arlecchino_entry = candidate.plan.entries
            assert navia_entry.banner == NAVI
            assert navia_entry.target_constellation == 0
            assert navia_entry.budget == 75
            assert arlecchino_entry.banner == ARLECCHINO
            assert arlecchino_entry.budget == wishes

        # 2. The Navia probability is the same candidate's probability:
        #    ~48% at both wish counts (analytic P(5-star within 75 pulls
        #    from pity 0) ~ 0.474 with soft pity at 74). The measured
        #    values are 0.4795 vs 0.4775 - a 0.002 gap from Monte Carlo
        #    stream divergence, ~0.2 standard errors, not the ~12-point
        #    drop the manual report saw (that was cap 71, not cap 75).
        for candidate in (candidate_200, candidate_201):
            assert 0.42 < candidate.outcome_probability < 0.53
        assert abs(
            candidate_200.outcome_probability - candidate_201.outcome_probability
        ) <= 0.01

        # 3./4. Arlecchino's protected standing for the 201-wish candidate:
        #       0.9395, at or above the 90% threshold, constraining (it
        #       outranks the Priority 2 Navia objective), and the candidate
        #       is feasible. Measured 0.9350 for the 200-wish candidate.
        arlecchino_200 = _arlecchino(candidate_200.protected)
        arlecchino_201 = _arlecchino(candidate_201.protected)
        for standing in (arlecchino_200, arlecchino_201):
            assert 0.90 <= standing.probability <= 0.97
            assert standing.meets_threshold is True
            assert standing.constraining is True
        # The extra unused wish moves protection slightly up, not down
        # (measured 0.9395 vs 0.9350): it is never spent by the candidate.
        assert abs(arlecchino_200.probability - arlecchino_201.probability) <= 0.02

        # 5. Both candidates are feasible: the 201-wish / 75-wish candidate
        #    is NOT the infeasible one - protection holds at both counts.
        assert candidate_200.feasible is True
        assert candidate_201.feasible is True

    def test_the_reported_75_vs_71_reproduces_from_the_coarse_cap_grid(self):
        """The dashboard's coarse cap grid alone explains 75 vs 71.

        The frontend scans caps stepping down by ceil(wishes / 8) (unless
        "exact scan" is checked). At 200 wishes that grid contains 75; at
        201 wishes it does not - it contains 97 (Arlecchino protection
        broken) and then 71. Reproducing both grids through `recommend`
        yields exactly the manually observed recommendations.
        """
        rec_200 = recommend(
            self._context(200),
            (),
            runs=RUNS,
            seed=SEED,
            budgets=[200, 175, 150, 125, 100, 75, 50, 25, 0],
        )
        rec_201 = recommend(
            self._context(201),
            (),
            runs=RUNS,
            seed=SEED,
            budgets=[201, 175, 149, 123, 97, 71, 45, 19, 0],
        )

        assert rec_200.action == "pursue"
        assert rec_200.budget == 75
        # The 75-cap coarse-grid winner IS the candidate test 1 evaluated.
        assert rec_200.outcome_probability == pytest.approx(0.4795, abs=0.01)
        assert _arlecchino(rec_200.protected).meets_threshold is True

        assert rec_201.action == "pursue"
        assert rec_201.budget == 71
        # 71 sits below the soft-pity start (74): the ~36% the manual
        # report saw is P(5-star within 71 pulls) - a smaller cap, not a
        # changed candidate.
        assert rec_201.outcome_probability == pytest.approx(0.3570, abs=0.01)
        assert rec_201.outcome_probability < 0.42
        assert _arlecchino(rec_201.protected).meets_threshold is True

    def test_the_exact_cap_scan_recommends_the_same_cap_for_both_accounts(self):
        """With the exact cap range, both accounts get cap 76.

        Feasibility is monotone down here (more Navia spend, less
        Arlecchino protection): 77 through 96 are all infeasible (~0.88 at
        77 down to ~0.80 at 96), 76 is feasible (~0.911 - verified against
        BOTH the scalar and the vectorized engine at 20k-100k runs). A
        descending scan over an exact-range window therefore lands on 76
        for BOTH wish counts - the optimizer's cap-selection logic does not
        treat the two accounts differently, and the 201-wish / 75-wish
        candidate sits strictly inside the chosen cap (76 >= 75).

        The 76/77 margin is ~2.7 points around the 90% threshold, i.e.
        only ~4 standard errors at the optimizer's 2,000-run default. This
        scan therefore runs at 20,000 runs (~13 standard errors of
        separation) so the boundary verdict does not depend on which
        engine's RNG stream samples the boundary: the vectorized engine
        consumes the rng differently from the scalar oracle (statistically
        equivalent, per simulation.engine), and its 2,000-run sample at
        cap 76 legitimately lands below 0.9.
        """
        # A descending window that brackets the 76/77 boundary (the full
        # 77..96 range was verified in the investigation; these probes hit
        # the same first-feasible answer).
        window = [96, 90, 85, 80, 79, 78, 77, 76]
        scan_runs = 20_000
        rec_200 = recommend(
            self._context(200), (), runs=scan_runs, seed=SEED, budgets=window
        )
        rec_201 = recommend(
            self._context(201), (), runs=scan_runs, seed=SEED, budgets=window
        )

        assert rec_200.action == "pursue"
        assert rec_201.action == "pursue"
        assert rec_200.budget == 76
        assert rec_201.budget == 76
        # The exact-scan recommendations are the same candidate again:
        # materially identical Navia probability (the extra wish is never
        # spent; measured 0.5685 vs 0.5685).
        assert abs(
            rec_200.outcome_probability - rec_201.outcome_probability
        ) <= 0.01
        assert _arlecchino(rec_200.protected).meets_threshold is True
        assert _arlecchino(rec_201.protected).meets_threshold is True
