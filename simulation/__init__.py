"""Monte Carlo simulation (Design Document §11, §18 Phase 4).

Phase 4 invariants:

1. `simulate_history` returns one possible future account history (§11); the
   per-run records (BannerResult, GoalOutcome, RunResult) are auditable, not
   just "success = True".
2. The simulator never mutates domain state: every transition rebuilds the
   frozen Account (Ownership.with_constellation, §4.2).
3. All per-pull rates come from the Phase 2 engine (`pull_rate`); mechanics
   stay data (§17) and no rate mathematics lives in this package.
4. Multi-copy targets work in copies, not constellations (§10.4, §4.2): a
   plan's `target_constellation` is a desired *resulting constellation*,
   never a number of copies to pull; `copies_needed` is derived from the
   simulated account and is 0 (nothing spent, target_met True) when the
   account already meets or exceeds the target.
5. Pity and guarantee propagate across banners (§11): featured copy ->
   pity 0 / guarantee off / ownership +1; lost 50/50 -> pity 0 / guarantee
   on; no 5-star -> pity +1, guarantee unchanged.
6. Income timing is the Phase 3/4 version-level approximation (§16):
   version-level income becomes available when the simulation first reaches
   the first processed banner in that version; banners in the same version
   never credit it twice; the current banner spends only `account.wishes`.
   This matches PlannerContext.income_credit and planner.protection exactly.
   Within-version arrival order is deliberately not modeled.
7. Strategies are input (§12): the simulator executes plans faithfully and
   never chooses between them. Candidate generation, preference chains,
   stop conditions and recommendations are Phase 5 (§13, §15).
8. An absent plan entry means skip, not stop: unplanned banners are still
   processed chronologically (income credited, pity/guarantee carried) and
   remain in the run's history (§11).
9. Spend caps: wishes_spent = min(budget, available, wishes the target
   still needs); wishes never go negative (§12, §4.1).
10. Results are deterministic for identical (context, plan, runs, seed);
    the default seed makes re-runs reproducible (§2).
11. Validation: the simulator must reproduce the Phase 2 analytical
    single-copy curves within sampling error, and the exact multi-copy
    reference built by convolving the single-copy completion-time PMF
    (f(n) = F(n) - F(n-1), derived from the Phase 2 CDF) K times and
    comparing CDFs (probability invariant 10, mirrored).
12. Weapon banners (§17) and banner uncertainty (§2) are out of Phase 4
    scope, matching §18's Phase 4 list.
"""

from simulation.conditioning import condition_on_pull
from simulation.engine import (
    DEFAULT_RUNS,
    DEFAULT_SEED,
    simulate,
    simulate_history,
)
from simulation.outcomes import aggregate_runs
from simulation.results import (
    BannerAggregate,
    BannerResult,
    GoalOutcome,
    GoalProbability,
    RunResult,
    SimulationResult,
)
from simulation.strategy import PlannedSpend, SpendPlan

__all__ = [
    "DEFAULT_RUNS",
    "DEFAULT_SEED",
    "BannerAggregate",
    "BannerResult",
    "GoalOutcome",
    "GoalProbability",
    "PlannedSpend",
    "RunResult",
    "SimulationResult",
    "SpendPlan",
    "aggregate_runs",
    "condition_on_pull",
    "simulate",
    "simulate_history",
]
