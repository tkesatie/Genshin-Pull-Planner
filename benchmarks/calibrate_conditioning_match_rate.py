"""TEMPORARY calibration: what conditioned_runs does conditioning.py produce?

Direct question for the MIN_CONDITIONED_RUNS measurement: take a fresh
planner run (2000 histories), then record one pull outcome with each
reporting convention, and see how many histories survive conditioning.

This bypasses the HTTP layer on purpose: it calls planner_evidence_cache
directly with a hand-built account, with both pity=0 (matches the
instrumentation tests' fresh-account fixture) and pity=30 (realistic).

Usage:
    python benchmarks/calibrate_conditioning_match_rate.py --samples 6
"""

import argparse
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.planner_cache import PlannerEvidenceCache, context_fingerprint  # noqa: E402
from api.routers.planner import MIN_CONDITIONED_RUNS  # noqa: E402
from domain import Account, Banner, Goal, Ownership, Roadmap  # noqa: E402
from optimizer import DEFAULT_RUNS  # noqa: E402
from optimizer.evaluation import evaluate_candidate  # noqa: E402
from optimizer.outcomes import OutcomeOption  # noqa: E402
from planner import PlannerContext  # noqa: E402
from simulation import PlannedSpend, SpendPlan  # noqa: E402


def make_context(pity: int) -> PlannerContext:
    account = Account(current_pity=pity, character_guarantee=False, wishes=400)
    banner = Banner("Vesna", "7.0", 1)
    roadmap = Roadmap(
        goals=[Goal("Vesna", 2, 1)],
        banners=[banner],
    )
    return PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0", current_phase=1
    )


def simulate_one(context: PlannerContext, seed: int):
    banner = context.roadmap.banners[0]
    plan = SpendPlan(
        entries=(PlannedSpend(banner, 2, context.account.wishes),),
        shared_current_phase_budget=context.account.wishes,
    )
    candidate = evaluate_candidate(
        context, OutcomeOption("Vesna", 2, 1), context.account.wishes,
        banner=banner, runs=DEFAULT_RUNS, seed=seed,
    )
    return plan, candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rng = random.Random(20260921)
    print(f"DEFAULT_RUNS={DEFAULT_RUNS} MIN_CONDITIONED_RUNS={MIN_CONDITIONED_RUNS}")
    for pity in (0, 30):
        print(f"\n=== starting pity={pity} ===")
        for sample in range(args.samples):
            seed = rng.randint(0, 10**6)
            context = make_context(pity)
            plan, candidate = simulate_one(context, seed)
            cache = PlannerEvidenceCache()
            cache.put_candidate(
                "acct", candidate, runs=DEFAULT_RUNS, seed=seed, context=context,
            )
            # pull the candidate result out of cache for conditioning
            from api.planner_cache import planner_evidence_cache  # noqa
            account = context.account
            budget = account.wishes
            for wishes_used, label in (
                (80, "spent-now=80"),
                (80 + pity, f"cumulative={80 + pity}"),
            ):
                conditioned = cache.condition_candidate(
                    "acct", character="Vesna", outcome="featured",
                    constellation=2, banner_version="7.0", banner_phase=1,
                    new_budget=budget - wishes_used, wishes_used=wishes_used,
                    context=context,
                )
                runs = conditioned.runs if conditioned is not None else 0
                verdict = "RETAINED" if runs >= MIN_CONDITIONED_RUNS else "thin"
                if args.verbose or True:
                    print(f"  seed={seed} {label}: conditioned_runs={runs} {verdict}")
                _ = plan  # plan unused beyond simulation context


if __name__ == "__main__":
    main()
