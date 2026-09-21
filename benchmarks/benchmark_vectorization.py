"""Benchmark: scalar Monte Carlo oracle vs the vectorized engine.

The vectorization goal (Design Document §11, Phase 4): banners stay
sequential in Python while the independent histories within each banner
run as NumPy arrays. The scalar implementation (`_simulate_scalar`) remains
the behavioral oracle; this script measures the speedup against it on the
project's realistic workload:

    * a multi-banner roadmap (8 banners across 4 versions, two slots each)
    * roughly 450 wishes of total resources (80 starting + ~370 forecast)
    * 5,000 and 10,000 Monte Carlo runs

Usage (from the repository root):

    python benchmarks/benchmark_vectorization.py               # 5k + 10k runs
    python benchmarks/benchmark_vectorization.py --runs 2000   # custom sizes
    python benchmarks/benchmark_vectorization.py --no-memory   # skip tracemalloc
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain import (  # noqa: E402
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    Ownership,
    Roadmap,
    VersionIncome,
)
from planner import PlannerContext  # noqa: E402
from simulation import PlannedSpend, SpendPlan  # noqa: E402
from simulation.engine import _simulate_scalar, simulate  # noqa: E402


def realistic_context() -> tuple[PlannerContext, SpendPlan]:
    """A ~450-wish, 8-banner roadmap with multi-copy targets."""
    banners = [
        Banner("Vesna", "7.0", 1),
        Banner("Tsaritsa", "7.0", 2),
        Banner("Vodynista", "7.1", 1),
        Banner("Miroslava", "7.1", 2),
        Banner("Yaroslav", "7.2", 1),
        Banner("Svetlana", "7.2", 2),
        Banner("Bogdan", "7.3", 1),
        Banner("Zoryana", "7.3", 2),
    ]
    roadmap_banners = {banner.character: banner for banner in banners}

    roadmap = Roadmap(
        goals=[
            Goal("Vesna", 0, 1),
            Goal("Tsaritsa", 0, 2),
            Goal("Vodynista", 1, 3),
            Goal("Miroslava", 0, 4),
            Goal("Yaroslav", 0, 5),
        ],
        banners=banners,
    )
    account = Account(
        current_pity=30,
        character_guarantee=False,
        owned_characters=Ownership({"Vodynista": 0}),
        wishes=80,
        capturing_radiance_counter=1,
    )
    income = IncomeForecast(
        versions=[
            VersionIncome("7.0", estimate=IncomeEstimate(90, 100, 110)),
            VersionIncome("7.1", estimate=IncomeEstimate(90, 100, 110)),
            VersionIncome("7.2", estimate=IncomeEstimate(80, 90, 100)),
            VersionIncome("7.3", estimate=IncomeEstimate(70, 80, 90)),
        ]
    )
    context = PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version="7.0",
        current_phase=1,
        income=income,
    )
    plan = SpendPlan(
        entries=(
            PlannedSpend(roadmap_banners["Vesna"], 2, 150),
            PlannedSpend(roadmap_banners["Tsaritsa"], 0, 60),
            PlannedSpend(roadmap_banners["Vodynista"], 1, 80),
            PlannedSpend(roadmap_banners["Miroslava"], 0, 60),
            PlannedSpend(roadmap_banners["Yaroslav"], 0, 70),
            PlannedSpend(roadmap_banners["Svetlana"], 0, 60),
            PlannedSpend(roadmap_banners["Bogdan"], 0, 70),
            PlannedSpend(roadmap_banners["Zoryana"], 0, 60),
        ),
        shared_current_phase_budget=150,
    )
    return context, plan


def _timed(call):
    start = time.perf_counter()
    result = call()
    return result, time.perf_counter() - start


def _peak_memory_mb(call) -> float:
    tracemalloc.start()
    call()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak / 1e6


def benchmark(runs_values, measure_memory: bool) -> None:
    context, plan = realistic_context()

    # Warm-up: imports, caches, and allocator pools.
    simulate(context, plan, runs=200, seed=0)
    _simulate_scalar(context, plan, runs=200, seed=0)

    total_wishes = context.account.wishes + context.income_credit("7.3")
    workload = (
        f"{len(plan.entries)}/{len(context.roadmap.banners)} planned banners, "
        f"~{total_wishes} wishes"
    )
    print(f"workload: {workload}")
    print()
    print(
        f"{'runs':>7}  {'scalar (s)':>12}  {'vectorized (s)':>15}  "
        f"{'speedup':>8}  {'scalar peak MB':>15}  {'vector peak MB':>15}"
    )

    for runs in runs_values:
        scalar, scalar_seconds = _timed(
            lambda: _simulate_scalar(context, plan, runs=runs, seed=42)
        )
        vectorized, vector_seconds = _timed(
            lambda: simulate(context, plan, runs=runs, seed=42)
        )
        assert scalar.runs == vectorized.runs == runs

        speedup = (
            scalar_seconds / vector_seconds if vector_seconds else float("inf")
        )

        scalar_mb = vector_mb = "-"
        if measure_memory:
            scalar_mb = f"{_peak_memory_mb(lambda: _simulate_scalar(context, plan, runs=runs, seed=42)):.1f}"
            vector_mb = f"{_peak_memory_mb(lambda: simulate(context, plan, runs=runs, seed=42)):.1f}"

        print(
            f"{runs:>7}  {scalar_seconds:>12.2f}  {vector_seconds:>15.2f}  "
            f"{speedup:>7.1f}x  {scalar_mb:>15}  {vector_mb:>15}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        nargs="+",
        default=[5_000, 10_000],
        help="run counts to benchmark (default: 5000 10000)",
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="skip the tracemalloc memory measurement",
    )
    args = parser.parse_args()
    benchmark(args.runs, measure_memory=not args.no_memory)