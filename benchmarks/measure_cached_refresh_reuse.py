"""TEMPORARY measurement: how often can cached-refresh reuse conditioned evidence?

Question
--------
/planner/cached-refresh answers a recorded pull from retained conditioned
evidence when at least MIN_CONDITIONED_RUNS histories survive the observation
filter, and otherwise falls back to a targeted CONDITIONED_FALLBACK_RUNS
resimulation. `MIN_CONDITIONED_RUNS = DEFAULT_RUNS // 2` is a judgement call, so
this harness measures how often each path actually occurs in realistic use.

The script drives the real HTTP surface exactly like the frontend
(refreshPlannerAfterPull / recordPullOutcome in frontend/index.html):

    Run Planner            GET  /accounts/{id}/planner/recommendation
                           GET  /accounts/{id}/planner/strategy
    record a pull result   POST /accounts/{id}/pull-result
    cached refresh         GET  /accounts/{id}/planner/cached-refresh?...

It only *reads* the counters in api/refresh_metrics.py and its DEBUG log: the
conditioning algorithm, both thresholds and every returned result are
untouched.

Two axes are configurable because both change whether conditioning can match
at all:

--pity realistic|zero   carried pity at account creation. conditioning.py
                        compares the recorded wishes_used against the index of
                        the 5-star inside each retained history; a history
                        carries the account's starting pity, so a reported
                        "wishes spent this session" only matches when pity is 0.
--observation-mode      spent-now  -> what the UI prompt asks for.
                        cumulative -> the within-banner index, which is what
                                       conditioning.py compares against.

Usage (from the repository root):

    python benchmarks/measure_cached_refresh_reuse.py
    python benchmarks/measure_cached_refresh_reuse.py --refreshes 120 --pity zero
    python benchmarks/measure_cached_refresh_reuse.py --observation-mode cumulative
"""

import argparse
import logging
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from api import create_app  # noqa: E402
from api.refresh_metrics import cached_refresh_metrics  # noqa: E402
from api.routers import planner as planner_router  # noqa: E402

# The design document's own cast, versions and phases (§4-§7).
CHARACTERS = [
    "Vesna",
    "Tsaritsa",
    "Vodynista",
    "Miroslava",
    "Yaroslav",
    "Svetlana",
    "Bogdan",
    "Zoryana",
]
VERSIONS = ["7.0", "7.1", "7.2", "7.3"]

GOAL_LINE = re.compile(
    r"^Cached refresh: observed=(\S+), wishes_used=(\d+), goal=(.+), "
    r"conditioned_runs=(\d+), threshold=(\d+), path=(\w+)$"
)
FINISH_LINE = re.compile(
    r"^Cached refresh: observed=(\S+), wishes_used=(\d+), path=(\w+), "
    r"fallback_resims=(\d+), candidate_fallbacks=(\d+), evidence=(\d+)$"
)


def build_scenario(
    rng: random.Random, index: int, pity_mode: str
) -> tuple[dict, str]:
    """One plausible account plus the featured character being pulled on."""
    roster = rng.sample(CHARACTERS, rng.randint(3, 5))
    featured = roster[0]

    banners = [
        {
            "character": name,
            "version": VERSIONS[min(position, len(VERSIONS) - 1)],
            "phase": 1 + (position % 2),
        }
        for position, name in enumerate(roster)
    ]

    goals = [
        {"character": name, "constellation": rng.choice([0, 0, 1, 2])}
        for name in roster
        if name != featured  # the featured character gets its own chase goal below
    ]
    goals.insert(1, {"character": featured, "constellation": 2})  # chasing copies
    roadmap_goals = [
        {
            "character": goal["character"],
            "constellation": goal["constellation"],
            "priority": rank,
        }
        for rank, goal in enumerate(goals, start=1)
    ]

    # Most players rank a preference chain for the character they chase (§15);
    # some never fill the page in at all.
    preferences = []
    if rng.random() < 0.8:
        preferences = [
            {"character": featured, "rank": 1, "constellation": 2, "weapon_refinement": 1},
            {"character": featured, "rank": 2, "constellation": 1, "weapon_refinement": 1},
            {"character": featured, "rank": 3, "constellation": 2},
            {"character": featured, "rank": 4, "constellation": 1},
            {"character": featured, "rank": 5, "constellation": 0},
        ]

    income = None
    if rng.random() < 0.7:
        income = {
            "versions": [
                {
                    "version": version,
                    "estimate": {
                        "low": expected - 15,
                        "expected": expected,
                        "high": expected + 15,
                    },
                }
                for version, expected in zip(VERSIONS, [90, 85, 80, 75])
            ]
        }

    owned: dict[str, int] = {}
    if rng.random() < 0.5:
        owned[featured] = 0
    if rng.random() < 0.3:
        owned[roster[-1]] = rng.choice([0, 1])

    payload = {
        "label": f"measurement account {index}",
        "account": {
            "current_pity": 0 if pity_mode == "zero" else rng.randint(0, 74),
            "character_guarantee": rng.random() < 0.35,
            "wishes": rng.choice([60, 90, 120, 160, 228, 300, 400]),
            "capturing_radiance_counter": rng.choice([0, 0, 1, 2, 3]),
            "owned_characters": owned,
        },
        "settings": {"current_version": "7.0", "current_phase": 1},
        "goals": roadmap_goals,
        "banners": banners,
        "preferences": preferences,
        "income": income,
    }
    return payload, featured


def sample_wishes_used(rng: random.Random, current_pity: int, mode: str) -> int:
    """What a player would type, drawn from the real 5-star wish distribution.

    The 5-star lands at some absolute within-banner index, mostly inside the
    soft-pity ramp (74-90) because that is where the probability mass is.
    """
    roll = rng.random()
    if roll < 0.12:
        target = rng.randint(1, 45)  # early luck
    elif roll < 0.20:
        target = rng.randint(46, 73)  # mid range
    else:
        target = min(90, 74 + int(abs(rng.gauss(0, 4.5))))  # soft-pity ramp
    if mode == "cumulative":
        return target
    return max(1, target - current_pity)


class _CaptureHandler(logging.Handler):
    """Keeps the endpoint's per-refresh DEBUG lines for the sample report."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(record.getMessage())


def run_planner(client: TestClient, account_id: str) -> dict:
    """The frontend's "Run Planner": recommendation plus strategy."""
    recommendation = client.get(f"/accounts/{account_id}/planner/recommendation")
    assert recommendation.status_code == 200, recommendation.text
    strategy = client.get(f"/accounts/{account_id}/planner/strategy")
    assert strategy.status_code == 200, strategy.text
    return recommendation.json()


def _sign_in(client: TestClient) -> None:
    client.post(
        "/auth/register",
        json={"username": "measure-user", "password": "measure-password"},
    )
    logged_in = client.post(
        "/auth/login",
        json={"username": "measure-user", "password": "measure-password"},
    )
    assert logged_in.status_code == 200, logged_in.text


def drive(client, rng, args, timings, failures, failure_samples):
    """Run realistic planner usage; return (accounts, planner_runs, attempts)."""
    accounts = 0
    planner_runs = 0
    attempts = 0

    while attempts < args.refreshes:
        accounts += 1
        if accounts > args.refreshes * 3:  # safety valve
            break
        payload, featured = build_scenario(rng, accounts, args.pity)
        created = client.post("/accounts", json=payload)
        assert created.status_code == 201, created.text
        view = created.json()
        account_id = view["id"]

        recommendation = run_planner(client, account_id)
        planner_runs += 1

        for _ in range(args.per_account):
            if attempts >= args.refreshes:
                break
            state = view["account"]
            wishes_used = sample_wishes_used(
                rng, state["current_pity"], args.observation_mode
            )
            if wishes_used > state["wishes"]:
                break  # cannot pay for a full attempt at the next 5-star
            got_character = state["character_guarantee"] or rng.random() < 0.5
            outcome = "featured" if got_character else "lost_50_50"
            pity_before = state["current_pity"]  # pull-result resets it to 0

            pull = client.post(
                f"/accounts/{account_id}/pull-result",
                json={
                    "outcome": outcome,
                    "wishes_used": wishes_used,
                    "character": featured,
                },
            )
            assert pull.status_code == 200, pull.text
            view = pull.json()

            params = {
                "character": featured,
                "outcome": outcome,
                "wishes_used": str(wishes_used),
            }
            # The frontend forwards the previous recommendation so the refresh
            # can condition that candidate too.
            if recommendation.get("outcome"):
                params["recommendation_budget"] = str(recommendation["budget"])
                params["recommendation_constellation"] = str(
                    recommendation["outcome"]["constellation"]
                )

            attempts += 1
            start = time.perf_counter()
            try:
                refresh = client.get(
                    f"/accounts/{account_id}/planner/cached-refresh", params=params
                )
            except Exception as exc:  # unhandled endpoint error -> HTTP 500
                # Known pre-existing crash in the candidate-fallback branch: it
                # dereferences the conditioned candidate even when
                # condition_candidate() returned None (no matching history, or
                # no cached candidate for that budget). Recording it - not
                # fixing it - is what this harness is for.
                refresh_ok = False
                kind = type(exc).__name__
            else:
                refresh_ok = refresh.status_code == 200
                kind = f"HTTP {refresh.status_code}"
            timings.append(time.perf_counter() - start)

            if not refresh_ok:
                failures[kind] = failures.get(kind, 0) + 1
                if len(failure_samples) < 5:
                    failure_samples.append(
                        f"{kind}: pity_before={pity_before}, "
                        f"{outcome} after {wishes_used} wishes, "
                        f"budget={params.get('recommendation_budget')}, "
                        f"wishes={state['wishes']}"
                    )

            if args.verbose:
                print(
                    f"  attempt {attempts}: pity_before={pity_before} "
                    f"{outcome}@{wishes_used} wishes "
                    f"{'ok' if refresh_ok else 'FAILED'} "
                    f"({timings[-1]:.2f}s)",
                    flush=True,
                )

            # After acquiring the character, after a failed refresh (the
            # frontend's expensive "Pull outcome recovery" rerun) or when the
            # player simply re-runs the planner, the recommendation is rebuilt -
            # that is what refills the evidence at DEFAULT_RUNS.
            if got_character or not refresh_ok or rng.random() < 0.5:
                recommendation = run_planner(client, account_id)
                planner_runs += 1

    return accounts, planner_runs, attempts


def measure(args) -> dict:
    """Collect counters plus the per-refresh log for the requested sample."""
    from api.planner_cache import planner_evidence_cache

    rng = random.Random(args.seed)
    timings: list[float] = []
    failures: dict[str, int] = {}
    failure_samples: list[str] = []

    with TestClient(create_app()) as client:
        _sign_in(client)
        cached_refresh_metrics.reset()
        planner_evidence_cache.clear()

        log = logging.getLogger("planner.cached_refresh")
        capture = _CaptureHandler()
        log.setLevel(logging.DEBUG)
        log.addHandler(capture)
        log.propagate = False
        try:
            accounts, planner_runs, attempts = drive(
                client, rng, args, timings, failures, failure_samples
            )
        finally:
            log.removeHandler(capture)

    goal_lines = []
    finish_lines = []
    for line in capture.lines:
        goal = GOAL_LINE.match(line)
        if goal:
            goal_lines.append(
                {
                    "outcome": goal.group(1),
                    "wishes_used": int(goal.group(2)),
                    "goal": goal.group(3),
                    "conditioned_runs": int(goal.group(4)),
                    "threshold": int(goal.group(5)),
                    "path": goal.group(6),
                }
            )
        finish = FINISH_LINE.match(line)
        if finish:
            finish_lines.append({"outcome": finish.group(1), "path": finish.group(3)})

    return {
        "accounts": accounts,
        "planner_runs": planner_runs,
        "attempts": attempts,
        "timings": timings,
        "failures": failures,
        "failure_samples": failure_samples,
        "goal_lines": goal_lines,
        "finish_lines": finish_lines,
        "debug_lines": capture.lines,
        "refreshes": cached_refresh_metrics.total,
    }


def _histogram(runs: list[int], threshold: int) -> list[str]:
    """Bucket the surviving-run counts so the distribution is readable."""
    buckets = [
        (1, 9),
        (10, 49),
        (50, 99),
        (100, 249),
        (250, 499),
        (500, 999),
        (1000, 1999),
        (2000, 10**9),
    ]
    lines = []
    for low, high in buckets:
        count = sum(1 for value in runs if low <= value <= high)
        label = f"{low}-{high}" if high < 10**9 else f"{low}+"
        marker = "  <- at/above threshold" if low == threshold else ""
        lines.append(
            f"    {label:>10}: {count:>6}  ({count / len(runs):5.1%}){marker}"
        )
    return lines


def report(result: dict, args) -> None:
    from optimizer import DEFAULT_RUNS

    metrics = cached_refresh_metrics
    threshold = planner_router.MIN_CONDITIONED_RUNS
    fallback_runs = planner_router.CONDITIONED_FALLBACK_RUNS
    attempts = result["attempts"]
    completed = metrics.total
    raised = sum(result["failures"].values())
    timings = result["timings"]
    goal_lines = result["goal_lines"]
    asked = [item["conditioned_runs"] for item in goal_lines]
    completed_runs = list(metrics.surviving_runs)

    # The endpoint calls finish() exactly once per refresh that reaches the end
    # of the function - so len(finish_lines) == completed - while every goal
    # evidence logged before that belongs to the refresh it ends. Parse the
    # ordered DEBUG stream into one group per completed refresh; attempts that
    # raised have no finish() line and form the trailing group.
    groups: list[dict] = []
    current = {"goal_paths": [], "finish": None}
    for line in result["debug_lines"]:
        goal = GOAL_LINE.match(line)
        if goal:
            current["goal_paths"].append(goal.group(6))
            continue
        finish = FINISH_LINE.match(line)
        if finish:
            current["finish"] = finish.group(3)
            groups.append(current)
            current = {"goal_paths": [], "finish": None}
    if current["goal_paths"] or raised:
        # Remaining goal lines belong to attempts that raised before finish();
        # the DEBUG stream carries no per-attempt ids, so count the trailing
        # group once (its goal evidence) and count every raised attempt as an
        # incomplete full-optimizer fallback below.
        groups.append({"goal_paths": current["goal_paths"], "finish": None})

    by_attempt = {
        "attempts": attempts,
        "retained": 0,
        "targeted_fallback": 0,
        "full_optimizer_fallback": 0,
        "groups_with_evidence": len(groups),
        "goal_evidence_total": sum(len(group["goal_paths"]) for group in groups),
        "goals_retained": 0,
        "goals_targeted": 0,
        "goals_no_plan": 0,
    }
    for group in groups:
        paths = group["goal_paths"]
        for path in paths:
            if path == "retained":
                by_attempt["goals_retained"] += 1
            elif path == "targeted_fallback":
                by_attempt["goals_targeted"] += 1
            else:
                by_attempt["goals_no_plan"] += 1

        if group["finish"] is None:
            continue  # counted via `raised` below, not via the log grouping
        elif group["finish"] == "full_optimizer_fallback":
            by_attempt["full_optimizer_fallback"] += 1
        elif group["finish"] == "targeted_fallback" or "targeted_fallback" in paths:
            by_attempt["targeted_fallback"] += 1
        else:
            by_attempt["retained"] += 1
    # Attempts that raised never reached finish(); the frontend reruns the full
    # optimizer for exactly those ("Pull outcome recovery"), so count them as
    # full-optimizer fallbacks.
    by_attempt["raised"] = raised
    by_attempt["full_optimizer_fallback"] += raised
    assert (
        by_attempt["retained"] + by_attempt["targeted_fallback"] + by_attempt["full_optimizer_fallback"]
        == attempts
    ), "attempt classification must cover every attempt exactly once"

    print()
    print("How often can cached-refresh reuse conditioned evidence?")
    print(f"  seed                     : {args.seed}")
    print(f"  refreshes requested      : {args.refreshes}")
    print(f"  accounts driven          : {result['accounts']}")
    print(f"  Run Planner calls        : {result['planner_runs']}")
    print(f"  pull-result refreshes    : {attempts} attempts")
    print(f"  carried pity at creation : {args.pity}")
    print(f"  observation convention   : {args.observation_mode}")
    print(f"  DEFAULT_RUNS             : {DEFAULT_RUNS}")
    print(f"  MIN_CONDITIONED_RUNS     : {threshold}")
    print(f"  CONDITIONED_FALLBACK_RUNS: {fallback_runs}")
    print(
        f"  endpoint reached finish(): {completed} of {attempts} attempts "
        f"({completed / max(attempts, 1):.1%});  {raised} raised"
    )
    for kind, count in sorted(result["failures"].items(), key=lambda kv: -kv[1]):
        print(f"    {kind}: {count} ({count / max(attempts, 1):.1%})")
    for sample in result["failure_samples"]:
        print(f"      e.g. {sample}")

    base = completed or 1
    print()
    print("Classification of completed refreshes (api/refresh_metrics.py counters):")
    print("| Metric                        | Count | Percentage |")
    print("| ----------------------------- | ----: | ---------: |")
    print(f"| Total refreshes               | {completed:>5} | {100.0:>9.1f}% |")
    print(
        f"| Retained conditioned evidence | {metrics.retained:>5} | "
        f"{metrics.retained / base:>9.1%} |"
    )
    print(
        f"| Targeted fallback             | {metrics.targeted_fallback:>5} | "
        f"{metrics.targeted_fallback / base:>9.1%} |"
    )
    print(
        f"| Full optimizer fallback       | {metrics.full_optimizer_fallback:>5} | "
        f"{metrics.full_optimizer_fallback / base:>9.1%} |"
    )
    print(f"  targeted goal resimulations: {metrics.fallback_resimulations}")
    print(f"  fresh candidate evaluations: {metrics.candidate_fallbacks}")
    return _report_evidence(
        result, args, by_attempt, asked, completed_runs, timings, threshold
    )


def _percentile(values: list[int], fraction: float) -> int:
    return values[int(fraction * (len(values) - 1))]


def _report_evidence(
    result: dict,
    args,
    by_attempt: dict,
    asked: list[int],
    completed_runs: list[int],
    timings: list[float],
    threshold: int,
) -> None:
    """The distribution this measurement exists for: how close to 1000 the
    surviving conditioned run counts actually land."""
    goal_lines = result["goal_lines"]

    if asked:
        ordered = sorted(asked)
        above = sum(1 for value in ordered if value >= threshold)
        print()
        print(f"Retained-evidence survival (per goal evidence, n={len(ordered)}):")
        print(
            f"  min={ordered[0]} p25={_percentile(ordered, 0.25)} "
            f"median={statistics.median(ordered)} p75={_percentile(ordered, 0.75)} "
            f"max={ordered[-1]} mean={statistics.fmean(ordered):.1f}"
        )
        print(
            f"  >= MIN_CONDITIONED_RUNS ({threshold}): {above} "
            f"({above / len(ordered):.1%}) - those goals were answered from retained "
            f"evidence instead of being resimulated"
        )
        print("  distribution:")
        print("\n".join(_histogram(ordered, threshold)))

        if args.verbose:
            print()
            print("  per-refresh goal evidence:")
            for item in goal_lines:
                print(
                    f"    {item['outcome']:<12} wishes_used={item['wishes_used']:<3} "
                    f"{item['goal']:<16} conditioned_runs={item['conditioned_runs']:<5} "
                    f"path={item['path']}"
                )
    else:
        print()
        print("no goal evidence survived conditioning in any refresh")

    if by_attempt:
        total_attempts = by_attempt["attempts"] or 1
        print()
        print(
            "Classification of every attempt, including those that raised "
            f"(n={by_attempt['attempts']}):"
        )
        print("| Metric                        | Count | Percentage |")
        print("| ----------------------------- | ----: | ---------: |")
        print(f"| Total refreshes               | {by_attempt['attempts']:>5} | {100.0:>9.1f}% |")
        print(
            f"| Retained conditioned evidence | {by_attempt['retained']:>5} | "
            f"{by_attempt['retained'] / total_attempts:>9.1%} |"
        )
        print(
            f"| Targeted fallback             | {by_attempt['targeted_fallback']:>5} | "
            f"{by_attempt['targeted_fallback'] / total_attempts:>9.1%} |"
        )
        print(
            f"| Full optimizer fallback       | {by_attempt['full_optimizer_fallback']:>5} | "
            f"{by_attempt['full_optimizer_fallback'] / total_attempts:>9.1%} |"
        )
        print(
            f"  (of the full-optimizer column, {by_attempt['raised']} were "
            f"attempts that raised before finish())"
        )

        goal_counts = {
            "retained": by_attempt["goals_retained"],
            "targeted": by_attempt["goals_targeted"],
            "no_plan": by_attempt["goals_no_plan"],
        }
        total_goals = sum(goal_counts.values()) or 1
        print()
        print(
            "Path per goal evidence across every attempt that logged evidence "
            f"(including attempts that then raised): n={total_goals}"
        )
        print(
            f"    retained          {goal_counts['retained']:>6} "
            f"({goal_counts['retained'] / total_goals:6.1%})\n"
            f"    targeted_fallback {goal_counts['targeted']:>6} "
            f"({goal_counts['targeted'] / total_goals:6.1%})\n"
            f"    no_plan (skipped) {goal_counts['no_plan']:>6} "
            f"({goal_counts['no_plan'] / total_goals:6.1%})"
        )

    if completed_runs:
        print()
        print(
            f"surviving runs as recorded by the counters (n={len(completed_runs)}): "
            f"min={min(completed_runs)} max={max(completed_runs)}"
        )

    if timings:
        print()
        print(
            f"cached-refresh latency: median={statistics.median(timings):.3f}s "
            f"mean={statistics.fmean(timings):.3f}s max={max(timings):.3f}s"
        )

    print()
    print("sample per-refresh DEBUG lines:")
    for line in result["debug_lines"][:6]:
        print(f"  {line}")
    print(f"  ... ({len(result['debug_lines'])} lines captured)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cached-refresh evidence reuse.")
    parser.add_argument(
        "--refreshes", type=int, default=120, help="pull-result refreshes to attempt"
    )
    parser.add_argument("--seed", type=int, default=20260921, help="scenario rng seed")
    parser.add_argument(
        "--per-account", type=int, default=4, help="pull updates per account"
    )
    parser.add_argument(
        "--pity",
        choices=("realistic", "zero"),
        default="realistic",
        help="carried pity at account creation",
    )
    parser.add_argument(
        "--observation-mode",
        choices=("spent-now", "cumulative"),
        default="spent-now",
        help="how the player reports wishes_used on a recorded pull",
    )
    parser.add_argument("--verbose", action="store_true", help="print each attempt")
    args = parser.parse_args()

    started = time.perf_counter()
    result = measure(args)
    report(result, args)
    print()
    print(f"measurement wall time: {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()




