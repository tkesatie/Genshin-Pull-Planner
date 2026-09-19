"""Validates the two "next" punch-list items together:

    * cache staleness: a confidence/mechanics/income/roadmap change must
      never let PlannerEvidenceCache hand back an evidence entry computed
      under the old settings
    * performance: recommend() must actually reuse cached CandidateStrategy
      (and skip-baseline) evidence across calls when nothing that matters
      has changed, instead of resimulating from scratch every time

Both are exercised end-to-end through `optimizer.recommend.recommend()`
wired to a real `PlannerEvidenceCache`, counting actual `simulate()` calls
via a wrapper - not just inspecting cache internals - so a change to either
module that breaks the wiring would fail this test, not just a change to
the cache class in isolation.
"""

from unittest.mock import patch

import pytest

from domain import Account, Banner, Goal, Roadmap
from domain.mechanics import CHARACTER_EVENT_BANNER, WishMechanics
from planner.context import PlannerContext
from optimizer import recommend

import optimizer.evaluation as evaluation_module
from api.planner_cache import PlannerEvidenceCache, context_fingerprint

ACCOUNT_ID = "test-account"


def _build_context(*, confidence: float = 0.9, mechanics: WishMechanics = CHARACTER_EVENT_BANNER) -> PlannerContext:
    """A scenario where the current-banner decision (Vesna C0, priority 2)
    is genuinely gated by a higher-priority future goal (Tsaritsa C0,
    priority 1) - i.e. `constraining_goals` is non-empty, so the cap scan
    actually runs (an unconstrained outcome short-circuits to a single cap
    regardless of caching, which would not exercise this test). With 140
    wishes and 90% confidence this roadmap is actually infeasible (Vesna
    would eat the wishes Tsaritsa needs), so it also exercises the
    expensive full-scan "skip" diagnostic path the punch list calls out.
    """
    account = Account(current_pity=0, character_guarantee=False, wishes=140)
    roadmap = Roadmap(
        goals=[Goal("Tsaritsa", 0, 1), Goal("Vesna", 0, 2)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)],
    )
    return PlannerContext(
        account=account,
        roadmap=roadmap,
        current_version="7.0",
        current_phase=1,
        confidence=confidence,
        mechanics=mechanics,
    )


def _wire_cache(cache: PlannerEvidenceCache, context: PlannerContext, runs: int, seed: int | None) -> dict:
    def lookup(outcome, cap, banner):
        evidence = cache.get_candidate(
            ACCOUNT_ID,
            character=outcome.character,
            constellation=outcome.constellation,
            banner_version=banner.version,
            banner_phase=banner.phase,
            budget=cap,
            context=context,
        )
        return None if evidence is None else evidence.candidate

    def sink(candidate):
        cache.put_candidate(ACCOUNT_ID, candidate, runs=runs, seed=seed, context=context)

    def skip_lookup(banner):
        return cache.get_skip_baseline(ACCOUNT_ID, banner, context=context)

    def skip_sink(banner, protected):
        cache.put_skip_baseline(ACCOUNT_ID, banner, protected, context=context)

    return dict(
        candidate_lookup=lookup,
        simulation_sink=sink,
        skip_baseline_lookup=skip_lookup,
        skip_baseline_sink=skip_sink,
    )


def _count_simulate_calls(fn):
    """Run `fn()` counting real calls to optimizer.evaluation.simulate."""
    original = evaluation_module.simulate
    calls = {"n": 0}

    def counting_simulate(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    with patch.object(evaluation_module, "simulate", counting_simulate):
        result = fn()
    return result, calls["n"]


def _run(context, cache, runs=200, seed=0):
    wiring = _wire_cache(cache, context, runs, seed)
    return recommend(context, (), runs=runs, seed=seed, **wiring)


def test_repeated_recommend_with_unchanged_context_reuses_cache():
    context = _build_context()
    cache = PlannerEvidenceCache()

    first_result, first_calls = _count_simulate_calls(lambda: _run(context, cache))
    assert first_result.action == "skip"
    assert first_calls > 100, (
        "sanity check: this scenario is the expensive full-scan skip "
        "diagnostic the punch list calls out (every cap from 140 down to "
        "0 is infeasible)"
    )

    second_result, second_calls = _count_simulate_calls(lambda: _run(context, cache))
    assert second_calls == 0, (
        f"expected the second identical-context call to reuse every cached "
        f"candidate AND the cached skip baseline (0 new simulations), got "
        f"{second_calls}"
    )
    assert second_result.action == first_result.action
    assert second_result.skip_reason == first_result.skip_reason
    assert len(second_result.rejected) == len(first_result.rejected)


def test_confidence_change_invalidates_cache():
    """A confidence override changes `meets_threshold` on every cached
    candidate, so it must never be served from a cache populated under a
    different confidence (module docstring of api.planner_cache).
    """
    base_context = _build_context(confidence=0.9)
    cache = PlannerEvidenceCache()

    _, first_calls = _count_simulate_calls(lambda: _run(base_context, cache))
    assert first_calls > 0

    changed_context = _build_context(confidence=0.5)
    _, second_calls = _count_simulate_calls(lambda: _run(changed_context, cache))
    assert second_calls > 0, (
        "a confidence change must be a cache miss, not a reuse of the old "
        "confidence's cached feasibility"
    )


def test_mechanics_change_invalidates_cache():
    base_context = _build_context()
    cache = PlannerEvidenceCache()
    _count_simulate_calls(lambda: _run(base_context, cache))

    changed_mechanics = WishMechanics(
        banner_type="character_event",
        hard_pity=80,
        soft_pity_start=64,
        base_rate=0.01,
        soft_pity_increment=0.08,
        featured_rate=0.5,
    )
    changed_context = _build_context(mechanics=changed_mechanics)
    _, second_calls = _count_simulate_calls(lambda: _run(changed_context, cache))
    assert second_calls > 0, "a mechanics change must invalidate the cache"


def test_roadmap_change_invalidates_cache():
    base_context = _build_context()
    cache = PlannerEvidenceCache()
    _count_simulate_calls(lambda: _run(base_context, cache))

    # Add a new protected goal - the roadmap itself changed.
    changed_roadmap = Roadmap(
        goals=list(base_context.roadmap.goals) + [Goal("Vodynista", 0, 3)],
        banners=list(base_context.roadmap.banners) + [Banner("Vodynista", "7.2", 1)],
    )
    changed_context = PlannerContext(
        account=base_context.account,
        roadmap=changed_roadmap,
        current_version=base_context.current_version,
        current_phase=base_context.current_phase,
        confidence=base_context.confidence,
        mechanics=base_context.mechanics,
    )
    _, second_calls = _count_simulate_calls(lambda: _run(changed_context, cache))
    assert second_calls > 0, "adding/removing a roadmap goal must invalidate the cache"


def test_fingerprint_excludes_account_state():
    """The account's own pity/guarantee/wishes must NOT be part of the
    fingerprint - a recorded pull legitimately changes those, and that
    change must be handled by conditioning, not by blanket invalidation.
    """
    context_a = _build_context()
    account_after_pull = Account(
        current_pity=0, character_guarantee=False, wishes=139,
    )
    context_b = PlannerContext(
        account=account_after_pull,
        roadmap=context_a.roadmap,
        current_version=context_a.current_version,
        current_phase=context_a.current_phase,
        confidence=context_a.confidence,
        mechanics=context_a.mechanics,
    )
    assert context_fingerprint(context_a) == context_fingerprint(context_b)


def test_get_candidate_rejects_stale_fingerprint_directly():
    """Unit-level check on the cache class itself, independent of recommend()."""
    cache = PlannerEvidenceCache()
    context = _build_context()
    banner = Banner("Vesna", "7.0", 1)
    from optimizer.outcomes import OutcomeOption
    from optimizer.evaluation import evaluate_candidate

    candidate = evaluate_candidate(context, OutcomeOption("Vesna", 0, 1), 50, banner=banner, runs=100, seed=0)
    cache.put_candidate(ACCOUNT_ID, candidate, runs=100, seed=0, context=context)

    hit = cache.get_candidate(
        ACCOUNT_ID, character="Vesna", constellation=0,
        banner_version="7.0", banner_phase=1, budget=50, context=context,
    )
    assert hit is not None

    stale_context = _build_context(confidence=0.5)
    miss = cache.get_candidate(
        ACCOUNT_ID, character="Vesna", constellation=0,
        banner_version="7.0", banner_phase=1, budget=50, context=stale_context,
    )
    assert miss is None


def test_skip_baseline_cache_hit_and_staleness():
    """Unit-level check for the skip-baseline cache slot specifically."""
    cache = PlannerEvidenceCache()
    context = _build_context()
    banner = Banner("Vesna", "7.0", 1)
    from optimizer.evaluation import evaluate_skip_baseline

    baseline = evaluate_skip_baseline(context, runs=100, seed=0)
    cache.put_skip_baseline(ACCOUNT_ID, banner, baseline, context=context)

    hit = cache.get_skip_baseline(ACCOUNT_ID, banner, context=context)
    assert hit == baseline

    stale_context = _build_context(confidence=0.5)
    miss = cache.get_skip_baseline(ACCOUNT_ID, banner, context=stale_context)
    assert miss is None
