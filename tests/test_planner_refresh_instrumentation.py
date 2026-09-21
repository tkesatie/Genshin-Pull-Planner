"""TEMPORARY tests for the cached-refresh instrumentation.

These verify that the measurement counters classify refreshes correctly and
that instrumenting the endpoint does not change its responses. The
instrumentation (api/refresh_metrics.py) is scheduled for removal once the
threshold data has been collected; these tests go with it.
"""

import pytest

from api.refresh_metrics import cached_refresh_metrics


@pytest.fixture(autouse=True)
def _reset_metrics():
    cached_refresh_metrics.reset()
    yield
    cached_refresh_metrics.reset()


def metrics_threshold():
    from api.routers import planner as planner_router

    return planner_router.MIN_CONDITIONED_RUNS


def make_account(api_client, wishes=228):
    """A fresh account with enough wishes for realistic pull sequences."""
    payload = {
        "label": "instrumentation probe",
        "account": {
            "current_pity": 0,
            "character_guarantee": False,
            "wishes": wishes,
            "owned_characters": {"Vodynista": 0},
        },
        "settings": {"current_version": "7.0", "current_phase": 1},
        "goals": [
            {"character": "Vesna", "constellation": 0, "priority": 1},
            {"character": "Vodynista", "constellation": 0, "priority": 2},
            {"character": "Vesna", "constellation": 2, "priority": 3},
            {"character": "Tsaritsa", "constellation": 0, "priority": 4},
        ],
        "banners": [
            {"character": "Vesna", "version": "7.0", "phase": 1},
            {"character": "Tsaritsa", "version": "7.1", "phase": 1},
            {"character": "Vodynista", "version": "7.2", "phase": 1},
        ],
        "preferences": [
            {"character": "Vesna", "rank": 1, "constellation": 2, "weapon_refinement": 1},
            {"character": "Vesna", "rank": 2, "constellation": 1, "weapon_refinement": 1},
            {"character": "Vesna", "rank": 3, "constellation": 2},
            {"character": "Vesna", "rank": 4, "constellation": 1},
            {"character": "Vesna", "rank": 5, "constellation": 0},
        ],
    }
    created = api_client.post("/accounts", json=payload)
    assert created.status_code == 201, created.text
    return created.json()["id"]


def run_planner(api_client, account_id):
    recommendation = api_client.get(
        f"/accounts/{account_id}/planner/recommendation"
    )
    assert recommendation.status_code == 200, recommendation.text
    strategy = api_client.get(f"/accounts/{account_id}/planner/strategy")
    assert strategy.status_code == 200, strategy.text


def record_pull(api_client, account_id, character="Vesna", wishes_used=80):
    pull = api_client.post(
        f"/accounts/{account_id}/pull-result",
        json={"outcome": "featured", "wishes_used": wishes_used, "character": character},
    )
    assert pull.status_code == 200, pull.text


def cached_refresh(api_client, account_id, character="Vesna", wishes_used=80):
    return api_client.get(
        f"/accounts/{account_id}/planner/cached-refresh",
        params={
            "character": character,
            "outcome": "featured",
            "wishes_used": wishes_used,
        },
    )



def replace_runs(evidence, runs):
    """Return a copy of the evidence with a thinned run count."""
    from dataclasses import replace

    thinned_result = replace(
        evidence.result,
        runs=runs,
        histories=evidence.result.histories[:runs],
    )
    return replace(evidence, result=thinned_result, runs=runs)


class TestCachedRefreshInstrumentation:
    def test_retained_evidence_is_counted_as_retained(self, api_client, monkeypatch):
        """A refresh whose surviving sample clears MIN_CONDITIONED_RUNS is
        served from retained evidence and counted as such."""
        from api.planner_cache import planner_evidence_cache

        account_id = make_account(api_client)
        run_planner(api_client, account_id)

        # Simulate conditioning that retains the whole sample (runs far above
        # the threshold) so the endpoint takes the retained path.
        def retain_everything(account_id, **kwargs):
            with planner_evidence_cache._lock:
                return {
                    evidence.goal: evidence
                    for evidence in planner_evidence_cache._entries.values()
                    if evidence.account_id == account_id
                }

        monkeypatch.setattr(
            planner_evidence_cache, "condition_account", retain_everything
        )

        record_pull(api_client, account_id)
        response = cached_refresh(api_client, account_id)
        assert response.status_code == 200, response.text
        assert response.json()["evidence"], "evidence must be returned"

        metrics = cached_refresh_metrics
        assert metrics.total == 1
        assert metrics.retained == 1
        assert metrics.targeted_fallback == 0
        assert metrics.full_optimizer_fallback == 0
        assert metrics.fallback_resimulations == 0
        assert metrics.surviving_runs, "per-goal run counts must be recorded"
        assert all(runs >= metrics_threshold() for runs in metrics.surviving_runs)



    def test_thin_sample_is_counted_as_targeted_fallback(
        self, api_client, monkeypatch
    ):
        """When conditioning leaves fewer than MIN_CONDITIONED_RUNS matching
        histories, the refresh takes the targeted resimulation path and is
        counted separately from the retained path."""
        from api.planner_cache import planner_evidence_cache

        account_id = make_account(api_client)
        run_planner(api_client, account_id)

        # Shrink the cached sample so conditioning survives fewer runs than
        # the threshold - without touching the thresholds themselves.
        with planner_evidence_cache._lock:
            for key, evidence in tuple(planner_evidence_cache._entries.items()):
                planner_evidence_cache._entries[key] = replace_runs(evidence, 1)

        # And let the real conditioning pass that thinned evidence through
        # (a featured observation matching every retained history).
        def retain_thinned(account_id, **kwargs):
            with planner_evidence_cache._lock:
                return {
                    evidence.goal: evidence
                    for evidence in planner_evidence_cache._entries.values()
                    if evidence.account_id == account_id
                }

        monkeypatch.setattr(
            planner_evidence_cache, "condition_account", retain_thinned
        )

        record_pull(api_client, account_id)
        response = cached_refresh(api_client, account_id)
        assert response.status_code == 200, response.text

        metrics = cached_refresh_metrics
        assert metrics.total == 1
        assert metrics.retained == 0
        assert metrics.targeted_fallback == 1
        assert metrics.full_optimizer_fallback == 0
        assert metrics.fallback_resimulations >= 1
        assert min(metrics.surviving_runs) < metrics_threshold()

    def test_full_optimizer_fallback_is_counted_separately(self, api_client):
        """With no usable evidence at all (empty cache), the refresh returns
        nothing reusable and the request is counted as the full-optimizer
        fallback path."""
        from api.planner_cache import planner_evidence_cache

        account_id = make_account(api_client)
        run_planner(api_client, account_id)
        planner_evidence_cache.clear_account(account_id)

        record_pull(api_client, account_id)
        response = cached_refresh(api_client, account_id)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["evidence"] == []
        assert body["recommendation_probability"] is None

        metrics = cached_refresh_metrics
        assert metrics.total == 1
        assert metrics.retained == 0
        assert metrics.targeted_fallback == 0
        assert metrics.full_optimizer_fallback == 1

    def test_instrumentation_does_not_alter_the_returned_result(self, api_client):
        """The same refresh performed with the metrics disabled must produce an
        identical response - the recorder only observes."""
        from api.routers import planner as planner_router

        def run_flow():
            account_id = make_account(api_client)
            run_planner(api_client, account_id)
            record_pull(api_client, account_id)
            response = cached_refresh(api_client, account_id)
            assert response.status_code == 200
            return response.json()

        instrumented = run_flow()
        cached_refresh_metrics.reset()

        # The endpoint binds the recorder at import time, so the swap must
        # target the router's namespace to actually disable recording.
        spy = _DiscardingMetrics()
        real_recorder = planner_router.cached_refresh_metrics
        planner_router.cached_refresh_metrics = spy
        try:
            discarded = run_flow()
        finally:
            planner_router.cached_refresh_metrics = real_recorder

        # The swap really did take effect, otherwise this proves nothing.
        assert spy.refreshes_started == 1
        assert spy.evidence_recorded >= 1
        assert spy.refreshes_finished == 1
        # ... and dropping every measurement leaves the response untouched.
        assert discarded == instrumented



class _DiscardingMetrics:
    """Drop-all recorder that counts its own calls, so a test can prove the
    instrumentation was genuinely in the call path while changing nothing."""

    def __init__(self):
        self.refreshes_started = 0
        self.refreshes_finished = 0
        self.evidence_recorded = 0

    class _Recorder:
        def __init__(self, owner):
            self._owner = owner

        def record_goal_evidence(self, *args, **kwargs):
            self._owner.evidence_recorded += 1

        def count_fallback_resimulation(self):
            pass

        def count_candidate_fallback(self):
            pass

        def finish(self, *args, **kwargs):
            self._owner.refreshes_finished += 1
            return "retained"

    def start_refresh(self, **kwargs):
        self.refreshes_started += 1
        return self._Recorder(self)

    def reset(self):
        pass
