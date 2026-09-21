"""TEMPORARY instrumentation for the /planner/cached-refresh path.

This module exists purely to measure how often pull-result refreshes can be
served from retained conditioned evidence versus needing the targeted
fallback resimulation, before deciding whether MIN_CONDITIONED_RUNS is an
appropriate threshold.

It records counters only - it never influences conditioning, thresholds,
recommendation results, or any stored state. Delete this module (and its
call sites in api/routers/planner.py) once the measurement data has been
collected.
"""

import logging
import threading

logger = logging.getLogger("planner.cached_refresh")


class RefreshRequestRecorder:
    """Per-request collector for one cached-refresh call."""

    def __init__(self, metrics, character, outcome, wishes_used, threshold):
        self._metrics = metrics
        self.character = character
        self.outcome = outcome
        self.wishes_used = wishes_used
        self.threshold = threshold
        self.goal_runs = []  # (conditioned_runs, has_plan) per goal evidence
        self.fallback_resimulations = 0
        self.candidate_fallbacks = 0

    def record_goal_evidence(self, goal_label, conditioned_runs, has_plan):
        """Log and retain the surviving run count for one goal's evidence."""
        self.goal_runs.append((conditioned_runs, has_plan))
        if conditioned_runs >= self.threshold:
            path = "retained"
        elif has_plan:
            path = "targeted_fallback"
        else:
            path = "no_plan"
        logger.debug(
            "Cached refresh: observed=%s, wishes_used=%d, goal=%s, "
            "conditioned_runs=%d, threshold=%d, path=%s",
            self.outcome,
            self.wishes_used,
            goal_label,
            conditioned_runs,
            self.threshold,
            path,
        )

    def count_fallback_resimulation(self):
        """One goal needed a CONDITIONED_FALLBACK_RUNS resimulation."""
        self.fallback_resimulations += 1

    def count_candidate_fallback(self):
        """The conditioned recommendation needed a fresh candidate run."""
        self.candidate_fallbacks += 1

    def finish(self, evidence_count, has_recommendation):
        """Classify and record the completed refresh."""
        if not evidence_count and not has_recommendation:
            path = "full_optimizer_fallback"
        elif self.fallback_resimulations or self.candidate_fallbacks:
            path = "targeted_fallback"
        else:
            path = "retained"
        self._metrics._record_request(self, path)
        logger.debug(
            "Cached refresh: observed=%s, wishes_used=%d, path=%s, "
            "fallback_resims=%d, candidate_fallbacks=%d, evidence=%d",
            self.outcome,
            self.wishes_used,
            path,
            self.fallback_resimulations,
            self.candidate_fallbacks,
            evidence_count,
        )
        return path


class CachedRefreshMetrics:
    """Aggregate counters for cached-refresh paths. Thread-safe."""

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.total = 0
            self.retained = 0
            self.targeted_fallback = 0
            self.full_optimizer_fallback = 0
            self.fallback_resimulations = 0
            self.candidate_fallbacks = 0
            self.surviving_runs = []  # conditioned_runs per goal evidence

    def start_refresh(self, *, character, outcome, wishes_used, threshold):
        return RefreshRequestRecorder(
            self, character, outcome, wishes_used, threshold
        )

    def _record_request(self, request, path):
        with self._lock:
            self.total += 1
            if path == "retained":
                self.retained += 1
            elif path == "targeted_fallback":
                self.targeted_fallback += 1
            else:
                self.full_optimizer_fallback += 1
            self.fallback_resimulations += request.fallback_resimulations
            self.candidate_fallbacks += request.candidate_fallbacks
            self.surviving_runs.extend(
                runs for runs, _ in request.goal_runs if runs is not None
            )

    def format_summary(self):
        """The temporary development log block, e.g.:

        Cached refresh:
          total: 100
          retained evidence: 94
          targeted fallback: 6
          full optimizer fallback: 0
        """
        with self._lock:
            total = self.total
            retained = self.retained
            targeted = self.targeted_fallback
            full = self.full_optimizer_fallback
            runs = sorted(self.surviving_runs)

        lines = [
            "Cached refresh:",
            f"  total: {total}",
            f"  retained evidence: {retained}",
            f"  targeted fallback: {targeted}",
            f"  full optimizer fallback: {full}",
        ]
        if runs:
            def pct(values, fraction):
                return values[int(fraction * (len(values) - 1))]

            lines.append(
                "  surviving conditioned runs: "
                f"min={runs[0]} p25={pct(runs, 0.25)} "
                f"median={pct(runs, 0.5)} p75={pct(runs, 0.75)} "
                f"max={runs[-1]} (n={len(runs)})"
            )
        return "\n".join(lines)


cached_refresh_metrics = CachedRefreshMetrics()
