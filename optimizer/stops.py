"""Stop conditions (Design Document §1, §18 Phase 5).

A recommendation is only as good as the discipline attached to it. The
spend cap is not a suggestion: it is the boundary at which the roadmap's
protected goals keep their required confidence (§13 step 5, §14). Stopping
is also a planner action - the tool is re-run after every meaningful
account update (§2), so every stop condition ends in "re-run".
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class StopConditions:
    """When to stop pursuing (and what to do then) (§1, §18 Phase 5).

    Attributes:
        action: "pursue", "discretionary" or "skip" - matching
            Recommendation.action.
        outcome_label: the pursued outcome's display label, or None when
            skipping.
        spend_cap: the current-banner spend cap in wishes (0 when skipping).
        rules: the stop rules, most immediate first.
    """

    action: str
    outcome_label: str | None
    spend_cap: int
    rules: tuple[str, ...]


def for_pursue(outcome_label: str, spend_cap: int) -> StopConditions:
    """Stop rules for a pursue recommendation (§13, §14, §2)."""
    return StopConditions(
        action="pursue",
        outcome_label=outcome_label,
        spend_cap=spend_cap,
        rules=(
            f"Stop as soon as {outcome_label} is achieved - the pursuit "
            "is complete.",
            f"Stop after {spend_cap} wish(es) spent on this banner, even "
            "if the outcome is not achieved: the cap is what keeps every "
            "protected future goal at or above the required confidence "
            "(§13, §14). Exceeding it voids the recommendation.",
            "Re-run the planner after this banner resolves, or whenever "
            "the account, roadmap or assumptions change (§2).",
        ),
    )


def for_discretionary(
    outcome_label: str,
    spend_cap: int,
    outcome_probability: float,
    minimum_outcome_probability: float,
) -> StopConditions:
    """Stop rules for a disclosed-gamble recommendation (§14 step 4).

    Feasible - protection holds and the outcome is empirically reachable -
    but below `minimum_outcome_probability`: the cap is still the boundary
    that keeps every higher-priority protected goal safe, but reaching
    `outcome_label` at that cap is unlikely enough that it is not presented
    as an ordinary recommendation (optimizer.recommend module docstring).
    """
    return StopConditions(
        action="discretionary",
        outcome_label=outcome_label,
        spend_cap=spend_cap,
        rules=(
            f"This is not a recommendation to pursue {outcome_label}: at "
            f"{spend_cap} wish(es) - the largest spend that still protects "
            f"every higher-priority goal - the chance of achieving it is "
            f"only {outcome_probability:.0%}, below the "
            f"{minimum_outcome_probability:.0%} minimum for an ordinary "
            "recommendation.",
            f"Up to {spend_cap} wish(es) may still be spent on this banner "
            "without endangering any higher-priority protected goal, if "
            "you choose to take the gamble.",
            "Re-run the planner after this banner resolves, or whenever "
            "the account, roadmap or assumptions change (§2).",
        ),
    )


def for_skip(reason: str) -> StopConditions:
    """Stop rules for a do-not-spend recommendation (§1, §2)."""
    return StopConditions(
        action="skip",
        outcome_label=None,
        spend_cap=0,
        rules=(
            f"Do not spend on this banner now: {reason}",
            "Re-run the planner when income arrives, the account changes, "
            "or the threshold or roadmap is relaxed (§2).",
        ),
    )
