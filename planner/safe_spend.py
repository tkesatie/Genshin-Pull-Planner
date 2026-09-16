"""Safe-spend approximation (Design Document §14, §18 Phase 3).

safe_spend is the largest spend on the current banner that keeps every
protected goal at or above the confidence threshold, under the sequential
independent-reserve approximation (planner.protection).

Closed form:

    safe_spend = clamp(W - max_i( sum_{j<=i} required_j - credit(v_i) ), 0, W)

where W is the account's wishes, required_j the full reserve of protected
goal j, and credit(v_i) the cumulative future income credited through
goal i's banner version. This is mathematically equivalent to walking
the protection model over spends ONLY under the Phase 3
independent-reserve assumptions - every protected goal shares the same
conservative starting state (pity 0, no guarantee) and the same
required_wishes. Once Phase 4 introduces pity/guarantee propagation and
multi-copy targets, the equivalence disappears and safe_spend must come
from simulation (§14).

An underfunded roadmap is reported as safe_spend 0 - "do not spend" is a
legitimate answer, not a negative budget (§1: the planner reasons in
probabilities, not affordability). When even zero spending cannot
protect the roadmap, safe_spend is 0 and every spend is unsafe.
"""

from planner.context import PlannerContext
from planner.protection import protected_goal_outcomes


def safe_spend(context: PlannerContext) -> int:
    """Largest current-banner spend that protects all protected goals
    (§14).

    With no protected goals the bound is the entire wish pool: nothing
    future constrains spending within this approximation. The result is
    clamped to [0, account.wishes]; see the module docstring for what a
    floor of 0 means.
    """
    wishes = context.account.wishes
    outcomes = protected_goal_outcomes(context, spent=0)
    if not outcomes:
        return wishes

    worst_deficit = 0
    for index, outcome in enumerate(outcomes, start=1):
        cumulative_required = index * outcome.required_wishes
        deficit = cumulative_required - context.income_credit(
            outcome.banner.version
        )
        worst_deficit = max(worst_deficit, deficit)
    return max(0, min(wishes, wishes - worst_deficit))