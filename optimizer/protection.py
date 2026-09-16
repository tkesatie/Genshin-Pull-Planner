"""Protected future goals, grouped per banner (Design Document §13, §14).

Classification reuses the planner's goal evaluations (§9) as the single
source of truth for "what remains" and "when is the next opportunity": a
goal is protected when it is unsatisfied and its next banner is strictly
after the current banner - the same rule planner.protection applies. An
agreement test pins the two classifications together so Phase 5 cannot
silently develop a different definition of "protected".

Grouping is a strategy-execution optimization only (§12): goals sharing a
banner are pursued through ONE plan entry whose target is the maximum
constellation among them - valid because constellation goals of one
character are strictly ordered (reaching C3 necessarily reaches C1).

GROUPING INVARIANT: grouping must never replace or merge the underlying
roadmap goals for outcome evaluation. The simulator reports satisfaction
per original Goal (§11); the optimizer reads those per-goal probabilities
back - the group only decides what single spend target executes them.
"""

from dataclasses import dataclass

from domain import Banner, Goal
from planner import PlannerContext, evaluate_goals
from planner.banners import current_banner


@dataclass(frozen=True)
class ProtectedGroup:
    """The roadmap goals pursued together on one future banner (§13).

    Attributes:
        banner: the strictly-future banner where these goals are pursued.
        goals: the protected roadmap goals on this banner, in priority
            order (§2). Evaluation stays per-goal (see module docstring).
        target_constellation: the plan target that satisfies every goal in
            the group: the maximum constellation among them (§12 - a
            desired resulting constellation, never a copy count).
        uncapped_budget: the strategic budget handed to the plan entry -
            sized so the entry's budget is never the limiting factor.
            This is NOT claimed to be the exact future balance: the
            simulator bounds actual spending to available wishes via
            min(budget, available) (§12), which is the real safety
            mechanism. The budget merely needs to be large enough that a
            protected goal never stops early for plan reasons.
    """

    banner: Banner
    goals: tuple[Goal, ...]
    target_constellation: int
    uncapped_budget: int


def protected_groups(context: PlannerContext) -> tuple[ProtectedGroup, ...]:
    """Protected goals grouped per banner, in chronological order (§13).

    Goals whose next banner is the current banner (including blocked ones
    such as Vesna C2 behind Vesna C0) are current-banner business, not
    protection; goals with no upcoming banner cannot be scheduled and are
    excluded while remaining visible through `evaluate_goals` - "not
    protectable" is not "does not exist" (§8).
    """
    current = current_banner(context)
    buckets: dict[Banner, list[Goal]] = {}
    for evaluation in evaluate_goals(context):
        if evaluation.copies_needed == 0:
            continue
        banner = evaluation.next_banner
        if banner is None or banner.order_key <= current.order_key:
            continue
        buckets.setdefault(banner, []).append(evaluation.goal)

    groups: list[ProtectedGroup] = []
    for banner in sorted(buckets, key=lambda item: item.order_key):
        goals = tuple(buckets[banner])  # evaluate_goals order: priority (§2)
        groups.append(
            ProtectedGroup(
                banner=banner,
                goals=goals,
                target_constellation=max(goal.constellation for goal in goals),
                uncapped_budget=context.account.wishes
                + context.income_credit(banner.version),
            )
        )
    return tuple(groups)
