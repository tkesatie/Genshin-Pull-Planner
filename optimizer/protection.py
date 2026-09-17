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

CLASSIFICATION vs. CONSTRAINT: which goals are *protected* (§13 step 5,
above) and which protected goals may *constrain the current decision* (§2)
are deliberately separate questions. §2 asks what spending now does to the
"higher-priority future objectives", so a protected goal the user ranked
below the objective the current banner serves does not get to veto spending
on it - otherwise "Priority 1 → Vesna C0" would mean nothing when Vesna is
available today and Tsaritsa is not. `constraining_goals` is that filter; it
is applied here in Phase 5 and never inside planner.protection, whose
Phase 3 contract is priority-independent by design.
"""

from dataclasses import dataclass

from domain import Banner, Goal
from planner import PlannerContext, actionable_goals, evaluate_goals
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


def current_goal_priority(context: PlannerContext) -> int | None:
    """Priority the current decision is anchored to (§2, §9).

    The objective a banner decision serves is the best-ranked ACTIVE goal
    on the current banner: the current banner's character's highest-priority
    actionable goal. Priorities are a strict protection order (§2), so this
    is the priority any future goal must beat to be allowed to constrain
    the spend.

    None when the current banner has no ACTIVE roadmap goal - a decision
    made from a preference chain alone is not anchored to a position in the
    roadmap, and every protected goal constrains it (the conservative
    reading; §13 step 5 is not weakened by an unanchored decision).

    Raises:
        ValueError: via `current_banner`, when the context has no roadmap
            banner at its (version, phase).
    """
    priorities = [
        evaluation.goal.priority for evaluation in actionable_goals(context)
    ]
    return min(priorities) if priorities else None


def constraining_goals(context: PlannerContext) -> frozenset[Goal]:
    """The protected goals whose probability gates the current decision (§2).

    §2 frames the tradeoff precisely: "If I spend wishes pursuing Tsaritsa,
    what happens to my probability of satisfying the *higher-priority*
    future objectives?" A protected goal therefore constrains spending only
    when its priority is higher (a lower number) than the current decision's
    own priority (`current_goal_priority`). A goal the user ranked below the
    current objective keeps its full place in the strategy - it is still
    pursued by candidate plans and still reported - but its probability can
    no longer force the current spend down to nothing.

    This is a Phase 5 filter over `protected_groups`; the classification
    itself (and planner.protection's Phase 3 contract) is untouched, so the
    two layers keep one definition of "protected" while only the decision
    layer knows about priority.
    """
    protected = [
        goal for group in protected_groups(context) for goal in group.goals
    ]
    priority = current_goal_priority(context)
    if priority is None:
        return frozenset(protected)
    return frozenset(goal for goal in protected if goal.priority < priority)
