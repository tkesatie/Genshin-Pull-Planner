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

from optimizer.outcomes import OutcomeOption


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


def protected_groups(context: PlannerContext, banner: Banner | None = None) -> tuple[ProtectedGroup, ...]:
    """Protected goals grouped per banner, in chronological order (§13).

    Goals whose next banner is the current banner (including blocked ones
    such as Vesna C2 behind Vesna C0) are current-banner business, not
    protection; goals with no upcoming banner cannot be scheduled and are
    excluded while remaining visible through `evaluate_goals` - "not
    protectable" is not "does not exist" (§8).
    """
    current = banner if banner is not None else current_banner(context)
    buckets: dict[Banner, list[Goal]] = {}
    for evaluation in evaluate_goals(context):
        if evaluation.copies_needed == 0:
            continue
        banner = evaluation.next_banner
        # A different banner in the same version/phase is an alternative
        # current opportunity. It must remain in the candidate plan so a
        # higher-priority goal there can constrain this decision. The
        # selected banner itself is current-banner business, not protection.
        if banner is None or banner.order_key < current.order_key or banner == current:
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
                uncapped_budget=(
                    context.account.wishes
                    if banner.order_key == current.order_key
                    else context.account.wishes + context.income_credit(banner.version)
                ),
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

    This is the fallback anchor for a candidate whose specific outcome has
    no roadmap counterpart; see `priority_for_outcome` for the per-outcome
    anchor `optimizer.recommend` actually gates on.

    Raises:
        ValueError: via `current_banner`, when the context has no roadmap
            banner at its (version, phase).
    """
    priorities = [
        evaluation.goal.priority for evaluation in actionable_goals(context)
    ]
    return min(priorities) if priorities else None


def priority_for_outcome(context: PlannerContext, outcome: OutcomeOption) -> int | None:
    """The priority anchor one specific candidate outcome is judged against (§2).

    `current_goal_priority` anchors the whole decision to a single active
    goal, which is correct as long as every candidate outcome represents
    the same objective. It stops being correct once a preference chain
    reaches past that goal to a *deeper* constellation of the same
    character (optimizer.outcomes: reaching C2 necessarily reaches C0).
    The roadmap can legitimately rank that deeper reach as its own,
    lower-priority objective - the design document's own example is
    exactly this pattern:

        Priority 1 -> Vesna C0
        Priority 2 -> Vodynista C0
        Priority 3 -> Vesna C2

    Here, pursuing the C2 *outcome* should be judged against priority 3,
    not priority 1, so that Vodynista (priority 2) can gate the reach to
    C2 without gating the reach to C0.

    When the outcome's exact (character, constellation) matches a roadmap
    Goal, that goal's own priority is the anchor. Preferences and
    priorities are still separate concepts (§2, §15): a preference chain
    with no roadmap counterpart at all falls back to
    `current_goal_priority`, unchanged from before this existed.
    """
    for goal in context.roadmap.goals:
        if goal.character == outcome.character and goal.constellation == outcome.constellation:
            return goal.priority
    return current_goal_priority(context)


_AUTO = object()


def constraining_goals(
    context: PlannerContext, priority: int | None = _AUTO  # type: ignore[assignment]
) -> frozenset[Goal]:
    """The protected goals whose probability gates a decision (§2).

    §2 frames the tradeoff precisely: "If I spend wishes pursuing Tsaritsa,
    what happens to my probability of satisfying the *higher-priority*
    future objectives?" A protected goal therefore constrains spending only
    when its priority is higher (a lower number) than the decision's own
    priority. A goal the user ranked below the current objective keeps its
    full place in the strategy - it is still pursued by candidate plans and
    still reported - but its probability can no longer force the current
    spend down to nothing.

    `priority` is the anchor to gate against. Left unset, it is computed
    from `current_goal_priority` (the whole-decision anchor, as before);
    callers evaluating one specific candidate outcome should instead pass
    `priority_for_outcome(context, outcome)`, so a deeper reach on the
    same character can be gated by its own, lower-priority objective
    without weakening protection for the base objective (see
    `priority_for_outcome`). Passing `None` explicitly means "unanchored":
    every protected goal constrains, the same conservative reading
    `current_goal_priority` documents for its own None.

    This is a Phase 5 filter over `protected_groups`; the classification
    itself (and planner.protection's Phase 3 contract) is untouched, so the
    two layers keep one definition of "protected" while only the decision
    layer knows about priority.
    """
    protected = [
        goal for group in protected_groups(context) for goal in group.goals
    ]
    if priority is _AUTO:
        priority = current_goal_priority(context)
    if priority is None:
        return frozenset(protected)
    return frozenset(goal for goal in protected if goal.priority < priority)
