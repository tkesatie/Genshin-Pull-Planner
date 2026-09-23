"""Goal matching and dependencies (Design Document §9, §18 Phase 3).

Three explicit levels of attention, kept separate:

    evaluate_goals()             all roadmap goals and their states
    relevant_goal_evaluations()  goals matching the current banner's
                                 featured character
    actionable_goals()           relevant goals in state ACTIVE

A goal is:

    SATISFIED  nothing remains (§6: copies_needed == 0)
    BLOCKED    an unsatisfied lower-constellation goal for the same
               character exists (§9: Vesna C2 cannot be treated as an
               independent objective while Vesna C0 is incomplete)
    ACTIVE     unsatisfied and unblocked

Dependency is constellation-based and deliberately independent of
priorities: priorities order roadmap protection (§2), they do not change
pull order on a banner. The lowest unsatisfied constellation for a
character is therefore the only active milestone for it; only degenerate
duplicate goals can leave more than one active goal per character, and
the spend table refuses to choose between them.
"""

from dataclasses import dataclass
from enum import Enum

from domain import Banner, Goal, copies_needed_for
from domain.targets import GoalTarget
from planner.banners import available_banners, current_banner
from planner.context import PlannerContext


class GoalState(Enum):
    """Planner state of one goal (§9)."""

    SATISFIED = "satisfied"
    ACTIVE = "active"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class GoalEvaluation:
    """One goal's planner state (§9).

    Attributes:
        goal: the roadmap objective.
        copies_needed: raw remaining copies (§6). For a blocked goal this
            is the full remaining count from current ownership (e.g.
            Vesna C2 while unowned -> 3); incremental-after-blocker
            accounting is multi-copy territory (§10.4, Phase 4).
        state: SATISFIED / ACTIVE / BLOCKED (see module docstring).
        blocked_by: the unsatisfied lower-constellation goal with the
            largest constellation blocking this one; None unless BLOCKED.
        next_banner: first banner for this target at-or-after the
            current banner; None when the target has no upcoming banner
            in the roadmap. Such a goal stays visible here - it is
            merely not schedulable, never silently dropped (§8).
    """

    goal: Goal
    copies_needed: int
    state: GoalState
    blocked_by: Goal | None
    next_banner: Banner | None


def _next_banner(
    context: PlannerContext, target: GoalTarget, banner: Banner | None = None
) -> Banner | None:
    """First banner for the target at-or-after the selected current slot.

    Target-based (character or weapon): a weapon goal's next opportunity
    is its weapon banner, matched by unified target identity - never by
    the character accessor, which weapon goals do not have.
    """
    current = banner
    if current is None:
        matches = available_banners(context)
        if not matches:
            current = current_banner(context)
        else:
            current = matches[0]
    upcoming = [
        banner
        for banner in context.roadmap.banners_for_target(target)
        if banner.order_key >= current.order_key
    ]
    return upcoming[0] if upcoming else None


def _evaluate_one(
    context: PlannerContext, goal: Goal, all_goals: list[Goal]
) -> GoalEvaluation:
    copies_needed = copies_needed_for(context.account, goal)
    blockers = [
        other
        for other in all_goals
        if other.target == goal.target
        and other.level < goal.level
        and copies_needed_for(context.account, other) > 0
    ]
    blocked_by = (
        max(blockers, key=lambda blocker: blocker.level)
        if blockers
        else None
    )
    if copies_needed == 0:
        state = GoalState.SATISFIED
    elif blocked_by is not None:
        state = GoalState.BLOCKED
    else:
        state = GoalState.ACTIVE
    return GoalEvaluation(
        goal=goal,
        copies_needed=copies_needed,
        state=state,
        blocked_by=blocked_by,
        next_banner=_next_banner(context, goal.target),
    )


def evaluate_goal(context: PlannerContext, goal: Goal) -> GoalEvaluation:
    """Evaluate one goal against the context (§9)."""
    return _evaluate_one(context, goal, context.roadmap.goals_in_priority_order())


def evaluate_goals(context: PlannerContext) -> list[GoalEvaluation]:
    """Every roadmap goal in priority order, with its planner state (§9)."""
    all_goals = context.roadmap.goals_in_priority_order()
    return [_evaluate_one(context, goal, all_goals) for goal in all_goals]


def relevant_goal_evaluations(
    context: PlannerContext, banner: Banner | None = None
) -> list[GoalEvaluation]:
    """Goals matching the current banner's featured character (§9).

    For the doc example at Vesna 7.0 this is Vesna C0 and Vesna C2: the
    C2 goal is relevant but (while C0 is incomplete) blocked, not
    actionable.
    """
    if banner is None:
        matches = available_banners(context)
        if len(matches) == 1:
            banner = matches[0]
        else:
            targets = {item.target for item in matches}
            return [
                evaluation
                for evaluation in evaluate_goals(context)
                if evaluation.goal.target in targets
            ]
    return [
        evaluation
        for evaluation in evaluate_goals(context)
        if evaluation.goal.target == banner.target
    ]


def actionable_goals(
    context: PlannerContext, banner: Banner | None = None
) -> list[GoalEvaluation]:
    """Relevant goals in state ACTIVE: what the user can act on now (§9).

    A relevant goal that is blocked (Vesna C2 behind Vesna C0) or already
    satisfied is not actionable, and unsatisfied goals for other
    characters are not actionable on this banner.
    """
    return [
        evaluation
        for evaluation in relevant_goal_evaluations(context, banner)
        if evaluation.state is GoalState.ACTIVE
    ]