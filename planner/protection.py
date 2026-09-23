"""Basic protected-goal calculation (Design Document §13 step 5, §14).

Sequential independent-reserve approximation (Phase 3):

    Protected goals are unsatisfied goals whose character's next banner
    is strictly after the current banner. Each receives its own
    independent full-confidence reserve - the wish count that reaches the
    context threshold from pity 0 with no guarantee, FOR THAT GOAL'S OWN
    `copies_needed` (§10.4) - and each reserve is consumed completely
    before the next goal is considered. Future income
    (PlannerContext.income_credit) is credited as it arrives between
    banners.

    The approximation is deliberately conservative:

    * no pity or guarantee carries from the current banner into
      protected goals - §14 requires the final planner to account for
      carried pity, which the Phase 4 simulator does;
    * each reserve is spent in full even when luck would leave wishes
      over.

    §14 explicitly warns that this must not become a permanent
    architectural assumption; it exists so Phase 3 can reason about the
    roadmap before simulation exists.

CORRECTNESS NOTE (multi-copy reserves): each protected goal's reserve is
computed from ITS OWN `copies_needed` via `multi_copy_wishes_for_confidence`
- not a single shared single-copy value. An earlier version of this module
computed one `wishes_for_confidence(...)` value (a single-copy reserve) and
reused it for every protected goal regardless of how many copies that goal
actually needed. For a goal like Skirk C2 from an unowned account (3
copies), that undercounted the reserve dramatically - reporting "100%
confidence" at a budget whose true probability of completing 3 copies was
closer to 58% - because it was answering "how likely is one copy", not
"how likely is this goal's actual target" (§4.2, §10.4). See
tests/test_reserve_accounting.py for the regression test that pins this.
"""

from dataclasses import dataclass

from domain import Banner, Goal, TargetKind, WEAPON_EVENT_BANNER
from planner.banners import current_banner
from planner.context import PlannerContext
from planner.goals import GoalEvaluation, evaluate_goals
from probability import (
    multi_copy_cumulative_probability,
    multi_copy_wishes_for_confidence,
    weapon_cumulative_probability,
    weapon_wishes_for_confidence,
)


def weapon_goal_reserve(context: PlannerContext, copies: int) -> int:
    """A weapon goal's full reserve: the wish count that reaches the
    context's confidence threshold from a FRESH weapon-banner state.

    The fresh state (pity 0, no guarantee, no Fate Points) is the
    weapon-banner mirror of the character reserve's "pity 0, no
    guarantee" starting point (planner.protection): obtaining the current
    target ends that banner at zero pity, while future guarantee or
    Fate-Point advantages should not be required to justify spending now.
    The reserve uses the exact weapon probability engine
    (probability.weapon) - never the character probability API.
    """
    return weapon_wishes_for_confidence(
        context.confidence, 0, False, 0, WEAPON_EVENT_BANNER, copies=copies
    )


def weapon_goal_confidence(context: PlannerContext, copies: int, budget: int) -> float:
    """P(copies designated copies within `budget` wishes) from the fresh
    weapon state, via the exact weapon probability engine."""
    if copies <= 0:
        return 1.0
    if budget <= 0:
        return 0.0
    curve = weapon_cumulative_probability(
        budget, 0, False, 0, WEAPON_EVENT_BANNER, copies=copies
    )
    return float(curve[budget])


@dataclass(frozen=True)
class ProtectedGoalOutcome:
    """One protected goal's standing under a spend decision (§13).

    Attributes:
        goal: the protected roadmap objective.
        banner: the strictly-future banner where the goal can be pursued.
        budget_at_banner: the sequential independent-reserve budget that
            remains when this goal's banner arrives, under the worst-case
            model. This is not "wishes currently in the account": earlier
            reserves are assumed consumed in full and future income is
            credited on the way.
        required_wishes: the goal's full reserve: the wish count that
            reaches the context's confidence threshold from pity 0 with
            no guarantee, FOR THIS GOAL'S OWN copies_needed (§10.3,
            §10.4) - e.g. a C2 goal from an unowned character needs the
            3-copy reserve, not the 1-copy reserve.
        confidence: P(this goal's full target - all of copies_needed -
            within budget_at_banner wishes) under the same conservative
            starting state.
        meets_threshold: budget_at_banner >= required_wishes.
    """

    goal: Goal
    banner: Banner
    budget_at_banner: int
    required_wishes: int
    confidence: float
    meets_threshold: bool


def protected_goal_outcomes(
    context: PlannerContext, spent: int = 0, banner: Banner | None = None
) -> list[ProtectedGoalOutcome]:
    """Protected future goals after spending `spent` wishes on the
    current banner (§13 step 5).

    Goals whose next banner is the current banner itself (including
    blocked ones such as Vesna C2 behind Vesna C0) are current-banner
    business, not protection; goals with no upcoming banner cannot be
    scheduled and are excluded here while remaining visible through
    `evaluate_goals` - "not protectable" must not become "does not
    exist".

    Raises:
        ValueError: if `spent` is negative or exceeds account wishes.
    """
    if spent < 0:
        raise ValueError(f"spent must be non-negative, got {spent}")
    if spent > context.account.wishes:
        raise ValueError(
            f"spent ({spent}) cannot exceed account wishes "
            f"({context.account.wishes})"
        )

    current = banner if banner is not None else current_banner(context)
    schedulable: list[tuple[Banner, GoalEvaluation]] = []
    for evaluation in evaluate_goals(context):
        if evaluation.copies_needed == 0:
            continue
        banner = evaluation.next_banner
        if banner is None or banner.order_key <= current.order_key:
            continue
        schedulable.append((banner, evaluation))
    schedulable.sort(key=lambda item: (item[0].order_key, item[1].goal.priority))

    outcomes: list[ProtectedGoalOutcome] = []
    budget = context.account.wishes - spent
    credited_through = 0  # cumulative income credit already folded in
    for banner, evaluation in schedulable:
        credit = context.income_available_before(banner.version, banner.phase)
        budget += credit - credited_through
        credited_through = max(credited_through, credit)

        copies = evaluation.copies_needed
        if evaluation.goal.target.kind is TargetKind.WEAPON:
            # Weapon goals use the weapon banner's own probability
            # mechanics (§10 weapon): the exact designated-copy DP from a
            # fresh weapon state - never the character probability API.
            # The reserve is computed over the SAME account wish pool:
            # character and weapon goals compete for one budget here, so
            # no separate weapon reserve can silently exceed the account.
            required = weapon_goal_reserve(context, copies)
            confidence = weapon_goal_confidence(context, copies, budget)
        else:
            required = multi_copy_wishes_for_confidence(
                context.confidence, copies, 0, False, context.mechanics
            )
            confidence = float(
                multi_copy_cumulative_probability(budget, copies, 0, False, context.mechanics)[
                    budget
                ]
            )
        outcomes.append(
            ProtectedGoalOutcome(
                goal=evaluation.goal,
                banner=banner,
                budget_at_banner=budget,
                required_wishes=required,
                confidence=confidence,
                meets_threshold=budget >= required,
            )
        )
        budget = max(0, budget - required)  # the reserve is consumed in full
    return outcomes
