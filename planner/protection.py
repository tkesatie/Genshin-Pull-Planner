"""Basic protected-goal calculation (Design Document §13 step 5, §14).

Sequential independent-reserve approximation (Phase 3):

    Protected goals are unsatisfied goals whose character's next banner
    is strictly after the current banner. Each receives its own
    independent full-confidence reserve - `wishes_for_confidence` at the
    context threshold starting from pity 0 with no guarantee - and each
    reserve is consumed completely before the next goal is considered.
    Future income (PlannerContext.income_credit) is credited as it
    arrives between banners.

    The approximation is deliberately conservative:

    * no pity or guarantee carries from the current banner into
      protected goals - §14 requires the final planner to account for
      carried pity, which the Phase 4 simulator will do;
    * each reserve is spent in full even when luck would leave wishes
      over.

    §14 explicitly warns that this must not become a permanent
    architectural assumption; it exists so Phase 3 can reason about the
    roadmap before simulation exists.
"""

from dataclasses import dataclass

from domain import Banner, Goal
from planner.banners import current_banner
from planner.context import PlannerContext
from planner.goals import GoalEvaluation, evaluate_goals
from probability import cumulative_probability, wishes_for_confidence


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
            no guarantee (§10.3).
        confidence: P(featured copy within budget_at_banner wishes) under
            the same conservative starting state.
        meets_threshold: budget_at_banner >= required_wishes.
    """

    goal: Goal
    banner: Banner
    budget_at_banner: int
    required_wishes: int
    confidence: float
    meets_threshold: bool


def protected_goal_outcomes(
    context: PlannerContext, spent: int = 0
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

    current = current_banner(context)
    schedulable: list[tuple[Banner, GoalEvaluation]] = []
    for evaluation in evaluate_goals(context):
        if evaluation.copies_needed == 0:
            continue
        banner = evaluation.next_banner
        if banner is None or banner.order_key <= current.order_key:
            continue
        schedulable.append((banner, evaluation))
    schedulable.sort(key=lambda item: (item[0].order_key, item[1].goal.priority))

    required = wishes_for_confidence(
        context.confidence, 0, False, context.mechanics
    )

    outcomes: list[ProtectedGoalOutcome] = []
    budget = context.account.wishes - spent
    credited_through = 0  # cumulative income credit already folded in
    for banner, evaluation in schedulable:
        credit = context.income_credit(banner.version)
        budget += credit - credited_through
        credited_through = credit
        outcomes.append(
            ProtectedGoalOutcome(
                goal=evaluation.goal,
                banner=banner,
                budget_at_banner=budget,
                required_wishes=required,
                confidence=float(
                    cumulative_probability(budget, 0, False, context.mechanics)[
                        budget
                    ]
                ),
                meets_threshold=budget >= required,
            )
        )
        budget = max(0, budget - required)  # the reserve is consumed in full
    return outcomes