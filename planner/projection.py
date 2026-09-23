"""Character-only projection of a unified planner context.

`simulation.engine` models the character event banner only: character
pity, guarantee and Capturing Radiance, character ownership, and
character goals. A roadmap that also contains weapon goals cannot be
executed by it (weapon goals have no `.character`), and forcing weapon
banners through the character simulator would misstate their mechanics -
exactly what the Phase 4 contract forbids ("do not force weapon
probability through the character probability API").

The strategy layer therefore runs the character Monte Carlo on a
PROJECTION of the context that contains only character goals and
character banners, and consumes weapon goals' probabilities from the
exact weapon engine (probability.weapon) instead.

One shared wish pool is preserved: weapon reserves are computed over the
same account wishes (planner.protection) and accounted analytically, so
character and weapon goals compete for one budget and no separate weapon
reserve can double-count the account's wishes.

Income timing is unaffected: income crediting is version-based
(PlannerContext.income_credit), so removing weapon banners from the
schedule cannot change when income becomes available to the character
roadmap.

Contexts without weapon goals and weapon banners are returned unchanged
(identity), so pure-character behavior is bit-for-bit identical to the
pre-unification engine.
"""

from dataclasses import replace

from domain import Roadmap, TargetKind
from planner.context import PlannerContext


def character_view(context: PlannerContext) -> PlannerContext:
    """The context restricted to character goals and character banners.

    Identity when the roadmap has no weapon content - the character
    simulator must behave exactly as before for character-only roadmaps.

    Slot-anchor retention: when the current (version, phase) slot contains
    ONLY weapon banners (a weapon-only current slot), the weapon banners at
    that slot are kept in the projection. Stripping them would leave the
    projected roadmap with no banner at the current position, and both
    `SpendPlan.require_valid_for` and `_banners_to_process` anchor through
    `available_banners`/`current_banner`, which raise in that case. The
    retained banners are never planned (plans contain character entries
    only), so the engine processes them as skipped BannerResults - income
    crediting stays version-based and identical.
    """
    goals = [
        goal
        for goal in context.roadmap.goals
        if goal.target.kind is TargetKind.CHARACTER
    ]
    current_slot_has_character_banner = any(
        banner.target.kind is TargetKind.CHARACTER
        and banner.version == context.current_version
        and banner.phase == context.current_phase
        for banner in context.roadmap.banners
    )
    banners = [
        banner
        for banner in context.roadmap.banners
        if banner.target.kind is TargetKind.CHARACTER
        or (
            not current_slot_has_character_banner
            and banner.version == context.current_version
            and banner.phase == context.current_phase
        )
    ]
    if len(goals) == len(context.roadmap.goals) and len(banners) == len(
        context.roadmap.banners
    ):
        return context
    return replace(context, roadmap=Roadmap(goals=goals, banners=banners))