"""Current-banner identification (Design Document §7, §18 Phase 3).

"Where is the user?" is planner input; which roadmap banners are available
at that position is identification logic. Chronological order is fundamental
(§7), but a single version/phase can contain multiple simultaneous
featured-character banners.
"""

from domain import Banner
from planner.context import PlannerContext


def available_banners(context: PlannerContext) -> tuple[Banner, ...]:
    """Return every roadmap banner available at the context's position.

    Multiple featured-character banners can share the same (version, phase)
    slot, and a version/phase can hold a character banner alongside a
    weapon banner (Phase 4). The planner must preserve all of them so
    higher layers can decide where the user's resources should go.
    """
    matches = [
        banner
        for banner in context.roadmap.banners
        if banner.version == context.current_version
        and banner.phase == context.current_phase
    ]
    # Target-based tie-break: weapon banners have no `.character`
    # accessor, so the unified ordering must go through target identity.
    return tuple(
        sorted(matches, key=lambda banner: (banner.order_key, banner.target.name))
    )


def current_banner(context: PlannerContext) -> Banner:
    """The roadmap banner at the context's (version, phase).

    This legacy singular helper remains strict: callers that still require
    exactly one banner must not silently choose when multiple are available.
    New decision-making code should use available_banners().
    """
    matches = available_banners(context)
    if not matches:
        scheduled = ", ".join(
            f"{banner.target.name} {banner.version}p{banner.phase}"
            for banner in context.roadmap.banners_in_chronological_order()
        )
        raise ValueError(
            f"no roadmap banner at {context.current_version} phase "
            f"{context.current_phase}; scheduled banners: [{scheduled}]"
        )
    if len(matches) > 1:
        listed = ", ".join(
            f"{banner.target.name} {banner.version}p{banner.phase}"
            for banner in matches
        )
        raise ValueError(
            f"ambiguous current banner: multiple roadmap banners at "
            f"{context.current_version} phase {context.current_phase} "
            f"({listed}); the planner will not silently choose"
        )
    return matches[0]
