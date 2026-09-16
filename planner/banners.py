"""Current-banner identification (Design Document §7, §18 Phase 3).

"Where is the user?" is planner input (`PlannerContext`); which roadmap
banner that corresponds to is identification logic. Chronological order
is fundamental (§7) but identification itself is positional: an exact
(version, phase) match.
"""

from domain import Banner
from planner.context import PlannerContext


def current_banner(context: PlannerContext) -> Banner:
    """The roadmap banner at the context's (version, phase).

    Raises:
        ValueError: if no roadmap banner matches, or if several do
            (e.g. two featured characters in the same slot): the Phase 3
            planner models one current banner and will not silently
            choose.
    """
    matches = [
        banner
        for banner in context.roadmap.banners
        if banner.version == context.current_version
        and banner.phase == context.current_phase
    ]
    if not matches:
        scheduled = ", ".join(
            f"{banner.character} {banner.version}p{banner.phase}"
            for banner in context.roadmap.banners_in_chronological_order()
        )
        raise ValueError(
            f"no roadmap banner at {context.current_version} phase "
            f"{context.current_phase}; scheduled banners: [{scheduled}]"
        )
    if len(matches) > 1:
        listed = ", ".join(
            f"{banner.character} {banner.version}p{banner.phase}"
            for banner in matches
        )
        raise ValueError(
            f"ambiguous current banner: multiple roadmap banners at "
            f"{context.current_version} phase {context.current_phase} "
            f"({listed}); the planner will not silently choose"
        )
    return matches[0]