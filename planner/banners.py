"""Banner opportunity identification (Design Document §7, §18 Phase 3-6).

"Where is the user?" is planner input; which roadmap banners are available at
that position, which are still ahead, and when a target's next opportunity
falls are identification logic. Chronological order is fundamental (§7), but a
single version/phase can contain several simultaneous featured banners, and a
character banner can coexist with a weapon banner in the same slot.

Three questions, three answers:

    available_banners()   what can be pulled RIGHT NOW (every opportunity)
    upcoming_banners()    what comes AFTER the current position, chronologically
    next_banner_for()     when a specific target's next opportunity is

`current_banner()` is the legacy strict singular helper: it refuses an
ambiguous slot instead of silently choosing one banner. Decision-making code
uses `available_banners()`.

Nothing here goes through `Banner.character`: a weapon banner has no such
accessor, so every match is made on unified target identity (`GoalTarget`).
"""

from datetime import datetime

from domain import Banner, GoalTarget, position_key
from planner.context import PlannerContext


def _opportunity_key(banner: Banner) -> tuple:
    """Order opportunities by position, then by target name.

    Weapon banners have no `.character` accessor, so the tie-break inside a
    slot goes through unified target identity rather than a character name.
    """
    return (banner.order_key, banner.target.name)


def available_banners(context: PlannerContext) -> tuple[Banner, ...]:
    """Return every roadmap banner available at the context's position.

    Multiple featured-character banners can share the same (version, phase)
    slot, and a version/phase can hold a character banner alongside a
    weapon banner (Phase 4). The planner must preserve all of them so
    higher layers can decide where the user's resources should go.
    """
    position = current_position(context)
    matches = [
        banner for banner in context.roadmap.banners if banner.order_key == position
    ]
    return tuple(sorted(matches, key=_opportunity_key))


def current_position(context: PlannerContext) -> tuple[int, int, int]:
    """Numeric ordering key of the planner's position: (major, minor, phase).

    The planner position is planner input (§7): a version compared
    numerically so "7.9" precedes "7.10", plus a 1-based phase.
    """
    return position_key(context.current_version, context.current_phase)


def position_anchor(context: PlannerContext) -> Banner:
    """A banner standing for the current (version, phase) position.

    Several callers need only the *position* of the current slot - to tell a
    strictly-future opportunity from a current one, or to start a
    chronological walk. Every banner in a slot shares that position, so an
    ambiguous slot (two simultaneous featured banners) is not ambiguous for
    these callers; they compare `order_key` only.

    Strict about an EMPTY slot: it raises through `current_banner`, so a
    position with no roadmap banner is still reported rather than silently
    treated as "everything is upcoming".
    """
    matches = available_banners(context)
    return matches[0] if matches else current_banner(context)


def upcoming_banners(context: PlannerContext) -> tuple[Banner, ...]:
    """Every roadmap banner strictly after the current (version, phase).

    The chronological counterpart of `available_banners`: a different banner
    in the SAME slot is an alternative current opportunity, not a future one.
    Derived from the numeric position rather than from a selected banner, so
    it is target-agnostic and still meaningful at a position whose slot has
    no banner at all.
    """
    position = current_position(context)
    return tuple(
        banner
        for banner in context.roadmap.banners_in_chronological_order()
        if banner.order_key > position
    )


def next_banner_for(
    context: PlannerContext, target: GoalTarget, *, after: Banner | None = None
) -> Banner | None:
    """The first opportunity for `target` at or after a position (§7).

    The default position is the planner's own, and the comparison is
    at-or-after: a target whose banner is in the current slot is available
    NOW, which is its next opportunity. Target-based (character or weapon) -
    matching goes through unified target identity, never through
    `Banner.character`, which a weapon banner does not have. Returns None
    when the target has no further opportunity, because "not scheduled" is
    not "does not exist" (§8).
    """
    floor = (
        current_position(context) if after is None else after.order_key
    )
    return next(
        (
            banner
            for banner in context.roadmap.banners_for_target(target)
            if banner.order_key >= floor
        ),
        None,
    )


def banners_active_at(context: PlannerContext, moment: datetime) -> tuple[Banner, ...]:
    """Every roadmap banner live at the real instant `moment`, chronological.

    The timestamp-based counterpart of `available_banners`, for the day the
    planner is given a real "now": (version, phase) stays the primary
    deterministic planning input, while this answers what a clock says is on
    screen. Only banners carrying `start`/`end` can be judged - a banner
    without dates is skipped, never assumed active. Empty when nothing
    (dated) covers the instant. Simultaneous opportunities come back in the
    same order `available_banners` returns them.

    Raises:
        ValueError: if `moment` is naive. The planner compares instants and
            will not assume UTC or local time for a wall-clock value.
    """
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(
            f"moment must be timezone-aware (include a UTC offset), got "
            f"naive {moment.isoformat()!r}"
        )
    return tuple(
        sorted(
            (
                banner
                for banner in context.roadmap.banners
                if banner.has_dates and banner.is_active_at(moment)
            ),
            key=_opportunity_key,
        )
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
