"""Spending plans (Design Document §12).

A strategy is a sequence of spending behavior across the roadmap (§12).
Phase 4 deliberately models it as the smallest structure that can be
*executed* - one decision per banner - because the logic that generates and
compares strategies is the Phase 5 optimizer (§13). The simulator executes
plans faithfully; it never chooses between them.

Terminology invariant (§4.2, §12):

    `target_constellation` is a desired *resulting constellation*, never a
    number of copies to pull. The simulator derives `copies_needed` from the
    simulated account state when the banner arrives (§10.4):

        Owned: C-1 (not owned)   Plan: C2   ->  3 copies
        Owned: C0                Plan: C2   ->  2 copies
        Owned: C2                Plan: C1   ->  0 copies (nothing spent)

    An account already at or above the target therefore pulls nothing.

Budget semantics (stated once, used everywhere):

    budget        maximum wishes this strategy permits spending
    available     wishes actually in the simulated account
    wishes_spent  min(budget, available, wishes needed to finish the target)

A budget is a cap, not a commitment: an unlucky run stops when the cap is
hit, a lucky run stops when the target is reached, and a cap beyond the
account's wishes never overspends them.

An absent plan entry means skip, not stop (§11): the banner is still
processed chronologically - income credited, pity/guarantee carried - it
just spends nothing and remains in the run's history.
"""

from dataclasses import dataclass

from domain import Banner
from planner.banners import available_banners, current_banner
from planner.context import PlannerContext


@dataclass(frozen=True)
class PlannedSpend:
    """One banner's spending decision (§12).

    Attributes:
        banner: the roadmap banner this entry applies to.
        target_constellation: the desired resulting constellation for the
            banner's character ("pursue C2" -> 2). A resulting constellation,
            never a copy count (§4.2): copies are derived from the simulated
            account (§10.4).
        budget: maximum wishes this entry permits spending - a cap, not a
            commitment (see module docstring).
    """

    banner: Banner
    target_constellation: int
    budget: int

    def __post_init__(self) -> None:
        if self.target_constellation < 0:
            raise ValueError(
                f"target_constellation must be >= 0, got {self.target_constellation}"
            )
        if self.budget < 0:
            raise ValueError(f"budget must be >= 0, got {self.budget}")


@dataclass(frozen=True)
class SpendPlan:
    """A strategy expressed as per-banner spending decisions (§12).

    Banners without an entry are skipped by the simulation (see module
    docstring): processed for income and pity/guarantee, spending nothing.
    """

    entries: tuple[PlannedSpend, ...] = ()
    # Optional cap shared by all planned spending on the current phase. This
    # lets a multi-banner current decision (e.g. Vodynista + Vesna) consume
    # one finite phase-1 budget rather than giving each banner its own cap.
    shared_current_budget: int | None = None

    def __post_init__(self) -> None:
        if self.shared_current_budget is not None and self.shared_current_budget < 0:
            raise ValueError(
                f"shared_current_budget must be >= 0, got {self.shared_current_budget}"
            )
        seen: set[Banner] = set()
        duplicates: set[Banner] = set()
        for entry in self.entries:
            if entry.banner in seen:
                duplicates.add(entry.banner)
            seen.add(entry.banner)
        if duplicates:
            listed = ", ".join(
                f"{banner.character} {banner.version}p{banner.phase}"
                for banner in sorted(duplicates, key=lambda item: item.order_key)
            )
            raise ValueError(f"duplicate plan entries for banner(s): [{listed}]")

    def entry_for(self, banner: Banner) -> PlannedSpend | None:
        """The plan entry for `banner`, or None when the banner is skipped."""
        for entry in self.entries:
            if entry.banner == banner:
                return entry
        return None

    def require_valid_for(self, context: PlannerContext) -> None:
        """Reject entries the simulator could never execute (§7, §11).

        Every entry's banner must exist in the roadmap and be chronologically
        at or after the current banner: the simulation walks the roadmap from
        the current banner forward, so a past or unknown banner is a plan
        that cannot be executed, not one to silently ignore.

        Raises:
            ValueError: if an entry references a banner outside the roadmap
                or chronologically before the current banner (via
                `current_banner`, also raised when the context has no
                matching roadmap banner at all).
        """
        matches = available_banners(context)
        if not matches:
            current = current_banner(context)
        else:
            current = matches[0]
        known = set(context.roadmap.banners)
        for entry in self.entries:
            banner = entry.banner
            label = f"{banner.character} {banner.version}p{banner.phase}"
            if banner not in known:
                raise ValueError(
                    f"plan entry for {label} is not a roadmap banner; the "
                    "simulator executes the roadmap's banners only (§7)"
                )
            if banner.order_key < current.order_key:
                raise ValueError(
                    f"plan entry for {label} is chronologically before the "
                    f"current banner ({current.character} "
                    f"{current.version}p{current.phase}); plans apply from "
                    "the current banner forward (§7)"
                )
