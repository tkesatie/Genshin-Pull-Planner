"""Available outcomes for the current banner (Design Document §13 step 2, §15).

Preferences answer "if I'm going after this character, what outcomes do I
prefer?" (§15); the optimizer turns the preference chain of the *current
banner's* character into an ordered list of outcomes it is allowed to
pursue. It never invents an outcome the user did not define (§2, §13).

Resolution order (character-restricted first, §15/§2):

    1. current banner character
    2. that character's preference chain, if the user defined one
    3. otherwise the single ACTIVE roadmap goal for that character
       (a goal is user-defined data, §5 - a fallback, not an invention)

Rules:

* All outcomes returned here share the current banner's character
  (module docstring, resolution order). Resulting constellations for one
  character are strictly nested - reaching C2 necessarily reaches C0
  along the way (§4.2, §12) - so a same-character chain is never a set
  of mutually exclusive alternatives to pick by rank. It is a menu of
  how far to go, and outcomes are therefore ordered by DESCENDING
  constellation within each scheduling class (see the later-progression
  rule below), not by preference rank: pursuing the higher constellation
  subsumes every lower one the chain also names, so it can never be
  worse than pursuing the lower one alone. `rank` is retained on each
  `OutcomeOption` for display and provenance (§15) and decides which
  constellations are in scope at all (see the duplicate-collapse rule
  below), but it is not the evaluation order - optimizer.recommend
  walks this tuple front-to-back and the front must be the
  most-inclusive current-banner target.
* A chain constellation whose roadmap goal is currently BLOCKED (§9: an
  unsatisfied lower-constellation goal for the same character exists)
  AND whose character also has a banner strictly after the current one
  is a LATER PROGRESSION objective, not current-banner business: the
  roadmap itself schedules that reach for the later opportunity, the
  current banner belongs to the active milestone, and the planner is
  re-run after every account update (§2). Such an outcome is still
  offered - the chain is a menu and the planner invents or drops
  nothing - but it sorts BEHIND the nearer objectives, tagged
  `later_progression`. optimizer.recommend promotes it back to the lead
  only under the eligibility rule: pursuing it NOW must be an ordinary
  recommendation (probability at or above MINIMUM_OUTCOME_PROBABILITY)
  at the largest cap that still keeps every higher-priority protected
  goal safe, and it must be the deeper constellation. That is the
  cumulative progression of §4.2/§12 - pulling toward the active
  milestone is also progress toward the later objective, and one plan
  entry targeting the deeper constellation lets the Phase 4 simulator
  pull through C0 toward C2 within each history. Below that bar the
  reach stays the gamble the roadmap scheduled for later: the active
  milestone leads and the objective is pursued on its own banner after
  the re-run (§2). Without a strictly later banner the deeper reach is
  this banner's only remaining chance and stays at the front (§2/§13's
  worked example: Vesna C0 at Priority 1, Vesna C2 at Priority 3,
  gated by an intervening Priority 2 goal). A chain constellation with
  no matching roadmap goal is never demoted: the chain alone defined
  it (§15).
* Duplicate constellations are collapsed to their best rank: "C2R1"
  before "C2" keeps the C2R1 outcome; both express the same target
  constellation. Labels are display-only (§15).
* Constellations at or below current ownership are dropped: a desired
  *resulting* constellation (§4.2, §12) the account already meets is
  nothing to pursue. The optimizer therefore never offers an outcome the
  account already has, and never re-offers one behind it.
* When a preference chain exists, ONLY its constellations are offered -
  even if a roadmap goal asks for a constellation the chain does not
  contain. A chain that is entirely already-owned therefore yields no
  outcomes ("preferences already satisfied"), not a silently upgraded
  target.
* `weapon_refinement` is carried as `None` when the preference does not
  ask for a refinement (domain `0` means none, §15). Refinement is
  displayed but never simulated: no weapon mechanics exist yet (§17,
  §19).
"""

from collections.abc import Iterable
from dataclasses import dataclass

from domain import Banner, Goal, Preference, TargetKind, WeaponTarget, sort_by_rank
from domain.targets import GoalTarget
from planner import (
    PlannerContext,
    GoalState,
    actionable_goals,
    relevant_goal_evaluations,
)
from planner.banners import available_banners, current_banner


@dataclass(frozen=True)
class OutcomeOption:
    """One outcome the optimizer may pursue on the current banner (§13, §15).

    Attributes:
        character: the current banner's featured character - or, for a
            weapon outcome, the featured weapon's name (unified targets
            are identified by name in both cases; `target` carries the
            kind).
        constellation: the desired *resulting* constellation - never a copy
            count (§4.2, §12); the simulator derives copies from the
            account state (§10.4). For a weapon outcome this is the
            desired resulting refinement (R1 -> 1), unified with
            constellation numbering through `Goal.level`.
        rank: preference rank; 1 is most preferred (§15).
        weapon_refinement: the wanted refinement ("R1" -> 1), or None when
            the preference asks for none. Display-only for character
            outcomes (§17, §19); weapon outcomes carry their refinement
            through `constellation`/`target` instead.
        later_progression: True when the roadmap schedules this objective
            for a strictly later banner (its goal is BLOCKED behind an
            unsatisfied lower milestone, §9). The outcome stays in the
            menu, sorted behind the nearer objectives; only
            optimizer.recommend's later-progression eligibility rule may
            promote it to the lead.
        target: the unified target (§4 Phase 2) for weapon outcomes; None
            for character outcomes, which were identified by
            `character` before weapon outcomes existed. The optimizer
            treats both kinds as members of the same ordered goal set.
    """

    character: str
    constellation: int
    rank: int
    weapon_refinement: int | None = None
    later_progression: bool = False
    target: "GoalTarget | None" = None

    @property
    def label(self) -> str:
        """Derived display label such as "C0", "C2", "C2R1" or "R1" (§15)."""
        if self.target is not None and self.target.kind is TargetKind.WEAPON:
            return f"R{self.constellation}"
        label = f"C{self.constellation}"
        if self.weapon_refinement is not None:
            label += f"R{self.weapon_refinement}"
        return label

    @property
    def target_name(self) -> str:
        """The pursued target's display name (character or weapon)."""
        return self.character


def goal_label(goal: Goal) -> str:
    """Display label for a roadmap goal: "Vesna C2" or "Wolf Fang R1"."""
    if goal.target.kind is TargetKind.WEAPON:
        return f"{goal.target.name} R{goal.level}"
    return f"{goal.target.name} C{goal.level}"


def _later_progression_constellations(
    context: PlannerContext, banner=None
) -> frozenset[int]:
    """Chain constellations the roadmap schedules as a later progression step.

    A roadmap goal that is BLOCKED (§9) names a progression objective that
    cannot be treated as independent while its lower milestone is
    incomplete. When the goal's target also has a banner strictly after
    the current one, that goal's pursuit belongs to the later banner - the
    current banner belongs to the active milestone, and the planner is
    re-run after every account update (§2). Without a strictly later
    banner the current banner is the goal's only remaining opportunity and
    nothing is demoted. Targets are matched by unified target identity
    (character or weapon), never by the character accessor.
    """
    current = banner if banner is not None else current_banner(context)
    if not any(
        banner.order_key > current.order_key
        for banner in context.roadmap.banners_for_target(current.target)
    ):
        return frozenset()
    return frozenset(
        evaluation.goal.level
        for evaluation in relevant_goal_evaluations(context)
        if evaluation.state is GoalState.BLOCKED
        and evaluation.goal.target == current.target
    )


def _weapon_outcomes(
    context: PlannerContext,
    preferences: Iterable[Preference],
    banner: Banner,
) -> tuple[OutcomeOption, ...]:
    """Outcomes for a weapon banner (Phase 4 unification).

    Mirror of the character resolution order: the weapon's preference
    chain (a weapon preference names the weapon in `character` and its
    refinement in `weapon_refinement`) first, then the single ACTIVE
    roadmap goal for the weapon as the user-defined fallback. Refinement
    levels for one weapon are strictly nested (reaching R2 necessarily
    reaches R1), so outcomes are ordered by descending level within each
    scheduling class, exactly like character constellations.
    """
    weapon = banner.target.name
    owned = context.account.owned_characters.owned_refinement(weapon)

    chain = sort_by_rank(
        [preference for preference in preferences if preference.character == weapon]
    )
    if chain:
        # Collapse duplicate levels to their best rank (§15). A weapon
        # preference's wanted level is its refinement when one is asked
        # for ("R1" -> 1), otherwise its constellation (a plain base-copy
        # request).
        best_by_level: dict[int, Preference] = {}
        for preference in chain:
            level = (
                preference.weapon_refinement
                if preference.weapon_refinement > 0
                else preference.constellation
            )
            best_by_level.setdefault(level, preference)
        later_progression = _later_progression_constellations(context, banner)
        eligible = tuple(
            OutcomeOption(
                character=weapon,
                constellation=level,
                rank=preference.rank,
                later_progression=level in later_progression,
                target=WeaponTarget(weapon),
            )
            for level, preference in best_by_level.items()
            if level > owned
        )
        return tuple(
            sorted(
                eligible,
                key=lambda option: (option.later_progression, -option.constellation),
            )
        )

    # No preference chain for this weapon: fall back to the roadmap goal
    # (§5) - user-defined data, never invented (§2).
    active = [
        evaluation
        for evaluation in actionable_goals(context, banner)
        if evaluation.goal.target == banner.target
    ]
    if len(active) > 1:
        listed = ", ".join(
            f"{evaluation.goal.weapon} R{evaluation.goal.level} "
            f"(priority {evaluation.goal.priority})"
            for evaluation in active
        )
        raise ValueError(
            "ambiguous roadmap: multiple active goals match the current "
            f"weapon banner ({listed}); the optimizer will not silently choose"
        )
    if not active:
        return ()
    goal = active[0].goal
    # ACTIVE means copies_needed > 0, i.e. goal.level > owned refinement (§9).
    return (
        OutcomeOption(
            character=weapon,
            constellation=goal.level,
            rank=1,
            target=WeaponTarget(weapon),
        ),
    )


def available_outcomes(
    context: PlannerContext,
    preferences: Iterable[Preference] = (),
    banner=None,
) -> tuple[OutcomeOption, ...]:
    """Outcomes the optimizer may pursue, in preference order (§13 step 2).

    Only the current banner character's chain is considered; preferences
    for other characters are ignored (they belong to other banners).

    Raises:
        ValueError: when no preference chain exists and the current banner
            character has several ACTIVE roadmap goals - degenerate
            duplicate goals; the optimizer will not silently choose (the
            same refusal as planner.spend_table).

    Weapon banners dispatch to `_weapon_outcomes` (Phase 4): both banner
    kinds are members of the same ordered opportunity set, resolved with
    their own target type's ownership and levels.
    """
    if banner is None:
        matches = available_banners(context)
        if len(matches) != 1:
            raise ValueError(
                "available_outcomes requires an explicit banner when multiple "
                "banners are available at the current version/phase"
            )
        banner = matches[0]
    if banner.target.kind is TargetKind.WEAPON:
        return _weapon_outcomes(context, preferences, banner)
    character = banner.character
    owned = context.account.owned_constellation(character)

    chain = sort_by_rank(
        [preference for preference in preferences if preference.character == character]
    )
    if chain:
        # Collapse duplicate constellations to their best rank (§15: the
        # chain is an ordering of outcomes, and equal targets are one
        # outcome with a better label).
        best_by_constellation: dict[int, Preference] = {}
        for preference in chain:
            best_by_constellation.setdefault(preference.constellation, preference)
        later_progression = _later_progression_constellations(context, banner)
        eligible = tuple(
            OutcomeOption(
                character=character,
                constellation=preference.constellation,
                rank=preference.rank,
                weapon_refinement=(
                    preference.weapon_refinement
                    if preference.weapon_refinement > 0
                    else None
                ),
                later_progression=preference.constellation in later_progression,
            )
            for preference in best_by_constellation.values()
            if preference.constellation > owned
        )
        # Descending constellation within each scheduling class (see the
        # module docstring): a same-character chain is a progression, and
        # the furthest attainable target subsumes every nearer one the
        # chain also names - except that a blocked goal with a strictly
        # later banner is a later progression objective. Such an outcome
        # sorts behind the current-banner objectives, tagged so
        # optimizer.recommend can apply the eligibility rule (it leads
        # only when pursuing it now is an ordinary recommendation at the
        # largest cap that keeps every higher-priority protected goal
        # safe - the cumulative progression, §4.2/§12).
        return tuple(
            sorted(
                eligible,
                key=lambda option: (option.later_progression, -option.constellation),
            )
        )

    # No preference chain for this character: fall back to the roadmap
    # goal (§5) - user-defined data, never invented (§2).
    active = [
        evaluation
        for evaluation in actionable_goals(context, banner)
        if evaluation.goal.character == character
    ]
    if len(active) > 1:
        listed = ", ".join(
            f"{evaluation.goal.character} C{evaluation.goal.constellation} "
            f"(priority {evaluation.goal.priority})"
            for evaluation in active
        )
        raise ValueError(
            "ambiguous roadmap: multiple active goals match the current "
            f"banner ({listed}); the optimizer will not silently choose"
        )
    if not active:
        return ()
    goal = active[0].goal
    # ACTIVE means copies_needed > 0, i.e. goal.constellation > owned (§9).
    return (
        OutcomeOption(character=character, constellation=goal.constellation, rank=1),
    )
