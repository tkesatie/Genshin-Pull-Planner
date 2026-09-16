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

* Outcomes are ordered by preference rank; rank 1 is pursued first (§13
  step 7 compares preference rank, never probability).
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

from domain import Preference, sort_by_rank
from planner import PlannerContext, actionable_goals
from planner.banners import current_banner


@dataclass(frozen=True)
class OutcomeOption:
    """One outcome the optimizer may pursue on the current banner (§13, §15).

    Attributes:
        character: the current banner's featured character.
        constellation: the desired *resulting* constellation - never a copy
            count (§4.2, §12); the simulator derives copies from the
            account state (§10.4).
        rank: preference rank; 1 is most preferred (§15).
        weapon_refinement: the wanted refinement ("R1" -> 1), or None when
            the preference asks for none. Display-only (§17, §19).
    """

    character: str
    constellation: int
    rank: int
    weapon_refinement: int | None = None

    @property
    def label(self) -> str:
        """Derived display label such as "C0", "C2" or "C2R1" (§15)."""
        label = f"C{self.constellation}"
        if self.weapon_refinement is not None:
            label += f"R{self.weapon_refinement}"
        return label


def available_outcomes(
    context: PlannerContext,
    preferences: Iterable[Preference] = (),
) -> tuple[OutcomeOption, ...]:
    """Outcomes the optimizer may pursue, in preference order (§13 step 2).

    Only the current banner character's chain is considered; preferences
    for other characters are ignored (they belong to other banners).

    Raises:
        ValueError: when no preference chain exists and the current banner
            character has several ACTIVE roadmap goals - degenerate
            duplicate goals; the optimizer will not silently choose (the
            same refusal as planner.spend_table).
    """
    character = current_banner(context).character
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
        options = tuple(
            OutcomeOption(
                character=character,
                constellation=preference.constellation,
                rank=preference.rank,
                weapon_refinement=(
                    preference.weapon_refinement
                    if preference.weapon_refinement > 0
                    else None
                ),
            )
            for preference in best_by_constellation.values()
        )
        return tuple(
            option for option in options if option.constellation > owned
        )

    # No preference chain for this character: fall back to the roadmap
    # goal (§5) - user-defined data, never invented (§2).
    active = [
        evaluation
        for evaluation in actionable_goals(context)
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
