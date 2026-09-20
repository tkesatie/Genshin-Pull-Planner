"""Account and ownership (Design Document §4).

Account state describes what the user currently owns, not what they want
(§4.1). Goals (§5) describe what the user wants and live in the roadmap,
never here.
"""

from dataclasses import dataclass, field

# Sentinel constellation value for a character the account does not own (§4.2).
NOT_OWNED = -1


def _validate_constellation(character: str, constellation: int) -> None:
    if constellation < NOT_OWNED:
        raise ValueError(
            f"constellation for {character!r} must be >= {NOT_OWNED} (NOT_OWNED), "
            f"got {constellation}"
        )


@dataclass(frozen=True)
class Ownership:
    """What the account currently possesses, per character (§4.2).

    A constellation is not a number of copies (§4.2):

        -1  character not owned (NOT_OWNED)
         0  C0
         1  C1
         2  C2
         ...

    Weapons will get their own record here later; Phase 1 tracks characters.
    """

    characters: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for character, constellation in self.characters.items():
            _validate_constellation(character, constellation)

    def owned_constellation(self, character: str) -> int:
        """Return the owned constellation, or NOT_OWNED (-1) if not owned (§4.2)."""
        return self.characters.get(character, NOT_OWNED)

    def get(self, character: str, default: int = NOT_OWNED) -> int:
        """Dict-style lookup, mirroring the §4.2 conceptual usage:
        `account.owned_characters.get("Vesna", -1)`.
        """
        return self.characters.get(character, default)

    def owns(self, character: str) -> bool:
        """True if the account owns the character at any constellation."""
        return self.owned_constellation(character) >= 0

    def with_constellation(self, character: str, constellation: int) -> "Ownership":
        """Return a new Ownership with `character` set to `constellation`.

        The original is left unchanged, so account state transitions later
        (e.g. the Phase 4 simulator) never mutate history.
        """
        _validate_constellation(character, constellation)
        updated = dict(self.characters)
        updated[character] = constellation
        return Ownership(updated)


@dataclass(frozen=True)
class Account:
    """The user's current account state (§4.1).

    Attributes:
        current_pity: pulls already made on the character banner since the
            last 5-star (0-based, non-negative).
        character_guarantee: True when the next 5-star is guaranteed to be
            the featured character.
        owned_characters: what the account possesses (see Ownership).
        wishes: wishes currently owned, as a resource.

    Note on `wishes`: this is a *possession*. It is deliberately distinct
    from a `wishes_to_spend` decision budget, which is a planner/strategy
    output (§12-§14), not account state.
    """

    current_pity: int = 0
    character_guarantee: bool = False
    owned_characters: Ownership = field(default_factory=Ownership)
    wishes: int = 0
    capturing_radiance_counter: int = 0

    def __post_init__(self) -> None:
        if self.current_pity < 0:
            raise ValueError(
                f"current_pity must be non-negative, got {self.current_pity}"
            )
        if self.wishes < 0:
            raise ValueError(f"wishes must be non-negative, got {self.wishes}")
        if not 0 <= self.capturing_radiance_counter <= 3:
            raise ValueError(
                "capturing_radiance_counter must be between 0 and 3, "
                f"got {self.capturing_radiance_counter}"
            )

    def owned_constellation(self, character: str) -> int:
        """Delegates to Ownership (§4.2): NOT_OWNED when the character is absent."""
        return self.owned_characters.owned_constellation(character)

    def owns(self, character: str) -> bool:
        """True if the account owns the character at any constellation."""
        return self.owned_characters.owns(character)


def next_capturing_radiance_counter(
    radiance: int, *, was_guaranteed: bool, featured: bool
) -> int:
    """Capturing Radiance counter after one 5-star pull.

    Single source of truth for this transition: `simulation.engine` and the
    account-update API both call this, so a live account's recorded state
    can never drift from what the simulator would produce for the same
    event. (This fixes a real bug: the account-update endpoint used to
    reimplement this transition inline without the radiance-3 special
    case, landing on the wrong counter value after a Capturing-Radiance-
    guaranteed win.)

    Paired with `probability.rates.capturing_radiance_rate`, which supplies
    the win probability at each state, the schedule is:

        radiance 0 or 1  ->  50% (base rate) chance of winning
        radiance 2       ->  6/11 (~54.5%) chance of winning
        radiance 3       ->  100% (guaranteed) chance of winning

    Transition rules:

    * a guaranteed pull never touches the counter - guarantee and
      Capturing Radiance do not interact (per the mechanic's official
      description: Capturing Radiance never applies while a guarantee is
      already active, and using the guarantee never resets or advances the
      counter);
    * a non-guaranteed loss increments the streak, capped at 3 - radiance 3
      always wins, so no more than 3 consecutive losses are possible;
    * a non-guaranteed win from radiance 0 or 1 fully resets the streak to
      0;
    * a non-guaranteed win from radiance 2 or 3 (a boosted or
      guaranteed-by-radiance win) leaves a residual mark at 1 rather than a
      full reset to 0.

    This schedule is not an arbitrary guess: solving the 4-state Markov
    chain formed by these transitions together with
    `capturing_radiance_rate`'s win probabilities gives a long-run average
    win rate across all states of exactly 0.55, matching the officially
    cited "55% overall" aggregate figure, while radiance 3 being absorbing
    on a win reproduces the documented "guaranteed by the 4th 5-star after
    3 consecutive losses" behavior.

    Raises:
        ValueError: if `radiance` is outside [0, 3].
    """
    if not 0 <= radiance <= 3:
        raise ValueError(f"radiance must be in [0, 3], got {radiance}")
    if was_guaranteed:
        return radiance
    if featured:
        return 1 if radiance >= 2 else 0
    return min(3, radiance + 1)
