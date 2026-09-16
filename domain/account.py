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

    def __post_init__(self) -> None:
        if self.current_pity < 0:
            raise ValueError(
                f"current_pity must be non-negative, got {self.current_pity}"
            )
        if self.wishes < 0:
            raise ValueError(f"wishes must be non-negative, got {self.wishes}")

    def owned_constellation(self, character: str) -> int:
        """Delegates to Ownership (§4.2): NOT_OWNED when the character is absent."""
        return self.owned_characters.owned_constellation(character)

    def owns(self, character: str) -> bool:
        """True if the account owns the character at any constellation."""
        return self.owned_characters.owns(character)
