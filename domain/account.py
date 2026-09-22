"""Account state and ownership."""

from dataclasses import dataclass, field

NOT_OWNED = -1


def _validate_constellation(character: str, constellation: int) -> None:
    if constellation < NOT_OWNED:
        raise ValueError(
            f"constellation for {character!r} must be >= {NOT_OWNED} (NOT_OWNED), "
            f"got {constellation}"
        )


def _validate_refinement(weapon: str, refinement: int) -> None:
    if refinement < NOT_OWNED:
        raise ValueError(
            f"refinement for {weapon!r} must be >= {NOT_OWNED} (NOT_OWNED), "
            f"got {refinement}"
        )


@dataclass(frozen=True)
class Ownership:
    """What the account currently possesses."""

    characters: dict[str, int] = field(default_factory=dict)
    weapons: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for character, constellation in self.characters.items():
            _validate_constellation(character, constellation)
        for weapon, refinement in self.weapons.items():
            _validate_refinement(weapon, refinement)

    def owned_constellation(self, character: str) -> int:
        return self.characters.get(character, NOT_OWNED)

    def owned_refinement(self, weapon: str) -> int:
        return self.weapons.get(weapon, NOT_OWNED)

    def get(self, character: str, default: int = NOT_OWNED) -> int:
        return self.characters.get(character, default)

    def owns(self, character: str) -> bool:
        return self.owned_constellation(character) >= 0

    def owns_weapon(self, weapon: str) -> bool:
        return self.owned_refinement(weapon) >= 0

    def with_constellation(self, character: str, constellation: int) -> "Ownership":
        _validate_constellation(character, constellation)
        updated = dict(self.characters)
        updated[character] = constellation
        return Ownership(updated, dict(self.weapons))

    def with_refinement(self, weapon: str, refinement: int) -> "Ownership":
        _validate_refinement(weapon, refinement)
        updated = dict(self.weapons)
        updated[weapon] = refinement
        return Ownership(dict(self.characters), updated)


@dataclass(frozen=True)
class CharacterWishState:
    """Current character-banner wish state."""

    pity: int = 0
    guarantee: bool = False
    capturing_radiance: int = 0

    def __post_init__(self) -> None:
        if self.pity < 0:
            raise ValueError(f"pity must be non-negative, got {self.pity}")
        if not 0 <= self.capturing_radiance <= 3:
            raise ValueError(
                "capturing_radiance must be between 0 and 3, "
                f"got {self.capturing_radiance}"
            )

    def after_five_star(self, *, was_guaranteed: bool, featured: bool) -> "CharacterWishState":
        return CharacterWishState(
            pity=0,
            guarantee=not featured,
            capturing_radiance=next_capturing_radiance_counter(
                self.capturing_radiance,
                was_guaranteed=was_guaranteed,
                featured=featured,
            ),
        )


@dataclass(frozen=True)
class WeaponWishState:
    """Current weapon-banner state.

    Phase 2 defines the state transition surface without implementing weapon
    probability mechanics.
    """

    pity: int = 0
    guarantee: bool = False
    fate_points: int = 0

    def __post_init__(self) -> None:
        if self.pity < 0:
            raise ValueError(f"pity must be non-negative, got {self.pity}")
        if self.fate_points < 0:
            raise ValueError(f"fate_points must be non-negative, got {self.fate_points}")

    def after_five_star(
        self,
        *,
        next_guarantee: bool,
        next_fate_points: int,
    ) -> "WeaponWishState":
        """Apply an observed five-star result.

        Weapon-specific mechanics determine the next guarantee and fate-point
        values. They are explicit here so this domain transition does not
        silently assume character-banner rules.
        """
        return WeaponWishState(
            pity=0,
            guarantee=next_guarantee,
            fate_points=next_fate_points,
        )


@dataclass(frozen=True)
class Account:
    """The user's current account state."""

    current_pity: int = 0
    character_guarantee: bool = False
    owned_characters: Ownership = field(default_factory=Ownership)
    wishes: int = 0
    capturing_radiance_counter: int = 0
    weapon_state: WeaponWishState = field(default_factory=WeaponWishState)

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

    @property
    def character_state(self) -> CharacterWishState:
        return CharacterWishState(
            pity=self.current_pity,
            guarantee=self.character_guarantee,
            capturing_radiance=self.capturing_radiance_counter,
        )

    def owned_constellation(self, character: str) -> int:
        return self.owned_characters.owned_constellation(character)

    def owns(self, character: str) -> bool:
        return self.owned_characters.owns(character)


def next_capturing_radiance_counter(
    radiance: int, *, was_guaranteed: bool, featured: bool
) -> int:
    if not 0 <= radiance <= 3:
        raise ValueError(f"radiance must be in [0, 3], got {radiance}")
    if was_guaranteed:
        return radiance
    if featured:
        return 1 if radiance >= 2 else 0
    return min(3, radiance + 1)
