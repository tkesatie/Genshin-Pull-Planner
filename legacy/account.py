from dataclasses import dataclass

@dataclass
class Account:
    current_pity: int
    character_guarantee: bool
    owned_characters: dict[str, int]
    wishes: int