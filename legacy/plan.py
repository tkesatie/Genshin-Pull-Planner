from dataclasses import dataclass

@dataclass
class PullDecision:
    character_goals: dict[str, int]
    wishes_to_spend: int

@dataclass
class Goal:
    character: str
    constellation: int
    priority: int

@dataclass
class Banner:
    character: str
    version: str
    phase: int

@dataclass
class Roadmap:
    goals: list[Goal]
    banners: list[Banner]

roadmap = Roadmap(
    goals=[
        Goal("Vesna", 0, 1),
        Goal("Vodynista", 0, 2),
        Goal("Vesna", 2, 3),
        Goal("Tsaritsa", 0, 4),
    ],
    banners=[
        Banner("Vesna", "7.0", 1),
        Banner("Tsaritsa", "7.1", 1),
        Banner("Vodynista", "7.2", 1),
    ]
)