"""Wire shapes for the probability engine (Design Document §10).

The engine answers isolated probability questions and knows nothing about
roadmaps (§10). These responses therefore echo the exact state the answer
was computed from - starting pity, guarantee, mechanics - so a number is
never separated from the question it answers.
"""

from pydantic import BaseModel, Field

from api.schemas.domain import MechanicsModel


class CharacterProbabilityView(BaseModel):
    """Single-copy character probability (§10.2)."""

    wishes: int = Field(..., description="Wishes looked ahead.")
    starting_pity: int
    guaranteed: bool
    probability: float = Field(
        ..., description="P(at least one featured copy within `wishes` wishes)."
    )
    curve: list[float] | None = Field(
        None,
        description=(
            "Cumulative curve when requested: index N is the probability "
            "within N additional wishes. Index 0 is always 0.0."
        ),
    )
    mechanics: MechanicsModel


class WishesNeededView(BaseModel):
    """Confidence inversion (§10.3)."""

    confidence: float
    starting_pity: int
    guaranteed: bool
    wishes_needed: int = Field(
        ...,
        description=(
            "Smallest wish count whose probability reaches the requested "
            "confidence from this state."
        ),
    )
    probability_at_wishes_needed: float
    mechanics: MechanicsModel
