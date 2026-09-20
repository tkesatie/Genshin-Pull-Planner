"""Probability engine endpoints (Design Document §10).

    GET /accounts/{id}/probability/character        §10.2
    GET /accounts/{id}/probability/wishes-needed    §10.3
    GET /accounts/{id}/probability/weapon           501 (§17, §19)

The engine answers isolated questions and is independent of roadmap logic
(§10). These endpoints hang off an account only because the account
supplies the *default* starting state: pity, guarantee, Capturing Radiance
counter and mechanics are all overridable per request, so the engine can
still be asked "what if I were at pity 70 with a guarantee?" without
touching stored state.

`/probability/weapon` answers 501. Weapon mechanics are deliberately absent
from the domain until verified independently (§17), and weapon pulling is
outside the initial scope (§19). Guessing them here would put domain logic
in the API and, worse, present a guess as an answer.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api.dependencies import get_record
from api.repository import AccountRecord
from api.schemas.domain import MechanicsModel
from api.schemas.probability import CharacterProbabilityView, WishesNeededView
from probability import cumulative_probability, wishes_for_confidence

router = APIRouter(tags=["probability"])

# An API-level bound, not a domain rule: the analytical curve costs
# O(wishes x hard_pity) to compute and `wishes + 1` floats to return, and an
# unbounded lookahead would be a denial-of-service surface. The engine
# itself accepts any non-negative lookahead.
MAX_LOOKAHEAD_WISHES = 10_000


@router.get(
    "/accounts/{account_id}/probability/character",
    response_model=CharacterProbabilityView,
    summary="Single-copy character probability",
    description=(
        "P(at least one featured copy within N wishes) from a pity/guarantee/"
        "Capturing Radiance state (§10.2). Defaults to the account's current "
        "state; pass `starting_pity`, `guaranteed` and/or `starting_radiance` "
        "to ask about any other state."
    ),
)
def character_probability(
    wishes: int = Query(..., description="Wishes to look ahead."),
    starting_pity: int | None = Query(
        None, description="Defaults to the account's current pity."
    ),
    guaranteed: bool | None = Query(
        None, description="Defaults to the account's guarantee state."
    ),
    starting_radiance: int | None = Query(
        None,
        description=(
            "Capturing Radiance loss-streak counter (0-3). Defaults to the "
            "account's current counter; omitting this for an account that "
            "currently carries a nonzero counter understates the true "
            "probability."
        ),
    ),
    include_curve: bool = Query(
        True,
        description=(
            "Include the full cumulative curve (length wishes + 1, index 0 "
            "is 0.0)."
        ),
    ),
    record: AccountRecord = Depends(get_record),
) -> CharacterProbabilityView:
    if wishes > MAX_LOOKAHEAD_WISHES:
        raise HTTPException(
            status_code=422,  # Unprocessable Entity
            detail=(
                f"wishes must be <= {MAX_LOOKAHEAD_WISHES} for this endpoint; "
                f"got {wishes}"
            ),
        )
    mechanics = record.settings.mechanics
    pity = (
        record.account.current_pity if starting_pity is None else starting_pity
    )
    guarantee = (
        record.account.character_guarantee if guaranteed is None else guaranteed
    )
    radiance = (
        record.account.capturing_radiance_counter
        if starting_radiance is None
        else starting_radiance
    )
    curve = cumulative_probability(wishes, pity, guarantee, mechanics, radiance)
    return CharacterProbabilityView(
        wishes=wishes,
        starting_pity=pity,
        guaranteed=guarantee,
        starting_radiance=radiance,
        probability=float(curve[wishes]),
        curve=[float(value) for value in curve] if include_curve else None,
        mechanics=MechanicsModel.from_domain(mechanics),
    )


@router.get(
    "/accounts/{account_id}/probability/wishes-needed",
    response_model=WishesNeededView,
    summary="Wishes needed for a confidence",
    description=(
        "Smallest wish count reaching the requested confidence from a "
        "pity/guarantee/Capturing Radiance state (§10.3). Defaults to the "
        "account's current state and its stored confidence threshold."
    ),
)
def wishes_needed(
    confidence: float | None = Query(
        None, description="Defaults to the account's stored threshold (§1)."
    ),
    starting_pity: int | None = Query(
        None, description="Defaults to the account's current pity."
    ),
    guaranteed: bool | None = Query(
        None, description="Defaults to the account's guarantee state."
    ),
    starting_radiance: int | None = Query(
        None,
        description=(
            "Capturing Radiance loss-streak counter (0-3). Defaults to the "
            "account's current counter."
        ),
    ),
    record: AccountRecord = Depends(get_record),
) -> WishesNeededView:
    mechanics = record.settings.mechanics
    threshold = (
        record.settings.confidence if confidence is None else confidence
    )
    pity = (
        record.account.current_pity if starting_pity is None else starting_pity
    )
    guarantee = (
        record.account.character_guarantee if guaranteed is None else guaranteed
    )
    radiance = (
        record.account.capturing_radiance_counter
        if starting_radiance is None
        else starting_radiance
    )
    needed = wishes_for_confidence(threshold, pity, guarantee, mechanics, radiance)
    curve = cumulative_probability(needed, pity, guarantee, mechanics, radiance)
    return WishesNeededView(
        confidence=threshold,
        starting_pity=pity,
        guaranteed=guarantee,
        starting_radiance=radiance,
        wishes_needed=needed,
        probability_at_wishes_needed=float(curve[needed]),
        mechanics=MechanicsModel.from_domain(mechanics),
    )


@router.get(
    "/accounts/{account_id}/probability/weapon",
    status_code=status.HTTP_501_NOT_IMPLEMENTED,
    summary="Weapon probability (not implemented)",
    description=(
        "Reserved. Weapon mechanics must be verified independently before "
        "the weapon simulator is implemented (§17), and weapon pulling is "
        "outside the initial scope (§19). This endpoint returns 501 rather "
        "than a guess."
    ),
)
def weapon_probability(record: AccountRecord = Depends(get_record)) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=(
            "weapon probability is not implemented: weapon banner mechanics "
            "are deliberately absent until verified independently (§17) and "
            "weapon pulling is outside the initial scope (§19)"
        ),
    )
