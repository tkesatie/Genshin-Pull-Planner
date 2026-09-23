"""Account resource (Design Document §18 Phase 6).

    POST   /accounts            create
    GET    /accounts            list
    GET    /accounts/{id}       read
    PUT    /accounts/{id}       replace state and settings (§2)
    DELETE /accounts/{id}       remove

The account is the aggregate the planner needs; its roadmap sub-resources
have their own endpoints (api.routers.roadmap) so updating goals never
requires resending income.

`PUT` replaces account state and settings because that is the operation the
planner is designed around: the tool is re-run after every meaningful
account update, and the previous recommendation is not a permanent plan
(§2).
"""

from dataclasses import replace

from fastapi import APIRouter, Depends, HTTPException, status
from api.auth import UserRecord

from api.dependencies import get_current_user, get_record, get_repository
from api.repository import AccountRecord, AccountRepository, new_account_id
from api.schemas.accounts import (
    AccountCreate,
    AccountSummary,
    AccountUpdate,
    AccountView,
    PullResultModel,
)
from domain import next_capturing_radiance_counter

router = APIRouter(tags=["accounts"])


def record_from_create(payload: AccountCreate) -> AccountRecord:
    """Build a record from a create request (domain validates the values)."""
    return AccountRecord(
        id=new_account_id(),
        label=payload.label,
        account=payload.account.to_domain(),
        settings=payload.settings.to_domain(),
        goals=tuple(goal.to_domain() for goal in payload.goals),
        banners=tuple(banner.to_domain() for banner in payload.banners),
        preferences=tuple(
            preference.to_domain() for preference in payload.preferences
        ),
        income=None if payload.income is None else payload.income.to_domain(),
    )


@router.post(
    "/accounts",
    response_model=AccountView,
    status_code=status.HTTP_201_CREATED,
    summary="Create an account",
    description=(
        "Creates an account with its state (§4.1), planner settings and, "
        "optionally, its whole roadmap (§8) in one call."
    ),
)
def create_account(
    payload: AccountCreate,
    repository: AccountRepository = Depends(get_repository),
    user: UserRecord = Depends(get_current_user),
) -> AccountView:
    record = repository.create(replace(record_from_create(payload), owner_id=user.id))
    return AccountView.from_record(record)


@router.get(
    "/accounts",
    response_model=list[AccountSummary],
    summary="List accounts",
)
def list_accounts(
    repository: AccountRepository = Depends(get_repository),
    user: UserRecord = Depends(get_current_user),
) -> list[AccountSummary]:
    return [AccountSummary.from_record(record) for record in repository.list(user.id)]


@router.get(
    "/accounts/{account_id}",
    response_model=AccountView,
    summary="Read an account",
)
def read_account(record: AccountRecord = Depends(get_record)) -> AccountView:
    return AccountView.from_record(record)


@router.put(
    "/accounts/{account_id}",
    response_model=AccountView,
    summary="Replace account state and settings",
    description=(
        "Replaces the account's state and planner settings; roadmap "
        "sub-resources are untouched. This is the update the planner is "
        "re-run after (§2)."
    ),
)
def replace_account(
    payload: AccountUpdate,
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> AccountView:
    updated = repository.save(
        replace(
            record,
            label=payload.label,
            account=payload.account.to_domain(),
            settings=payload.settings.to_domain(),
        )
    )
    # A manual account edit is not a recorded observation that can be
    # conditioned onto retained simulation evidence. Invalidate disposable
    # planner evidence so it cannot be reused against unrelated state.
    from api.planner_cache import planner_evidence_cache
    planner_evidence_cache.clear_account(record.id)
    return AccountView.from_record(updated)


@router.delete(
    "/accounts/{account_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an account",
)
def delete_account(
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> None:
    repository.delete(record.id)


@router.post(
    "/accounts/{account_id}/pull-result",
    response_model=AccountView,
    summary="Record a character-banner outcome",
    description=(
        "Records a meaningful character-banner outcome without requiring "
        "one update per individual wish. Use featured for a pulled featured "
        "character, lost_50_50 for an off-banner 5-star, or stopped for "
        "wishes spent without a 5-star."
    ),
)
def record_pull_result(
    payload: PullResultModel,
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> AccountView:
    if payload.outcome not in {"featured", "lost_50_50"}:
        raise HTTPException(status_code=400, detail="outcome must be featured or lost_50_50")
    if payload.wishes_used > record.account.wishes:
        raise HTTPException(status_code=400, detail="wishes_used cannot exceed the account's wishes")

    account = record.account
    wishes = account.wishes - payload.wishes_used

    if payload.outcome == "lost_50_50":
        updated_account = replace(
            account,
            wishes=wishes,
            current_pity=0,
            character_guarantee=True,
            capturing_radiance_counter=next_capturing_radiance_counter(
                account.capturing_radiance_counter,
                was_guaranteed=account.character_guarantee,
                featured=False,
            ),
        )
    else:
        if not payload.character:
            raise HTTPException(status_code=400, detail="character is required for a featured outcome")
        ownership = account.owned_characters
        current = ownership.owned_constellation(payload.character)
        updated_account = replace(
            account,
            wishes=wishes,
            current_pity=0,
            character_guarantee=False,
            capturing_radiance_counter=next_capturing_radiance_counter(
                account.capturing_radiance_counter,
                was_guaranteed=account.character_guarantee,
                featured=True,
            ),
            owned_characters=ownership.with_constellation(payload.character, current + 1),
        )

    updated = repository.save(replace(record, account=updated_account))
    return AccountView.from_record(updated)
