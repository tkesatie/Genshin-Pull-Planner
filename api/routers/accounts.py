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

from fastapi import APIRouter, Depends, status
from api.auth import UserRecord

from api.dependencies import get_current_user, get_record, get_repository
from api.repository import AccountRecord, AccountRepository, new_account_id
from api.schemas.accounts import (
    AccountCreate,
    AccountSummary,
    AccountUpdate,
    AccountView,
)

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
