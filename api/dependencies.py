"""Shared dependencies (Design Document §18 Phase 6).

Two kinds of shared wiring:

* access to the process-wide repository and job store, read from
  `app.state` so an application built by `create_app` is self-contained
  and tests never share storage;
* per-request planner overrides: the confidence threshold and income
  scenario are assumptions (§2, §16), so a client must be able to ask
  "what if I required 80% instead?" without writing anything to the stored
  settings.
"""

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Path, Query, Request, status

from api.auth import SESSION_COOKIE, AuthRepository, UserRecord
from api.jobs import SimulationJobStore
from api.repository import AccountNotFound, AccountRecord, AccountRepository


def get_auth_repository(request: Request) -> AuthRepository:
    return request.app.state.auth_repository


def get_current_user(
    request: Request,
    auth_repository: AuthRepository = Depends(get_auth_repository),
) -> UserRecord:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    session = auth_repository.get_session(token)
    if session is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    user_id, expires_at = session
    from datetime import datetime, timezone
    if expires_at <= datetime.now(timezone.utc):
        auth_repository.delete_session(token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="session expired")
    user = auth_repository.get_user_by_id(user_id)
    if user is None:
        auth_repository.delete_session(token)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="authentication required")
    return user


def get_repository(request: Request) -> AccountRepository:
    """The application's account repository."""
    return request.app.state.repository


def get_jobs(request: Request) -> SimulationJobStore:
    """The application's simulation job store."""
    return request.app.state.jobs


def get_record(
    account_id: str = Path(..., description="Account identifier."),
    repository: AccountRepository = Depends(get_repository),
    user: UserRecord = Depends(get_current_user),
) -> AccountRecord:
    """The stored account record, or 404."""
    try:
        record = repository.get(account_id)
        if record.owner_id != user.id:
            raise AccountNotFound(f"no account with id {account_id!r}")
        return record
    except AccountNotFound as missing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(missing)
        ) from missing


@dataclass(frozen=True)
class ContextOverrides:
    """Per-request planner assumptions; None means "use the stored setting"."""

    confidence: float | None = None
    income_scenario: str | None = None


def context_overrides(
    confidence: float | None = Query(
        None,
        description=(
            "Override the stored confidence threshold for this request only "
            "(§1). Nothing is written."
        ),
    ),
    income_scenario: str | None = Query(
        None,
        description=(
            'Override the stored income scenario ("low", "expected", "high") '
            "for this request only (§16). Nothing is written."
        ),
    ),
) -> ContextOverrides:
    """Assumption overrides that apply to one call (§2: assumptions vary)."""
    return ContextOverrides(
        confidence=confidence, income_scenario=income_scenario
    )
