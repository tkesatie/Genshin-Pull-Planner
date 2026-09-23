"""Roadmap sub-resources (Design Document §5, §7, §15, §16).

    GET/PUT /accounts/{id}/goals         objectives and their priorities (§5)
    GET/PUT /accounts/{id}/banners       the expected schedule (§7)
    GET/PUT /accounts/{id}/preferences   preference chains (§15)
    GET/PUT /accounts/{id}/income        expected future income (§16)
    DELETE  /accounts/{id}/income        stop tracking income

Each collection is replaced whole rather than patched item by item. Goals
are a *prioritized* list whose priorities must be unique (§8) and
preferences are a *chain* whose ranks express an ordering (§15): both are
sets whose members constrain each other, so a whole-collection write is the
operation that can always be validated as a unit.

Reads are returned in the roadmap's own orders - goals by priority (§5),
banners chronologically (§7) - so a client never has to re-sort to see what
the planner sees.
"""

from dataclasses import replace

from fastapi import APIRouter, Depends, Path, status

from api.dependencies import get_record, get_repository
from api.repository import AccountRecord, AccountRepository
from api.schemas.accounts import (
    BannerListModel,
    BannerListView,
    GoalListModel,
    GoalListView,
    GoalStatusListView,
    GoalStatusView,
    PreferenceListModel,
    PreferenceListView,
)
from api.schemas.domain import (
    BannerModel,
    GoalModel,
    IncomeForecastModel,
    IncomeForecastView,
    PreferenceView,
)
from domain import GoalState, sort_by_rank
from planner.goals import evaluate_goals

router = APIRouter(tags=["roadmap"])


@router.get(
    "/accounts/{account_id}/goals",
    response_model=GoalListView,
    summary="Read roadmap goals",
    description="Goals in priority order (§5); priority 1 is protected first.",
)
def read_goals(record: AccountRecord = Depends(get_record)) -> GoalListView:
    return GoalListView(
        goals=[
            GoalModel.from_domain(goal)
            for goal in record.roadmap().goals_in_priority_order()
        ]
    )


@router.put(
    "/accounts/{account_id}/goals",
    response_model=GoalListView,
    summary="Replace roadmap goals",
    description=(
        "Replaces every goal. Priorities must be unique within the roadmap "
        "(§8); the same character may appear in several goals (§5)."
    ),
)
def replace_goals(
    payload: GoalListModel,
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> GoalListView:
    updated = repository.save(
        replace(record, goals=tuple(goal.to_domain() for goal in payload.goals))
    )
    return GoalListView(
        goals=[
            GoalModel.from_domain(goal)
            for goal in updated.roadmap().goals_in_priority_order()
        ]
    )


@router.get(
    "/accounts/{account_id}/goals/status",
    response_model=GoalStatusListView,
    summary="Read unified goal status",
    description="Returns completion, blocking state, remaining copies/refinements, and next opportunity for every goal.",
)
def read_goal_status(record: AccountRecord = Depends(get_record)) -> GoalStatusListView:
    return GoalStatusListView(
        goals=[
            GoalStatusView(
                goal=GoalModel.from_domain(evaluation.goal),
                status=evaluation.state.value,
                copies_needed=evaluation.copies_needed,
                blocked_by=(
                    None
                    if evaluation.blocked_by is None
                    else GoalModel.from_domain(evaluation.blocked_by)
                ),
                next_banner=(
                    None
                    if evaluation.next_banner is None
                    else BannerModel.from_domain(evaluation.next_banner)
                ),
            )
            for evaluation in evaluate_goals(record.context())
        ]
    )


@router.delete(
    "/accounts/{account_id}/goals/{priority}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete one goal by priority",
)
def delete_goal(
    priority: int = Path(..., ge=1),
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> None:
    goals = tuple(goal for goal in record.goals if goal.priority != priority)
    if len(goals) == len(record.goals):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"no goal with priority {priority}")
    repository.save(replace(record, goals=goals))


@router.get(
    "/accounts/{account_id}/banners",
    response_model=BannerListView,
    summary="Read the banner schedule",
    description="Banners in chronological order (§7): version, then phase.",
)
def read_banners(record: AccountRecord = Depends(get_record)) -> BannerListView:
    return BannerListView(
        banners=[
            BannerModel.from_domain(banner)
            for banner in record.roadmap().banners_in_chronological_order()
        ]
    )


@router.put(
    "/accounts/{account_id}/banners",
    response_model=BannerListView,
    summary="Replace the banner schedule",
    description=(
        "Replaces the expected schedule. Banners are user assumptions, not "
        "guaranteed game information (§8)."
    ),
)
def replace_banners(
    payload: BannerListModel,
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> BannerListView:
    updated = repository.save(
        replace(
            record,
            banners=tuple(banner.to_domain() for banner in payload.banners),
        )
    )
    return BannerListView(
        banners=[
            BannerModel.from_domain(banner)
            for banner in updated.roadmap().banners_in_chronological_order()
        ]
    )


@router.get(
    "/accounts/{account_id}/preferences",
    response_model=PreferenceListView,
    summary="Read preference chains",
    description="Preferences ordered by rank (§15); labels are derived.",
)
def read_preferences(
    record: AccountRecord = Depends(get_record),
) -> PreferenceListView:
    return PreferenceListView(
        preferences=[
            PreferenceView.from_domain(preference)
            for preference in sort_by_rank(record.preferences)
        ]
    )


@router.put(
    "/accounts/{account_id}/preferences",
    response_model=PreferenceListView,
    summary="Replace preference chains",
    description=(
        "Replaces every preference, for every character. Preferences answer "
        '"if I am going after this character, what outcomes do I prefer?" '
        "and are separate from goals (§15)."
    ),
)
def replace_preferences(
    payload: PreferenceListModel,
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> PreferenceListView:
    updated = repository.save(
        replace(
            record,
            preferences=tuple(
                preference.to_domain() for preference in payload.preferences
            ),
        )
    )
    return PreferenceListView(
        preferences=[
            PreferenceView.from_domain(preference)
            for preference in sort_by_rank(updated.preferences)
        ]
    )


@router.get(
    "/accounts/{account_id}/income",
    response_model=IncomeForecastView | None,
    summary="Read the income forecast",
    description="Future income per version (§16), or null when untracked.",
)
def read_income(
    record: AccountRecord = Depends(get_record),
) -> IncomeForecastView | None:
    if record.income is None:
        return None
    return IncomeForecastView.from_domain(record.income)


@router.put(
    "/accounts/{account_id}/income",
    response_model=IncomeForecastView,
    summary="Replace the income forecast",
    description=(
        "Replaces the forecast. Values are income arriving *after* the "
        "current account state, and the current version's forecast still "
        "arrives (planner.context); earlier versions are presumed banked in "
        "the account's wishes."
    ),
)
def replace_income(
    payload: IncomeForecastModel,
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> IncomeForecastView:
    updated = repository.save(replace(record, income=payload.to_domain()))
    return IncomeForecastView.from_domain(updated.income)


@router.delete(
    "/accounts/{account_id}/income",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Stop tracking income",
    description=(
        "Clears the forecast. An account with no forecast credits no future "
        "income at all - distinct from a forecast of zero wishes (§16)."
    ),
)
def delete_income(
    record: AccountRecord = Depends(get_record),
    repository: AccountRepository = Depends(get_repository),
) -> None:
    repository.save(replace(record, income=None))
