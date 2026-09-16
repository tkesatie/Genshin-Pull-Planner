"""Request and response shapes for the API (Design Document §18 Phase 6).

Shape only. Value rules belong to the domain: a schema builds the frozen
domain object and lets it validate, so each rule has exactly one home and
clients see the domain's own message (api package invariant 2).

    api.schemas.domain      mirrors of the domain model (§4-§8, §15-§17)
    api.schemas.accounts    the stored account aggregate (§18 Phase 6)
    api.schemas.probability the probability engine (§10)
    api.schemas.planner     planner and optimizer views (§9, §13-§15)
    api.schemas.simulation  plans, results and jobs (§11, §12)
"""

from api.schemas.accounts import (
    AccountCreate,
    AccountSummary,
    AccountUpdate,
    AccountView,
    BannerListModel,
    BannerListView,
    GoalListModel,
    GoalListView,
    PlannerSettingsModel,
    PreferenceListModel,
    PreferenceListView,
)
from api.schemas.domain import (
    AccountStateModel,
    BannerModel,
    GoalModel,
    IncomeEstimateModel,
    IncomeForecastModel,
    IncomeForecastView,
    IncomeSourceModel,
    MechanicsModel,
    PreferenceModel,
    PreferenceView,
    StrictModel,
    VersionIncomeModel,
    VersionIncomeView,
)
from api.schemas.planner import (
    GoalEvaluationsView,
    GoalEvaluationView,
    GoalStandingView,
    OutcomeView,
    ProtectedGoalOutcomeView,
    RecommendationView,
    RejectedOutcomeView,
    SafeSpendView,
    SpendRowView,
    SpendTableView,
    StopConditionsView,
)
from api.schemas.probability import CharacterProbabilityView, WishesNeededView
from api.schemas.simulation import (
    BannerAggregateView,
    GoalProbabilityView,
    PlannedSpendModel,
    SimulationJobView,
    SimulationResultView,
    SimulationRunRequest,
    SpendPlanModel,
)

__all__ = [
    "AccountCreate",
    "AccountStateModel",
    "AccountSummary",
    "AccountUpdate",
    "AccountView",
    "BannerAggregateView",
    "BannerListModel",
    "BannerListView",
    "BannerModel",
    "CharacterProbabilityView",
    "GoalEvaluationView",
    "GoalEvaluationsView",
    "GoalListModel",
    "GoalListView",
    "GoalModel",
    "GoalProbabilityView",
    "GoalStandingView",
    "IncomeEstimateModel",
    "IncomeForecastModel",
    "IncomeForecastView",
    "IncomeSourceModel",
    "MechanicsModel",
    "OutcomeView",
    "PlannedSpendModel",
    "PlannerSettingsModel",
    "PreferenceListModel",
    "PreferenceListView",
    "PreferenceModel",
    "PreferenceView",
    "ProtectedGoalOutcomeView",
    "RecommendationView",
    "RejectedOutcomeView",
    "SafeSpendView",
    "SimulationJobView",
    "SimulationResultView",
    "SimulationRunRequest",
    "SpendPlanModel",
    "SpendRowView",
    "SpendTableView",
    "StopConditionsView",
    "StrictModel",
    "VersionIncomeModel",
    "VersionIncomeView",
    "WishesNeededView",
]
