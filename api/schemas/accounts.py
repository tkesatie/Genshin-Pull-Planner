"""Wire shapes for the account resource (Design Document §18 Phase 6).

The account resource is the aggregate the planner needs: account state
(§4.1) plus the roadmap sub-resources - goals (§5), banners (§7),
preferences (§15), income (§16) - plus planner settings that are neither
account state nor roadmap.

Sub-resources can be supplied at creation for convenience and are
maintained afterwards through their own endpoints, so a client updating
goals never has to resend an account's income.
"""

from pydantic import BaseModel, Field

from api.repository import AccountRecord, PlannerSettings
from api.schemas.domain import (
    AccountStateModel,
    BannerModel,
    GoalModel,
    IncomeForecastModel,
    IncomeForecastView,
    MechanicsModel,
    PreferenceModel,
    PreferenceView,
    StrictModel,
)
from domain import CHARACTER_EVENT_BANNER


class PlannerSettingsModel(StrictModel):
    """Planner input that is not account state (§4.1, §16, §17).

    `mechanics` defaults to the character event banner (§17); supplying it
    explicitly is how updated mechanics reach the engines without a code
    change.
    """

    current_version: str = Field(..., description='Version the user is in, e.g. "7.0".')
    current_phase: int = 1
    confidence: float = Field(
        0.9, description="Required confidence threshold for protected goals (§1)."
    )
    income_scenario: str = Field(
        "expected", description='"low", "expected" or "high" (§16).'
    )
    mechanics: MechanicsModel | None = None

    def to_domain(self) -> PlannerSettings:
        return PlannerSettings(
            current_version=self.current_version,
            current_phase=self.current_phase,
            confidence=self.confidence,
            income_scenario=self.income_scenario,
            mechanics=(
                CHARACTER_EVENT_BANNER
                if self.mechanics is None
                else self.mechanics.to_domain()
            ),
        )

    @classmethod
    def from_domain(cls, settings: PlannerSettings) -> "PlannerSettingsModel":
        return cls(
            current_version=settings.current_version,
            current_phase=settings.current_phase,
            confidence=settings.confidence,
            income_scenario=settings.income_scenario,
            mechanics=MechanicsModel.from_domain(settings.mechanics),
        )


class AccountCreate(StrictModel):
    """Create an account, optionally with its roadmap in one call."""

    label: str = ""
    account: AccountStateModel = Field(default_factory=AccountStateModel)
    settings: PlannerSettingsModel
    goals: list[GoalModel] = Field(default_factory=list)
    banners: list[BannerModel] = Field(default_factory=list)
    preferences: list[PreferenceModel] = Field(default_factory=list)
    income: IncomeForecastModel | None = None


class AccountUpdate(StrictModel):
    """Replace an account's state and settings (§2: re-run after updates).

    Roadmap sub-resources are untouched; they have their own endpoints.
    """

    label: str = ""
    account: AccountStateModel
    settings: PlannerSettingsModel


class AccountSummary(BaseModel):
    """List view of a stored account."""

    id: str
    label: str
    wishes: int
    current_version: str
    current_phase: int
    goal_count: int
    banner_count: int

    @classmethod
    def from_record(cls, record: AccountRecord) -> "AccountSummary":
        return cls(
            id=record.id,
            label=record.label,
            wishes=record.account.wishes,
            current_version=record.settings.current_version,
            current_phase=record.settings.current_phase,
            goal_count=len(record.goals),
            banner_count=len(record.banners),
        )


class AccountView(BaseModel):
    """Full view of a stored account and its roadmap."""

    id: str
    label: str
    account: AccountStateModel
    settings: PlannerSettingsModel
    goals: list[GoalModel]
    banners: list[BannerModel]
    preferences: list[PreferenceView]
    income: IncomeForecastView | None

    @classmethod
    def from_record(cls, record: AccountRecord) -> "AccountView":
        return cls(
            id=record.id,
            label=record.label,
            account=AccountStateModel.from_domain(record.account),
            settings=PlannerSettingsModel.from_domain(record.settings),
            goals=[GoalModel.from_domain(goal) for goal in record.goals],
            banners=[BannerModel.from_domain(banner) for banner in record.banners],
            preferences=[
                PreferenceView.from_domain(preference)
                for preference in record.preferences
            ],
            income=(
                None
                if record.income is None
                else IncomeForecastView.from_domain(record.income)
            ),
        )


class GoalListModel(StrictModel):
    """Replacement set of roadmap goals (§5)."""

    goals: list[GoalModel]


class BannerListModel(StrictModel):
    """Replacement banner schedule (§7)."""

    banners: list[BannerModel]


class PreferenceListModel(StrictModel):
    """Replacement preference chains (§15)."""

    preferences: list[PreferenceModel]


class GoalListView(BaseModel):
    """Stored roadmap goals, in priority order (§5)."""

    goals: list[GoalModel]


class BannerListView(BaseModel):
    """Stored banners, in chronological order (§7)."""

    banners: list[BannerModel]


class PreferenceListView(BaseModel):
    """Stored preferences with derived labels (§15)."""

    preferences: list[PreferenceView]
