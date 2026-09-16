"""Wire shapes for the domain model (Design Document §4-§8, §15-§17).

These models describe the *shape* of a request or response - field names
and types. They deliberately do not re-state the domain's value rules
(§4.2's NOT_OWNED sentinel, §8's unique priorities, §16's
low <= expected <= high, §17's mechanics bounds). `to_domain()` builds the
frozen domain object and lets it validate, so there is exactly one place
where each rule lives and the client sees the domain's own message.

Labels such as "C2R1" are derived for display and never accepted as input
(§15).
"""

from pydantic import BaseModel, ConfigDict, Field

from domain import (
    Account,
    Banner,
    Goal,
    IncomeEstimate,
    IncomeForecast,
    IncomeSource,
    Ownership,
    Preference,
    VersionIncome,
    WishMechanics,
)


class StrictModel(BaseModel):
    """Request shape: unknown fields are a client error, not silence."""

    model_config = ConfigDict(extra="forbid")


class AccountStateModel(StrictModel):
    """Current account state (§4.1)."""

    current_pity: int = Field(
        0, description="Pulls made on the character banner since the last 5-star."
    )
    character_guarantee: bool = Field(
        False, description="True when the next 5-star is guaranteed featured."
    )
    wishes: int = Field(0, description="Wishes currently owned, as a resource.")
    owned_characters: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Owned constellation per character: -1 not owned, 0 is C0, 1 is C1. "
            "A constellation is not a number of copies (§4.2)."
        ),
    )

    def to_domain(self) -> Account:
        return Account(
            current_pity=self.current_pity,
            character_guarantee=self.character_guarantee,
            owned_characters=Ownership(dict(self.owned_characters)),
            wishes=self.wishes,
        )

    @classmethod
    def from_domain(cls, account: Account) -> "AccountStateModel":
        return cls(
            current_pity=account.current_pity,
            character_guarantee=account.character_guarantee,
            wishes=account.wishes,
            owned_characters=dict(account.owned_characters.characters),
        )


class GoalModel(StrictModel):
    """A roadmap objective (§5)."""

    character: str
    constellation: int = Field(
        ..., description='Goal constellation ("C2" is 2). Not a copy count.'
    )
    priority: int = Field(
        ..., description="Protection order; 1 is protected first. Unique per roadmap."
    )

    def to_domain(self) -> Goal:
        return Goal(
            character=self.character,
            constellation=self.constellation,
            priority=self.priority,
        )

    @classmethod
    def from_domain(cls, goal: Goal) -> "GoalModel":
        return cls(
            character=goal.character,
            constellation=goal.constellation,
            priority=goal.priority,
        )


class BannerModel(StrictModel):
    """An opportunity to pull (§7)."""

    character: str
    version: str = Field(..., description='Game version, e.g. "7.0".')
    phase: int = Field(1, description="1-based phase within the version.")

    def to_domain(self) -> Banner:
        return Banner(
            character=self.character, version=self.version, phase=self.phase
        )

    @classmethod
    def from_domain(cls, banner: Banner) -> "BannerModel":
        return cls(
            character=banner.character, version=banner.version, phase=banner.phase
        )


class PreferenceModel(StrictModel):
    """One acceptable outcome for a character (§15)."""

    character: str
    rank: int = Field(..., description="Position in the chain; 1 is most preferred.")
    constellation: int
    weapon_refinement: int = Field(
        0,
        description=(
            "0 means no refinement wanted, 1 means R1. Displayed, never "
            "simulated (§17, §19)."
        ),
    )
    notes: str = ""

    def to_domain(self) -> Preference:
        return Preference(
            character=self.character,
            rank=self.rank,
            constellation=self.constellation,
            weapon_refinement=self.weapon_refinement,
            notes=self.notes,
        )


class PreferenceView(BaseModel):
    """A stored preference with its derived display label (§15)."""

    character: str
    rank: int
    constellation: int
    weapon_refinement: int
    notes: str
    label: str = Field(..., description='Derived for display, e.g. "C2R1".')

    @classmethod
    def from_domain(cls, preference: Preference) -> "PreferenceView":
        return cls(
            character=preference.character,
            rank=preference.rank,
            constellation=preference.constellation,
            weapon_refinement=preference.weapon_refinement,
            notes=preference.notes,
            label=preference.label,
        )


class IncomeEstimateModel(StrictModel):
    """A ranged wish estimate (§16)."""

    low: int
    expected: int
    high: int

    def to_domain(self) -> IncomeEstimate:
        return IncomeEstimate(low=self.low, expected=self.expected, high=self.high)

    @classmethod
    def from_domain(cls, estimate: IncomeEstimate) -> "IncomeEstimateModel":
        return cls(
            low=estimate.low, expected=estimate.expected, high=estimate.high
        )


class IncomeSourceModel(StrictModel):
    """Optional income breakdown by source (§16)."""

    name: str
    estimate: IncomeEstimateModel

    def to_domain(self) -> IncomeSource:
        return IncomeSource(name=self.name, estimate=self.estimate.to_domain())

    @classmethod
    def from_domain(cls, source: IncomeSource) -> "IncomeSourceModel":
        return cls(
            name=source.name,
            estimate=IncomeEstimateModel.from_domain(source.estimate),
        )


class VersionIncomeModel(StrictModel):
    """Income expected for one version (§16).

    Supply `estimate`, `sources`, or both when they agree - the domain
    decides what a valid combination is.
    """

    version: str
    estimate: IncomeEstimateModel | None = None
    sources: list[IncomeSourceModel] = Field(default_factory=list)

    def to_domain(self) -> VersionIncome:
        return VersionIncome(
            version=self.version,
            estimate=None if self.estimate is None else self.estimate.to_domain(),
            sources=[source.to_domain() for source in self.sources],
        )


class VersionIncomeView(BaseModel):
    """Stored version income plus its derived aggregate forecast (§16)."""

    version: str
    estimate: IncomeEstimateModel | None
    sources: list[IncomeSourceModel]
    aggregate: IncomeEstimateModel

    @classmethod
    def from_domain(cls, entry: VersionIncome) -> "VersionIncomeView":
        return cls(
            version=entry.version,
            estimate=(
                None
                if entry.estimate is None
                else IncomeEstimateModel.from_domain(entry.estimate)
            ),
            sources=[
                IncomeSourceModel.from_domain(source) for source in entry.sources
            ],
            aggregate=IncomeEstimateModel.from_domain(entry.aggregate),
        )


class IncomeForecastModel(StrictModel):
    """Income expected across future versions (§16)."""

    versions: list[VersionIncomeModel] = Field(default_factory=list)

    def to_domain(self) -> IncomeForecast:
        return IncomeForecast(
            versions=[entry.to_domain() for entry in self.versions]
        )


class IncomeForecastView(BaseModel):
    """A stored forecast, in version order (§16)."""

    versions: list[VersionIncomeView]

    @classmethod
    def from_domain(cls, forecast: IncomeForecast) -> "IncomeForecastView":
        return cls(
            versions=[
                VersionIncomeView.from_domain(entry)
                for entry in forecast.in_version_order()
            ]
        )


class MechanicsModel(StrictModel):
    """Wish mechanics as data (§17)."""

    banner_type: str
    hard_pity: int
    soft_pity_start: int
    base_rate: float
    soft_pity_increment: float
    featured_rate: float

    def to_domain(self) -> WishMechanics:
        return WishMechanics(
            banner_type=self.banner_type,
            hard_pity=self.hard_pity,
            soft_pity_start=self.soft_pity_start,
            base_rate=self.base_rate,
            soft_pity_increment=self.soft_pity_increment,
            featured_rate=self.featured_rate,
        )

    @classmethod
    def from_domain(cls, mechanics: WishMechanics) -> "MechanicsModel":
        return cls(
            banner_type=mechanics.banner_type,
            hard_pity=mechanics.hard_pity,
            soft_pity_start=mechanics.soft_pity_start,
            base_rate=mechanics.base_rate,
            soft_pity_increment=mechanics.soft_pity_increment,
            featured_rate=mechanics.featured_rate,
        )
