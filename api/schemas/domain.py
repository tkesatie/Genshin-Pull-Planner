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

from pydantic import BaseModel, ConfigDict, Field, model_validator

from datetime import datetime

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
    TargetKind,
    WeaponTarget,
    CharacterTarget,
)


class StrictModel(BaseModel):
    """Request shape: unknown fields are a client error, not silence."""

    model_config = ConfigDict(extra="forbid")


class WeaponWishStateModel(StrictModel):
    """Current weapon-banner pity and Epitomized Path state."""

    pity: int = Field(0, ge=0, lt=80)
    guarantee: bool = False
    fate_points: int = Field(0, ge=0, le=2)

    def to_domain(self):
        from domain import WeaponWishState
        return WeaponWishState(
            pity=self.pity,
            guarantee=self.guarantee,
            fate_points=self.fate_points,
        )

    @classmethod
    def from_domain(cls, state) -> "WeaponWishStateModel":
        return cls(
            pity=state.pity,
            guarantee=state.guarantee,
            fate_points=state.fate_points,
        )


class AccountStateModel(StrictModel):
    """Current unified character + weapon account state."""

    current_pity: int = Field(0, ge=0)
    character_guarantee: bool = False
    wishes: int = Field(0, ge=0)
    capturing_radiance_counter: int = Field(0, ge=0, le=3)
    owned_characters: dict[str, int] = Field(default_factory=dict)
    owned_weapons: dict[str, int] = Field(default_factory=dict)
    weapon_state: WeaponWishStateModel = Field(default_factory=WeaponWishStateModel)

    def to_domain(self) -> Account:
        return Account(
            current_pity=self.current_pity,
            character_guarantee=self.character_guarantee,
            owned_characters=Ownership(
                dict(self.owned_characters),
                dict(self.owned_weapons),
            ),
            wishes=self.wishes,
            capturing_radiance_counter=self.capturing_radiance_counter,
            weapon_state=self.weapon_state.to_domain(),
        )

    @classmethod
    def from_domain(cls, account: Account) -> "AccountStateModel":
        return cls(
            current_pity=account.current_pity,
            character_guarantee=account.character_guarantee,
            wishes=account.wishes,
            capturing_radiance_counter=account.capturing_radiance_counter,
            owned_characters=dict(account.owned_characters.characters),
            owned_weapons=dict(account.owned_characters.weapons),
            weapon_state=WeaponWishStateModel.from_domain(account.weapon_state),
        )


class GoalModel(StrictModel):
    """Unified character/weapon goal with legacy character fields accepted."""

    target_kind: TargetKind | None = None
    target_name: str | None = None
    level: int | None = Field(None, ge=0)
    priority: int = Field(..., ge=1)

    # Compatibility fields for the existing character UI/API.
    character: str | None = None
    constellation: int | None = Field(None, ge=0)
    weapon: str | None = None
    refinement: int | None = Field(None, ge=0)

    @model_validator(mode="after")
    def normalize_target(self):
        if self.target_kind is None:
            if self.weapon is not None:
                self.target_kind = TargetKind.WEAPON
                self.target_name = self.weapon
                self.level = self.refinement if self.level is None else self.level
            elif self.character is not None:
                self.target_kind = TargetKind.CHARACTER
                self.target_name = self.character
                self.level = self.constellation if self.level is None else self.level
            else:
                raise ValueError("goal requires a character or weapon target")
        if self.target_name is None or not self.target_name.strip():
            raise ValueError("goal target_name must not be empty")
        if self.level is None:
            raise ValueError("goal requires level")
        if self.target_kind is TargetKind.CHARACTER:
            if self.character is None:
                self.character = self.target_name
            if self.constellation is None:
                self.constellation = self.level
            if self.weapon is not None or self.refinement is not None:
                raise ValueError("character goals cannot specify weapon/refinement")
        else:
            if self.weapon is None:
                self.weapon = self.target_name
            if self.refinement is None:
                self.refinement = self.level
            if self.character is not None or self.constellation is not None:
                raise ValueError("weapon goals cannot specify character/constellation")
        return self

    def to_domain(self) -> Goal:
        target = (
            CharacterTarget(self.target_name)
            if self.target_kind is TargetKind.CHARACTER
            else WeaponTarget(self.target_name)
        )
        return Goal(target=target, level=self.level, priority=self.priority)

    @classmethod
    def from_domain(cls, goal: Goal) -> "GoalModel":
        if goal.target.kind is TargetKind.CHARACTER:
            return cls(
                target_kind=TargetKind.CHARACTER,
                target_name=goal.target.name,
                level=goal.level,
                priority=goal.priority,
                character=goal.target.name,
                constellation=goal.level,
            )
        return cls(
            target_kind=TargetKind.WEAPON,
            target_name=goal.target.name,
            level=goal.level,
            priority=goal.priority,
            weapon=goal.target.name,
            refinement=goal.level,
        )


class BannerModel(StrictModel):
    """Unified character/weapon banner with legacy fields accepted."""

    target_kind: TargetKind | None = None
    target_name: str | None = None
    version: str
    phase: int = Field(1, ge=1)
    start: datetime | None = Field(
        None,
        description=(
            "When the banner goes live, inclusive. Timezone-aware ISO 8601 "
            "only: the planner compares instants, so a naive wall-clock "
            "value is rejected rather than assumed to be UTC or local time."
        ),
    )
    end: datetime | None = Field(
        None,
        description=(
            "When the banner stops being live, exclusive: at this instant "
            "the next banner has taken over. Timezone-aware ISO 8601 only."
        ),
    )
    character: str | None = None
    weapon: str | None = None

    @model_validator(mode="after")
    def normalize_target(self):
        if self.target_kind is None:
            if self.weapon is not None:
                self.target_kind = TargetKind.WEAPON
                self.target_name = self.weapon
            elif self.character is not None:
                self.target_kind = TargetKind.CHARACTER
                self.target_name = self.character
            else:
                raise ValueError("banner requires a character or weapon target")
        if self.target_name is None or not self.target_name.strip():
            raise ValueError("banner target_name must not be empty")
        if self.target_kind is TargetKind.CHARACTER:
            if self.character is None:
                self.character = self.target_name
            if self.weapon is not None:
                raise ValueError("character banners cannot specify weapon")
        else:
            if self.weapon is None:
                self.weapon = self.target_name
            if self.character is not None:
                raise ValueError("weapon banners cannot specify character")
        return self

    def to_domain(self) -> Banner:
        target = (
            CharacterTarget(self.target_name)
            if self.target_kind is TargetKind.CHARACTER
            else WeaponTarget(self.target_name)
        )
        return Banner(
            target=target,
            version=self.version,
            phase=self.phase,
            start=self.start,
            end=self.end,
        )

    @classmethod
    def from_domain(cls, banner: Banner) -> "BannerModel":
        if banner.target.kind is TargetKind.CHARACTER:
            return cls(
                target_kind=TargetKind.CHARACTER,
                target_name=banner.target.name,
                version=banner.version,
                phase=banner.phase,
                start=banner.start,
                end=banner.end,
                character=banner.target.name,
            )
        return cls(
            target_kind=TargetKind.WEAPON,
            target_name=banner.target.name,
            version=banner.version,
            phase=banner.phase,
            start=banner.start,
            end=banner.end,
            weapon=banner.target.name,
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
