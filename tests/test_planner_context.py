"""PlannerContext: planner input validation and the future-income
interpretation (Design Document §16, §18 Phase 3)."""

import pytest

from domain import (
    CHARACTER_EVENT_BANNER,
    Account,
    Banner,
    IncomeEstimate,
    IncomeForecast,
    Roadmap,
    VersionIncome,
)
from planner import PlannerContext


def make(**overrides) -> PlannerContext:
    values = dict(
        account=Account(wishes=40),
        roadmap=Roadmap(banners=[Banner("Vesna", "7.0", 1)]),
        current_version="7.0",
        current_phase=1,
    )
    values.update(overrides)
    return PlannerContext(**values)


@pytest.fixture
def forecast() -> IncomeForecast:
    """A forecast that includes one PAST version (6.9) to exercise the
    future-income interpretation: only 7.0 onward is creditable."""
    return IncomeForecast(
        versions=[
            VersionIncome("6.9", estimate=IncomeEstimate(50, 50, 50)),
            VersionIncome("7.0", estimate=IncomeEstimate(20, 30, 40)),
            VersionIncome("7.1", estimate=IncomeEstimate(20, 30, 40)),
        ]
    )


class TestDefaults:
    def test_documented_defaults(self):
        context = make()
        assert context.confidence == 0.9
        assert context.income_scenario == "expected"
        assert context.mechanics == CHARACTER_EVENT_BANNER
        assert context.current_phase == 1
        assert context.income is None


class TestValidation:
    def test_zero_confidence_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            make(confidence=0.0)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValueError, match="confidence"):
            make(confidence=1.01)

    def test_unknown_income_scenario_rejected(self):
        with pytest.raises(ValueError, match="income scenario"):
            make(income_scenario="optimistic")

    def test_malformed_current_version_rejected(self):
        with pytest.raises(ValueError, match="version"):
            make(current_version="7")

    def test_zero_phase_rejected(self):
        with pytest.raises(ValueError, match="current_phase"):
            make(current_phase=0)

    def test_pity_at_hard_pity_rejected(self):
        with pytest.raises(ValueError, match="current_pity"):
            make(account=Account(current_pity=90, wishes=40))


class TestIncomeCredit:
    def test_no_forecast_means_zero_credit(self):
        assert make().income_credit("7.1") == 0

    def test_current_version_income_counts_as_future(self, forecast):
        """The 7.0 entry is credited: forecast income arrives after the
        current account state, including the current version's share."""
        assert make(income=forecast).income_credit("7.0") == 30

    def test_credit_is_cumulative_through_the_version(self, forecast):
        assert make(income=forecast).income_credit("7.1") == 60

    def test_versions_before_current_are_never_credited(self, forecast):
        """The 6.9 entry is presumed already reflected in account wishes
        - crediting it would double-count."""
        assert make(income=forecast).income_credit("7.1") == 60  # not 110

    def test_scenario_selects_the_bracket(self, forecast):
        low = make(income=forecast, income_scenario="low")
        high = make(income=forecast, income_scenario="high")
        assert low.income_credit("7.1") == 40
        assert high.income_credit("7.1") == 80

    def test_credit_before_current_version_rejected(self, forecast):
        with pytest.raises(ValueError, match="earlier"):
            make(income=forecast).income_credit("6.8")