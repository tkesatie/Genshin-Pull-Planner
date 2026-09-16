"""Income (Design Document §16)."""

import pytest

from domain import IncomeEstimate, IncomeForecast, IncomeSource, VersionIncome


class TestIncomeEstimate:
    def test_scenario_accessor(self):
        estimate = IncomeEstimate(low=10, expected=20, high=30)
        assert estimate.scenario("low") == 10
        assert estimate.scenario("expected") == 20
        assert estimate.scenario("high") == 30

    def test_unknown_scenario_rejected(self):
        with pytest.raises(ValueError, match="scenario"):
            IncomeEstimate(10, 20, 30).scenario("maximum")

    def test_ordering_is_validated(self):
        with pytest.raises(ValueError):
            IncomeEstimate(low=30, expected=20, high=10)

    def test_negative_values_rejected(self):
        with pytest.raises(ValueError):
            IncomeEstimate(low=-1, expected=0, high=10)

    def test_addition_combines_scenarios(self):
        combined = IncomeEstimate(10, 20, 30) + IncomeEstimate(1, 2, 3)
        assert combined == IncomeEstimate(11, 22, 33)


class TestVersionIncome:
    def test_direct_estimate(self):
        entry = VersionIncome("7.1", estimate=IncomeEstimate(10, 20, 30))
        assert entry.aggregate == IncomeEstimate(10, 20, 30)

    def test_sources_are_summed_into_the_aggregate(self):
        entry = VersionIncome(
            "7.1",
            sources=[
                IncomeSource("Commissions", IncomeEstimate(5, 6, 7)),
                IncomeSource("Events", IncomeEstimate(5, 8, 13)),
            ],
        )
        assert entry.aggregate == IncomeEstimate(10, 14, 20)

    def test_matching_estimate_and_sources_are_accepted(self):
        entry = VersionIncome(
            "7.1",
            estimate=IncomeEstimate(10, 14, 20),
            sources=[
                IncomeSource("Commissions", IncomeEstimate(5, 6, 7)),
                IncomeSource("Events", IncomeEstimate(5, 8, 13)),
            ],
        )
        assert entry.aggregate == IncomeEstimate(10, 14, 20)

    def test_mismatching_estimate_and_sources_are_rejected(self):
        with pytest.raises(ValueError, match="does not match"):
            VersionIncome(
                "7.1",
                estimate=IncomeEstimate(10, 14, 20),
                sources=[IncomeSource("Commissions", IncomeEstimate(5, 6, 7))],
            )

    def test_requires_estimate_or_sources(self):
        with pytest.raises(ValueError):
            VersionIncome("7.1")

    def test_malformed_version_rejected(self):
        with pytest.raises(ValueError):
            VersionIncome("7", estimate=IncomeEstimate(0, 0, 0))


class TestIncomeForecast:
    @pytest.fixture
    def forecast(self) -> IncomeForecast:
        return IncomeForecast(
            versions=[
                VersionIncome("7.0", estimate=IncomeEstimate(30, 40, 50)),
                VersionIncome("7.1", estimate=IncomeEstimate(10, 20, 30)),
                VersionIncome("7.2", estimate=IncomeEstimate(5, 5, 5)),
            ]
        )

    def test_for_version(self, forecast):
        assert forecast.for_version("7.1").aggregate.expected == 20

    def test_for_unknown_version_raises_key_error(self, forecast):
        with pytest.raises(KeyError):
            forecast.for_version("8.0")

    def test_versions_order_numerically(self):
        """String sorting would order "7.10" before "7.9"."""
        forecast = IncomeForecast(
            versions=[
                VersionIncome("7.10", estimate=IncomeEstimate(1, 1, 1)),
                VersionIncome("7.9", estimate=IncomeEstimate(1, 1, 1)),
            ]
        )
        assert [e.version for e in forecast.in_version_order()] == ["7.9", "7.10"]

    def test_cumulative_through_includes_the_named_version(self, forecast):
        assert forecast.cumulative_through("7.1") == 60  # 40 + 20 expected

    def test_cumulative_through_supports_scenarios(self, forecast):
        assert forecast.cumulative_through("7.1", "low") == 40
        assert forecast.cumulative_through("7.1", "high") == 80
        assert forecast.cumulative_through("7.2", "expected") == 65

    def test_cumulative_through_rejects_unknown_scenario(self, forecast):
        with pytest.raises(ValueError, match="scenario"):
            forecast.cumulative_through("7.1", "bogus")

    def test_cumulative_through_before_any_income_is_zero(self, forecast):
        assert forecast.cumulative_through("6.9") == 0

    def test_duplicate_versions_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            IncomeForecast(
                versions=[
                    VersionIncome("7.0", estimate=IncomeEstimate(1, 1, 1)),
                    VersionIncome("7.0", estimate=IncomeEstimate(2, 2, 2)),
                ]
            )
