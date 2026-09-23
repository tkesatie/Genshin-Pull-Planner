"""Phase 6 banner dates: intervals, timezones and transitions (§7).

Dates are optional, auxiliary metadata: (version, phase) stays the primary
deterministic planning input, and a real timestamp answers "what is live
right now" through the same banner schedule. The interval is half-open
(`start` inclusive, `end` exclusive) so a hand-off instant belongs to exactly
one banner, and every bound is timezone-aware - the planner compares instants
and will not guess UTC or local time for a naive wall-clock value.
"""

from datetime import datetime, timedelta, timezone

import pytest

from api.repository import AccountRecord, PlannerSettings
from api.sqlite_repository import SQLiteAccountRepository
from domain import Account, Banner, Goal, Roadmap, WeaponTarget
from planner import PlannerContext, available_banners, banners_active_at

UTC = timezone.utc
SERVER_ZONE = timezone(timedelta(hours=8))  # e.g. an Asia server's own zone


def moment(day: int, *, month: int = 1, hour: int = 12, zone=UTC) -> datetime:
    """A concrete instant in 2026, for readable boundary tests."""
    return datetime(2026, month, day, hour, tzinfo=zone)


class TestIntervalRules:
    def test_dates_are_optional(self):
        plain = Banner("Vesna", "7.0", 1)
        assert plain.start is None
        assert plain.end is None
        assert plain.has_dates is False

    def test_a_valid_aware_interval_is_accepted(self):
        dated = Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21))
        assert dated.has_dates is True
        assert dated.start == moment(1)
        assert dated.end == moment(21)

    @pytest.mark.parametrize("zone", [UTC, SERVER_ZONE])
    def test_any_real_offset_is_accepted(self, zone):
        """The planner stores the caller's offset instead of imposing one."""
        dated = Banner("Vesna", "7.0", 1, start=moment(1, zone=zone), end=moment(21, zone=zone))
        assert dated.is_active_at(moment(10, zone=zone)) is True

    def test_naive_bounds_are_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            Banner("Vesna", "7.0", 1, start=datetime(2026, 1, 1), end=moment(21))
        with pytest.raises(ValueError, match="timezone-aware"):
            Banner("Vesna", "7.0", 1, start=moment(1), end=datetime(2026, 1, 21))

    def test_half_specified_interval_is_rejected(self):
        with pytest.raises(ValueError, match="provided together"):
            Banner("Vesna", "7.0", 1, start=moment(1))
        with pytest.raises(ValueError, match="provided together"):
            Banner("Vesna", "7.0", 1, end=moment(21))

    def test_reversed_interval_is_rejected(self):
        with pytest.raises(ValueError, match="before end"):
            Banner("Vesna", "7.0", 1, start=moment(21), end=moment(1))

    def test_empty_interval_is_rejected(self):
        """A zero-length banner is a hand-off mistake, not a schedule."""
        with pytest.raises(ValueError, match="before end"):
            Banner("Vesna", "7.0", 1, start=moment(1), end=moment(1))


class TestActiveAt:
    def test_before_during_and_after(self):
        dated = Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21))
        assert dated.is_active_at(moment(1) - timedelta(minutes=1)) is False
        assert dated.is_active_at(moment(1)) is True
        assert dated.is_active_at(moment(10)) is True
        assert dated.is_active_at(moment(21)) is False
        assert dated.is_active_at(moment(21) + timedelta(minutes=1)) is False

    def test_the_same_instant_in_another_zone_agrees(self):
        """Instants, not wall clocks: equal moments compare equal."""
        dated = Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21))
        # 2026-01-10 12:00+00:00 is 2026-01-10 20:00+08:00.
        same_instant = moment(10, hour=20, zone=SERVER_ZONE)
        assert same_instant == moment(10)
        assert dated.is_active_at(same_instant) is True

    def test_a_banner_without_dates_cannot_be_judged(self):
        with pytest.raises(ValueError, match="has no dates"):
            Banner("Vesna", "7.0", 1).is_active_at(moment(10))

    def test_a_naive_moment_is_rejected(self):
        dated = Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21))
        with pytest.raises(ValueError, match="timezone-aware"):
            dated.is_active_at(datetime(2026, 1, 10, 12))


def schedule_context(*banners: Banner) -> PlannerContext:
    return PlannerContext(
        account=Account(wishes=40),
        roadmap=Roadmap(banners=list(banners)),
        current_version="7.0",
        current_phase=1,
    )


def two_phase_schedule() -> PlannerContext:
    """7.0 phase 1 hands over to 7.0 phase 2, which hands over to 7.1 phase 1."""
    return schedule_context(
        Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21)),
        Banner("Astra", "7.0", 2, start=moment(21), end=moment(10, month=2)),
        Banner("Tsaritsa", "7.1", 1, start=moment(10, month=2), end=moment(2, month=3)),
    )


class TestTransitions:
    def test_before_the_first_start_nothing_is_live(self):
        assert banners_active_at(two_phase_schedule(), moment(1) - timedelta(days=1)) == ()

    def test_during_a_banner_that_banner_is_live(self):
        assert banners_active_at(two_phase_schedule(), moment(10)) == (
            Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21)),
        )

    def test_the_hand_off_instant_belongs_to_the_next_phase_only(self):
        """Half-open intervals: phase 1 ends exactly where phase 2 begins."""
        assert banners_active_at(two_phase_schedule(), moment(21)) == (
            Banner("Astra", "7.0", 2, start=moment(21), end=moment(10, month=2)),
        )

    def test_a_new_version_transitions_like_a_new_phase(self):
        assert banners_active_at(two_phase_schedule(), moment(10, month=2)) == (
            Banner(
                "Tsaritsa",
                "7.1",
                1,
                start=moment(10, month=2),
                end=moment(2, month=3),
            ),
        )

    def test_after_the_last_end_nothing_is_live(self):
        assert banners_active_at(two_phase_schedule(), moment(2, month=3)) == ()
        assert banners_active_at(two_phase_schedule(), moment(3, month=3)) == ()

    def test_simultaneous_dated_banners_are_both_live(self):
        """A character and a weapon banner sharing a slot both count as now."""
        context = schedule_context(
            Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21)),
            Banner(
                target=WeaponTarget("Astra"),
                version="7.0",
                phase=1,
                start=moment(5),
                end=moment(21),
            ),
        )
        live = banners_active_at(context, moment(10))
        assert [item.target.name for item in live] == ["Astra", "Vesna"]

    def test_a_banner_without_dates_is_skipped_not_assumed_live(self):
        context = schedule_context(
            Banner("Vesna", "7.0", 1),
            Banner("Astra", "7.0", 1, start=moment(1), end=moment(21)),
        )
        assert [
            item.target.name for item in banners_active_at(context, moment(10))
        ] == ["Astra"]

    def test_a_naive_moment_is_rejected(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            banners_active_at(two_phase_schedule(), datetime(2026, 1, 10, 12))

        with pytest.raises(ValueError, match="before end"):
            Banner("Vesna", "7.0", 1, start=moment(1), end=moment(1))



class TestDatesDoNotReplaceVersions:
    def test_positional_lookup_ignores_dates(self):
        """(version, phase) stays the deterministic planning input (§7)."""
        context = PlannerContext(
            account=Account(wishes=40),
            roadmap=Roadmap(
                banners=[
                    Banner("Vesna", "7.0", 2, start=moment(1), end=moment(21)),
                    Banner("Tsaritsa", "7.0", 1, start=moment(21), end=moment(31)),
                ]
            ),
            current_version="7.0",
            current_phase=1,
        )
        # Where dates disagree with the declared phase, the phase wins.
        assert available_banners(context) == (
            Banner("Tsaritsa", "7.0", 1, start=moment(21), end=moment(31)),
        )

    def test_chronological_order_remains_version_then_phase(self):
        dated_late = Banner("Vesna", "7.0", 1, start=moment(21), end=moment(31))
        dated_early = Banner("Tsaritsa", "7.0", 2, start=moment(1), end=moment(21))
        roadmap = Roadmap(banners=[dated_late, dated_early])
        assert roadmap.banners_in_chronological_order() == [dated_late, dated_early]


class TestPersistence:
    def test_sqlite_round_trips_banner_dates(self, tmp_path):
        record = AccountRecord(
            id="dated",
            label="dated schedule",
            account=Account(wishes=40),
            settings=PlannerSettings(current_version="7.0"),
            goals=(Goal("Vesna", 0, 1),),
            banners=(Banner("Vesna", "7.0", 1, start=moment(1), end=moment(21)),),
        )
        path = tmp_path / "planner.sqlite3"
        SQLiteAccountRepository(path).create(record)
        assert SQLiteAccountRepository(path).get("dated") == record

    def test_sqlite_round_trips_a_non_utc_offset(self, tmp_path):
        """The caller's offset survives storage; it is not normalized away."""
        record = AccountRecord(
            id="zoned",
            label="server zone",
            account=Account(wishes=40),
            settings=PlannerSettings(current_version="7.0"),
            banners=(
                Banner(
                    "Vesna",
                    "7.0",
                    1,
                    start=moment(1, zone=SERVER_ZONE),
                    end=moment(21, zone=SERVER_ZONE),
                ),
            ),
        )
        path = tmp_path / "planner.sqlite3"
        SQLiteAccountRepository(path).create(record)
        loaded = SQLiteAccountRepository(path).get("zoned")
        (stored_banner,) = loaded.banners
        assert stored_banner.start == moment(1, zone=SERVER_ZONE)
        assert stored_banner.start.utcoffset() == timedelta(hours=8)
