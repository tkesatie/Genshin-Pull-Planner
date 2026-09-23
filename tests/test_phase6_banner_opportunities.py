"""Phase 6 banner opportunities: what exists now, what is next, in what order.

The planner answers three questions about the banner schedule (§7): which
opportunities are live at the current (version, phase), which come after it,
and when a specific target's next opportunity falls. All of it is
target-agnostic - a weapon banner has no `.character` - so none of these paths
may go through a character accessor.
"""

import pytest

from domain import (
    Account,
    Banner,
    CharacterTarget,
    Goal,
    Roadmap,
    TargetKind,
    WeaponTarget,
)
from planner import (
    PlannerContext,
    available_banners,
    current_banner,
    evaluate_goals,
    next_banner_for,
    position_anchor,
    protected_goal_outcomes,
    single_copy_active_goal,
    upcoming_banners,
)


def weapon_banner(name: str, version: str = "7.0", phase: int = 1) -> Banner:
    return Banner(target=WeaponTarget(name), version=version, phase=phase)


def schedule(
    banners: list[Banner],
    *,
    goals: list[Goal] | None = None,
    version: str = "7.0",
    phase: int = 1,
    wishes: int = 40,
) -> PlannerContext:
    return PlannerContext(
        account=Account(wishes=wishes),
        roadmap=Roadmap(banners=banners, goals=goals or []),
        current_version=version,
        current_phase=phase,
    )


class TestCurrentOpportunities:
    def test_one_current_character_banner(self):
        context = schedule([Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)])
        assert available_banners(context) == (Banner("Vesna", "7.0", 1),)
        assert current_banner(context) == Banner("Vesna", "7.0", 1)
        # A banner in another slot is not a current opportunity.
        assert [banner.target.name for banner in upcoming_banners(context)] == [
            "Tsaritsa"
        ]

    def test_multiple_simultaneous_character_banners(self):
        context = schedule([Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.0", 1)])
        assert available_banners(context) == (
            Banner("Tsaritsa", "7.0", 1),
            Banner("Vesna", "7.0", 1),
        )
        # The legacy singular helper still refuses to choose between them.
        with pytest.raises(ValueError, match="ambiguous"):
            current_banner(context)

    def test_character_and_weapon_banner_in_the_same_slot(self):
        context = schedule([Banner("Vesna", "7.0", 1), weapon_banner("Astra")])
        available = available_banners(context)
        assert [(banner.target.kind, banner.target.name) for banner in available] == [
            (TargetKind.WEAPON, "Astra"),
            (TargetKind.CHARACTER, "Vesna"),
        ]
        # Nothing on this path touched `.character`: the weapon banner has none.
        assert [banner.target.name for banner in available] == ["Astra", "Vesna"]
        assert position_anchor(context).order_key == (7, 0, 1)

    def test_weapon_only_current_slot(self):
        context = schedule([weapon_banner("Astra"), Banner("Vesna", "7.1", 1)])
        assert available_banners(context) == (weapon_banner("Astra"),)
        # Singular because the slot really holds one opportunity - not because
        # the planner assumed every banner must be a character banner.
        assert current_banner(context).target == WeaponTarget("Astra")

    def test_no_current_opportunity_is_an_empty_tuple(self):
        context = schedule([Banner("Vesna", "7.0", 1)], phase=2)
        assert available_banners(context) == ()


class TestUpcoming:
    def test_versions_are_ordered_numerically(self):
        """7.9 precedes 7.10: string comparison would invert them (§7)."""
        context = schedule(
            [
                Banner("Late", "7.10", 1),
                Banner("Early", "7.9", 2),
                Banner("Earliest", "7.9", 1),
            ],
            version="7.9",
            phase=1,
        )
        assert [
            (banner.target.name, banner.version, banner.phase)
            for banner in upcoming_banners(context)
        ] == [("Early", "7.9", 2), ("Late", "7.10", 1)]

    def test_same_slot_alternatives_are_current_not_upcoming(self):
        context = schedule([Banner("Vesna", "7.0", 1), weapon_banner("Astra")])
        assert upcoming_banners(context) == ()

    def test_upcoming_is_empty_after_the_schedule_ends(self):
        context = schedule([Banner("Vesna", "7.0", 1)], version="8.0", phase=1)
        assert upcoming_banners(context) == ()
        assert available_banners(context) == ()

    def test_position_anchor_refuses_an_empty_slot(self):
        context = schedule([Banner("Vesna", "7.1", 1)], version="7.0", phase=1)
        with pytest.raises(ValueError, match="no roadmap banner"):
            position_anchor(context)


class TestNextBannerFor:
    def test_a_target_available_now_is_its_next_opportunity(self):
        context = schedule([Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)])
        assert next_banner_for(context, CharacterTarget("Vesna")) == Banner(
            "Vesna", "7.0", 1
        )

    def test_a_future_target_gets_its_later_banner(self):
        context = schedule([Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)])
        assert next_banner_for(context, CharacterTarget("Tsaritsa")) == Banner(
            "Tsaritsa", "7.1", 1
        )

    def test_weapon_targets_are_matched_by_unified_target_identity(self):
        context = schedule([Banner("Vesna", "7.0", 1), weapon_banner("Astra", "7.1", 2)])
        assert next_banner_for(context, WeaponTarget("Astra")) == weapon_banner(
            "Astra", "7.1", 2
        )

    def test_an_unscheduled_target_has_no_next_banner(self):
        context = schedule([Banner("Vesna", "7.0", 1)])
        assert next_banner_for(context, CharacterTarget("Skirk")) is None

    def test_an_explicit_anchor_moves_the_search_forward(self):
        context = schedule([Banner("Vesna", "7.0", 1), Banner("Vesna", "7.1", 1)])
        assert next_banner_for(
            context, CharacterTarget("Vesna"), after=Banner("Vesna", "7.1", 1)
        ) == Banner("Vesna", "7.1", 1)


class TestPhaseTransitions:
    def test_walking_the_schedule_position_by_position(self):
        roadmap_banners = [
            Banner("Vesna", "7.0", 1),
            Banner("Astra", "7.0", 2),
            Banner("Tsaritsa", "7.1", 1),
            Banner("Vodynista", "7.1", 2),
        ]
        seen: list[str] = []
        for version, phase in (
            ("7.0", 1),
            ("7.0", 2),
            ("7.1", 1),
            ("7.1", 2),
        ):
            context = schedule(roadmap_banners, version=version, phase=phase)
            (live,) = available_banners(context)
            seen.append(f"{live.version}p{live.phase}:{live.target.name}")
            assert all(
                banner.order_key > live.order_key
                for banner in upcoming_banners(context)
            )

        assert seen == [
            "7.0p1:Vesna",
            "7.0p2:Astra",
            "7.1p1:Tsaritsa",
            "7.1p2:Vodynista",
        ]


class TestTargetIndependence:
    def test_the_same_helpers_serve_characters_and_weapons(self):
        context = schedule([Banner("Vesna", "7.0", 1), weapon_banner("Astra")])
        assert next_banner_for(context, CharacterTarget("Vesna")) == Banner(
            "Vesna", "7.0", 1
        )
        assert next_banner_for(context, WeaponTarget("Astra")) == weapon_banner("Astra")
        assert {banner.target for banner in available_banners(context)} == {
            CharacterTarget("Vesna"),
            WeaponTarget("Astra"),
        }
        # The character accessor really is unusable for a weapon banner, so
        # the assertions above cannot have gone through it.
        with pytest.raises(AttributeError):
            _ = weapon_banner("Astra").character

    def test_goal_evaluation_reports_each_targets_next_opportunity(self):
        context = schedule(
            [Banner("Vesna", "7.0", 1), weapon_banner("Astra", "7.1", 1)],
            goals=[
                Goal("Vesna", 0, 1),
                Goal(target=WeaponTarget("Astra"), level=1, priority=2),
            ],
        )
        by_target = {
            evaluation.goal.target: evaluation
            for evaluation in evaluate_goals(context)
        }
        assert by_target[CharacterTarget("Vesna")].next_banner == Banner(
            "Vesna", "7.0", 1
        )
        assert by_target[WeaponTarget("Astra")].next_banner == weapon_banner(
            "Astra", "7.1", 1
        )

    def test_a_multi_copy_weapon_goal_is_refused_with_its_own_label(self):
        """The refusal names a weapon goal without a character accessor."""
        context = schedule(
            [weapon_banner("Astra")],
            goals=[Goal(target=WeaponTarget("Astra"), level=2, priority=1)],
        )
        with pytest.raises(ValueError, match="Astra R2"):
            single_copy_active_goal(context)

    def test_protection_works_in_a_slot_with_several_opportunities(self):
        """Protection anchors on the position, not on one chosen banner."""
        context = schedule(
            [
                Banner("Vesna", "7.0", 1),
                Banner("Tsaritsa", "7.0", 1),
                Banner("Vodynista", "7.1", 1),
            ],
            goals=[
                Goal("Vesna", 0, 1),
                Goal("Tsaritsa", 0, 2),
                Goal("Vodynista", 0, 3),
            ],
            wishes=200,
        )
        outcomes = protected_goal_outcomes(context, spent=0)
        assert [outcome.goal.target.name for outcome in outcomes] == ["Vodynista"]
        assert outcomes[0].banner == Banner("Vodynista", "7.1", 1)


class TestApiBannerExposure:
    def test_planner_goals_expose_current_and_upcoming_banners(
        self, api_client, doc_account_id
    ):
        body = api_client.get(f"/accounts/{doc_account_id}/planner/goals").json()
        current = body["current_banner"]
        assert current["target_kind"] == "character"
        assert current["target_name"] == "Vesna"
        assert current["character"] == "Vesna"
        assert current["version"] == "7.0"
        assert current["phase"] == 1
        assert current["start"] is None
        assert current["end"] is None

        # Everything after the current position, chronologically - which is
        # exactly what a "what's next" list needs.
        assert [
            (banner["target_name"], banner["version"], banner["phase"])
            for banner in body["upcoming_banners"]
        ] == [("Tsaritsa", "7.1", 1), ("Vodynista", "7.2", 1)]

        # Per-goal next opportunity, so the UI never has to match banners
        # against goals itself. Planner goal payloads keep their legacy shape
        # (character/constellation or weapon/refinement, Phase 5).
        by_goal = {
            goal["goal"].get("character") or goal["goal"].get("weapon"): goal
            for goal in body["goals"]
        }
        assert by_goal["Tsaritsa"]["next_banner"]["version"] == "7.1"
        assert by_goal["Vodynista"]["next_banner"]["target_name"] == "Vodynista"

    def test_a_slot_with_a_character_and_a_weapon_banner_reports_both(
        self, api_client
    ):
        created = api_client.post(
            "/accounts",
            json={
                "account": {"wishes": 80},
                "settings": {"current_version": "7.1", "current_phase": 1},
                "banners": [
                    {"character": "Vesna", "version": "7.1", "phase": 1},
                    {"weapon": "Astra", "version": "7.1", "phase": 1},
                    {"character": "Tsaritsa", "version": "7.1", "phase": 2},
                ],
                "goals": [
                    {"character": "Vesna", "constellation": 0, "priority": 1},
                    {"weapon": "Astra", "refinement": 1, "priority": 2},
                    {"character": "Tsaritsa", "constellation": 0, "priority": 3},
                ],
            },
        )
        assert created.status_code == 201, created.text
        account_id = created.json()["id"]

        body = api_client.get(f"/accounts/{account_id}/planner/goals").json()
        assert body["current_banner"] is None
        assert {
            (banner["target_kind"], banner["target_name"])
            for banner in body["available_banners"]
        } == {("character", "Vesna"), ("weapon", "Astra")}
        assert [
            (banner["target_kind"], banner["target_name"], banner["version"], banner["phase"])
            for banner in body["upcoming_banners"]
        ] == [("character", "Tsaritsa", "7.1", 2)]

    def test_dated_banners_round_trip_with_their_offset(self, api_client, doc_account_id):
        response = api_client.put(
            f"/accounts/{doc_account_id}/banners",
            json={
                "banners": [
                    {
                        "target_kind": "weapon",
                        "target_name": "Astra",
                        "version": "7.0",
                        "phase": 1,
                        "start": "2026-01-01T11:00:00+08:00",
                        "end": "2026-01-21T17:59:59+08:00",
                    }
                ]
            },
        )
        assert response.status_code == 200, response.text
        (banner,) = response.json()["banners"]
        assert banner["target_kind"] == "weapon"
        assert banner["target_name"] == "Astra"
        assert banner["weapon"] == "Astra"
        assert banner["character"] is None
        assert banner["start"] == "2026-01-01T11:00:00+08:00"
        assert banner["end"] == "2026-01-21T17:59:59+08:00"

    def test_invalid_dates_are_rejected_with_the_domain_message(
        self, api_client, doc_account_id
    ):
        def put(start: str, end: str):
            return api_client.put(
                f"/accounts/{doc_account_id}/banners",
                json={
                    "banners": [
                        {
                            "character": "Vesna",
                            "version": "7.0",
                            "phase": 1,
                            "start": start,
                            "end": end,
                        }
                    ]
                },
            )

        naive = put("2026-01-01T12:00:00", "2026-01-21T12:00:00")
        assert naive.status_code == 422
        assert "timezone-aware" in naive.json()["detail"]

        reversed_interval = put(
            "2026-01-21T12:00:00+00:00", "2026-01-01T12:00:00+00:00"
        )
        assert reversed_interval.status_code == 422
        assert "before end" in reversed_interval.json()["detail"]
