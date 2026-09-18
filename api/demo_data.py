"""Development/demo data for the local dashboard.

This data exists only to make the default in-memory application immediately
usable during local UI development. A caller that injects its own repository
is never seeded by this module.
"""

from domain import Account, Banner, Goal, IncomeEstimate, IncomeForecast, Ownership, VersionIncome
from api.repository import AccountRecord, PlannerSettings


DEMO_ACCOUNT_ID = "demo"


def create_demo_account() -> AccountRecord:
    """Return a representative current-account demo for UI development."""
    return AccountRecord(
        id=DEMO_ACCOUNT_ID,
        label="Demo — Current Account",
        account=Account(
            wishes=450,
            current_pity=27,
            character_guarantee=False,
            owned_characters=Ownership({"Skirk": 0}),
        ),
        settings=PlannerSettings(
            current_version="7.1",
            current_phase=1,
            confidence=0.90,
            income_scenario="expected",
        ),
        goals=(
            Goal(character="Vodynista", constellation=0, priority=1),
            Goal(character="Vesna", constellation=0, priority=2),
            Goal(character="Skirk", constellation=2, priority=3),
            Goal(character="Vesna", constellation=2, priority=4),
        ),
        banners=(
            Banner(character="Vesna", version="7.1", phase=1),
            Banner(character="Vodynista", version="7.1", phase=1),
            Banner(character="Skirk", version="7.1", phase=2),
        ),
        preferences=(),
        income=IncomeForecast(
            versions=(
                VersionIncome(
                    version="7.1",
                    estimate=IncomeEstimate(low=60, expected=90, high=120),
                ),
            )
        ),
    )
