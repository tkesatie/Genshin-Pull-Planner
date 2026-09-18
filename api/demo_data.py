"""Development/demo data for the local dashboard.

This data exists only to make the default in-memory application immediately
usable during local UI development. A caller that injects its own repository
is never seeded by this module.
"""

from domain import Account, Banner, Goal, IncomeEstimate, IncomeForecast, Preference, Ownership, VersionIncome
from api.repository import AccountRecord, PlannerSettings


DEMO_ACCOUNT_ID = "demo"


def create_demo_account() -> AccountRecord:
    """Return the representative Navia current-banner demo account."""
    return AccountRecord(
        id=DEMO_ACCOUNT_ID,
        label="Demo — Vesna Roadmap",
        account=Account(
            wishes=180,
            current_pity=0,
            character_guarantee=False,
            owned_characters=Ownership({}),
        ),
        settings=PlannerSettings(
            current_version="7.0",
            current_phase=1,
            confidence=0.90,
            income_scenario="expected",
        ),
        goals=(
            Goal(character="Vesna", constellation=0, priority=1),
            Goal(character="Tsaritsa", constellation=0, priority=2),
            Goal(character="Vesna", constellation=2, priority=3),
        ),
        banners=(
            Banner(character="Vesna", version="7.0", phase=1),
            Banner(character="Tsaritsa", version="7.1", phase=1),
        ),
        preferences=(
            Preference(character="Vesna", rank=1, constellation=2),
        ),
        income=IncomeForecast(
            versions=(
                VersionIncome(
                    version="7.1",
                    estimate=IncomeEstimate(low=60, expected=90, high=120),
                ),
            )
        ),
    )
