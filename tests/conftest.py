"""Shared Phase 1 fixtures built from the design document's running example."""

import pytest

from domain import Account, Banner, Goal, Ownership, Roadmap


@pytest.fixture
def doc_account() -> Account:
    """Account state for the design-document example (§4.1, §8).

    Vesna and Tsaritsa are not owned; Vodynista is owned at C0.
    """
    return Account(
        current_pity=0,
        character_guarantee=False,
        owned_characters=Ownership({"Vodynista": 0}),
        wishes=40,
    )


@pytest.fixture
def doc_roadmap() -> Roadmap:
    """The roadmap from §8 (Vesna appears in two separate goals)."""
    return Roadmap(
        goals=[
            Goal("Vesna", 0, 1),
            Goal("Vodynista", 0, 2),
            Goal("Vesna", 2, 3),
            Goal("Tsaritsa", 0, 4),
        ],
        banners=[
            Banner("Vesna", "7.0", 1),
            Banner("Tsaritsa", "7.1", 1),
            Banner("Vodynista", "7.2", 1),
        ],
    )
