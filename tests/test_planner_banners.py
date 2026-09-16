"""current_banner: identification from the roadmap schedule (§7)."""

import pytest

from domain import Account, Banner, Roadmap
from planner import PlannerContext, current_banner


def test_identifies_the_current_banner(doc_context):
    assert current_banner(doc_context) == Banner("Vesna", "7.0", 1)


def test_identifies_mid_roadmap_banners(doc_account, doc_roadmap):
    """The planner can be re-run from any position in the schedule (§2):
    identification is positional, not "always the first banner"."""
    context = PlannerContext(
        account=doc_account, roadmap=doc_roadmap, current_version="7.1"
    )
    assert current_banner(context) == Banner("Tsaritsa", "7.1", 1)


def test_phase_mismatch_is_no_match(doc_account, doc_roadmap):
    context = PlannerContext(
        account=doc_account,
        roadmap=doc_roadmap,
        current_version="7.0",
        current_phase=2,
    )
    with pytest.raises(ValueError, match="no roadmap banner"):
        current_banner(context)


def test_unknown_version_is_no_match(doc_account, doc_roadmap):
    context = PlannerContext(
        account=doc_account, roadmap=doc_roadmap, current_version="6.8"
    )
    with pytest.raises(ValueError, match="no roadmap banner"):
        current_banner(context)


def test_ambiguous_slot_is_rejected(doc_account):
    """Two characters in the same (version, phase) slot cannot be
    resolved silently: the Phase 3 planner models one current banner."""
    roadmap = Roadmap(
        banners=[Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.0", 1)]
    )
    context = PlannerContext(
        account=doc_account, roadmap=roadmap, current_version="7.0"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        current_banner(context)