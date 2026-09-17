"""Available outcomes from preference chains (Design Document §13 step 2, §15)."""

import pytest

from domain import Account, Banner, Goal, Ownership, Preference, Roadmap
from optimizer import OutcomeOption, available_outcomes
from planner import PlannerContext


def context_with(
    account: Account, goals: list[Goal] | None = None
) -> PlannerContext:
    roadmap = Roadmap(
        goals=goals if goals is not None else [Goal("Vesna", 0, 1)],
        banners=[Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)],
    )
    return PlannerContext(
        account=account, roadmap=roadmap, current_version="7.0"
    )


class TestPreferenceChain:
    def test_ranks_and_labels_follow_the_chain(self, doc_context, doc_preferences):
        outcomes = available_outcomes(doc_context, doc_preferences)
        assert [(o.rank, o.label) for o in outcomes] == [
            (1, "C2R1"),
            (2, "C1R1"),
            (5, "C0"),
        ]

    def test_duplicate_constellations_collapse_to_the_best_rank(
        self, doc_context, doc_preferences
    ):
        """C2R1 (rank 1) beats C2 (rank 3); C1R1 (rank 2) beats C1 (rank 4)."""
        outcomes = available_outcomes(doc_context, doc_preferences)
        constellations = [o.constellation for o in outcomes]
        assert constellations == [2, 1, 0]
        assert outcomes[0].rank == 1
        assert outcomes[1].rank == 2

    def test_constellations_at_or_below_ownership_are_dropped(self):
        account = Account(wishes=40, owned_characters=Ownership({"Vesna": 1}))
        chain = (
            Preference("Vesna", 1, 2),
            Preference("Vesna", 2, 1),  # already owned: dropped
            Preference("Vesna", 3, 0),  # below ownership: dropped
        )
        outcomes = available_outcomes(context_with(account), chain)
        assert [o.label for o in outcomes] == ["C2"]

    def test_fully_owned_chain_yields_no_outcomes(self):
        """Preferences already satisfied: nothing to pursue - and never a
        silently upgraded target (§2: no invented outcomes)."""
        account = Account(wishes=40, owned_characters=Ownership({"Vesna": 2}))
        chain = (
            Preference("Vesna", 1, 2),
            Preference("Vesna", 2, 1),
        )
        assert available_outcomes(context_with(account), chain) == ()

    def test_no_invention_beyond_the_chain(self):
        """A chain without C0 never offers C0, even with a C0 roadmap goal."""
        account = Account(wishes=40, owned_characters=Ownership({"Vesna": 1}))
        chain = (Preference("Vesna", 1, 2),)
        outcomes = available_outcomes(context_with(account), chain)
        assert [o.constellation for o in outcomes] == [2]

    def test_other_characters_preferences_are_ignored(self, doc_context):
        """Review point 6: another character's chain contributes nothing -
        the chain is restricted to the current banner character first. With
        no Vesna chain, the roadmap-goal fallback applies instead (§15, §5).
        """
        chain = (Preference("Tsaritsa", 1, 2), Preference("Tsaritsa", 2, 0))
        outcomes = available_outcomes(doc_context, chain)
        assert outcomes == (
            OutcomeOption(character="Vesna", constellation=0, rank=1),
        )
        assert all(outcome.character == "Vesna" for outcome in outcomes)

    def test_same_character_chain_orders_by_descending_constellation(self):
        """Reported bug: a same-character chain is a progression (reaching
        C2 necessarily reaches C0), so the furthest target is offered
        first regardless of the user's stated rank order. Here C0 is
        ranked ahead of C2 (rank 1 vs rank 3), but C2 still comes first."""
        chain = (
            Preference("Vesna", 1, 0),
            Preference("Vesna", 3, 2),
        )
        outcomes = available_outcomes(context_with(Account(wishes=40)), chain)
        assert [(o.label, o.rank) for o in outcomes] == [("C2", 3), ("C0", 1)]

    def test_no_refinement_stays_none(self, doc_context):
        outcomes = available_outcomes(doc_context, (Preference("Vesna", 1, 2),))
        assert outcomes[0].weapon_refinement is None
        assert outcomes[0].label == "C2"


class TestGoalFallback:
    def test_falls_back_to_the_active_goal_without_preferences(self, doc_context):
        outcomes = available_outcomes(doc_context)
        assert outcomes == (
            OutcomeOption(character="Vesna", constellation=0, rank=1),
        )

    def test_satisfied_goal_yields_no_outcomes(self):
        account = Account(wishes=40, owned_characters=Ownership({"Vesna": 0}))
        assert available_outcomes(context_with(account)) == ()

    def test_blocked_goal_is_not_actionable(self):
        """Vesna C2 behind an unsatisfied C0: the fallback offers C0 (§9)."""
        goals = [Goal("Vesna", 0, 1), Goal("Vesna", 2, 2)]
        outcomes = available_outcomes(context_with(Account(wishes=40), goals))
        assert [o.constellation for o in outcomes] == [0]

    def test_multiple_active_current_character_goals_rejected(self):
        """Degenerate duplicate goals for the CURRENT character are refused
        (§9); other characters' active goals must not create the error."""
        account = Account(wishes=40)
        goals = [Goal("Vesna", 0, 1), Goal("Vesna", 0, 2)]
        with pytest.raises(ValueError, match="multiple active goals"):
            available_outcomes(context_with(account, goals))

    def test_other_characters_active_goals_do_not_trigger_ambiguity(self):
        """Review point 6: ambiguity is current-banner-character only."""
        account = Account(wishes=40)
        goals = [Goal("Vesna", 0, 1), Goal("Tsaritsa", 0, 2)]
        outcomes = available_outcomes(context_with(account, goals))
        assert [o.constellation for o in outcomes] == [0]

    def test_preferences_override_the_goal(self):
        """With a chain present, only the chain's constellations are
        offered - the roadmap goal does not add targets (§15)."""
        account = Account(wishes=40)
        goals = [Goal("Vesna", 0, 1)]
        chain = (Preference("Vesna", 1, 2),)
        outcomes = available_outcomes(context_with(account, goals), chain)
        assert [o.constellation for o in outcomes] == [2]
