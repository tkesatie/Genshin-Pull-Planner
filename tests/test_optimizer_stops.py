"""Stop conditions (Design Document §1, §18 Phase 5)."""

from optimizer import for_pursue, for_skip


class TestPursueStopConditions:
    def test_rules_stop_on_outcome_at_the_cap_and_rerun(self):
        stops = for_pursue("C2R1", 20)
        assert stops.action == "pursue"
        assert stops.outcome_label == "C2R1"
        assert stops.spend_cap == 20
        assert len(stops.rules) == 3
        assert "C2R1" in stops.rules[0]
        assert "20" in stops.rules[1]
        assert "protected" in stops.rules[1]
        assert "Re-run" in stops.rules[2]

    def test_the_cap_rule_names_its_consequence(self):
        stops = for_pursue("C0", 7)
        assert "7 wish(es)" in stops.rules[1]
        assert "voids the recommendation" in stops.rules[1]


class TestSkipStopConditions:
    def test_rules_say_do_not_spend_and_rerun(self):
        stops = for_skip("Tsaritsa C0 cannot be protected at 90%")
        assert stops.action == "skip"
        assert stops.outcome_label is None
        assert stops.spend_cap == 0
        assert len(stops.rules) == 2
        assert "Do not spend" in stops.rules[0]
        assert "Tsaritsa C0 cannot be protected at 90%" in stops.rules[0]
        assert "Re-run" in stops.rules[1]
