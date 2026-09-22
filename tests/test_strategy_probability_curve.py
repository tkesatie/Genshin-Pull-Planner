"""Probability-curve diagnostics for the Vesna and Skirk spend frontiers.

This is intentionally a diagnostic test rather than a brittle regression
against exact Monte Carlo percentages. It measures how increasing the Vesna
spend cap changes Vesna C2 and protected Skirk C2 outcomes under the same
shared 450-wish pool, including the expected 90 wishes of 7.1 income.
"""
import numpy as np
from dataclasses import replace

from domain import Account, Banner, Goal, IncomeEstimate, IncomeForecast, Ownership, Roadmap, VersionIncome
from planner import PlannerContext
from simulation import PlannedSpend, SpendPlan, simulate
from dataclasses import replace
import numpy as np

VODYNISTA = Banner("Vodynista", "7.1", 1)
VESNA = Banner("Vesna", "7.1", 1)
SKIRK = Banner("Skirk", "7.1", 2)


def make_context() -> PlannerContext:
    return PlannerContext(
        account=Account(
            wishes=450,
            current_pity=27,
            owned_characters=Ownership({"Skirk": 0}),
        ),
        roadmap=Roadmap(
            goals=[
                Goal("Vodynista", 0, 1),
                Goal("Vesna", 0, 2),
                Goal("Skirk", 2, 3),
                Goal("Vesna", 2, 4),
            ],
            banners=[VESNA, VODYNISTA, SKIRK],
        ),
        current_version="7.1",
        current_phase=1,
        confidence=0.90,
        income=IncomeForecast(
            versions=[
                VersionIncome(
                    "7.1",
                    estimate=IncomeEstimate(90, 90, 90),
                )
            ]
        ),
    )


def test_vesna_spend_curve(capsys):
    """Print the probability tradeoff across Vesna spend caps."""
    context = make_context()
    budgets = (0, 100, 150, 200, 219, 250, 300, 350, 400, 450)

    print("\\nVesna spend curve with 90 expected 7.1 income (10,000 runs, seed=0)")
    print("budget | Vesna C2 | Skirk C2 | all goals | final wishes")
    print("-------+-----------+----------+-----------+-------------")

    rows = []
    for budget in budgets:
        plan = SpendPlan(
            entries=(
                PlannedSpend(VODYNISTA, target_constellation=0, budget=450),
                PlannedSpend(VESNA, target_constellation=2, budget=budget),
                PlannedSpend(SKIRK, target_constellation=2, budget=450),
            )
        )
        result = simulate(context, plan, runs=10_000, seed=0)

        vesna = next(
            item.probability
            for item in result.goals
            if item.goal == Goal("Vesna", 2, 4)
        )
        skirk = next(
            item.probability
            for item in result.goals
            if item.goal == Goal("Skirk", 2, 3)
        )

        rows.append((budget, vesna, skirk, result.all_goals_probability))
        print(
            f"{budget:6d} | {vesna:9.2%} | {skirk:8.2%} | "
            f"{result.all_goals_probability:9.2%} | {result.final_wishes_mean:12.1f}"
        )

    assert rows[-1][1] > rows[0][1]
    assert rows[-1][2] < rows[0][2]
    assert rows[-1][0] == 450


def test_skirk_c2_probability_sanity_curve(capsys):
    """Measure raw Skirk C2 probability at the Hu Tao calculator comparison points."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    budgets = (240, 264, 265, 280)

    print("\\nSkirk C2 raw probability sanity curve (10,000 runs, seed=0)")
    print("budget | Skirk C2")
    print("-------+----------")

    rows = []
    for budget in budgets:
        success = 0
        rng = np.random.default_rng(0)

        for _ in range(10_000):
            account = context.account
            _, copies, _, _, _ = _pull_toward_target(
                account, "Skirk", 2, budget, mechanics, rng
            )
            success += copies == 2

        probability = success / 10_000
        rows.append((budget, probability))
        print(f"{budget:6d} | {probability:8.2%}")

    assert rows[-1][1] > rows[0][1]


def test_multi_copy_confidence_is_explicit(capsys):
    """Measure the actual simulated probability of two featured copies."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    budgets = (155, 180, 240, 264, 265)
    runs = 10_000
    rng = np.random.default_rng(0)

    print("\\nMulti-copy simulation confidence diagnostic")
    print(f"runs: {runs}, seed: 0")
    print("budget | Skirk C2")
    print("-------+----------")

    results = []
    for budget in budgets:
        successes = 0
        for _ in range(runs):
            account = replace(context.account, wishes=budget)
            spent, obtained, _, _, _ = _pull_toward_target(
                account,
                "Skirk",
                2,
                budget,
                context.mechanics,
                rng,
            )
            if obtained >= 2 and spent <= budget:
                successes += 1
        probability = successes / runs
        results.append(probability)
        print(f"{budget:6d} | {probability:8.2%}")

    assert results[0] < results[-1]
    assert results[1] < results[2] < results[3] < results[4]


def test_skirk_c2_probability_by_starting_pity(capsys):
    """Compare the C2 probability from zero pity versus the account's current pity."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    budgets = (240, 264, 265)
    runs = 10_000
    print("\\nSkirk C2 probability by starting pity (10,000 runs, seed=0)")
    print("budget | pity 0 | pity 27")
    print("-------+---------+--------")

    rows = []
    for budget in budgets:
        probabilities = []
        for starting_pity in (0, 27):
            successes = 0
            rng = np.random.default_rng(0)
            for _ in range(runs):
                account = replace(
                    context.account,
                    wishes=budget,
                    current_pity=starting_pity,
                    character_guarantee=False,
                )
                _, copies, _, _, _ = _pull_toward_target(
                    account, "Skirk", 2, budget, context.mechanics, rng
                )
                successes += copies == 2
            probabilities.append(successes / runs)
        rows.append(probabilities)
        print(f"{budget:6d} | {probabilities[0]:7.2%} | {probabilities[1]:7.2%}")

    for zero_pity, current_pity in rows:
        assert current_pity > zero_pity


def test_planner_protection_matches_skirk_threshold(capsys):
    """Inspect the planner's protected Skirk reserve at representative spend levels."""
    from planner.protection import protected_goal_outcomes

    context = make_context()
    spends = (0, 50, 100, 150, 200, 240)

    print("\\nPlanner Skirk protection diagnostic")
    print("spent | budget_at_banner | required | confidence | meets")
    print("------+-------------------+----------+------------+------")

    rows = []
    for spent in spends:
        outcomes = protected_goal_outcomes(context, spent=spent, banner=VODYNISTA)
        skirk = next(
            outcome for outcome in outcomes
            if outcome.goal == Goal("Skirk", 2, 3)
        )
        rows.append((spent, skirk))
        print(
            f"{spent:5d} | {skirk.budget_at_banner:17d} | "
            f"{skirk.required_wishes:8d} | {skirk.confidence:10.2%} | "
            f"{str(skirk.meets_threshold):5s}"
        )

    assert rows[-1][1].budget_at_banner < rows[0][1].budget_at_banner
    assert rows[-1][1].confidence <= rows[0][1].confidence


def test_combined_current_banner_frontier_respects_skirk_reserve(capsys):
    """Measure Skirk C2 using the reserve actually advertised by the frontier."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    caps = (200, 210, 220, 225, 230, 231, 235, 240)
    mechanics = context.mechanics

    print("\\nCombined frontier with actual Skirk reserve (10,000 runs, seed=0)")
    print("cap | reserve | Skirk C2")
    print("----+---------+----------")

    rows = []
    for cap in caps:
        skirk_success = 0
        rng = np.random.default_rng(0)
        reserve = 450 - cap

        for _ in range(10_000):
            account = context.account

            vod_spent, _, _, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )

            account = replace(account, wishes=account.wishes + 90)
            vesna_budget = max(cap - vod_spent, 0)
            _, _, _, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )

            _, skirk_copies, _, _, _ = _pull_toward_target(
                account, "Skirk", 2, reserve, mechanics, rng
            )
            skirk_success += skirk_copies == 2

        probability = skirk_success / 10_000
        rows.append((cap, reserve, probability))
        print(f"{cap:3d} | {reserve:7d} | {probability:8.2%}")

    assert rows[-1][2] < rows[0][2]


def test_combined_current_banner_frontier(capsys):
    """Measure the real frontier: Vod C0 + Vesna C2 share the 450-wish pool."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    caps = (200, 210, 220, 225, 230, 231, 235, 240)
    mechanics = context.mechanics

    print("\\nCombined phase-1 spend frontier (Vod C0 + Vesna C2)")
    print("cap | reserve | Vesna C2 | Skirk C2 | all three")
    print("----+---------+----------+----------+----------")

    rows = []
    for cap in caps:
        vesna_success = 0
        skirk_success = 0
        all_success = 0
        rng = np.random.default_rng(0)

        for _ in range(10_000):
            account = context.account

            vod_spent, vod_copies, _, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )

            vesna_budget = max(cap - vod_spent, 0)
            _, vesna_copies, _, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )

            vesna_met = vesna_copies == 3
            vesna_success += vesna_met

            _, skirk_copies, _, _, _ = _pull_toward_target(
                account, "Skirk", 2, 450, mechanics, rng
            )
            skirk_met = skirk_copies == 2
            skirk_success += skirk_met
            all_success += vod_copies == 1 and vesna_met and skirk_met

        rows.append((cap, vesna_success / 10_000, skirk_success / 10_000))
        print(
            f"{cap:3d} | {450 - cap:7d} | {vesna_success / 10_000:8.2%} | "
            f"{skirk_success / 10_000:8.2%} | {all_success / 10_000:8.2%}"
        )

    assert rows[-1][1] > rows[0][1]
    assert rows[-1][2] < rows[0][2]


def test_skirk_protected_boundary_convergence(capsys):
    """Compare the exact protected-future state against the live account state."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    budgets = (259, 263, 265)
    run_counts = (2_000, 5_000, 10_000, 20_000)

    print("\\nSkirk C2 boundary: zero pity vs current 27 pity")
    print("runs  | budget | pity 0 | pity 27")
    print("------+--------+--------+--------")

    rows = []
    for runs in run_counts:
        for budget in budgets:
            probabilities = []
            for starting_pity in (0, 27):
                successes = 0
                rng = np.random.default_rng(0)

                account = replace(
                    context.account,
                    wishes=budget,
                    current_pity=starting_pity,
                    character_guarantee=False,
                    capturing_radiance_counter=0,
                )

                for _ in range(runs):
                    _, copies, _, _, _ = _pull_toward_target(
                        account,
                        "Skirk",
                        2,
                        budget,
                        context.mechanics,
                        rng,
                    )
                    successes += copies == 2

                probabilities.append(successes / runs)

            rows.append((runs, budget, *probabilities))
            print(
                f"{runs:5d} | {budget:6d} | {probabilities[0]:6.2%} | "
                f"{probabilities[1]:7.2%}"
            )

    for runs, budget, zero_pity, current_pity in rows:
        assert current_pity > zero_pity


def test_skirk_reserve_from_actual_vesna_outcome_states(capsys):
    """Measure Skirk C2 protection after carrying forward the Vesna path state."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    current_spend_caps = (275, 277)
    skirk_reserves = (259, 263, 265)
    runs = 20_000

    print("\\nSkirk C2 after actual Vesna outcome states (20,000 runs, seed=0)")
    print("phase-1 cap | Skirk reserve | probability")
    print("------------+---------------+------------")

    rows = []
    for phase1_cap in current_spend_caps:
        for skirk_reserve in skirk_reserves:
            successes = 0
            rng = np.random.default_rng(0)

            for _ in range(runs):
                account = context.account
                vod_spent, _, _, _, account = _pull_toward_target(
                    account, "Vodynista", 1, phase1_cap, mechanics, rng
                )
                vesna_budget = max(phase1_cap - vod_spent, 0)
                _, _, _, _, account = _pull_toward_target(
                    account, "Vesna", 3, vesna_budget, mechanics, rng
                )

                account = replace(account, wishes=skirk_reserve)
                _, skirk_copies, _, _, _ = _pull_toward_target(
                    account, "Skirk", 2, skirk_reserve, mechanics, rng
                )
                successes += skirk_copies == 2

            probability = successes / runs
            rows.append((phase1_cap, skirk_reserve, probability))
            print(f"{phase1_cap:11d} | {skirk_reserve:13d} | {probability:10.2%}")

    assert rows
    assert all(0.0 <= probability <= 1.0 for _, _, probability in rows)


def test_vodynista_c0_vesna_c2_then_skirk_c2_with_future_income(capsys):
    """Measure Skirk C2 after fully pursuing Vodynista C0 and Vesna C2."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    runs = 20_000
    skirk_success = 0
    vodynista_success = 0
    vesna_success = 0
    all_success = 0

    rng = np.random.default_rng(0)

    for _ in range(runs):
        account = context.account

        # Spend the shared current pool on Vodynista C0 first.
        _, vod_copies, _, _, account = _pull_toward_target(
            account, "Vodynista", 1, account.wishes, mechanics, rng
        )
        vod_met = vod_copies == 1
        vodynista_success += vod_met

        # Whatever remains is then used for Vesna C2 (three copies from C0).
        _, vesna_copies, _, _, account = _pull_toward_target(
            account, "Vesna", 3, account.wishes, mechanics, rng
        )
        vesna_met = vesna_copies == 3
        vesna_success += vesna_met

        # 90 wishes of future 7.1 income arrive before Skirk's Phase 2.
        account = replace(account, wishes=account.wishes + 90)
        _, skirk_copies, _, _, _ = _pull_toward_target(
            account, "Skirk", 2, account.wishes, mechanics, rng
        )
        skirk_met = skirk_copies == 2
        skirk_success += skirk_met
        all_success += vod_met and vesna_met and skirk_met

    print("\\nVodynista C0 -> Vesna C2 -> +90 income -> Skirk C2")
    print(f"runs: {runs}, seed: 0")
    print(f"Vodynista C0: {vodynista_success / runs:.2%}")
    print(f"Vesna C2:     {vesna_success / runs:.2%}")
    print(f"Skirk C2:     {skirk_success / runs:.2%}")
    print(f"All three:    {all_success / runs:.2%}")

    assert 0.0 <= skirk_success / runs <= 1.0
    assert 0.0 <= all_success / runs <= 1.0


def test_strategy_vesna_c2_probability_is_joint_with_vodynista(capsys):
    """The Vesna C2 strategy probability must require Vodynista C0 too."""
    from optimizer.strategy import build_strategy

    context = make_context()
    strategy = build_strategy(context, runs=10_000, seed=0)
    vesna_c2 = next(
        step for step in strategy.steps
        if step.goal == Goal("Vesna", 2, 4)
    )

    print("\\nVesna C2 joint strategy probability")
    print(f"safe spend: {vesna_c2.safe_spend}")
    print(f"reserve: {vesna_c2.reserve_wishes}")
    print(f"joint probability: {vesna_c2.outcome_probability:.2%}")

    assert vesna_c2.safe_spend in range(275, 278)
    assert vesna_c2.reserve_wishes in range(173, 176)
    assert vesna_c2.future_income == 90
    assert vesna_c2.protected_total_wishes in range(263, 266)
    assert 0.15 < vesna_c2.outcome_probability < 0.21


def test_future_income_cannot_fund_current_phase(capsys):
    """Current-phase spending uses only wishes already available now."""
    from optimizer.strategy import build_strategy

    context = make_context()
    strategy = build_strategy(context, runs=10_000, seed=0)
    vesna_c2 = next(step for step in strategy.steps if step.goal == Goal("Vesna", 2, 4))

    assert vesna_c2.safe_spend in range(275, 278)
    assert vesna_c2.reserve_wishes in range(173, 176)
    assert vesna_c2.future_income == 90
    assert vesna_c2.protected_total_wishes in range(263, 266)
    assert 0.15 < vesna_c2.outcome_probability < 0.21


def test_zero_wishes_cannot_spend_current_phase_income():
    """Forecast income is unavailable until the roadmap reaches a later phase."""
    from simulation.engine import simulate_history

    context = make_context()
    zero_context = replace(context, account=replace(context.account, wishes=0))
    plan = SpendPlan(
        entries=(
            PlannedSpend(VODYNISTA, target_constellation=0, budget=90),
            PlannedSpend(VESNA, target_constellation=0, budget=90),
            PlannedSpend(SKIRK, target_constellation=2, budget=90),
        )
    )
    history = simulate_history(zero_context, plan, np.random.default_rng(0))
    assert all(result.income_credited == 0 for result in history.banner_results[:2])
    assert all(result.wishes_spent == 0 for result in history.banner_results[:2])
    assert history.banner_results[2].income_credited == 90


def test_vodynista_vesna_spend_cap_for_90_percent_skirk(capsys):
    """Find the current-phase cap whose downstream Skirk C2 rate is near 90%."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    runs = 20_000
    caps = (280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400)

    print("\nCurrent-phase cap vs downstream Skirk C2 (20,000 runs, seed=0)")
    print("cap | Vesna C2 | avg wishes at Skirk | Skirk C2")
    print("----+----------+----------------------+----------")

    rows = []
    for cap in caps:
        vesna_success = 0
        skirk_success = 0
        skirk_wishes = 0
        rng = np.random.default_rng(0)

        for _ in range(runs):
            account = context.account
            _, _, _, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )
            vod_spent = 450 - account.wishes
            vesna_budget = max(cap - vod_spent, 0)
            _, vesna_copies, _, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )
            vesna_success += vesna_copies == 3

            account = replace(account, wishes=account.wishes + 90)
            skirk_wishes += account.wishes
            _, skirk_copies, _, _, _ = _pull_toward_target(
                account, "Skirk", 2, account.wishes, mechanics, rng
            )
            skirk_success += skirk_copies == 2

        vesna_probability = vesna_success / runs
        skirk_probability = skirk_success / runs
        average_skirk_wishes = skirk_wishes / runs
        rows.append((cap, vesna_probability, average_skirk_wishes, skirk_probability))
        print(
            f"{cap:3d} | {vesna_probability:8.2%} | "
            f"{average_skirk_wishes:20.1f} | {skirk_probability:8.2%}"
        )

    qualifying = [row for row in rows if row[3] >= 0.90]
    assert qualifying
    max_safe_cap = max(row[0] for row in qualifying)
    print(f"\nLargest tested cap with >=90% Skirk C2: {max_safe_cap}")


def test_carried_state_distribution_near_skirk_frontier(capsys):
    """Inspect the full post-Vesna state distribution near the 90% Skirk frontier."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    caps = (320, 325, 330, 335)
    runs = 50_000

    print("\nCarried state distribution near Skirk C2 frontier (50,000 runs, seed=0)")
    print(
        "cap | Vesna C2 | Skirk C2 | avg wishes | median | p10-p90 | "
        "avg pity | guarantee | avg radiance"
    )
    print(
        "----+----------+----------+-------------+--------+----------+"
        "----------+-----------+------------"
    )

    rows = []
    for cap in caps:
        vesna_success = 0
        skirk_success = 0
        wishes = []
        pities = []
        guarantees = 0
        radiance = []

        rng = np.random.default_rng(0)

        for _ in range(runs):
            account = context.account

            vod_spent, vod_copies, _, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )
            assert vod_copies == 1

            vesna_budget = max(cap - vod_spent, 0)
            _, vesna_copies, _, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )
            vesna_success += vesna_copies == 3

            account = replace(account, wishes=account.wishes + 90)

            wishes.append(account.wishes)
            pities.append(account.current_pity)
            guarantees += account.character_guarantee
            radiance.append(account.capturing_radiance_counter)

            _, skirk_copies, _, _, _ = _pull_toward_target(
                account, "Skirk", 2, account.wishes, mechanics, rng
            )
            skirk_success += skirk_copies == 2

        vesna_probability = vesna_success / runs
        skirk_probability = skirk_success / runs
        average_wishes = np.mean(wishes)
        median_wishes = np.median(wishes)
        p10_wishes, p90_wishes = np.percentile(wishes, (10, 90))
        average_pity = np.mean(pities)
        guarantee_probability = guarantees / runs
        average_radiance = np.mean(radiance)

        rows.append(
            (
                cap,
                vesna_probability,
                skirk_probability,
                average_wishes,
                median_wishes,
                p10_wishes,
                p90_wishes,
                average_pity,
                guarantee_probability,
                average_radiance,
            )
        )
        print(
            f"{cap:3d} | {vesna_probability:8.2%} | {skirk_probability:8.2%} | "
            f"{average_wishes:11.1f} | {median_wishes:6.1f} | "
            f"{p10_wishes:4.0f}-{p90_wishes:4.0f} | "
            f"{average_pity:8.1f} | {guarantee_probability:9.2%} | "
            f"{average_radiance:10.2f}"
        )

    assert len(rows) == len(caps)
    assert all(0.0 <= row[1] <= 1.0 for row in rows)
    assert all(0.0 <= row[2] <= 1.0 for row in rows)
    assert all(row[3] >= 0 for row in rows)
    assert all(0 <= row[4] <= 540 for row in rows)


def test_candidate_plan_shares_current_phase_budget_and_full_roadmap_is_subset():
    """The displayed full-roadmap probability cannot exceed a required goal."""
    from optimizer.candidates import candidate_plan
    from optimizer.outcomes import OutcomeOption

    context = make_context()
    outcome = OutcomeOption("Vodynista", 0, 1)
    plan = candidate_plan(context, outcome, 275, banner=VODYNISTA)

    assert plan.shared_current_phase_budget == 275

    result = simulate(context, plan, runs=10_000, seed=0)
    vesna_c2 = next(
        item.probability
        for item in result.goals
        if item.goal == Goal("Vesna", 2, 4)
    )

    # Completing the entire roadmap necessarily includes Vesna C2.
    assert result.all_goals_probability <= vesna_c2
