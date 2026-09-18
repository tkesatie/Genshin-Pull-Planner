

def test_vodynista_vesna_spend_cap_for_90_percent_skirk(capsys):
    """Find the current-phase cap whose downstream Skirk C2 rate is near 90%."""
    from simulation.engine import _pull_toward_target

    context = make_context()
    mechanics = context.mechanics
    runs = 20_000
    caps = (180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 275)

    print("\\nCurrent-phase cap vs downstream Skirk C2 (20,000 runs, seed=0)")
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

            # The shared current-phase cap is consumed in priority order:
            # Vodynista C0 first, then Vesna C2 with whatever remains.
            _, vod_copies, _, account = _pull_toward_target(
                account, "Vodynista", 1, cap, mechanics, rng
            )
            vesna_budget = max(cap - (450 - account.wishes), 0)
            _, vesna_copies, _, account = _pull_toward_target(
                account, "Vesna", 3, vesna_budget, mechanics, rng
            )
            vesna_met = vesna_copies == 3
            vesna_success += vesna_met

            # The 90 future wishes arrive before Skirk. Carry forward all
            # actual pity/guarantee/Radiance state from Phase 1.
            account = replace(account, wishes=account.wishes + 90)
            skirk_wishes += account.wishes
            _, skirk_copies, _, _ = _pull_toward_target(
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

    # Diagnostic only: identify the empirical 90% boundary rather than baking
    # a particular answer into the test.
    qualifying = [row for row in rows if row[3] >= 0.90]
    assert qualifying
    max_safe_cap = max(row[0] for row in qualifying)
    print(f"\\nLargest tested cap with >=90% Skirk C2: {max_safe_cap}")
