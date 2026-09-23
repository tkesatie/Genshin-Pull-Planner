"""Scratch probe 2 for Phase 4 scenario verification (deleted after use)."""

from domain import Account, Banner, Goal, Roadmap, WeaponTarget, WeaponWishState
from optimizer.evaluation import evaluate_candidate
from optimizer.outcomes import available_outcomes
from optimizer.recommend import recommend
from planner import PlannerContext


def w(name, level, priority):
    return Goal(target=WeaponTarget(name), level=level, priority=priority)


def wb(name, version, phase=1):
    return Banner(target=WeaponTarget(name), version=version, phase=phase)


def ctx(goals, banners, account=None, cv="7.0"):
    return PlannerContext(
        account=account if account is not None else Account(wishes=300),
        roadmap=Roadmap(goals=goals, banners=banners),
        current_version=cv,
    )


# M1: mixed same slot - weapon P1 + character P2, both banners at 7.0
c = ctx(
    [w("Wolf Fang", 1, 1), Goal("Vesna", 0, 2)],
    [wb("Wolf Fang", "7.0"), Banner("Vesna", "7.0", 1)],
)
r = recommend(c, runs=400, seed=1)
print("M1", r.action, r.target_kind, r.target_name, "banner",
      r.banner.target.name, r.banner.target.kind.value, "budget", r.budget)

# M2: mixed same slot - character P1 + weapon P2, both banners at 7.0
c = ctx(
    [Goal("Vesna", 0, 1), w("Wolf Fang", 1, 2)],
    [wb("Wolf Fang", "7.0"), Banner("Vesna", "7.0", 1)],
)
r = recommend(c, runs=400, seed=1)
print("M2", r.action, r.target_kind, r.target_name, "budget", r.budget,
      "prot_goal", r.protected_goal)

# S1: weapon outcome probability at fixed budget: fresh vs loaded weapon state
base = dict(goals=[w("Wolf Fang", 1, 1)], banners=[wb("Wolf Fang", "7.0")])
fresh = ctx(
    [w("Wolf Fang", 1, 1)], [wb("Wolf Fang", "7.0")]
)
loaded = ctx(
    [w("Wolf Fang", 1, 1)], [wb("Wolf Fang", "7.0")],
    account=Account(wishes=300,
                    weapon_state=WeaponWishState(pity=70, guarantee=True,
                                                 fate_points=1)),
)
for label, context in (("fresh", fresh), ("loaded", loaded)):
    outcomes = available_outcomes(context, (), banner=None)
    cand = evaluate_candidate(context, outcomes[0], 60,
                               banner=None, runs=400, seed=1)
    print(f"S1 {label}: p={cand.outcome_probability:.4f}")

# S2: character outcome probability at fixed budget: fresh vs loaded char state
fresh_c = ctx([Goal("Vesna", 0, 1)], [Banner("Vesna", "7.0", 1)])
loaded_c = ctx(
    [Goal("Vesna", 0, 1)], [Banner("Vesna", "7.0", 1)],
    account=Account(wishes=300, current_pity=75, character_guarantee=True,
                    capturing_radiance_counter=2),
)
for label, context in (("fresh", fresh_c), ("loaded", loaded_c)):
    outcomes = available_outcomes(context, (), banner=None)
    cand = evaluate_candidate(context, outcomes[0], 30,
                               banner=None, runs=400, seed=1)
    print(f"S2 {label}: p={cand.outcome_probability:.4f}")
