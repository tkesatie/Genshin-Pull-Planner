"""Scratch probe 3: per-cap feasibility for the weapon-protected scenario."""

from domain import Account, Banner, Goal, Roadmap, WeaponTarget
from optimizer.evaluation import evaluate_candidate
from optimizer.outcomes import available_outcomes
from planner import PlannerContext
from planner.protection import weapon_goal_reserve, weapon_goal_confidence


def w(name, level, priority):
    return Goal(target=WeaponTarget(name), level=level, priority=priority)


def wb(name, version, phase=1):
    return Banner(target=WeaponTarget(name), version=version, phase=phase)


ctx = PlannerContext(
    account=Account(wishes=300),
    roadmap=Roadmap(
        goals=[w("Fav Sword", 1, 1), w("Wolf Fang", 1, 2)],
        banners=[wb("Wolf Fang", "7.0"), wb("Fav Sword", "7.1")],
    ),
    current_version="7.0",
)

print("reserve(1 copy, 90%) =", weapon_goal_reserve(ctx, 1))
print("conf @250 =", weapon_goal_confidence(ctx, 1, 250))
print("conf @140 =", weapon_goal_confidence(ctx, 1, 140))

outcomes = available_outcomes(ctx, (), banner=None)
print("outcomes:", [(o.character, o.constellation) for o in outcomes])
for cap in (300, 250, 200, 160, 150, 140, 100, 80, 60, 50, 49):
    cand = evaluate_candidate(ctx, outcomes[0], cap, banner=None, runs=400, seed=1)
    gating = [
        (s.goal.target.name, round(s.probability, 3), s.meets_threshold, s.constraining)
        for s in cand.protected
    ]
    print(
        f"cap={cap:3d} feasible={cand.feasible} "
        f"p={cand.outcome_probability:.4f} min_prot={cand.min_protected_probability} "
        f"standings={gating}"
    )
