"""Scratch probe 6: shared mixed reserve + Fate-Points-only sensitivity."""

from domain import Account, Banner, Goal, Ownership, Roadmap, WeaponTarget, WeaponWishState
from optimizer.evaluation import evaluate_candidate
from optimizer.outcomes import available_outcomes
from optimizer.recommend import recommend
from planner import PlannerContext
from planner.protection import weapon_goal_reserve
from probability import multi_copy_wishes_for_confidence
from domain import CHARACTER_EVENT_BANNER


def w(name, level, priority):
    return Goal(target=WeaponTarget(name), level=level, priority=priority)


def wb(name, version, phase=1):
    return Banner(target=WeaponTarget(name), version=version, phase=phase)


print("char 1-copy 90% reserve =",
      multi_copy_wishes_for_confidence(0.9, 1, 0, False, CHARACTER_EVENT_BANNER))

base = PlannerContext(
    account=Account(wishes=300),
    roadmap=Roadmap(goals=[], banners=[]),
    current_version="7.0",
)
print("weapon reserve copies=1 =", weapon_goal_reserve(base, 1))

# X1: current character P3, protected future CHARACTER P1 + protected future
# WEAPON P2 (owned R1 -> R2 needs 1 copy -> reserve 136). Sequential reserves
# over ONE pool: expect budget = 300 - 155 - 136 = 9, spend_down = 291.
x1 = PlannerContext(
    account=Account(
        wishes=300,
        owned_characters=Ownership(weapons={"Fav Sword": 1}),
    ),
    roadmap=Roadmap(
        goals=[
            Goal("Tsaritsa", 0, 1),
            w("Fav Sword", 2, 2),
            Goal("Vesna", 0, 3),
        ],
        banners=[
            Banner("Vesna", "7.0", 1),
            Banner("Tsaritsa", "7.1", 1),
            wb("Fav Sword", "7.2"),
        ],
    ),
    current_version="7.0",
)
r = recommend(x1, runs=400, seed=1)
print("X1", r.action, "budget", r.budget, "down", r.spend_down_to,
      "prot_res", r.protected_reserve, "prot_goal", r.protected_goal,
      "conf", r.confidence, "banner", r.banner.target.name)
print("   protected:", [
    (s.goal.target.name, s.goal.target.kind.value, s.goal.priority,
     round(s.probability, 3), s.meets_threshold, s.constraining)
    for s in r.protected
])
print("   reasons:", r.reasons)

# FP-only: plain vs fate_points=1 (no other state difference), fixed budget.
for label, st in (
    ("plain", WeaponWishState()),
    ("fp1", WeaponWishState(fate_points=1)),
):
    c = PlannerContext(
        account=Account(wishes=300, weapon_state=st),
        roadmap=Roadmap(
            goals=[w("Wolf Fang", 1, 1)],
            banners=[wb("Wolf Fang", "7.0")],
        ),
        current_version="7.0",
    )
    outcomes = available_outcomes(c, (), banner=None)
    cand = evaluate_candidate(c, outcomes[0], 40, banner=None, runs=400, seed=1)
    print(f"FP {label}: p={cand.outcome_probability:.4f}")


from domain import Account, Banner, Goal, Roadmap, WeaponTarget
from optimizer.evaluation import _analytic_goal_standing
from planner import PlannerContext
from planner.protection import protected_goal_outcomes, weapon_goal_confidence


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

for spent in (49, 50, 140, 250):
    outcomes = protected_goal_outcomes(ctx, spent=spent)
    print(f"spent={spent} outcomes:", [
        (o.goal.target.name, o.budget_at_banner, round(o.confidence, 4),
         o.required_wishes, o.meets_threshold)
        for o in outcomes
    ])
    standing = _analytic_goal_standing(
        ctx, ctx.roadmap.goals[0], spent, None
    )
    print(f"  analytic standing: {standing}")

print("conf direct @136 =", weapon_goal_confidence(ctx, 1, 136))
print("conf direct @251 =", weapon_goal_confidence(ctx, 1, 251))
print("income credit 7.1 =", ctx.income_available_before("7.1", 1))
