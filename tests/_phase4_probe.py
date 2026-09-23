"""Scratch probe for Phase 4 scenario verification (deleted after use)."""

from domain import Account, Banner, Goal, Roadmap, WeaponTarget, WeaponWishState
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


# P1: mixed priorities weapon + char + weapon (global order)
c = ctx(
    [w("Wolf Fang", 1, 1), Goal("Vesna", 0, 2), w("Fav Sword", 1, 3)],
    [wb("Wolf Fang", "7.0"), Banner("Vesna", "7.1", 1), wb("Fav Sword", "7.2", 1)],
)
r = recommend(c, runs=400, seed=1)
print("P1", r.action, r.target_kind, r.target_name, "budget", r.budget,
      "down", r.spend_down_to, "prot_res", r.protected_reserve,
      "prot_goal", r.protected_goal, "conf", r.confidence)
print("   protected:", [(str(s.goal), s.goal.target.kind.value, s.constraining)
                        for s in r.protected])
print("   reasons:", r.reasons)

# P9b debug: protected future character + current character, fresh vs loaded state
goals2 = [Goal("Vesna", 0, 2), Goal("Tsaritsa", 0, 1)]
banners2 = [Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)]
c0 = ctx(goals2, banners2)
c1 = ctx(goals2, banners2,
         account=Account(wishes=300, current_pity=75,
                         character_guarantee=True,
                         capturing_radiance_counter=2))
r3 = recommend(c0, runs=400, seed=1)
r4 = recommend(c1, runs=400, seed=1)
print("P9b fresh ", r3.action, r3.target_name, r3.budget, r3.spend_down_to,
      "conf", r3.confidence)
print("   protected:", [(str(s.goal), s.probability, s.meets_threshold,
                         s.constraining) for s in r3.protected])
print("P9b loaded", r4.action, r4.target_name, r4.budget, r4.spend_down_to,
      "conf", r4.confidence)
print("   protected:", [(str(s.goal), s.probability, s.meets_threshold,
                         s.constraining) for s in r4.protected])

# P8b debug: protected future weapon + current weapon, fresh vs loaded weapon state
goals = [w("Fav Sword", 1, 1), w("Wolf Fang", 1, 2)]
banners = [wb("Wolf Fang", "7.0"), wb("Fav Sword", "7.1")]
f = ctx(goals, banners)
ld = ctx(goals, banners,
         account=Account(wishes=300,
                         weapon_state=WeaponWishState(pity=70, guarantee=True,
                                                      fate_points=1)))
rf = recommend(f, runs=400, seed=1)
rl = recommend(ld, runs=400, seed=1)
print("P8b fresh ", rf.action, rf.target_name, rf.budget, rf.spend_down_to,
      "conf", rf.confidence)
print("P8b loaded", rl.action, rl.target_name, rl.budget, rl.spend_down_to,
      "conf", rl.confidence)
