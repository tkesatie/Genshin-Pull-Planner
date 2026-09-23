"""Scratch probe 4: remaining Phase 4 scenarios (deleted after use)."""

from domain import (
    Account,
    Banner,
    Goal,
    Ownership,
    Roadmap,
    WeaponTarget,
    WeaponWishState,
)
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


# Q1: protected CHARACTER (P1, future) + active WEAPON (P2, current)
c = ctx(
    [Goal("Tsaritsa", 0, 1), w("Wolf Fang", 1, 2)],
    [wb("Wolf Fang", "7.0"), Banner("Tsaritsa", "7.1", 1)],
)
r = recommend(c, runs=400, seed=1)
print("Q1", r.action, r.target_kind, r.target_name, "budget", r.budget,
      "down", r.spend_down_to, "prot_res", r.protected_reserve,
      "prot_goal", r.protected_goal, "conf", r.confidence)
print("   protected:", [(str(s.goal.target), s.goal.priority, round(s.probability, 3),
                         s.meets_threshold, s.constraining) for s in r.protected])

# Q2: protected WEAPON (P1, future) + active CHARACTER (P2, current)
c = ctx(
    [w("Fav Sword", 1, 1), Goal("Vesna", 0, 2)],
    [Banner("Vesna", "7.0", 1), wb("Fav Sword", "7.1")],
)
r = recommend(c, runs=400, seed=1)
print("Q2", r.action, r.target_kind, r.target_name, "budget", r.budget,
      "down", r.spend_down_to, "prot_res", r.protected_reserve,
      "prot_goal", r.protected_goal, "conf", r.confidence)
print("   protected:", [(str(s.goal.target), s.goal.priority, round(s.probability, 3),
                         s.meets_threshold, s.constraining) for s in r.protected])

# Q3: completion - completed character + incomplete weapon (current)
c = ctx(
    [Goal("Vesna", 0, 1), w("Wolf Fang", 1, 2)],
    [wb("Wolf Fang", "7.0"), Banner("Vesna", "7.1", 1)],
    account=Account(wishes=300, owned_characters=Ownership(characters={"Vesna": 0})),
)
r = recommend(c, runs=400, seed=1)
print("Q3", r.action, r.target_kind, r.target_name)

# Q4: completion - completed weapon + incomplete character (current)
c = ctx(
    [w("Wolf Fang", 1, 1), Goal("Vesna", 0, 2)],
    [Banner("Vesna", "7.0", 1), wb("Wolf Fang", "7.1")],
    account=Account(wishes=300, owned_characters=Ownership(weapons={"Wolf Fang": 1})),
)
r = recommend(c, runs=400, seed=1)
print("Q4", r.action, r.target_kind, r.target_name)

# Q5: all goals completed -> skip
c = ctx(
    [Goal("Vesna", 0, 1), w("Wolf Fang", 1, 2)],
    [Banner("Vesna", "7.0", 1), wb("Wolf Fang", "7.1")],
    account=Account(
        wishes=300,
        owned_characters=Ownership(characters={"Vesna": 0}, weapons={"Wolf Fang": 1}),
    ),
)
r = recommend(c, runs=400, seed=1)
print("Q5", r.action, "|", r.skip_reason, "| reasons:", r.reasons)

# Q6: future weapon goal only (banner not currently available) -> skip
c = ctx(
    [w("Fav Sword", 1, 1)],
    [Banner("Vesna", "7.0", 1), wb("Fav Sword", "7.1")],
)
r = recommend(c, runs=400, seed=1)
print("Q6", r.action, "|", r.skip_reason)

# Q7: future character goal only -> skip (character availability regression)
c = ctx(
    [Goal("Tsaritsa", 0, 1)],
    [Banner("Vesna", "7.0", 1), Banner("Tsaritsa", "7.1", 1)],
)
r = recommend(c, runs=400, seed=1)
print("Q7", r.action, "|", r.skip_reason)

# Q8: Capturing Radiance sensitivity at fixed character state/budget
base_goals = [Goal("Vesna", 0, 1)]
base_banners = [Banner("Vesna", "7.0", 1)]
for label, cr in (("cr0", 0), ("cr2", 2)):
    c = ctx(
        base_goals,
        base_banners,
        account=Account(wishes=300, current_pity=70,
                        character_guarantee=False,
                        capturing_radiance_counter=cr),
    )
    outcomes = available_outcomes(c, (), banner=None)
    cand = evaluate_candidate(c, outcomes[0], 40, banner=None, runs=400, seed=1)
    print(f"Q8 {label}: p={cand.outcome_probability:.4f}")

# Q9: weapon guarantee (no fate points) sensitivity at fixed budget
for label, st in (
    ("plain", WeaponWishState()),
    ("guaranteed", WeaponWishState(guarantee=True)),
):
    c = ctx(
        [w("Wolf Fang", 1, 1)],
        [wb("Wolf Fang", "7.0")],
        account=Account(wishes=300, weapon_state=st),
    )
    outcomes = available_outcomes(c, (), banner=None)
    cand = evaluate_candidate(c, outcomes[0], 40, banner=None, runs=400, seed=1)
    print(f"Q9 {label}: p={cand.outcome_probability:.4f}")

# Q10: response contract fields on a pursue rec
c = ctx(
    [Goal("Tsaritsa", 0, 1), w("Wolf Fang", 1, 2)],
    [wb("Wolf Fang", "7.0"), Banner("Tsaritsa", "7.1", 1)],
)
r = recommend(c, runs=400, seed=1)
print("Q10", r.action, "spend_limit", r.spend_limit, "spend_down", r.spend_down_to,
      "prot_res", r.protected_reserve, "kind", r.target_kind,
      "conf", r.confidence is not None, "reasons", len(r.reasons))
