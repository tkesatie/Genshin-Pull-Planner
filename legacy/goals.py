@dataclass
class GoalStatus:
    goal: Goal
    copies_needed: int

def find_copies_needed(account, goals):
    statuses = []

    for goal in goals:
        owned = account.owned_characters.get(goal.character, -1)
        copies_needed = max(goal.constellation - owned, 0)

        statuses.append(
            GoalStatus(goal, copies_needed)
        )

    return statuses