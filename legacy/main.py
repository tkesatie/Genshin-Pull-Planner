from account import Account
from goals import find_copies_needed
from character_probability import calculate_target_probability
from plan import Roadmap

my_account = Account(0, False, {"Mavuika": 0, "Vesna": 0})

roadmap = Roadmap(
    goals=[
        Goal("Vesna", 0, 1),
        Goal("Vodynista", 0, 2),
        Goal("Vesna", 2, 3),
        Goal("Tsaritsa", 0, 4),
    ],
    banners=[
        Banner("Vesna", "7.0", 1),
        Banner("Tsaritsa", "7.1", 1),
        Banner("Vodynista", "7.2", 1),
    ]
)

all_targets = find_copies_needed(my_account, pull_plan.character_goals)

target = all_targets['Vesna']

prob = calculate_target_probability(pull_plan.wishes_to_spend, my_account.current_pity, my_account.character_guarantee, target)

print(prob)