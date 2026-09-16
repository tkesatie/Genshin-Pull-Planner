import random
from collections import defaultdict, Counter

def calculate_probability_distribution(
    max_wishes,
    current_pity=0,
    guarantee=False,
    simulations=100,
    target=1
):
    all_results = []

    for _ in range(simulations):
        result = simulate_character_pulls(
            max_wishes,
            current_pity,
            guarantee,
            target
        )
        all_results.append(result)

    probability_distribution = {}

    for wish_count in range(1, max_wishes + 1):
        successes = 0

        for result in all_results:
            if result["reached_target"] and result["wishes_used"] <= wish_count:
                successes += 1

        probability = successes / simulations * 100
        probability_distribution[wish_count] = probability

    return probability_distribution

def simulate_character_pulls(max_wishes, current_pity, guarantee, target):
    featured_characters = 0
    total_wishes = 0
    featured_wishes = []
    reached_target = False

    while total_wishes < max_wishes:
        featured_pulled, guarantee, current_pity = character_pull_odds(
            current_pity, guarantee
        )

        total_wishes += 1

        if featured_pulled:
            featured_characters += 1
            featured_wishes.append(total_wishes)

            if featured_characters >= target:
                reached_target = True
                break

    return {
        "wishes_used": total_wishes,
        "featured_pulled": featured_characters,
        "current_pity": current_pity,
        "guarantee": guarantee,
        "reached_target": reached_target,
        "featured_wishes": featured_wishes,
    }

def character_pull_odds(current_pity, guarantee):
    featured_character = False

    if current_pity == 89:
        current_pity = 0
        if guarantee:
            featured_character = True
            guarantee = False
        else:
            if random.random() < 0.5:
                featured_character = True
            else:
                guarantee = True
    else:
        if current_pity < 73:
            if random.random() < 0.006:
                if guarantee:
                    featured_character = True
                    current_pity = 0
                    guarantee = False
                else:
                    current_pity = 0
                    if random.random() < 0.5:
                        featured_character = True
                    else:
                        guarantee = True
            else:
                current_pity += 1
        else: 
            if random.random() < 0.006 + (current_pity - 72) * 0.06:
                if guarantee:
                    featured_character = True
                    current_pity = 0
                    guarantee = False
                else:
                    current_pity = 0
                    if random.random() < 0.5:
                        featured_character = True
                    else:
                        guarantee = True
            else:
                current_pity += 1
    
    return featured_character, guarantee, current_pity

def calculate_target_probability(wishes, current_pity, guarantee, target, simulations=100):

    results = calculate_probability_distribution(
        max_wishes=wishes,
        current_pity=current_pity,
        guarantee=guarantee,
        target=target,
        simulations=simulations
    )

    return results[wishes]

if __name__ == "__main__":

    print(calculate_target_probability(150, 0, False, 1))
    print(calculate_target_probability(400, 0, False, 5))
    print(calculate_target_probability(200, 0, False, 1))
    print(calculate_target_probability(90, 0, True, 1))