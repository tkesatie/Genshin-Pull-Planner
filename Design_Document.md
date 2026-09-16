# Genshin Pull Strategy Planner

## 1. Purpose

The Genshin Pull Strategy Planner is a probability-based resource-allocation tool for deciding **what to spend wishes on, how far to pursue it, and what that spending puts at risk later**.

It is not primarily an affordability calculator.

The central question is:

> Given my current account state, my prioritized goals, my preferences for each character, and my assumptions about future banners and income, what is the most preferred outcome I can pursue now while maintaining my required confidence in protected future goals?

The planner is intended to be **re-run after every meaningful account update**. It does not produce a fixed pull plan that becomes obsolete once actual pulls differ from expectations.

The user provides:

* Current account state
* Character and weapon ownership
* Prioritized goals
* Preferred/fallback outcomes
* Expected future banners
* Expected future wish income
* Required confidence threshold

The planner provides:

* Current actionable goals
* Possible spending strategies
* Probability of achieving outcomes
* Downstream consequences
* Safe spending boundaries
* Stop conditions
* A current recommendation based on the user's stated preferences and confidence threshold

---

# 2. Core Design Principles

## Probability-oriented

Recommendations are expressed in probabilities rather than deterministic affordability.

The planner should answer:

> "Pursuing this outcome leaves me with a 92% probability of satisfying the protected roadmap."

rather than:

> "You can afford this."

---

## Preferences, not simple character rankings

A character is not simply "wanted" or "not wanted."

For each character, the user can define an ordered preference chain such as:

1. C2R1
2. C1R1
3. C2
4. C1
5. C0

The planner evaluates these outcomes in preference order and finds the most preferred outcome that satisfies the user's roadmap constraints.

The planner must never invent an outcome that the user did not define as a preference.

---

## Priorities are separate from preferences

**Priority** answers:

> Which objectives should be protected first?

**Preference** answers:

> If I am pursuing this character, which outcome do I want most?

For example:

```text
Priority 1 → Vesna C0
Priority 2 → Vodynista C0
Priority 3 → Vesna C2
Priority 4 → Tsaritsa C0
```

These are separate objectives even though Vesna appears twice.

This distinction is important because the same character can occur at multiple points in the roadmap.

---

## Future goals are constraints

A current-banner decision must consider its effect on future objectives.

For example:

```text
Current banner: Tsaritsa
Future goal:    Vodynista C0
```

The planner should not simply ask:

> "Can I afford Tsaritsa?"

It should ask:

> "If I spend wishes pursuing Tsaritsa, what happens to my probability of satisfying the higher-priority future objectives?"

Spending more now therefore represents an explicit tradeoff against future probability.

---

## Uncertainty is explicit

Future banners and future income are assumptions rather than facts.

The planner should eventually support uncertainty such as:

```text
Banner confidence:
High
Medium
Low
```

and:

```text
Income:
Low
Expected
High
```

The simulation engine can then evaluate outcomes across those uncertainties.

---

## Re-run after account updates

The planner is a living decision tool.

After a banner:

```text
Actual account state
        ↓
Re-run planner
        ↓
New recommendation
```

The previous recommendation is not treated as a permanent plan.

---

# 3. System Architecture

The eventual system consists of four major logical layers.

```text
┌─────────────────────────────────────────────┐
│                 Frontend                    │
│      Dashboard / Account / Goals / etc.     │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│                  API Layer                  │
│                  FastAPI                    │
└──────────────────────┬──────────────────────┘
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
┌────────────────────┐  ┌────────────────────┐
│    Domain /        │  │ Probability /     │
│    Planner Logic   │  │ Simulation Engine  │
└────────────────────┘  └────────────────────┘
             │                   │
             └─────────┬─────────┘
                       ▼
                 Data / Database
```

The immediate development work is focused on the **domain and probability logic**.

Database, API, and frontend infrastructure should not dictate the domain model prematurely.

---

# 4. Domain Model

## 4.1 Account

`Account` represents the user's current state.

Conceptually:

```python
Account(
    current_pity,
    character_guarantee,
    owned_characters
)
```

The eventual persistent model will also contain wish resources, version/phase, weapon state, and other account information.

The important rule is:

> Account state describes what the user currently owns, not what they want.

---

## 4.2 Ownership

Ownership records what characters and weapons the account currently possesses.

For characters:

```text
-1 → character not owned
 0 → C0
 1 → C1
 2 → C2
 ...
```

The `-1` state is important.

If the account does not own Vesna:

```python
account.owned_characters.get("Vesna", -1)
```

returns `-1`.

Therefore:

```text
Owned: C-1
Goal:  C0
Needed: 1 copy
```

Whereas:

```text
Owned: C0
Goal:  C2
Needed: 2 copies
```

### Terminology rule

The project must consistently distinguish:

* `owned_constellation`
* `goal_constellation`
* `copies_needed`
* `target_copies`

A constellation is **not** a number of copies.

---

# 5. Goals

A `Goal` represents a specific objective in the user's prioritized roadmap.

Current conceptual model:

```python
@dataclass
class Goal:
    character: str
    constellation: int
    priority: int
```

Example:

```python
Goal("Vesna", 0, 1)
Goal("Vodynista", 0, 2)
Goal("Vesna", 2, 3)
Goal("Tsaritsa", 0, 4)
```

The repeated Vesna goals are intentional.

A goal does not contain banner information.

The goal says:

> "I want Vesna at C2."

The banner says:

> "Vesna is available in version 7.0."

Those are separate concepts.

---

# 6. Goal Status

`GoalStatus` represents the current state of a goal relative to the account.

```python
@dataclass
class GoalStatus:
    goal: Goal
    copies_needed: int
```

The calculation is:

```python
owned_constellation = account.owned_characters.get(
    goal.character,
    -1
)

copies_needed = max(
    goal.constellation - owned_constellation,
    0
)
```

Examples:

| Owned     | Goal | Copies needed |
| --------- | ---- | ------------: |
| Not owned | C0   |             1 |
| C0        | C0   |             0 |
| C0        | C2   |             2 |
| C1        | C2   |             1 |
| C2        | C2   |             0 |

This layer answers:

> "How much remains to satisfy this goal?"

It does not decide whether the user should pursue it.

---

# 7. Banners

A `Banner` represents an opportunity to pull.

Current conceptual model:

```python
@dataclass
class Banner:
    character: str
    version: str
    phase: int
```

The eventual model can contain both character and weapon banners.

A banner is chronological.

For example:

```text
7.0 → Vesna
7.1 → Tsaritsa
7.2 → Vodynista
```

This chronological ordering is fundamental to the planner.

Priority order and banner order are deliberately allowed to differ.

---

# 8. Roadmap

`Roadmap` contains the user's objectives and expected banner schedule.

```python
@dataclass
class Roadmap:
    goals: list[Goal]
    banners: list[Banner]
```

Example:

```python
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
```

The roadmap represents **user assumptions**, not guaranteed future game information.

---

# 9. Identifying Actionable Goals

Given:

```text
Account
Roadmap
Current Banner
```

the planner must determine which goals are relevant to the current banner.

For the initial example:

```text
Current banner: Vesna 7.0
```

the relevant goals are:

```text
Priority 1 → Vesna C0
Priority 3 → Vesna C2
```

But the C2 goal cannot be treated as an independent objective while the C0 goal is incomplete.

Conceptually:

```text
Vesna C0 → active
Vesna C2 → dependent on Vesna C0
```

After obtaining Vesna C0:

```text
Vesna C0 → complete
Vesna C2 → active
```

This dependency should be represented by the goal state rather than duplicated manually.

---

# 10. Probability Engine

The probability engine answers isolated probability questions.

It should remain independent from roadmap strategy logic.

Its fundamental question is:

> Given this starting pity/guarantee state, how likely am I to obtain a specified number of featured copies within N wishes?

---

## 10.1 Pull Rate

The pity curve is represented by mechanics data.

```python
def pull_rate(
    pity: int,
    mechanics: WishMechanics
) -> float:
    ...
```

---

## 10.2 Single-Copy Probability

For a single featured character target, probability can be calculated analytically.

```python
def cumulative_probability(
    wishes: int,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
) -> np.ndarray:
    ...
```

The result is:

```text
index N → probability of obtaining the target within N additional wishes
```

No Monte Carlo sampling is required for this case.

---

## 10.3 Confidence Inversion

The engine can invert the probability curve:

```python
def wishes_for_confidence(
    confidence: float,
    starting_pity: int,
    guaranteed: bool,
    mechanics: WishMechanics,
) -> int:
    ...
```

This answers:

> "How many wishes are required to reach at least 90% probability?"

---

## 10.4 Multi-Copy Targets

The probability engine must distinguish:

```text
C0 for an unowned character → 1 copy
C2 from C0               → 2 copies
C2 from C1               → 1 copy
```

The simulator works in terms of **target copies**, not constellation numbers.

Multi-copy targets become increasingly stateful because each obtained 5-star resets pity and may change guarantee state.

---

# 11. Simulation Engine

Monte Carlo simulation is used when multiple roadmap variables interact.

The simulation eventually models:

* Multiple banners
* Character and weapon pity
* Guarantee state
* Multiple copies
* Income between banners
* Banner uncertainty
* Different spending strategies
* Account state transitions
* Downstream goal outcomes

A simulation run represents one possible future account history.

Conceptually:

```text
Starting Account
      ↓
Banner 1
      ↓
Decision
      ↓
Pull outcomes
      ↓
Updated Account
      ↓
Income
      ↓
Banner 2
      ↓
Decision
      ↓
...
```

The important output is not merely whether a pull succeeded.

It is the **resulting account state and roadmap outcome**.

---

# 12. Strategy

A strategy represents a sequence of spending behavior across the roadmap.

A strategy may effectively say:

```text
At Vesna:
    pursue C1

At Tsaritsa:
    stop / skip

At Vodynista:
    pursue C0
```

The exact class structure should be designed when the optimizer is implemented.

We should not over-design `Strategy` before the decision logic exists.

---

# 13. Strategy Optimization

This is the core planner logic.

At a current banner, the planner:

1. Identifies goals relevant to the banner.
2. Determines which outcomes are available according to the user's Preferences.
3. Generates candidate spending strategies.
4. Evaluates their effect on the rest of the roadmap.
5. Checks protected future goals against the confidence threshold.
6. Determines which strategies satisfy the constraints.
7. Selects the highest-preference feasible outcome.

Conceptually:

```text
Generate candidate strategies
            ↓
       Evaluate each
            ↓
  ┌─────────┴─────────┐
  │                   │
Meets confidence?   Doesn't
  │                   │
  ▼                   ▼
Feasible            Reject
  │
  ▼
Compare preference rank
```

The planner does not choose based on a generic notion of "best."

It follows the user's defined priorities, preferences, and confidence threshold.

---

# 14. Safe Spending

"Safe spending" is the amount that can be spent on the current banner while maintaining the required confidence in protected future objectives.

This should ultimately be determined by roadmap-level evaluation.

A simple reserve calculation can be used during early development, but the final planner should account for:

* Starting pity
* Guarantee state
* Pity carried between banners
* Actual spending decisions
* Income timing
* Multiple future goals
* Multi-copy targets
* Banner uncertainty

Therefore:

```text
safe_spend ≠ simply available_wishes - sum(independent_reserves)
```

That equation may be useful as an early approximation, but it should not become a permanent architectural assumption.

---

# 15. Preferences

Preferences define the user's acceptable outcomes for each character.

Example:

```text
Vesna
1. C2R1
2. C1R1
3. C2
4. C1
5. C0
```

Each preference has:

```python
character
rank
constellation
weapon_refinement
tier
notes
```

Rank is assigned from the user's ordering.

Outcome labels such as:

```text
C1R1
C2
C0
```

are derived for display and should not be stored as authoritative data.

Preferences answer:

> "If I'm going after this character, what outcomes do I prefer?"

Goals answer:

> "Where does this objective sit in my roadmap?"

These should not be collapsed into one concept.

---

# 16. Income

Future income is modeled as a range.

```text
Low
Expected
High
```

Income is associated with future versions and can optionally be broken down into sources.

The simulation engine uses the aggregate forecast.

Sources such as:

```text
Commissions
Events
Abyss
Exploration
```

are supporting detail.

---

# 17. Wish Mechanics

Wish mechanics are data rather than hard-coded assumptions.

```python
@dataclass
class WishMechanics:
    banner_type: str
    hard_pity: int
    soft_pity_start: int
    base_rate: float
    soft_pity_increment: float
    featured_rate: float
```

This allows the mechanics to be updated without changing probability logic.

The exact weapon-banner mechanics should be verified independently before implementing the weapon simulator.

---

# 18. Development Architecture

The project should be developed in layers.

## Current Layer: Domain Model

First establish and test:

```text
Account
Goal
GoalStatus
Banner
Roadmap
```

Then establish:

```text
Account + Goals
        ↓
GoalStatus
```

and:

```text
Roadmap + Current Banner
        ↓
Relevant / actionable goals
```

This is the current development stage.

---

## Phase 1 — Domain and Data Layer

Implement and test:

* Account
* Ownership
* Goal
* GoalStatus
* Preference
* Banner
* Roadmap
* Income
* WishMechanics

Database/API infrastructure can be added around a stable domain model.

---

## Phase 2 — Analytical Probability

Implement and test:

* `pull_rate`
* `cumulative_probability`
* `wishes_for_confidence`
* Single-copy character probability
* Starting pity
* Guarantee state

Validate against known mechanics and expected results.

---

## Phase 3 — Basic Planner

Build the first version of:

* Current-banner identification
* Goal matching
* Goal dependencies
* Basic protected-goal calculation
* Safe-spend approximation
* Spending table

Initially restrict this to simpler single-copy scenarios.

---

## Phase 4 — Monte Carlo

Implement:

* Full account state transitions
* Multi-copy targets
* Multiple banners
* Income
* Pity/guarantee propagation
* Strategy simulation
* Roadmap outcome tracking

Validate the simulator against analytical results wherever an equivalent analytical case exists.

---

## Phase 5 — Strategy Optimizer

Implement:

* Preference chains
* Candidate strategy generation
* Future-goal protection
* Confidence constraints
* Strategy comparison
* Stop conditions
* Recommendation generation

This is where the planner becomes the actual decision-making tool.

---

## Phase 6 — API

Build the FastAPI layer around the proven domain and planner logic.

Potential endpoints:

```text
/accounts
/accounts/{id}

/accounts/{id}/goals
/accounts/{id}/preferences
/accounts/{id}/banners
/accounts/{id}/income

/accounts/{id}/probability/character
/accounts/{id}/probability/weapon
/accounts/{id}/probability/wishes-needed

/accounts/{id}/planner/recommendation
/accounts/{id}/planner/safe-spend
/accounts/{id}/planner/spend-table
/accounts/{id}/planner/stop-conditions

/accounts/{id}/simulation/run
/accounts/{id}/simulation/{job_id}
```

The API should expose the domain logic rather than contain the domain logic itself.

---

## Phase 7 — Frontend

Build after the backend logic is proven.

Primary dashboard:

```text
Current recommendation
        ↓
Why
        ↓
Safe spending
        ↓
Spend table
        ↓
Protected future goals
        ↓
Roadmap
```

Account, Goals, Preferences, Banner, and Income editing are secondary views.

---

# 19. Initial Scope Exclusions

The planner does not initially attempt to:

* Predict banners automatically
* Scrape game data
* Evaluate character strength
* Compare teams
* Optimize DPS
* Recommend artifacts
* Recommend weapons outside pull planning
* Compare multiple accounts

The user supplies assumptions.

The planner's job is to **reason correctly from those assumptions**.

---

# 20. Guiding Architecture

The most important separation is:

```text
ACCOUNT
"What do I have?"

        +

ROADMAP
"What do I want, in what priority, and when might I have the opportunity?"

        ↓

GOAL STATUS
"What remains to accomplish?"

        ↓

CURRENT BANNER
"What can I pursue right now?"

        ↓

STRATEGY
"What could I spend and what outcome would I pursue?"

        ↓

PROBABILITY / SIMULATION
"What could happen if I do that?"

        ↓

ROADMAP EVALUATION
"What does that do to everything else I care about?"

        ↓

RECOMMENDATION
"What is the highest-preference outcome that satisfies my constraints?"
```

The probability engine is therefore **not the planner**.

The simulation engine is **not the planner**.

The roadmap is **not the strategy**.

The account is **not the user's goals**.

The planner is the layer that eventually connects all of them.
