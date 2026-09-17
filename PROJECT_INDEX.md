# Genshin Pull Strategy Planner — Project Index

> Compact context for developers and AI coding agents.
>
> **Read this file first. Do not read the entire repository by default.** Identify the relevant subsystem, then inspect its implementation and tests. `Design_Document.md` is the detailed specification; this file is the routing/index layer.

## 1. Purpose

The planner is a probability-based **resource-allocation and decision tool** for deciding what to spend wishes on, how far to pursue it, and what that spending puts at risk later.

The central question is:

> Given the current account, prioritized goals, preferred outcomes, future banners/income, and required confidence threshold, what is the most preferred outcome that can be pursued without violating the relevant higher-priority roadmap constraints?

It is not primarily an affordability calculator. The planner is intended to be rerun after meaningful account updates.

## 2. Core invariants

### Priority and preference are separate

- **Priority** says which roadmap objectives constrain one another.
- **Preference** says which outcome is desired for a character.
- A character may appear in multiple goals at different priorities.
- The planner must never invent an outcome that the user did not define.

Example:

```text
Priority 1 → Vesna C0
Priority 2 → Vodynista C0
Priority 3 → Vesna C2
Priority 4 → Tsaritsa C0
```

### Constellation is not copies

Character ownership uses:

```text
-1 = not owned
 0 = C0
 1 = C1
 2 = C2
```

A target constellation is a **resulting constellation**, not a copy count. Copies needed are derived from current ownership.

### Higher-priority protection

A future goal constrains a current decision only when its priority outranks the objective represented by that decision. Lower-priority future goals are still simulated and reported; they do not veto a higher-priority current objective.

`optimizer.protection.constraining_goals()` implements this Phase 5 filtering.

### Preference selection is lexicographic

The intended decision order is:

```text
1. Most preferred feasible outcome
2. Within that outcome, largest feasible spending cap
```

There is no global "best" score combining preference and probability.

### Spending caps are not assumed monotonic

The optimizer currently scans candidate caps in descending order rather than binary-searching. Extra spending can change pity/guarantee state, so feasibility can theoretically change non-monotonically.

### Skip is valid

If no available outcome is feasible, the recommendation is `action="skip"`, with diagnostics. Do not fabricate an outcome.

## 3. Current architecture

```text
API / future frontend
        ↓
    PlannerContext
        ↓
   ┌────┴────┐
   ↓         ↓
planner   probability
   ↓         ↓
   └────┬────┘
        ↓
   simulation
        ↓
    optimizer
        ↓
 Recommendation
```

More precisely:

- `domain/` — what the entities and data mean
- `probability/` — exact pull-probability calculations
- `planner/` — converts account + roadmap + current position into planning inputs
- `simulation/` — Monte Carlo future account histories
- `optimizer/` — candidate generation, evaluation, protection, and final recommendation
- `api/` — FastAPI boundary
- `tests/` — behavioral contract by subsystem

Keep strategy decisions out of `probability/`, and keep API concerns out of core planning logic.

## 4. Important current entry points

### `planner.context.PlannerContext`

The main input object for planner/simulation/optimizer logic.

Contains:

```text
account
roadmap
current_version
current_phase
income
income_scenario
confidence
mechanics
```

`current_version` and `current_phase` are currently planner inputs, not account fields.

Income semantics currently are:

- forecast values represent wishes that arrive after the current account state
- current-version forecast is still future income
- earlier-version income is assumed already reflected in `account.wishes`
- `income_credit(version)` sums configured scenario income from the current version through that version

Source: `planner/context.py`.

### `optimizer.recommend.recommend()`

Final decision boundary.

Current flow:

```text
available_outcomes()
        ↓
_caps()
        ↓
for each outcome:
    evaluate_candidate(outcome, cap)
        ↓
    first feasible candidate wins
        ↓
Recommendation
```

If no candidate is feasible, it returns a skip recommendation with rejected-outcome diagnostics.

### `optimizer.evaluation.evaluate_candidate()`

Evaluates one `(outcome, budget)` candidate through the Monte Carlo simulator.

Feasibility currently means:

```text
all constraining protected goals meet confidence threshold
AND
current-banner outcome probability > 0
```

Default optimizer evaluation runs: `2,000`.

This is deliberately lower than the simulation module's default of `10,000` because the optimizer evaluates many candidates.

### `optimizer.outcomes.available_outcomes()`

Determines what outcomes the optimizer is allowed to pursue on the current banner.

Resolution:

1. current banner's character
2. that character's preference chain, if present
3. otherwise the single active roadmap goal for that character

Important current behavior:

- preferences for other characters are ignored
- duplicate preference constellations are collapsed to the best-ranked preference
- already-owned target constellations are removed
- if a preference chain exists, only its constellations are offered
- if no preference chain exists and multiple active goals match the current banner character, it raises `ValueError` rather than choosing silently
- weapon refinement is carried in the outcome but is currently display-only; weapon mechanics are not simulated

**Important:** for same-character preference chains, eligible outcomes are currently sorted by **descending constellation**, not preference rank. This is intentional in the current implementation because a deeper constellation subsumes lower constellations for the same character. Do not change this casually; it is a key behavior to understand when debugging recommendation selection.

### `optimizer.protection`

Two separate concepts live here:

1. `protected_groups()` — identifies unsatisfied goals whose next banner is strictly after the current banner and groups goals sharing a future banner for execution.
2. `constraining_goals()` — filters those protected goals by priority for the current decision.

Grouping is only an execution optimization. Simulation still evaluates every original roadmap `Goal` independently.

`priority_for_outcome()` exists because a deeper same-character outcome can correspond to a lower-priority roadmap goal. Example:

```text
Priority 1 → Vesna C0
Priority 2 → Vodynista C0
Priority 3 → Vesna C2
```

For a Vesna C2 candidate, the protection anchor can be Priority 3, allowing Priority 2 Vodynista to constrain the C2 reach without incorrectly constraining the base Vesna C0 objective.

### `simulation.engine`

Monte Carlo future-history engine.

Each history walks roadmap banners chronologically from the current banner onward:

```text
starting account
→ current/future banner
→ spending according to SpendPlan
→ pity/guarantee/ownership transition
→ income
→ next banner
→ ...
```

Important mechanics:

- no mutation of domain account state
- featured copy resets pity to 0 and guarantee off
- lost 50/50 resets pity to 0 and sets guarantee on
- non-5-star increments pity
- spend is bounded by `min(budget, available wishes)`
- target constellation is converted to copies needed from the simulated account state
- income is credited at the first processed banner of a version; the current banner receives no future income, matching the planner's budget accounting
- all pull rates come from `probability.pull_rate`

Simulation defaults:

```text
runs = 10,000
seed = 0
```

## 5. Repository routing

### `domain/`

Read when changing the meaning or stored state of core entities.

Relevant concepts:

```text
Account
Ownership
Goal
GoalStatus / goal evaluation
Banner
Roadmap
Preference
IncomeForecast
WishMechanics
```

Especially important: account ownership uses `-1` for not owned, and goal calculations distinguish target constellation from copies needed.

### `probability/`

Read for mathematical pull-probability behavior.

Use exact probability for isolated tractable questions such as:

- cumulative probability within N wishes
- wishes required for a confidence level
- multi-copy probability when the DP supports it
- pity/rate calculations

Do not move roadmap strategy logic into this layer.

### `planner/`

Read for interpreting the current account/roadmap position.

Key files:

```text
planner/context.py
planner/banners.py
planner/goals.py
planner/protection.py
planner/safe_spend.py
planner/spend_table.py
```

### `simulation/`

Read when the question concerns future account state, banner-to-banner consequences, pity/guarantee carry, income timing, or Monte Carlo aggregation.

Key files:

```text
simulation/engine.py
simulation/strategy.py
simulation/outcomes.py
simulation/results.py
```

### `optimizer/`

Read first for recommendation/strategy bugs.

Key files:

```text
optimizer/outcomes.py       # what can be pursued
optimizer/candidates.py     # candidate SpendPlan generation
optimizer/evaluation.py     # candidate simulation + feasibility
optimizer/protection.py     # protected/constraining goals
optimizer/recommend.py      # final choice
optimizer/stops.py          # stop conditions
```

### `api/`

Read for FastAPI request/response behavior. Do not duplicate planner or optimizer rules in routers.

### `tests/`

Read the corresponding tests before changing behavior. The optimizer tests are especially important because many rules are edge-case dependent.

### `legacy/`

Historical implementation only. Not the current source of truth.

### Scratch / experimental files

Files with names such as `scratch_*` are development aids, not architectural sources of truth.

## 6. Debugging a surprising recommendation

Do not assume the bug is in `recommend()` just because the final answer is wrong.

Trace in this order:

```text
1. current_banner(context)
2. actionable/evaluated goals
3. preferences supplied to recommend()
4. available_outcomes()
5. priority_for_outcome()
6. constraining_goals()
7. candidate_plan()
8. evaluate_candidate()
9. simulation result
10. recommend()
```

For an unexpected "skip", determine whether:

- the expected outcome was absent from `available_outcomes()`
- the outcome was present but sorted differently than expected
- a goal was incorrectly classified as constraining
- the candidate simulation produced an unexpected probability
- the confidence threshold was actually violated
- the outcome probability was zero in the finite Monte Carlo sample

This sequence is more useful than immediately modifying the final recommendation logic.

## 7. Probability vs. simulation

Use **exact probability** when asking an isolated mathematical question.

Use **simulation** when choices interact across banners.

Example:

```text
Exact:
"What is the probability of getting C2 in 150 wishes?"

Simulation:
"If I spend up to 150 wishes on this banner, what happens to
my future C0/C2 goals after pity, guarantee, and income carry forward?"
```

The optimizer currently evaluates candidates through simulation rather than substituting the earlier analytic approximation.

## 8. Source-of-truth hierarchy

When information conflicts:

1. Current executable implementation
2. Current tests describing intended behavior
3. `Design_Document.md`
4. Legacy code
5. Scratch/experimental code

If implementation and design disagree, identify the conflict explicitly rather than silently assuming one is correct.

## 9. Change discipline

For a normal coding task:

1. Read this index.
2. Identify the relevant subsystem.
3. Read the implementation entry point.
4. Read the directly relevant tests.
5. Trace dependencies only as needed.
6. Add or update the smallest test demonstrating the desired behavior.
7. Make the implementation change in the layer where the behavior actually originates.
8. Run focused tests, then the full suite when practical.
9. Update this index only when architecture, routing, or a durable invariant changes.

Do not turn this file into a changelog or duplicate the full design document.

## 10. Detailed specification

For deeper requirements, examples, design rationale, and phased architecture, read `Design_Document.md`.

**Default rule for AI agents:** `PROJECT_INDEX.md` first; relevant implementation + tests second; `Design_Document.md` only when the index and code/tests do not answer the question.
