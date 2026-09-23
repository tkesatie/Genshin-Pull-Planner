# UI ↔ Backend Contract

Primary reference for frontend work. The UI should treat API responses as the source of truth. Return to backend source only when this contract is incomplete or a real backend change is required.

**Current scope:** character planning only. Weapon/refinement fields exist in the data model for future use, but weapon mechanics and weapon optimization are not implemented.

## 1. API surface

The FastAPI app serves the static frontend at / and the API from the same origin.

| Endpoint | Purpose |
|---|---|
| GET /health | Liveness |
| GET /accounts | Account summaries |
| POST /accounts | Create account + optional complete roadmap |
| GET /accounts/{id} | Full account + roadmap |
| PUT /accounts/{id} | Replace account state/settings |
| DELETE /accounts/{id} | Delete account |
| GET/PUT /accounts/{id}/goals | Roadmap goals |
| GET/PUT /accounts/{id}/banners | Banner schedule |
| GET/PUT /accounts/{id}/preferences | Preference chains |
| GET/PUT /accounts/{id}/income | Future income |
| DELETE /accounts/{id}/income | Remove income forecast |
| GET /accounts/{id}/planner/goals | Goal states |
| GET /accounts/{id}/planner/safe-spend | Analytical safe-spend approximation |
| GET /accounts/{id}/planner/spend-table | Spend/probability table |
| GET /accounts/{id}/planner/recommendation | Central optimizer decision |
| GET /accounts/{id}/planner/stop-conditions | Stop instructions for current decision |
| GET /accounts/{id}/probability/character | Exact single-copy probability |
| GET /accounts/{id}/probability/wishes-needed | Exact wishes needed for confidence |
| POST /accounts/{id}/simulation/run | Submit lower-level Monte Carlo simulation |
| GET /accounts/{id}/simulation/{job_id} | Poll simulation |

/probability/weapon currently returns 501 and is not a UI feature.

The current frontend is a single static frontend/index.html that already contains a basic dashboard and direct API calls. It can be refactored/replaced during the UI phase.

## 2. Account data

Account state:

~~~json
{
  "wishes": 180,
  "current_pity": 32,
  "character_guarantee": false,
  "owned_characters": {"Navia": 0}
}
~~~

- wishes: currently held wishes.
- current_pity: character-banner pulls since the last 5-star.
- character_guarantee: whether the next 5-star is guaranteed featured.
- owned_characters: character → resulting constellation; -1 means not owned. A constellation is not a copy count.

Planner settings:

~~~json
{
  "current_version": "7.0",
  "current_phase": 1,
  "confidence": 0.9,
  "income_scenario": "expected"
}
~~~

confidence is the required probability threshold for protected roadmap goals. income_scenario is low, expected, or high.

PUT /accounts/{id} replaces account state/settings but does not modify roadmap collections. Re-run the planner after meaningful account changes.

## 3. Roadmap data

### Goals

~~~json
{
  "goals": [
    {"character": "Navia", "constellation": 0, "priority": 1},
    {"character": "Navia", "constellation": 1, "priority": 2},
    {"character": "Navia", "constellation": 2, "priority": 3},
    {"character": "Tsaritsa", "constellation": 0, "priority": 4}
  ]
}
~~~

A Goal is a **roadmap priority/progression milestone**. Priority 1 is protected before priority 2, etc.

**Critical:** same-character C0 → C1 → C2 goals are cumulative progression, not competing alternatives.

### Banners

~~~json
{
  "target_kind": "character",
  "target_name": "Navia",
  "character": "Navia",
  "version": "7.0",
  "phase": 1,
  "start": "2026-01-01T11:00:00+08:00",
  "end": "2026-01-21T17:59:59+08:00"
}
~~~

A weapon banner is the same shape with `"target_kind": "weapon"`, `"target_name"`/`"weapon"` set, and `"character": null`.

The banner schedule is a user assumption. Banners are ordered chronologically by version/phase, compared numerically ("7.9" precedes "7.10") and then by phase. One banner carries exactly one featured target: several simultaneous opportunities - two character banners in one phase, or a character banner next to a weapon banner - are separate entries in the same version/phase.

`start`/`end` are optional and describe when the banner is live. When supplied they must be timezone-aware ISO 8601 (a naive timestamp is rejected rather than assumed UTC or local time), and the interval is half-open: `start` is inclusive, `end` is exclusive, so a hand-off instant belongs to exactly one banner. Dates are display/real-time metadata; (version, phase) remains what the planner simulates.

### Preferences

~~~json
{
  "preferences": [
    {
      "character": "Navia",
      "rank": 1,
      "constellation": 2,
      "weapon_refinement": 0,
      "notes": "Main target",
      "label": "C2"
    },
    {
      "character": "Navia",
      "rank": 2,
      "constellation": 1,
      "weapon_refinement": 0,
      "notes": "Fallback",
      "label": "C1"
    }
  ]
}
~~~

A Preference is an alternative acceptable way of pursuing an objective, ordered by rank. It is **not** roadmap priority.

Do not visually merge Goal priority and Preference rank into one ordering.

weapon_refinement/labels may be displayed as stored data, but refinement is currently display-only and must not be presented as optimized behavior.

### Income

~~~json
{
  "versions": [
    {
      "version": "7.0",
      "estimate": {"low": 60, "expected": 75, "high": 90},
      "sources": [],
      "aggregate": {"low": 60, "expected": 75, "high": 90}
    }
  ]
}
~~~

Income is future income arriving after the current account state. null income means no future income is credited; that differs from a forecast of zero wishes.

Roadmap PUT requests replace the complete collection.

## 4. Planner query overrides

Planner endpoints accept:

- confidence — temporary override of stored confidence.
- income_scenario — low, expected, or high.

These do not modify stored account settings.

## 5. Goal-state response

GET /accounts/{id}/planner/goals

~~~json
{
  "current_banner": null,
  "available_banners": [
    {"character": "Navia", "version": "7.0", "phase": 1},
    {"character": "Arlecchino", "version": "7.0", "phase": 1}
  ],
  "upcoming_banners": [
    {"character": "Tsaritsa", "version": "7.1", "phase": 1}
  ],
  "goals": [
    {
      "goal": {"character": "Navia", "constellation": 0, "priority": 1},
      "copies_needed": 1,
      "state": "active",
      "blocked_by": null,
      "next_banner": {"character": "Navia", "version": "7.0", "phase": 1},
      "relevant": true,
      "actionable": true
    },
    {
      "goal": {"character": "Navia", "constellation": 2, "priority": 3},
      "copies_needed": 3,
      "state": "blocked",
      "blocked_by": {"character": "Navia", "constellation": 0, "priority": 1},
      "next_banner": {"character": "Navia", "version": "7.0", "phase": 1},
      "relevant": true,
      "actionable": false
    },
    {
      "goal": {"character": "Tsaritsa", "constellation": 0, "priority": 4},
      "copies_needed": 1,
      "state": "active",
      "blocked_by": null,
      "next_banner": {"character": "Tsaritsa", "version": "7.1", "phase": 1},
      "relevant": false,
      "actionable": false
    }
  ]
}
~~~

States:

- satisfied: account already meets the target.
- active: unsatisfied and not blocked by an earlier same-character constellation.
- blocked: an earlier same-character constellation must be reached first.
- next_banner null: the goal still exists, but no matching scheduled banner exists.
- relevant: goal character matches one of the currently available banners.
- actionable: relevant + active.
- when multiple banners share the current version/phase, current_banner is null and available_banners lists every simultaneous opportunity.
- upcoming_banners: every scheduled banner strictly after the current version/phase, chronologically. The current slot's alternatives are current opportunities, not upcoming ones.
- next_banner: the goal target's next opportunity, at or after the current position (null when nothing is scheduled for it).

A blocked goal remains part of the roadmap.

## 6. Safe-spend response

GET /accounts/{id}/planner/safe-spend?character=Navia

When multiple banners are active, `character` selects the banner being analyzed. The response includes all `available_banners`.

~~~json
{
  "current_banner": {"character": "Navia", "version": "7.0", "phase": 1},
  "available_banners": [{"character": "Navia", "version": "7.0", "phase": 1}],
  "safe_spend": 84,
  "account_wishes": 180,
  "confidence": 0.9,
  "approximation": "sequential independent reserves (Phase 3); the recommendation endpoint evaluates spending through simulation (§14)",
  "protected": [
    {
      "goal": {"character": "Tsaritsa", "constellation": 0, "priority": 4},
      "banner": {"character": "Tsaritsa", "version": "7.1", "phase": 1},
      "budget_at_banner": 96,
      "required_wishes": 80,
      "confidence": 0.93,
      "meets_threshold": true
    }
  ]
}
~~~

safe_spend is an analytical Phase 3 approximation, not the optimizer's exact maximum feasible cap. It may disagree with recommendation.budget. A value of 0 is valid.

## 7. Spending analysis

GET /accounts/{id}/planner/spend-table?character=Vesna&step=10&runs=2000&seed=0

Shows how different current-banner spending levels affect the selected current-banner outcome and every protected future goal. It uses the same Monte Carlo simulation and multi-copy outcome semantics as the recommendation; it is not limited to single-copy goals.

Response:

~~~json
{
  "current_banner": {"character": "Vesna", "version": "7.0", "phase": 1},
  "available_banners": [
    {"character": "Vesna", "version": "7.0", "phase": 1},
    {"character": "Navia", "version": "7.0", "phase": 1}
  ],
  "outcomes": [
    {"character": "Vesna", "constellation": 0, "rank": 1, "weapon_refinement": null, "label": "C0"},
    {"character": "Vesna", "constellation": 2, "rank": 3, "weapon_refinement": null, "label": "C2"}
  ],
  "step": 10,
  "confidence": 0.9,
  "runs": 2000,
  "seed": 0,
  "rows": [
    {
      "wishes_spent": 80,
      "outcomes": [
        {"outcome": {"character": "Vesna", "constellation": 0, "rank": 1, "weapon_refinement": null, "label": "C0"}, "probability": 0.94},
        {"outcome": {"character": "Vesna", "constellation": 2, "rank": 3, "weapon_refinement": null, "label": "C2"}, "probability": 0.31}
      ],
      "protected": [],
      "all_protected_meet_threshold": true
    }
  ]
}
~~~

`outcome_probability` is the probability of reaching the selected resulting constellation on the current banner within the displayed spend cap. For example, Vesna C2 from an unowned account is evaluated as a three-copy cumulative target; the simulator handles the intermediate C0/C1 progression.

`all_protected_meet_threshold` considers only protected goals that constrain the selected outcome. Lower-priority future goals remain in `protected` and are reported but do not veto a higher-priority current decision.

`step` controls the displayed spending increments. The endpoint always includes both 0 and the account's full current wish balance.

This section is explanatory decision analysis. The recommendation endpoint remains the authoritative decision.

**Known gap (tracked, not yet redesigned):** the current implementation only evaluates the current banner's own character. It does not show how spending on the current banner moves probabilities for other characters' future goals beyond the aggregate `protected` list (i.e. no per-row breakdown of "how does this spend change my Tsaritsa C0 chance specifically" beyond what `protected` already reports). This is being reconsidered given that pull-history conditioning (`/planner/cached-refresh`, §12 below) now keeps evidence fresh after each recorded pull, which changes what this table needs to show. Do not build UI against a redesigned shape until that's settled.

## 8. Recommendation response

GET /accounts/{id}/planner/recommendation

Query parameters:

- runs: Monte Carlo histories per candidate; optimizer default is 2000.
- seed: reproducibility seed; default 0; null uses OS entropy.
- budgets: optional explicit candidate caps. If omitted, every cap from available wishes down to 0 is scanned.
- minimum_outcome_probability: probability floor separating pursue from discretionary.

### Pursue

~~~json
{
  "banner": {"character": "Navia", "version": "7.0", "phase": 1},
  "action": "pursue",
  "outcome": {
    "character": "Navia",
    "constellation": 2,
    "rank": 1,
    "weapon_refinement": 0,
    "label": "C2"
  },
  "budget": 150,
  "plan": {
    "entries": [
      {
        "banner": {"character": "Navia", "version": "7.0", "phase": 1},
        "target_constellation": 2,
        "budget": 150
      }
    ]
  },
  "outcome_probability": 0.91,
  "all_goals_probability": 0.86,
  "confidence": 0.9,
  "minimum_outcome_probability": 0.5,
  "protected": [
    {
      "goal": {"character": "Tsaritsa", "constellation": 0, "priority": 4},
      "banner": {"character": "Tsaritsa", "version": "7.1", "phase": 1},
      "probability": 0.93,
      "meets_threshold": true,
      "constraining": true
    }
  ],
  "rejected": [],
  "skip_reason": null,
  "discretionary_reason": null,
  "stops": {
    "action": "pursue",
    "outcome_label": "C2",
    "spend_cap": 150,
    "rules": ["Stop after reaching C2.", "Do not spend more than 150 wishes."]
  },
  "runs": 2000,
  "seed": 0
}
~~~

### Discretionary

~~~json
{
  "banner": {"character": "Navia", "version": "7.0", "phase": 1},
  "action": "discretionary",
  "outcome": {
    "character": "Navia",
    "constellation": 2,
    "rank": 1,
    "weapon_refinement": 0,
    "label": "C2"
  },
  "budget": 120,
  "plan": {
    "entries": [
      {
        "banner": {"character": "Navia", "version": "7.0", "phase": 1},
        "target_constellation": 2,
        "budget": 120
      }
    ]
  },
  "outcome_probability": 0.41,
  "all_goals_probability": 0.37,
  "confidence": 0.9,
  "minimum_outcome_probability": 0.5,
  "protected": [],
  "rejected": [],
  "skip_reason": null,
  "discretionary_reason": "The outcome is feasible, but its estimated probability is below the minimum ordinary-recommendation threshold.",
  "stops": {
    "action": "discretionary",
    "outcome_label": "C2",
    "spend_cap": 120,
    "rules": ["Treat this as discretionary spending.", "Do not spend more than 120 wishes."]
  },
  "runs": 2000,
  "seed": 0
}
~~~

### Skip

~~~json
{
  "banner": {"character": "Navia", "version": "7.0", "phase": 1},
  "action": "skip",
  "outcome": null,
  "budget": 0,
  "plan": null,
  "outcome_probability": 0.0,
  "all_goals_probability": 0.58,
  "confidence": 0.9,
  "minimum_outcome_probability": 0.5,
  "protected": [],
  "rejected": [
    {
      "outcome": {
        "character": "Navia",
        "constellation": 2,
        "rank": 1,
        "weapon_refinement": 0,
        "label": "C2"
      },
      "best_budget": 150,
      "outcome_probability": 0.92,
      "min_protected_probability": 0.72,
      "shortfalls": [
        {
          "goal": {"character": "Tsaritsa", "constellation": 0, "priority": 4},
          "banner": {"character": "Tsaritsa", "version": "7.1", "phase": 1},
          "probability": 0.72,
          "meets_threshold": false,
          "constraining": true
        }
      ]
    }
  ],
  "skip_reason": "No available outcome can be pursued while satisfying the required protection constraints.",
  "discretionary_reason": null,
  "stops": {
    "action": "skip",
    "outcome_label": null,
    "spend_cap": 0,
    "rules": ["Do not spend on this banner."]
  },
  "runs": 2000,
  "seed": 0
}
~~~

For a skip, `all_goals_probability` is the do-nothing baseline's roadmap-wide figure (§1, §2): "if I spend nothing on this banner, what's my chance of completing the whole roadmap?" It is not derived from `rejected` and is computed from its own simulation of the baseline plan.

Recommendation semantics:

- pursue = feasible and outcome_probability meets minimum_outcome_probability.
- discretionary = feasible, but outcome_probability is below minimum_outcome_probability.
- skip = no available outcome is feasible.
- outcome is null for skip.
- budget is the largest feasible **spend cap**, not a commitment or necessarily an exact amount the user will spend.
- optimizer selection is lexicographic: most-preferred feasible outcome first, then largest feasible cap for that outcome.
- protected[].constraining identifies goals whose thresholds actually gate the decision.
- rejected explains more-preferred outcomes that were infeasible.
- all_goals_probability is roadmap-wide (every goal, not just protected ones) and is NOT the product of the individual protected probabilities - goals share one simulated history and are not independent.

## 9. Probability distinctions

Do not collapse these into one number:

| Field/concept | Meaning |
|---|---|
| outcome_probability | Chance of achieving the selected current-banner outcome |
| all_goals_probability | Chance every roadmap goal (protected or not) ends satisfied under the recommendation, or under the do-nothing baseline when skipping |
| protected[].probability | Chance of satisfying a future protected goal |
| confidence | Required probability for protected goals |
| minimum_outcome_probability | Probability floor for an ordinary pursue recommendation |
| safe_spend | Analytical approximation |
| recommendation.budget | Largest feasible optimizer cap |
| spend-table wishes_spent | Hypothetical current-banner spend |
| planned_budget | Cap encoded in a simulation plan |

**Important:** minimum_outcome_probability is not the confidence/protection threshold.

**Important:** all_goals_probability is not derivable from protected[] by multiplying probabilities together; it comes from the simulator counting histories where every goal held at once.

**Important:** a displayed/coarse spend amount is not necessarily the exact maximum safe spend. If the UI supplies a coarse budgets list, it can miss a narrow feasible cap window.

Recommendation probabilities and simulation probabilities are Monte Carlo estimates. Preserve runs and seed in detailed/provenance UI.

## 10. Stop conditions

The recommendation contains stops, and GET /planner/stop-conditions returns the same data independently.

~~~json
{
  "action": "pursue",
  "outcome_label": "C2",
  "spend_cap": 150,
  "rules": [
    "Stop after reaching C2.",
    "Do not spend more than 150 wishes."
  ]
}
~~~

Present these as operational instructions attached to the recommendation, not as a separate optimization result.

## 11. Exact probability endpoints

GET /accounts/{id}/probability/character?wishes=N

Optional: starting_pity, guaranteed, include_curve.

~~~json
{
  "wishes": 80,
  "starting_pity": 32,
  "guaranteed": false,
  "probability": 0.84,
  "curve": [0.0, 0.006, "..."],
  "mechanics": {
    "banner_type": "character_event",
    "hard_pity": 90,
    "soft_pity_start": 74,
    "base_rate": 0.006,
    "soft_pity_increment": 0.06,
    "featured_rate": 0.5
  }
}
~~~

This is exact analytical probability, not Monte Carlo.

GET /accounts/{id}/probability/wishes-needed

Optional: confidence, starting_pity, guaranteed.

~~~json
{
  "confidence": 0.9,
  "starting_pity": 32,
  "guaranteed": false,
  "wishes_needed": 91,
  "probability_at_wishes_needed": 0.901,
  "mechanics": {
    "banner_type": "character_event",
    "hard_pity": 90,
    "soft_pity_start": 74,
    "base_rate": 0.006,
    "soft_pity_increment": 0.06,
    "featured_rate": 0.5
  }
}
~~~

## 12. Lower-level simulation

POST /accounts/{id}/simulation/run accepts a SpendPlan and returns a queued job. GET /accounts/{id}/simulation/{job_id} reports queued, running, succeeded, or failed.

A successful result contains runs, seed, the plan, per-goal probabilities, per-banner aggregates, all_goals_probability, and final-wishes mean/min/max.

This is not the initial dashboard's primary decision endpoint.

## 13. Full account example

~~~json
{
  "id": "demo",
  "label": "Main Account",
  "account": {
    "wishes": 180,
    "current_pity": 32,
    "character_guarantee": false,
    "owned_characters": {"Navia": 0}
  },
  "settings": {
    "current_version": "7.0",
    "current_phase": 1,
    "confidence": 0.9,
    "income_scenario": "expected",
    "mechanics": {
      "banner_type": "character_event",
      "hard_pity": 90,
      "soft_pity_start": 74,
      "base_rate": 0.006,
      "soft_pity_increment": 0.06,
      "featured_rate": 0.5
    }
  },
  "goals": [
    {"character": "Navia", "constellation": 0, "priority": 1},
    {"character": "Navia", "constellation": 1, "priority": 2},
    {"character": "Navia", "constellation": 2, "priority": 3},
    {"character": "Tsaritsa", "constellation": 0, "priority": 4}
  ],
  "banners": [
    {"character": "Navia", "version": "7.0", "phase": 1},
    {"character": "Tsaritsa", "version": "7.1", "phase": 1}
  ],
  "preferences": [
    {"character": "Navia", "rank": 1, "constellation": 2, "weapon_refinement": 0, "notes": "Main target", "label": "C2"}
  ],
  "income": null
}
~~~

## 14. Initial UI information hierarchy

The application should answer the practical question in this order:

1. **Current decision** — pursue / discretionary / skip, target, cap, probability.
2. **Why** — more-preferred rejected outcomes and the protected goals that constrained them.
3. **When to stop** — stop rules and cap.
4. **What is at risk** — future protected-goal probabilities against confidence.
5. **Spending detail** — safe-spend approximation and spend/probability table.
6. **Roadmap** — chronological banners and goal state/priority.
7. **Inputs/settings** — account state, goals, preferences, banners, income, confidence, scenario.

The first screen should not require users to understand optimizer terminology before seeing the decision.

## 15. UI development boundary

Use this contract and the dummy responses as the working reference for UI implementation.

Do not redesign optimizer/backend behavior during UI work.

Only return to backend implementation when:

1. a required UI datum is absent from this contract/API, or
2. the UI exposes a real backend defect, or
3. a genuinely required backend capability does not exist.

Do not implement weapon/refinement optimization until explicitly brought back into scope.
