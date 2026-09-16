# Legacy Prototypes

Pre-Phase-1 sketches, archived when the `domain/` package was introduced.
They are kept for history only and are not imported by anything.

- `account.py` - original `Account` stub (no validation, plain dict ownership)
- `goals.py` - original `GoalStatus` sketch (incomplete: missing imports and
  the `Goal` definition)
- `plan.py` - original `Goal` / `Banner` / `Roadmap` sketch plus a sample
  roadmap, now covered by the Phase 1 test fixtures
- `main.py` - original demo script (was broken: referenced an undefined
  `pull_plan` and an unimported `Banner`)

`character_probability.py` deliberately stays at the project root: it is a
working Monte Carlo prototype and the starting point for Phase 4, not Phase 1
scope.
