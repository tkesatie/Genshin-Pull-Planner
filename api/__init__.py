"""FastAPI layer (Design Document §3, §18 Phase 6).

The API is built around the proven domain and planner logic: it exposes the
domain rather than containing it (§18 Phase 6). Every answer this layer
returns is produced by `domain`, `probability`, `planner`, `simulation` or
`optimizer`; the routers translate HTTP into those calls and translate the
results back.

    HTTP request
          |
          v
    schemas  (shape only - values are validated by the domain)
          |
          v
    AccountRecord -> Roadmap -> PlannerContext      (composition)
          |
          v
    probability / planner / simulation / optimizer  (all the logic)
          |
          v
    schemas  (views)
          |
          v
    HTTP response

Run it with:

    uvicorn api.main:app --reload

Phase 6 invariants:

1. No domain logic lives in this package: no rate mathematics (§10), no
   goal dependency rules (§9), no protection classification (§13), no
   strategy selection (§13 step 7). A router that needs an answer calls
   the layer that owns it.
2. Validation belongs to the domain. Request schemas describe shape and
   type; value rules (pity below hard pity, unique goal priorities,
   low <= expected <= high) are enforced by the frozen domain dataclasses
   and surfaced as HTTP 422 with the domain's own message.
3. The account resource is the aggregate the planner needs: account state
   (§4.1) plus the roadmap sub-resources - goals (§5), banners (§7),
   preferences (§15) and income (§16). `Roadmap` and `PlannerContext` are
   derived per request, never stored: they are views over the record.
4. Planner input that is not account state - current version/phase,
   confidence threshold, income scenario, mechanics - is stored as
   settings (§4.1 defers version/phase to the persistent model) and can be
   overridden per request without writing anything.
5. Nothing is mutated in place. Updates replace records; domain objects
   stay frozen (§4.2, §11 invariant 2).
6. Probability endpoints default to the account's pity/guarantee state but
   accept explicit overrides: the engine stays independent of roadmap
   state (§10), so the API must let it be asked isolated questions.
7. `/probability/weapon` answers 501 until weapon mechanics are verified
   independently (§17, §19). An endpoint that invents mechanics is worse
   than one that says "not yet".
8. Simulation is a job (§11): `/simulation/run` validates the plan
   synchronously - an unexecutable plan fails immediately, not 10,000
   histories later - and samples in the background; results are polled by
   job id.
9. Every Monte Carlo response carries `runs` and `seed` (§2, §11 invariant
   10), so a sampled probability is never presented as an exact value.
10. Stop conditions are a property of a recommendation (§13, Phase 5
    invariant 12): `/planner/stop-conditions` returns the stops of an
    actual `recommend()` call and never composes rules of its own.
11. Persistence sits behind a repository protocol and is in-memory for
    now; a database drops in without touching routers (§18: infrastructure
    is added around a stable domain model).
12. Frontend concerns are Phase 7 (§18) and are absent here.
"""

from api.main import app, create_app
from api.repository import (
    AccountNotFound,
    AccountRecord,
    AccountRepository,
    InMemoryAccountRepository,
    PlannerSettings,
)

__all__ = [
    "AccountNotFound",
    "AccountRecord",
    "AccountRepository",
    "InMemoryAccountRepository",
    "PlannerSettings",
    "app",
    "create_app",
]
