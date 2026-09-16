"""HTTP routers (Design Document §18 Phase 6).

One router per layer the API exposes, matching the design document's
endpoint list:

    accounts      the stored aggregate (§4.1)
    roadmap       goals, banners, preferences, income (§5, §7, §15, §16)
    probability   the analytical engine (§10)
    planner       goal states, safe spend, spend table, recommendation (§9,
                  §13, §14)
    simulation    Monte Carlo jobs (§11)

Routers translate HTTP into calls on those layers and translate results
back. They contain no planning logic of their own (api invariant 1).
"""

from api.routers import accounts, planner, probability, roadmap, simulation

__all__ = ["accounts", "planner", "probability", "roadmap", "simulation"]
