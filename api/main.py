"""The application (Design Document §3, §18 Phase 6).

`create_app` wires storage and routers into a FastAPI application. Each
application owns its repository and job store through `app.state`, so tests
build isolated applications instead of sharing process-wide state, and a
database-backed repository is injected here rather than imported by routers
(§18).

Error translation is the other thing that belongs at this level. The domain
signals invalid values by raising `ValueError` from frozen dataclasses and
planner functions - an unknown income scenario, duplicate goal priorities,
no banner at the current version, a multi-copy goal in the spend table. Each
becomes 422 carrying the domain's own message, because the domain's message
already explains the rule and the section it comes from. Re-wording them
here would be a second, drifting copy of the domain's vocabulary.
"""

import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from api.demo_data import create_demo_account
from api.jobs import SimulationJobStore
from api.repository import AccountNotFound, AccountRepository, InMemoryAccountRepository
from api.routers import accounts, planner, probability, roadmap, simulation

# The Phase 7 dashboard (§18) is a static page served from the same origin,
# so it is a client of this API rather than a second implementation of it:
# it can only show what an endpoint returns. Serving it here also means no
# CORS configuration and no build step.
DASHBOARD = Path(__file__).resolve().parent.parent / "frontend" / "index.html"

DESCRIPTION = """
Probability-based pull planning: what to spend wishes on, how far to pursue
it, and what that spending puts at risk later.

The API exposes the domain, probability, planner, simulation and optimizer
layers; it contains none of their logic. Probabilities from the simulator
are Monte Carlo estimates and always carry their `runs` and `seed`.
""".strip()


async def _value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Domain validation failures become 422 with the domain's message."""
    return JSONResponse(
        status_code=422,  # Unprocessable Entity
        content={"detail": str(exc)},
    )


async def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    """A repository lookup miss becomes 404 wherever it is raised."""
    return JSONResponse(
        status_code=404, content={"detail": str(exc)}
    )


async def _planner_timing_middleware(request: Request, call_next):
    """Measure server-side planner request time for local performance diagnostics."""
    if "/planner/" not in request.url.path or request.url.path.endswith("/planner/timing"):
        return await call_next(request)

    started = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - started
    response.headers["X-Planner-Process-Time"] = f"{elapsed:.6f}"
    print(f"Planner API {request.url.path}: {elapsed:.2f}s")
    return response


def create_app(
    repository: AccountRepository | None = None,
    jobs: SimulationJobStore | None = None,
) -> FastAPI:
    """Build an application with its own storage.

    Args:
        repository: account storage; defaults to a fresh in-memory
            repository (§18 - a database implementation drops in here).
        jobs: simulation job store; defaults to a fresh in-process store.
    """
    app = FastAPI(
        title="Genshin Pull Strategy Planner",
        description=DESCRIPTION,
        version="0.6.0",
    )
    app.middleware("http")(_planner_timing_middleware)

    if repository is None:
        # Each app instance gets an isolated repository; test fixtures and
        # callers that need seeded data must inject it explicitly.
        repository = InMemoryAccountRepository()
    app.state.repository = repository
    app.state.jobs = SimulationJobStore() if jobs is None else jobs

    app.add_exception_handler(AccountNotFound, _not_found_handler)
    app.add_exception_handler(ValueError, _value_error_handler)

    app.include_router(accounts.router)
    app.include_router(roadmap.router)
    app.include_router(probability.router)
    app.include_router(planner.router)
    app.include_router(simulation.router)

    @app.get("/health", tags=["service"], summary="Liveness check")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    if DASHBOARD.is_file():

        @app.get(
            "/",
            tags=["service"],
            summary="The dashboard",
            response_class=FileResponse,
            include_in_schema=False,
        )
        def dashboard() -> FileResponse:
            return FileResponse(DASHBOARD)

    return app


app = create_app()
# The module-level application is the local dashboard entry point. Seed its
# isolated in-memory repository with the development account; create_app()
# itself remains unseeded so tests and injected callers stay isolated.
app.state.repository.create(create_demo_account())
