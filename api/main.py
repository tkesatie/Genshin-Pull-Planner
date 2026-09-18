"""The application (Design Document §3, §18 Phase 6).

create_app wires storage and routers into a FastAPI application. Tests can
inject an in-memory repository, while the normal application uses SQLite so
saved accounts survive process restarts.
"""

import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from api.demo_data import create_demo_account
from api.jobs import SimulationJobStore
from api.repository import AccountNotFound, AccountRepository, InMemoryAccountRepository
from api.sqlite_repository import SQLiteAccountRepository
from api.routers import accounts, planner, probability, roadmap, simulation

DASHBOARD = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
DEFAULT_DATABASE_PATH = Path(os.environ.get("GENSHIN_PLANNER_DB", "data/planner.sqlite3"))

DESCRIPTION = """
Probability-based pull planning: what to spend wishes on, how far to pursue
it, and what that spending puts at risk later.

The API exposes the domain, probability, planner, simulation and optimizer
layers; it contains none of their logic. Probabilities from the simulator
are Monte Carlo estimates and always carry their runs and seed.
""".strip()


async def _value_error_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


async def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


async def _planner_timing_middleware(request: Request, call_next):
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
    database_path: str | Path | None = None,
) -> FastAPI:
    """Build an application with injectable storage.

    Tests continue to receive isolated in-memory storage by default. The
    module-level production app uses SQLite persistence.
    """
    app = FastAPI(
        title="Genshin Pull Strategy Planner",
        description=DESCRIPTION,
        version="0.6.0",
    )
    app.middleware("http")(_planner_timing_middleware)

    if repository is None:
        repository = (
            SQLiteAccountRepository(database_path or DEFAULT_DATABASE_PATH)
            if database_path is not None
            else InMemoryAccountRepository()
        )
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


app = create_app(database_path=DEFAULT_DATABASE_PATH)

# Keep the local development account available on first startup, but don't
# overwrite it on subsequent restarts.
if not app.state.repository.list():
    app.state.repository.create(create_demo_account())
