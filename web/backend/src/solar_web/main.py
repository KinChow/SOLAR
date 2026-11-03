"""FastAPI application factory for the SOLAR Web demo."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from solar_web.api.routes import router
from solar_web.services.run_manager import RunManager


def create_app(
    *,
    runtime_root: Optional[Path] = None,
    solar_root: Optional[Path] = None,
    static_dir: Optional[Path] = None,
) -> FastAPI:
    module_path = Path(__file__).resolve()
    repo_root = Path(solar_root) if solar_root is not None else module_path.parents[4]
    run_root = Path(runtime_root) if runtime_root is not None else repo_root / "web" / ".runtime" / "runs"
    app = FastAPI(title="SOLAR Web API", version="0.1.0")
    app.state.run_manager = RunManager(run_root, repo_root)
    app.state.static_dir = static_dir
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    if static_dir is not None and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")

    return app


app = create_app(
    static_dir=Path(__file__).resolve().parents[4] / "web" / "frontend" / "dist"
)
