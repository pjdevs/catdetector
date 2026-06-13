import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from catdetector_api.observability import (
    api_log_level,
    api_logging_dict_config,
    configure_api_logging,
)
from catdetector_api.routes import router


def create_app(web_dist_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Cat Detector API")
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    mount_frontend(app, web_dist_dir or default_web_dist_dir())
    return app


def default_web_dist_dir() -> Path:
    if web_dist_dir := os.environ.get("CATDETECTOR_WEB_DIST"):
        return Path(web_dist_dir)

    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "apps" / "catdetector-web" / "dist"


def mount_frontend(app: FastAPI, web_dist_dir: Path) -> None:
    index_path = web_dist_dir / "index.html"
    assets_dir = web_dist_dir / "assets"
    if not index_path.exists():
        return

    if assets_dir.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_dir),
            name="frontend-assets",
        )

    @app.get("/", include_in_schema=False)
    def frontend_index() -> FileResponse:
        return FileResponse(index_path)


app = create_app()


def api_host() -> str:
    return os.environ.get("CATDETECTOR_API_HOST", "0.0.0.0")


def api_port() -> int:
    return int(os.environ.get("CATDETECTOR_API_PORT", "8000"))


def main() -> None:
    from granian import Granian
    from granian.constants import Interfaces
    from granian.log import LogLevels

    configure_api_logging()
    log_level = LogLevels(api_log_level().lower())
    Granian(
        "catdetector_api.main:app",
        address=api_host(),
        port=api_port(),
        interface=Interfaces.ASGI,
        log_enabled=True,
        log_level=log_level,
        log_dictconfig=api_logging_dict_config(),
        log_access=True,
    ).serve()
