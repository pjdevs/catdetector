from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from wireup.integration.fastapi import setup

from catdetector_api.dependencies import create_container
from catdetector_api.observability import (
    api_logging_dict_config,
    configure_api_logging,
)
from catdetector_api.routes import router
from catdetector_api.settings import ApiSettings


def create_app(
    *,
    settings: ApiSettings | None = None,
    web_dist_dir: Path | None = None,
) -> FastAPI:
    api_settings = settings or ApiSettings()
    app = FastAPI(title="Cat Detector API")
    app.include_router(router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    mount_frontend(app, web_dist_dir or api_settings.web_dist_dir)
    setup(create_container(api_settings), app)
    return app


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


def default_web_dist_dir() -> Path:
    return ApiSettings().web_dist_dir


def api_host() -> str:
    return ApiSettings().api_host


def api_port() -> int:
    return ApiSettings().api_port


def main() -> None:
    from granian import Granian
    from granian.constants import Interfaces
    from granian.log import LogLevels

    settings = ApiSettings()
    configure_api_logging(settings)
    log_level = LogLevels(settings.log_level.lower())
    Granian(
        "catdetector_api.main:app",
        address=settings.api_host,
        port=settings.api_port,
        interface=Interfaces.ASGI,
        log_enabled=True,
        log_level=log_level,
        log_dictconfig=api_logging_dict_config(settings),
        log_access=settings.log_access,
    ).serve()
