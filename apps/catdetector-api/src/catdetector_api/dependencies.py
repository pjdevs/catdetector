from wireup import create_async_container, injectable, instance
from wireup.ioc.container.async_container import AsyncContainer

from catdetector_api.inference import CheckpointCatPredictor
from catdetector_api.predictions import CatPredictor
from catdetector_api.settings import ApiSettings


@injectable
def create_cat_predictor(settings: ApiSettings) -> CatPredictor:
    return CheckpointCatPredictor(
        checkpoint=settings.checkpoint,
        checkpoints_dir=settings.checkpoints_dir,
        vickie_threshold=settings.vickie_threshold,
        oka_threshold=settings.oka_threshold,
        device=settings.device,
    )


def create_container(settings: ApiSettings | None = None) -> AsyncContainer:
    api_settings = settings or ApiSettings()
    return create_async_container(
        injectables=[
            instance(api_settings, as_type=ApiSettings),
            create_cat_predictor,
        ]
    )
