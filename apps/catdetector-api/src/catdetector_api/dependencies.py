from functools import lru_cache

from catdetector_api.predictions import CatPredictor


def get_cat_predictor() -> CatPredictor:
    return _get_checkpoint_cat_predictor()


@lru_cache(maxsize=1)
def _get_checkpoint_cat_predictor() -> CatPredictor:
    from catdetector_api.inference import CheckpointCatPredictor

    return CheckpointCatPredictor()
