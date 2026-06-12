from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from catdetector_api.dependencies import get_cat_predictor
from catdetector_api.predictions import CatPrediction, CatPredictor

router = APIRouter()


@router.post("/api/predictions")
async def create_prediction(
    image: Annotated[UploadFile, File()],
    predictor: Annotated[CatPredictor, Depends(get_cat_predictor)],
) -> CatPrediction:
    return predictor.predict(await image.read(), image.filename)
