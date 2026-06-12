from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from catdetector_api.dependencies import get_cat_predictor
from catdetector_api.predictions import (
    CatPrediction,
    CatPredictor,
    InvalidImageError,
    PredictorUnavailableError,
)

router = APIRouter()


@router.post("/api/predictions")
async def create_prediction(
    image: Annotated[UploadFile, File()],
    predictor: Annotated[CatPredictor, Depends(get_cat_predictor)],
) -> CatPrediction:
    try:
        return predictor.predict(await image.read(), image.filename)
    except InvalidImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc
    except PredictorUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
