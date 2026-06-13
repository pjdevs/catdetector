import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from catdetector_api.dependencies import get_cat_predictor
from catdetector_api.observability import bind_request_id
from catdetector_api.predictions import (
    CatPrediction,
    CatPredictor,
    InvalidImageError,
    PredictorUnavailableError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/predictions")
async def create_prediction(
    image: Annotated[UploadFile, File()],
    predictor: Annotated[CatPredictor, Depends(get_cat_predictor)],
) -> CatPrediction:
    request_id = uuid.uuid4().hex
    with bind_request_id(request_id):
        image_bytes = await image.read()
        logger.info(
            "Prediction request received",
            extra={
                "event": "prediction.request.received",
                "image_filename": image.filename,
                "content_type": image.content_type,
                "image_size_bytes": len(image_bytes),
            },
        )
        try:
            prediction = predictor.predict(image_bytes, image.filename)
            logger.info(
                "Prediction request completed",
                extra={
                    "event": "prediction.request.completed",
                    "image_filename": image.filename,
                    "label": prediction.label,
                    "vickie_probability": prediction.probabilities.vickie,
                    "oka_probability": prediction.probabilities.oka,
                    "vickie_detected": prediction.detected.vickie,
                    "oka_detected": prediction.detected.oka,
                },
            )
            return prediction
        except InvalidImageError as exc:
            logger.warning(
                "Prediction request rejected",
                extra={
                    "event": "prediction.request.invalid_image",
                    "image_filename": image.filename,
                    "error": str(exc),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc
        except PredictorUnavailableError as exc:
            logger.error(
                "Prediction request unavailable",
                extra={
                    "event": "prediction.request.predictor_unavailable",
                    "image_filename": image.filename,
                    "error": str(exc),
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc
