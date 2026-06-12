from fastapi import HTTPException, status

from catdetector_api.predictions import CatPrediction, CatPredictor


class UnavailableCatPredictor:
    def predict(self, image: bytes, filename: str | None = None) -> CatPrediction:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cat predictor is not configured.",
        )


def get_cat_predictor() -> CatPredictor:
    return UnavailableCatPredictor()
