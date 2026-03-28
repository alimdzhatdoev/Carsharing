"""Маршруты версии v1 API."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_inference_service
from app.core.payload_examples import BATCH_PREDICT_EXAMPLE, TRIP_FEATURES_EXAMPLE
from app.core.logger import get_logger
from app.core.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TripFeaturesRequest,
)
from app.services.inference_service import InferenceService

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Проверка готовности сервиса")
def health(svc: Annotated[InferenceService, Depends(get_inference_service)]) -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=svc.ready)


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Метаданные загруженной модели и версии API",
)
def model_info(svc: Annotated[InferenceService, Depends(get_inference_service)]) -> ModelInfoResponse:
    return ModelInfoResponse.model_validate(svc.get_model_info())


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Предсказание для одной поездки",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "high_risk_night_trip": {
                            "summary": "Ночная поездка, высокий контекст риска",
                            "value": TRIP_FEATURES_EXAMPLE,
                        }
                    }
                }
            }
        }
    },
)
def predict(
    body: TripFeaturesRequest,
    svc: Annotated[InferenceService, Depends(get_inference_service)],
) -> PredictionResponse:
    if not svc.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model and place files under artifacts/.",
        )
    try:
        out = svc.predict_one(body.model_dump())
    except ValueError as e:
        logger.warning("predict validation_error %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict failed")
        raise HTTPException(status_code=400, detail=str(e)) from e
    return PredictionResponse(**out)


@router.post(
    "/predict_batch",
    response_model=BatchPredictionResponse,
    summary="Пакетное предсказание",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "two_trips": {
                            "summary": "Две поездки",
                            "value": BATCH_PREDICT_EXAMPLE,
                        }
                    }
                }
            }
        }
    },
)
def predict_batch(
    body: BatchPredictionRequest,
    svc: Annotated[InferenceService, Depends(get_inference_service)],
) -> BatchPredictionResponse:
    if not svc.ready:
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not found. Train the model and place files under artifacts/.",
        )
    rows = [item.model_dump() for item in body.items]
    try:
        preds = svc.predict_batch(rows)
    except ValueError as e:
        logger.warning("predict_batch validation_error %s", e)
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("predict_batch failed")
        raise HTTPException(status_code=400, detail=str(e)) from e
    return BatchPredictionResponse(predictions=[PredictionResponse(**p) for p in preds])
