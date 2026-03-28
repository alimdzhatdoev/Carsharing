"""Зависимости FastAPI (инференс-сервис из состояния приложения)."""

from __future__ import annotations

from fastapi import HTTPException, Request

from app.services.inference_service import InferenceService


def get_inference_service(request: Request) -> InferenceService:
    svc = getattr(request.app.state, "inference_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Inference service not initialized")
    return svc
