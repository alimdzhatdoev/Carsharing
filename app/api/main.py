"""FastAPI inference service: версионирование v1, middleware, обработчики ошибок."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_exception_handlers
from app.api.middleware import RequestLoggingMiddleware
from app.api.v1.routes import router as v1_router
from app.core.api_constants import API_ROUTE_VERSION, SERVICE_SEMANTIC_VERSION
from app.core.logger import get_logger, setup_logging
from app.services.inference_service import default_service

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.inference_service = default_service()
    logger.info("Inference service initialized (API %s)", API_ROUTE_VERSION)
    yield


app = FastAPI(
    title="Carsharing trip risk API",
    version=SERVICE_SEMANTIC_VERSION,
    lifespan=lifespan,
    openapi_tags=[
        {"name": "v1", "description": "Версия 1 — стабильный контракт URL и схем"},
    ],
)

app.add_middleware(RequestLoggingMiddleware)
register_exception_handlers(app)


@app.get("/", summary="Корень сервиса", tags=["meta"])
def service_root() -> dict:
    return {
        "service": "carsharing-trip-risk-api",
        "service_semantic_version": SERVICE_SEMANTIC_VERSION,
        "api": {
            "current_route_version": API_ROUTE_VERSION,
            "base_path": f"/{API_ROUTE_VERSION}",
            "endpoints": {
                "health": f"/{API_ROUTE_VERSION}/health",
                "model_info": f"/{API_ROUTE_VERSION}/model-info",
                "predict": f"/{API_ROUTE_VERSION}/predict",
                "predict_batch": f"/{API_ROUTE_VERSION}/predict_batch",
            },
        },
        "documentation": {"openapi": "/docs", "redoc": "/redoc"},
    }


app.include_router(v1_router, prefix=f"/{API_ROUTE_VERSION}", tags=["v1"])
