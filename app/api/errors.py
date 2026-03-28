"""Единый формат ошибок и регистрация обработчиков исключений."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException, RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logger import get_logger

logger = get_logger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def _json_error(request: Request, status_code: int, payload: dict) -> JSONResponse:
    resp = JSONResponse(status_code=status_code, content=payload)
    resp.headers["X-Request-ID"] = _request_id(request)
    return resp


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RequestValidationError)
    async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        rid = _request_id(request)
        logger.warning("validation_error request_id=%s errors=%s", rid, exc.errors())
        return _json_error(
            request,
            422,
            {
                "error": {
                    "code": "VALIDATION_ERROR",
                    "message": "Request body or parameters failed validation",
                    "request_id": rid,
                    "details": jsonable_encoder(exc.errors()),
                }
            },
        )

    @app.exception_handler(HTTPException)
    async def http_handler(request: Request, exc: HTTPException) -> JSONResponse:
        rid = _request_id(request)
        detail: Any = exc.detail
        if isinstance(detail, dict):
            message = str(detail.get("message", detail))
        else:
            message = str(detail)
        code = "HTTP_ERROR"
        if exc.status_code == 503:
            code = "SERVICE_UNAVAILABLE"
        elif exc.status_code == 404:
            code = "NOT_FOUND"
        return _json_error(
            request,
            exc.status_code,
            {
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": rid,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        rid = _request_id(request)
        logger.exception("unhandled_error request_id=%s", rid)
        return _json_error(
            request,
            500,
            {
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "request_id": rid,
                }
            },
        )
