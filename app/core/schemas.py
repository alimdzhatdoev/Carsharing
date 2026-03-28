"""Pydantic schemas for API and shared typing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.payload_examples import BATCH_PREDICT_EXAMPLE, TRIP_FEATURES_EXAMPLE
from app.core.api_constants import API_ROUTE_VERSION, SERVICE_SEMANTIC_VERSION


class TripFeaturesRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": TRIP_FEATURES_EXAMPLE},
    )

    user_age: int = Field(ge=16, le=90)
    driving_experience_years: float = Field(ge=0, le=70)
    account_age_days: int = Field(ge=0)
    payment_delay_count: int = Field(ge=0)
    previous_incidents_count: int = Field(ge=0)
    rating: float = Field(ge=0, le=5)
    tariff_type: str
    trip_duration_min: float = Field(ge=0)
    trip_distance_km: float = Field(ge=0)
    average_speed_kmh: float = Field(ge=0)
    max_speed_kmh: float = Field(ge=0)
    start_hour: int = Field(ge=0, le=23)
    day_of_week: int = Field(ge=0, le=6)
    trip_cost: float = Field(ge=0)
    late_finish_flag: int = Field(ge=0, le=1)
    trip_start_zone: str
    trip_end_zone: str
    car_category: str
    car_age_years: float = Field(ge=0, le=40)
    mileage_km: float = Field(ge=0)
    fuel_or_charge_level_start: float = Field(ge=0, le=100)
    fuel_or_charge_level_end: float = Field(ge=0, le=100)
    weather_condition: str
    traffic_level: str
    city_zone_risk: str
    night_trip_flag: int = Field(ge=0, le=1)
    weekend_flag: int = Field(ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_class: int = Field(ge=0, le=1)
    probability_positive: float = Field(ge=0, le=1)
    threshold_used: float


class BatchPredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": BATCH_PREDICT_EXAMPLE},
    )

    items: list[TripFeaturesRequest] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_route_version: str = Field(default=API_ROUTE_VERSION, description="Версия префикса маршрутов")
    service_semantic_version: str = Field(
        default=SERVICE_SEMANTIC_VERSION,
        description="SemVer развёрнутого сервиса",
    )


class ModelInfoResponse(BaseModel):
    """Сведения о версии API и загруженных артефактах модели."""

    model_config = ConfigDict(extra="allow")

    model_loaded: bool
    api_route_version: str = Field(default=API_ROUTE_VERSION)
    service_semantic_version: str = Field(default=SERVICE_SEMANTIC_VERSION)
    artifact_schema_version: int | None = Field(
        default=None,
        description="Версия схемы training_config.json",
    )
    trained_at_utc: str | None = Field(
        default=None,
        description="Время сохранения метаданных при обучении (UTC, ISO-8601)",
    )
    model_weights_modified_at_utc: str | None = Field(
        default=None,
        description="mtime файла весов на диске (UTC, ISO-8601)",
    )
    input_dim: int | None = None
    classification_threshold: float | None = None
    architecture: dict[str, Any] | None = Field(
        default=None,
        description="Параметры MLP из конфига обучения",
    )
    target_column: str | None = None
    random_seed: int | None = None
    model_checkpoint_basename: str | None = None
    training_config_basename: str | None = None
