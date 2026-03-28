"""Feature column definitions for trip risk classification."""

IDENTIFIER_COLUMNS = ("user_id", "trip_id", "car_id")

TARGET_COLUMN = "target_class"

NUMERIC_COLUMNS = (
    "user_age",
    "driving_experience_years",
    "account_age_days",
    "payment_delay_count",
    "previous_incidents_count",
    "rating",
    "trip_duration_min",
    "trip_distance_km",
    "average_speed_kmh",
    "max_speed_kmh",
    "start_hour",
    "day_of_week",
    "trip_cost",
    "late_finish_flag",
    "car_age_years",
    "mileage_km",
    "fuel_or_charge_level_start",
    "fuel_or_charge_level_end",
    "night_trip_flag",
    "weekend_flag",
)

CATEGORICAL_COLUMNS = (
    "tariff_type",
    "trip_start_zone",
    "trip_end_zone",
    "car_category",
    "weather_condition",
    "traffic_level",
    "city_zone_risk",
)


def feature_columns() -> tuple[str, ...]:
    return NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
