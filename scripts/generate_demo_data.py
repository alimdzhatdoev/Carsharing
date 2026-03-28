"""
Generate synthetic carsharing trip dataset with domain-informed target and noise.

Usage:
  python scripts/generate_demo_data.py [--rows 8000] [--output data/raw/trips_demo.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import uuid

import numpy as np
import pandas as pd

from app.utils.common import get_project_root


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def generate_dataset(n_rows: int, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    tariff_type = rng.choice(["economy", "standard", "premium"], size=n_rows, p=[0.5, 0.35, 0.15])
    zones = ["north", "south", "east", "west", "center"]
    trip_start_zone = rng.choice(zones, size=n_rows)
    trip_end_zone = rng.choice(zones, size=n_rows)
    car_category = rng.choice(["compact", "sedan", "suv", "van"], size=n_rows, p=[0.35, 0.35, 0.25, 0.05])
    weather_condition = rng.choice(
        ["clear", "rain", "snow", "fog"], size=n_rows, p=[0.65, 0.18, 0.1, 0.07]
    )
    traffic_level = rng.choice(["low", "medium", "high"], size=n_rows, p=[0.35, 0.45, 0.2])
    city_zone_risk = rng.choice(["low", "medium", "high"], size=n_rows, p=[0.55, 0.3, 0.15])

    user_age = rng.integers(18, 71, size=n_rows)
    driving_experience_years = np.clip(
        rng.normal(user_age - 18, 4, size=n_rows).astype(float), 0, 55
    )
    account_age_days = rng.integers(1, 2000, size=n_rows)
    payment_delay_count = rng.poisson(0.35, size=n_rows)
    previous_incidents_count = rng.poisson(0.15, size=n_rows)
    rating = np.clip(rng.normal(4.2, 0.7, size=n_rows), 1.0, 5.0)

    weekend_flag = rng.binomial(1, 0.42, size=n_rows)
    start_hour = rng.integers(0, 24, size=n_rows)
    night_trip_flag = ((start_hour >= 22) | (start_hour <= 5)).astype(np.int64)

    trip_duration_min = rng.lognormal(mean=3.2, sigma=0.65, size=n_rows)
    trip_distance_km = np.clip(trip_duration_min * rng.uniform(0.2, 1.4, size=n_rows), 0.5, 300)
    average_speed_kmh = np.clip(trip_distance_km / np.maximum(trip_duration_min / 60.0, 0.05), 5, 130)
    max_speed_kmh = np.clip(average_speed_kmh + rng.exponential(12, size=n_rows), 10, 200)
    late_finish_flag = rng.binomial(1, 0.08 + 0.04 * night_trip_flag, size=n_rows)
    trip_cost = np.clip(
        trip_duration_min * 3.5 + trip_distance_km * 12.0 + rng.normal(0, 15, size=n_rows),
        0,
        5000,
    )
    day_of_week = rng.integers(0, 7, size=n_rows)

    car_age_years = rng.uniform(0.5, 12, size=n_rows)
    mileage_km = rng.uniform(10_000, 250_000, size=n_rows)
    fuel_or_charge_level_start = rng.uniform(15, 100, size=n_rows)
    fuel_or_charge_level_end = np.clip(
        fuel_or_charge_level_start - rng.uniform(5, 60, size=n_rows), 0, 100
    )

    high_speed = (max_speed_kmh - 70) / 25.0
    bad_weather = np.isin(weather_condition, ["rain", "snow", "fog"]).astype(float)
    high_traffic = (traffic_level == "high").astype(float)
    high_risk_zone = (city_zone_risk == "high").astype(float) + 0.55 * (city_zone_risk == "medium").astype(
        float
    )
    young_user = (user_age < 23).astype(float)
    fresh_account = np.clip(1.0 - account_age_days / 120.0, 0, 1)
    low_rating = (rating < 3.6).astype(float)

    logit = (
        -2.2
        + 0.95 * night_trip_flag
        + 0.55 * high_speed
        + 0.75 * high_risk_zone
        + 0.55 * late_finish_flag
        + 0.12 * previous_incidents_count
        + 0.07 * payment_delay_count
        + 0.45 * low_rating
        + 0.55 * fresh_account
        + 0.35 * bad_weather
        + 0.25 * high_traffic
        + 0.2 * young_user
        + 0.15 * (trip_cost > np.percentile(trip_cost, 85)).astype(float)
        + rng.normal(0, 0.65, size=n_rows)
    )

    p_problem = _sigmoid(logit)
    target_class = rng.binomial(1, p_problem)

    df = pd.DataFrame(
        {
            "user_id": [str(uuid.uuid4()) for _ in range(n_rows)],
            "user_age": user_age,
            "driving_experience_years": driving_experience_years,
            "account_age_days": account_age_days,
            "payment_delay_count": payment_delay_count,
            "previous_incidents_count": previous_incidents_count,
            "rating": rating,
            "tariff_type": tariff_type,
            "trip_id": [str(uuid.uuid4()) for _ in range(n_rows)],
            "trip_duration_min": trip_duration_min,
            "trip_distance_km": trip_distance_km,
            "average_speed_kmh": average_speed_kmh,
            "max_speed_kmh": max_speed_kmh,
            "start_hour": start_hour,
            "day_of_week": day_of_week,
            "trip_cost": trip_cost,
            "late_finish_flag": late_finish_flag,
            "trip_start_zone": trip_start_zone,
            "trip_end_zone": trip_end_zone,
            "car_id": [str(uuid.uuid4()) for _ in range(n_rows)],
            "car_category": car_category,
            "car_age_years": car_age_years,
            "mileage_km": mileage_km,
            "fuel_or_charge_level_start": fuel_or_charge_level_start,
            "fuel_or_charge_level_end": fuel_or_charge_level_end,
            "weather_condition": weather_condition,
            "traffic_level": traffic_level,
            "city_zone_risk": city_zone_risk,
            "night_trip_flag": night_trip_flag,
            "weekend_flag": weekend_flag,
            "target_class": target_class,
        }
    )

    # Introduce missing values (MCAR-ish) in a subset of numeric columns
    miss_cols = [
        "average_speed_kmh",
        "fuel_or_charge_level_end",
        "driving_experience_years",
        "rating",
    ]
    for col in miss_cols:
        mask = rng.random(n_rows) < 0.04
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8000)
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/trips_demo.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = get_project_root()
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.rows, random_seed=args.seed)
    df.to_csv(out, index=False)
    pos_rate = df["target_class"].mean()
    print(f"Wrote {len(df)} rows to {out} (positive rate={pos_rate:.3f})")


if __name__ == "__main__":
    main()
