import math
from pathlib import Path

import numpy as np
import pandas as pd

LAT_MIN, LAT_MAX = 4.7, 11.2
LON_MIN, LON_MAX = -3.3, 1.2


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the straight-line distance between two GPS coordinates.
    The Haversine formula accounts for the curvature of the Earth.
    Returns distance in kilometres.
    """
    R = 6371  # Earth's radius in km
    lat1, lon1 = math.radians(lat1), math.radians(lon1)
    lat2, lon2 = math.radians(lat2), math.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def generate_logistics_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic logistics ETA dataset for Ghana.
    Coordinates are within the Ghana bounding box.

    Features (inputs to the model):
      - origin_lat, origin_lon: GPS of pickup location
      - dest_lat, dest_lon:     GPS of delivery location
      - cargo_weight_kg:        weight of the shipment
      - hour_of_day:            departure hour (0-23)
      - day_of_week:            0=Monday to 6=Sunday
      - num_stops:              number of intermediate stops
      - traffic_index:          1.0=normal, 2.0=double travel time

    Target (what we want to predict):
      - eta_minutes:            actual delivery time in minutes
    """
    rng = np.random.default_rng(seed)

    # Generate random coordinates
    origin_lat = rng.uniform(LAT_MIN, LAT_MAX, n_samples)
    origin_lon = rng.uniform(LON_MIN, LON_MAX, n_samples)
    dest_lat = rng.uniform(LAT_MIN, LAT_MAX, n_samples)
    dest_lon = rng.uniform(LON_MIN, LON_MAX, n_samples)

    # Generate feature values with realistic distributions
    cargo_weight_kg = rng.exponential(scale=600, size=n_samples).clip(10, 20000)
    hour_of_day = rng.integers(0, 24, n_samples)
    day_of_week = rng.integers(0, 7, n_samples)
    num_stops = rng.integers(1, 8, n_samples)
    # Traffic index: most trips are normal (1.0), some are heavy (2.0-3.0)
    traffic_index = 1.0 + rng.exponential(scale=0.3, size=n_samples).clip(0, 2.5)

    # Calculate straight-line distance for each row
    distances = [
        haversine_distance(origin_lat[i], origin_lon[i], dest_lat[i], dest_lon[i])
        for i in range(n_samples)
    ]
    distances = np.array(distances)

    # Calculate ETA with realistic formula:
    # Trucks travel ~60 km/h on average in Ghana
    base_speed_kmh = 60.0
    # Distance/speed = hours, * 60 = minutes
    travel_time = (distances / base_speed_kmh) * 60

    # Traffic slows travel down
    traffic_penalty = travel_time * (traffic_index - 1.0)

    # Each stop adds 15-40 minutes
    stop_time = num_stops * rng.uniform(15, 40, n_samples)

    # Rush hour adds 20-60% more time
    rush_hours = list(range(7, 10)) + list(range(17, 20))
    is_rush = np.isin(hour_of_day, rush_hours).astype(float)
    rush_penalty = travel_time * is_rush * rng.uniform(0.2, 0.6, n_samples)

    # Heavier cargo = slightly longer loading/unloading
    weight_penalty = (cargo_weight_kg / 10000) * rng.uniform(5, 20, n_samples)

    # Random noise (weather, unexpected delays, driver behaviour)
    noise = rng.normal(0, 12, n_samples)

    # Total ETA
    eta_minutes = (
        travel_time
        + traffic_penalty
        + stop_time
        + rush_penalty
        + weight_penalty
        + noise
    ).clip(5, 2000)

    return pd.DataFrame(
        {
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "dest_lat": dest_lat,
            "dest_lon": dest_lon,
            "cargo_weight_kg": cargo_weight_kg,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "num_stops": num_stops,
            "traffic_index": traffic_index,
            "distance_km": distances,
            "eta_minutes": eta_minutes,
        }
    )


if __name__ == "__main__":
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    print("Generating dataset...")
    df = generate_logistics_dataset(n_samples=6000)

    output_path = "data/raw/logistics_eta.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} samples to {output_path}")
    print()
    print("=== DATASET SUMMARY ===")
    print(df.describe().round(2))
    print()
    print("Sample rows:")
    print(df.head(3).to_string())