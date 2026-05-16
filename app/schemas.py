# app/schemas.py
# All Pydantic models for input validation and output structuring

import math
from datetime import datetime
from typing import ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class ETARequest(BaseModel):
    """
    Validated input for ETA prediction.

    All fields are validated automatically when this object is created.
    Invalid data raises a ValidationError with clear error messages.
    """

    # model_config controls Pydantic behaviour
    model_config = ConfigDict(
        str_strip_whitespace=True,  # Strip spaces from strings automatically
        extra="forbid",  # Reject any extra fields not defined here
    )

    # Field(...) means REQUIRED — no default value
    # ge = greater than or equal to (>=)
    # le = less than or equal to (<=)
    origin_lat: float = Field(
        ..., ge=-90, le=90, description="Origin latitude (-90 to 90)"
    )
    origin_lon: float = Field(
        ..., ge=-180, le=180, description="Origin longitude (-180 to 180)"
    )
    dest_lat: float = Field(
        ..., ge=-90, le=90, description="Destination latitude (-90 to 90)"
    )
    dest_lon: float = Field(
        ..., ge=-180, le=180, description="Destination longitude (-180 to 180)"
    )

    # gt = greater than (>) — excludes zero, so weight must be strictly positive
    cargo_weight_kg: float = Field(
        ..., gt=0, le=20000, description="Cargo weight in kg (max 20 tonnes)"
    )
    hour_of_day: int = Field(
        ..., ge=0, le=23, description="Departure hour (0=midnight, 23=11pm)"
    )
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Departure day (0=Monday, 6=Sunday)"
    )
    # Fields with defaults become optional in the request body
    num_stops: int = Field(
        1, ge=1, le=20, description="Number of delivery stops (default: 1)"
    )
    traffic_index: float = Field(
        1.0, ge=0.5, le=5.0, description="Traffic multiplier (1.0=normal, 2.0=double)"
    )
    # Literal restricts to exactly these string values — anything else fails validation
    vehicle_type: Literal["truck", "van", "motorcycle"] = Field(
        "truck", description="Vehicle type — affects weight limits and speed"
    )

    # Field Validators

    # mode='before' runs BEFORE Pydantic's type coercion — v could still be a string
    # @classmethod is required by Pydantic V2 for all field validators
    @field_validator("origin_lat", "origin_lon", "dest_lat", "dest_lon", mode="before")
    @classmethod
    def round_coordinates_to_6dp(cls, v) -> float:
        """GPS coordinates beyond 6 decimal places are meaningless (~0.1mm precision)."""
        return round(float(v), 6)

    # ── Cross-Field Validators (model_validator) ────────────────────

    # mode='after' runs after ALL fields are validated and coerced — safe to compare them
    # Must return self; raising ValueError triggers a Pydantic ValidationError
    @model_validator(mode="after")
    def origin_and_destination_must_differ(self) -> "ETARequest":
        """A delivery cannot start and end at the same spot."""
        same_lat = abs(self.origin_lat - self.dest_lat) < 0.001
        same_lon = abs(self.origin_lon - self.dest_lon) < 0.001
        if same_lat and same_lon:
            raise ValueError("Origin and destination must be different locations")
        return self

    @model_validator(mode="after")
    def motorcycle_weight_limit(self) -> "ETARequest":
        """Motorcycles have a 100kg weight limit."""
        if self.vehicle_type == "motorcycle" and self.cargo_weight_kg > 100:
            raise ValueError(
                f"Motorcycle cannot carry {self.cargo_weight_kg:.1f}kg (limit: 100kg). "
                "Use van or truck for heavier cargo."
            )
        return self

    # Computed Fields (auto-calculated from other fields)

    # @computed_field tells Pydantic to include this property in .model_dump() / JSON ( learned)
    @computed_field
    @property
    def distance_km(self) -> float:
        """Straight-line distance from origin to destination (Haversine formula)."""
        R = 6371  # Earth's radius in km
        lat1 = math.radians(self.origin_lat)
        lat2 = math.radians(self.dest_lat)
        dlat = lat2 - lat1
        dlon = math.radians(self.dest_lon - self.origin_lon)
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        return round(R * 2 * math.asin(math.sqrt(a)), 3)

    @computed_field
    @property
    def is_rush_hour(self) -> bool:
        """True if departure is during morning or evening rush hour."""
        morning_rush = list(range(7, 10))  # 7:00 - 9:59
        evening_rush = list(range(17, 20))  # 17:00 - 19:59
        return self.hour_of_day in morning_rush + evening_rush

    def to_feature_vector(self) -> list[float]:
        """
        Flatten this request into a list of numbers the ML model can process.
        The ORDER of features here must match the order used during training.
        """
        return [
            self.distance_km,  # feature 0
            self.cargo_weight_kg,  # feature 1
            float(self.is_rush_hour),  # feature 2 (1.0 or 0.0)
            float(self.hour_of_day),  # feature 3
            float(self.day_of_week),  # feature 4
            float(self.num_stops),  # feature 5
            self.traffic_index,  # feature 6
            1.0 if self.vehicle_type == "truck" else 0.0,  # feature 7
            1.0 if self.vehicle_type == "van" else 0.0,  # feature 8
            1.0 if self.vehicle_type == "motorcycle" else 0.0,  # feature 9
        ]

    # Class-level constant listing feature names in same order as to_feature_vector()
    FEATURE_NAMES: ClassVar[list[str]] = [
        "distance_km",
        "cargo_weight_kg",
        "is_rush_hour",
        "hour_of_day",
        "day_of_week",
        "num_stops",
        "traffic_index",
        "vehicle_truck",
        "vehicle_van",
        "vehicle_motorcycle",
    ]


#  Response Schema (what the API returns)
class ETAResponse(BaseModel):
    """Structured, validated output from the ETA prediction endpoint."""

    eta_minutes: float  # e.g. 185.4
    eta_human_readable: str  # e.g. '3 hours 5 minutes'
    distance_km: float
    confidence_low: float  # 80% confidence interval lower bound
    confidence_high: float  # 80% confidence interval upper bound
    is_rush_hour: bool
    model_version: str
    # default_factory=datetime.utcnow auto-sets the timestamp if not provided
    prediction_timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health Check Response
class HealthResponse(BaseModel):
    """Health status returned by the /health endpoint."""

    # Literal restricts status to these exact values — no free-form strings
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    api_version: str