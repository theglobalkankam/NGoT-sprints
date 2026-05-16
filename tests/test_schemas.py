
# Tests for our Pydantic schemas 
# Run with: poetry run pytest tests/test_schemas.py -v 
  
import pytest 
from pydantic import ValidationError 
from app.schemas import ETARequest 
"""Tests for correctly valid requests.""" 
  
def test_basic_valid_request(self): 
        """A request with all valid fields should not raise an error.""" 
        req = ETARequest( 
            origin_lat=5.6037,   origin_lon=-0.1870,  # Accra 
            dest_lat=6.6885,     dest_lon=-1.6244,    # Kumasi 
            cargo_weight_kg=500, hour_of_day=10, day_of_week=1, 
        ) 
        # No exception means it's valid 
        assert req.cargo_weight_kg == 500 
        assert req.vehicle_type == 'truck'  # Default value 
        assert req.num_stops == 1            # Default value 
  
def test_computed_distance_accra_to_kumasi(self): 
        """Accra to Kumasi straight-line distance should be approximately 250 km.""" 
        req = ETARequest( 
            origin_lat=5.6037, origin_lon=-0.1870, 
            dest_lat=6.6885,   dest_lon=-1.6244, 
            cargo_weight_kg=200, hour_of_day=9, day_of_week=0, 
        ) 
        # Actual road distance is ~253 km, Haversine (straight-line) is slightly less 
        assert 240 < req.distance_km < 270, f'Expected ~250km, got {req.distance_km}' 
  
def test_rush_hour_detection_morning(self): 
        """Hour 8 (8am) should be detected as rush hour.""" 
        req = ETARequest( 
            origin_lat=5.6, origin_lon=-0.2, 
            dest_lat=6.7,   dest_lon=-1.6, 
            cargo_weight_kg=100, hour_of_day=8, day_of_week=1, 
        ) 
        assert req.is_rush_hour is True 
  
def test_rush_hour_detection_midday(self): 
        """Hour 14 (2pm) should NOT be rush hour.""" 
        req = ETARequest( 
            origin_lat=5.6, origin_lon=-0.2, 
            dest_lat=6.7,   dest_lon=-1.6, 
            cargo_weight_kg=100, hour_of_day=14, day_of_week=1, 
        ) 
        assert req.is_rush_hour is False 
  
def test_feature_vector_length(self): 
        """Feature vector must have exactly 10 elements.""" 
        req = ETARequest( 
            origin_lat=5.6, origin_lon=-0.2, 
            dest_lat=6.7,   dest_lon=-1.6, 
            cargo_weight_kg=100, hour_of_day=10, day_of_week=2, 
        ) 
        features = req.to_feature_vector() 
        assert len(features) == 10 
        # All elements must be numbers 
        assert all(isinstance(f, float) for f in features) 
  
def test_feature_vector_rush_hour_flag(self): 
        """Feature vector index 2 should be 1.0 during rush hour.""" 
        rush_req = ETARequest( 
  origin_lat=5.6, origin_lon=-0.2, 
            dest_lat=6.7,   dest_lon=-1.6, 
            cargo_weight_kg=100, hour_of_day=8, day_of_week=1, 
        ) 
        off_req = ETARequest( 
            origin_lat=5.6, origin_lon=-0.2, 
            dest_lat=6.7,   dest_lon=-1.6, 
            cargo_weight_kg=100, hour_of_day=14, day_of_week=1, 
        ) 
        assert rush_req.to_feature_vector()[2] == 1.0 
        assert off_req.to_feature_vector()[2]  == 0.0 
  
  
# ── Invalid Request Tests ────────────────────────────────────────── 
class TestETARequestInvalid: 
       """Tests that invalid requests are properly rejected.""" 
  
def test_latitude_out_of_range(self): 
        """Latitude 999 is impossible — must be rejected.""" 
        with pytest.raises(ValidationError) as exc_info: 
            ETARequest( 
                origin_lat=999,   # INVALID 
                origin_lon=-0.2, dest_lat=6.7, dest_lon=-1.6, 
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1, 
            ) 
        # Check the error mentions the right field 
        assert 'origin_lat' in str(exc_info.value) 
  
def test_negative_cargo_weight(self): 
        """Negative weight is impossible — must be rejected.""" 
        with pytest.raises(ValidationError): 
            ETARequest( 
                origin_lat=5.6, origin_lon=-0.2, 
                dest_lat=6.7,   dest_lon=-1.6, 
                cargo_weight_kg=-100,   # INVALID 
                hour_of_day=10, day_of_week=1, 
            ) 
  
def test_hour_out_of_range(self): 
        """Hour 25 does not exist — must be rejected.""" 
        with pytest.raises(ValidationError): 
            ETARequest( 
                origin_lat=5.6, origin_lon=-0.2, 
                dest_lat=6.7,   dest_lon=-1.6, 
                cargo_weight_kg=100, 
                hour_of_day=25,   # INVALID — max is 23 
                day_of_week=1, 
            ) 
  
def test_motorcycle_overloaded(self): 
        """Motorcycle with 500kg cargo should be rejected.""" 
        with pytest.raises(ValidationError) as exc_info: 
            ETARequest( 
                origin_lat=5.6, origin_lon=-0.2, 
                dest_lat=6.7,   dest_lon=-1.6, 
                cargo_weight_kg=500,     # Too heavy for motorcycle 
                vehicle_type='motorcycle', 
                hour_of_day=10, day_of_week=1, 
            )
            assert 'motorcycle' in str(exc_info.value).lower() 
  
def test_same_origin_and_destination(self): 
        """Origin same as destination should be rejected.""" 
        with pytest.raises(ValidationError): 
            ETARequest( 
                origin_lat=5.6037, origin_lon=-0.1870, 
                dest_lat=5.6037,   dest_lon=-0.1870,   # SAME 
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1, 
            ) 
  
def test_missing_required_field(self): 
        """Missing a required field (no default) should be rejected.""" 
        with pytest.raises(ValidationError): 
            ETARequest( 
                # origin_lat is missing 
                origin_lon=-0.2, dest_lat=6.7, dest_lon=-1.6, 
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1, 
            ) 
  
def test_extra_fields_forbidden(self): 
        """Extra unknown fields should be rejected (extra='forbid' in config).""" 
        with pytest.raises(ValidationError): 
            ETARequest( 
                origin_lat=5.6, origin_lon=-0.2, 
                dest_lat=6.7,   dest_lon=-1.6, 
                cargo_weight_kg=100, hour_of_day=10, day_of_week=1, 
                unknown_field='this should fail',   # INVALID — extra field 
                ) 