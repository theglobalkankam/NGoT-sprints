
import time 
from datetime import datetime 
from fastapi import FastAPI, HTTPException, status 
from fastapi.middleware.cors import CORSMiddleware 
from app.schemas import ETARequest, ETAResponse, HealthResponse 
from app.predictor import ETAPredictor 
  
  
# ── App Setup ────────────────────────────────────────────────────── 
  
# Create the FastAPI app with metadata for the auto-generated docs 
app = FastAPI( 
    title="ETA Predictor API", 
    description=""" 
    Production-grade logistics ETA prediction service. 
  
    ## How to use 
    1. POST your delivery request to `/predict` 
    2. Receive the estimated delivery time in minutes 
    3. Use `/health` to check the service status 
    """, 
    version="1.0.0", 
    docs_url="/docs",       # Interactive documentation at /docs 
    redoc_url="/redoc",     # Alternative docs at /redoc 
) 
  
# Allow requests from any origin (important for frontend apps) 
app.add_middleware( 
    CORSMiddleware, 
    allow_origins=['*'], 
    allow_credentials=True, 
    allow_methods=['GET', 'POST'], 
    allow_headers=['*'], 
) 
  
# Create a single predictor instance (loaded once at startup) 
# Module-level singleton: one predictor shared across all requests
predictor = ETAPredictor()
START_TIME = time.time() 
  
  
# ── Startup Event ────────────────────────────────────────────────── 
  
@app.on_event('startup') 
async def load_model_on_startup(): 
    """Load the ML model when the server starts.""" 
    success = predictor.load() 
    if not success: 
        print('WARNING: API starting without a loaded model.') 
        print('POST /predict will return 503 until a model is loaded.') 
  
  
# ── Endpoints ────────────────────────────────────────────────────── 
  
@app.get('/', include_in_schema=False) 
async def root(): 
    return {"message": "ETA Predictor API", "docs": "/docs"} 
  
  
@app.get('/health', response_model=HealthResponse, tags=['System']) 
async def health_check(): 
    """ 
    Check if the API and ML model are healthy. 
    Returns 'healthy' if the model is loaded, 'degraded' if not. 
    """ 
    return HealthResponse( 
        status='healthy' if predictor.is_loaded else 'degraded', 
        model_loaded=predictor.is_loaded, 
        api_version='1.0.0', 
    ) 
  
  
@app.post('/predict', response_model=ETAResponse, tags=['Predictions']) 
async def predict_eta(request: ETARequest): 
    """ 
    Predict the estimated delivery time. 
  
    Provide origin/destination GPS coordinates, cargo details, and departure time. 
    Returns estimated delivery time in minutes plus a confidence interval. 
    """ 
    # Check model is available 
    if not predictor.is_loaded: 
        raise HTTPException( 
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail='Model not loaded. Check server logs.', 
        ) 
  
    try: 
        # Convert the validated request to a feature vector 
        features = request.to_feature_vector() 
  
        # Run prediction 
        eta_min, ci_low, ci_high = predictor.predict(features) 
  
        # Format ETA as human-readable string 
        hours   = int(eta_min // 60)
        minutes = int(eta_min % 60) 
        if hours > 0: 
            eta_human = f'{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}' 
        else: 
            eta_human = f'{minutes} minute{'s' if minutes != 1 else ''}' 
  
        return ETAResponse( 
            eta_minutes=eta_min, 
            eta_human_readable=eta_human, 
            distance_km=request.distance_km, 
            confidence_low=ci_low, 
            confidence_high=ci_high, 
            is_rush_hour=request.is_rush_hour, 
            model_version=predictor.version, 
        ) 
  
    except Exception as e: 
        raise HTTPException( 
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f'Prediction failed: {str(e)}', 
        )