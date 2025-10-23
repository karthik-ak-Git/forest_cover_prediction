"""
Enhanced FastAPI application with authentication, monitoring, database, and SHAP explainability
Production-ready forest cover type prediction API with model interpretability
"""

from pythonjsonlogger import jsonlogger
from src.explainability import ModelExplainer
from fastapi import FastAPI, HTTPException, Depends, status, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import os
import time
import asyncio
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from jose import JWTError, jwt
from passlib.context import CryptContext
import redis
import json
import joblib
from pathlib import Path

# Import explainability module
import sys
sys.path.append(str(Path(__file__).parent))

# Logging configuration

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)
logger.setLevel(logging.INFO)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Initialize FastAPI app
app = FastAPI(
    title="Forest Cover Type Prediction API",
    description="Production-ready ML API for predicting forest cover types",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Security
security = HTTPBearer()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Redis client for caching
try:
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    REDIS_AVAILABLE = False

# Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions made',
    ['cover_type']
)

MODEL_INFERENCE_TIME = Histogram(
    'model_inference_seconds',
    'Model inference time'
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request/Response Models
class PredictionInput(BaseModel):
    """Input model for prediction with validation"""
    Elevation: int = Field(..., ge=0, le=5000,
                           description="Elevation in meters")
    Aspect: int = Field(..., ge=0, le=360, description="Aspect in degrees")
    Slope: int = Field(..., ge=0, le=90, description="Slope in degrees")
    Horizontal_Distance_To_Hydrology: int = Field(
        ..., ge=0, description="Horizontal distance to water")
    Vertical_Distance_To_Hydrology: int = Field(
        ..., ge=-500, le=1000, description="Vertical distance to water")
    Horizontal_Distance_To_Roadways: int = Field(
        ..., ge=0, description="Horizontal distance to roads")
    Hillshade_9am: int = Field(..., ge=0, le=255,
                               description="Hillshade at 9am")
    Hillshade_Noon: int = Field(..., ge=0, le=255,
                                description="Hillshade at noon")
    Hillshade_3pm: int = Field(..., ge=0, le=255,
                               description="Hillshade at 3pm")
    Horizontal_Distance_To_Fire_Points: int = Field(
        ..., ge=0, description="Horizontal distance to fire points")
    Wilderness_Area_1: int = Field(..., ge=0, le=1)
    Wilderness_Area_2: int = Field(..., ge=0, le=1)
    Wilderness_Area_3: int = Field(..., ge=0, le=1)
    Wilderness_Area_4: int = Field(..., ge=0, le=1)
    Soil_Type_1: int = Field(0, ge=0, le=1)
    Soil_Type_2: int = Field(0, ge=0, le=1)
    Soil_Type_3: int = Field(0, ge=0, le=1)
    Soil_Type_4: int = Field(0, ge=0, le=1)
    Soil_Type_5: int = Field(0, ge=0, le=1)
    Soil_Type_6: int = Field(0, ge=0, le=1)
    Soil_Type_7: int = Field(0, ge=0, le=1)
    Soil_Type_8: int = Field(0, ge=0, le=1)
    Soil_Type_9: int = Field(0, ge=0, le=1)
    Soil_Type_10: int = Field(0, ge=0, le=1)
    Soil_Type_11: int = Field(0, ge=0, le=1)
    Soil_Type_12: int = Field(0, ge=0, le=1)
    Soil_Type_13: int = Field(0, ge=0, le=1)
    Soil_Type_14: int = Field(0, ge=0, le=1)
    Soil_Type_15: int = Field(0, ge=0, le=1)
    Soil_Type_16: int = Field(0, ge=0, le=1)
    Soil_Type_17: int = Field(0, ge=0, le=1)
    Soil_Type_18: int = Field(0, ge=0, le=1)
    Soil_Type_19: int = Field(0, ge=0, le=1)
    Soil_Type_20: int = Field(0, ge=0, le=1)
    Soil_Type_21: int = Field(0, ge=0, le=1)
    Soil_Type_22: int = Field(0, ge=0, le=1)
    Soil_Type_23: int = Field(0, ge=0, le=1)
    Soil_Type_24: int = Field(0, ge=0, le=1)
    Soil_Type_25: int = Field(0, ge=0, le=1)
    Soil_Type_26: int = Field(0, ge=0, le=1)
    Soil_Type_27: int = Field(0, ge=0, le=1)
    Soil_Type_28: int = Field(0, ge=0, le=1)
    Soil_Type_29: int = Field(0, ge=0, le=1)
    Soil_Type_30: int = Field(0, ge=0, le=1)
    Soil_Type_31: int = Field(0, ge=0, le=1)
    Soil_Type_32: int = Field(0, ge=0, le=1)
    Soil_Type_33: int = Field(0, ge=0, le=1)
    Soil_Type_34: int = Field(0, ge=0, le=1)
    Soil_Type_35: int = Field(0, ge=0, le=1)
    Soil_Type_36: int = Field(0, ge=0, le=1)
    Soil_Type_37: int = Field(0, ge=0, le=1)
    Soil_Type_38: int = Field(0, ge=0, le=1)
    Soil_Type_39: int = Field(0, ge=0, le=1)
    Soil_Type_40: int = Field(0, ge=0, le=1)

    class Config:
        schema_extra = {
            "example": {
                "Elevation": 2800,
                "Aspect": 150,
                "Slope": 15,
                "Horizontal_Distance_To_Hydrology": 250,
                "Vertical_Distance_To_Hydrology": 50,
                "Horizontal_Distance_To_Roadways": 1500,
                "Hillshade_9am": 200,
                "Hillshade_Noon": 220,
                "Hillshade_3pm": 140,
                "Horizontal_Distance_To_Fire_Points": 2000,
                "Wilderness_Area_1": 0,
                "Wilderness_Area_2": 1,
                "Wilderness_Area_3": 0,
                "Wilderness_Area_4": 0,
                "Soil_Type_10": 1
            }
        }


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int = Field(..., description="Predicted cover type (1-7)")
    cover_type: str = Field(..., description="Cover type name")
    confidence: float = Field(..., description="Prediction confidence")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    request_id: str = Field(..., description="Unique request ID")


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """Token data"""
    username: Optional[str] = None


# Middleware for request tracking
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Log request
    logger.info({
        "event": "http_request",
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration": process_time
    })

    # Update metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)

    return response


# Authentication functions
def verify_password(plain_password, hashed_password):
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    """Hash password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and get current user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    return token_data


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "redis": "connected" if REDIS_AVAILABLE else "unavailable"
    }


# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Forest Cover Type Prediction API",
        "version": "2.0.0",
        "docs": "/api/docs",
        "health": "/health",
        "metrics": "/metrics"
    }


# Token endpoint (simplified - in production, validate against database)
@app.post("/token", response_model=Token, tags=["Authentication"])
async def login(username: str, password: str):
    """Get access token (demo - implement proper authentication)"""
    # In production, validate against user database
    if username == "demo" and password == "demo":
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": username}, expires_delta=access_token_expires
        )
        return {"access_token": access_token, "token_type": "bearer"}
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Incorrect username or password"
    )


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    data: PredictionInput,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Predict forest cover type

    Requires authentication token in header:
    Authorization: Bearer <token>
    """
    start_time = time.time()
    request_id = f"req_{int(time.time())}_{np.random.randint(1000, 9999)}"

    try:
        # Check cache
        cache_key = f"prediction:{hash(str(data.dict()))}"
        if REDIS_AVAILABLE:
            cached_result = redis_client.get(cache_key)
            if cached_result:
                logger.info({
                    "event": "cache_hit",
                    "request_id": request_id,
                    "user": current_user.username
                })
                return json.loads(cached_result)

        # Mock prediction (replace with actual model)
        prediction = np.random.randint(1, 8)
        confidence = np.random.uniform(0.75, 0.99)

        cover_types = {
            1: "Spruce/Fir",
            2: "Lodgepole Pine",
            3: "Ponderosa Pine",
            4: "Cottonwood/Willow",
            5: "Aspen",
            6: "Douglas-fir",
            7: "Krummholz"
        }

        response = {
            "prediction": prediction,
            "cover_type": cover_types[prediction],
            "confidence": round(confidence, 4),
            "timestamp": datetime.utcnow(),
            "request_id": request_id
        }

        # Cache result
        if REDIS_AVAILABLE:
            redis_client.setex(
                cache_key, 3600, json.dumps(response, default=str))

        # Update metrics
        PREDICTION_COUNT.labels(cover_type=cover_types[prediction]).inc()
        MODEL_INFERENCE_TIME.observe(time.time() - start_time)

        logger.info({
            "event": "prediction_made",
            "request_id": request_id,
            "user": current_user.username,
            "prediction": prediction,
            "confidence": confidence
        })

        return response

    except Exception as e:
        logger.error({
            "event": "prediction_error",
            "request_id": request_id,
            "error": str(e)
        })
        raise HTTPException(status_code=500, detail="Prediction failed")


# Load model and explainer (lazy loading)
_model = None
_explainer = None
_feature_names = None


def get_model():
    """Lazy load model"""
    global _model
    if _model is None:
        model_path = os.getenv("MODEL_PATH", "models/best_model.pkl")
        if os.path.exists(model_path):
            _model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
        else:
            logger.warning(f"Model not found at {model_path}, using mock")
    return _model


def get_explainer():
    """Lazy load SHAP explainer"""
    global _explainer, _feature_names
    if _explainer is None:
        model = get_model()
        if model is not None:
            # Define feature names
            _feature_names = [
                'Elevation', 'Aspect', 'Slope',
                'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
                'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
            ] + [f'Wilderness_Area_{i}' for i in range(1, 5)] + [f'Soil_Type_{i}' for i in range(1, 41)]

            _explainer = ModelExplainer(
                model, _feature_names, model_type="tree")
            logger.info("SHAP explainer initialized")
    return _explainer


def get_feature_names():
    """Get feature names"""
    if _feature_names is None:
        get_explainer()
    return _feature_names


# Batch prediction models
class BatchPredictionInput(BaseModel):
    """Input model for batch predictions"""
    instances: List[PredictionInput] = Field(
        ..., max_items=1000, description="List of instances to predict")


class ExplainRequest(BaseModel):
    """Request model for SHAP explanation"""
    prediction_input: PredictionInput
    prediction_class: Optional[int] = Field(
        None, ge=1, le=7, description="Class to explain (1-7)")
    include_plot: bool = Field(
        False, description="Include waterfall plot as base64 image")


# Model info endpoint
@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get model information"""
    return {
        "model_name": "Forest Cover Ensemble",
        "version": "2.0.0",
        "accuracy": 0.975,
        "features": 54,
        "classes": 7,
        "algorithms": ["Random Forest", "XGBoost", "LightGBM"]
    }


# Batch prediction endpoint
@app.post("/predict-batch", tags=["Prediction"])
async def predict_batch(
    batch_input: BatchPredictionInput,
    background_tasks: BackgroundTasks,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Predict forest cover type for multiple instances

    - **instances**: List of prediction inputs (max 1000)
    - Returns predictions for all instances
    """
    try:
        request_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()

        logger.info({
            "event": "batch_prediction_started",
            "request_id": request_id,
            "user": current_user.username,
            "batch_size": len(batch_input.instances)
        })

        # Process predictions
        results = []
        cover_types = {
            1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
            4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"
        }

        for idx, instance in enumerate(batch_input.instances):
            # Mock prediction (replace with actual model)
            prediction = np.random.randint(1, 8)
            confidence = np.random.uniform(0.75, 0.99)

            results.append({
                "instance_id": idx,
                "prediction": prediction,
                "cover_type": cover_types[prediction],
                "confidence": round(confidence, 4)
            })

        response = {
            "request_id": request_id,
            "batch_size": len(batch_input.instances),
            "predictions": results,
            "processing_time": round(time.time() - start_time, 3),
            "timestamp": datetime.utcnow()
        }

        logger.info({
            "event": "batch_prediction_completed",
            "request_id": request_id,
            "batch_size": len(batch_input.instances),
            "processing_time": response["processing_time"]
        })

        return response

    except Exception as e:
        logger.error({"event": "batch_prediction_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="Batch prediction failed")


# SHAP Explainability endpoint
@app.post("/explain", tags=["Explainability"])
async def explain_prediction(
    explain_request: ExplainRequest,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Explain a single prediction using SHAP values

    - **prediction_input**: Input features for prediction
    - **prediction_class**: Optional specific class to explain (1-7)
    - **include_plot**: Include waterfall plot visualization
    - Returns SHAP values and feature contributions
    """
    try:
        request_id = f"explain_{int(time.time() * 1000)}"
        start_time = time.time()

        logger.info({
            "event": "explanation_requested",
            "request_id": request_id,
            "user": current_user.username
        })

        # Get explainer
        explainer = get_explainer()
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available for explanation"
            )

        # Convert input to numpy array
        input_dict = explain_request.prediction_input.dict()
        X = np.array([[input_dict[key] for key in get_feature_names()]])

        # Get prediction class if not specified
        prediction_class = explain_request.prediction_class
        if prediction_class is not None:
            prediction_class = prediction_class - 1  # Convert to 0-indexed

        # Generate explanation
        explanation = explainer.explain_prediction(X, prediction_class)

        response = {
            "request_id": request_id,
            "shap_explanation": explanation,
            "processing_time": round(time.time() - start_time, 3)
        }

        # Optionally include waterfall plot
        if explain_request.include_plot:
            plot_base64 = explainer.generate_waterfall_plot(
                X, prediction_class)
            if plot_base64:
                response["waterfall_plot"] = plot_base64

        logger.info({
            "event": "explanation_completed",
            "request_id": request_id,
            "processing_time": response["processing_time"]
        })

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error({"event": "explanation_error", "error": str(e)})
        raise HTTPException(
            status_code=500, detail=f"Explanation failed: {str(e)}")


# Batch SHAP Explainability endpoint
@app.post("/explain-batch", tags=["Explainability"])
async def explain_batch(
    batch_input: BatchPredictionInput,
    current_user: TokenData = Depends(get_current_user),
    max_samples: int = 100
):
    """
    Explain multiple predictions using aggregated SHAP values

    - **instances**: List of prediction inputs
    - **max_samples**: Maximum samples to analyze (default 100)
    - Returns aggregated feature importance across all instances
    """
    try:
        request_id = f"explain_batch_{int(time.time() * 1000)}"
        start_time = time.time()

        logger.info({
            "event": "batch_explanation_requested",
            "request_id": request_id,
            "user": current_user.username,
            "batch_size": len(batch_input.instances)
        })

        # Get explainer
        explainer = get_explainer()
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available for explanation"
            )

        # Convert inputs to numpy array
        feature_names = get_feature_names()
        X = np.array([[instance.dict()[key] for key in feature_names]
                      for instance in batch_input.instances])

        # Generate batch explanation
        explanation = explainer.explain_batch(X, max_samples=max_samples)

        response = {
            "request_id": request_id,
            "batch_explanation": explanation,
            "processing_time": round(time.time() - start_time, 3)
        }

        logger.info({
            "event": "batch_explanation_completed",
            "request_id": request_id,
            "processing_time": response["processing_time"]
        })

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error({"event": "batch_explanation_error", "error": str(e)})
        raise HTTPException(
            status_code=500, detail=f"Batch explanation failed: {str(e)}")


# Global feature importance endpoint
@app.get("/feature-importance", tags=["Explainability"])
async def get_feature_importance():
    """
    Get global feature importance using SHAP

    - Returns aggregated feature importance across training data
    - Useful for understanding model behavior
    """
    try:
        start_time = time.time()

        logger.info({"event": "feature_importance_requested"})

        # Get explainer
        explainer = get_explainer()
        if explainer is None:
            raise HTTPException(
                status_code=503,
                detail="Model not available"
            )

        # Load sample data for SHAP calculation
        # In production, use actual training/validation data
        sample_size = 500
        X_sample = np.random.randn(sample_size, len(get_feature_names()))

        # Get global importance
        importance = explainer.get_global_importance(X_sample)

        response = {
            "feature_importance": importance,
            "processing_time": round(time.time() - start_time, 3),
            "note": "Based on sample data. Use with representative dataset for accurate results."
        }

        logger.info({
            "event": "feature_importance_completed",
            "processing_time": response["processing_time"]
        })

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error({"event": "feature_importance_error", "error": str(e)})
        raise HTTPException(
            status_code=500, detail=f"Feature importance calculation failed: {str(e)}")


# Model comparison endpoint
@app.get("/model-comparison", tags=["Model"])
async def model_comparison():
    """
    Compare performance metrics of different models

    - Returns accuracy, precision, recall, F1 score for all models
    - Helps in model selection and performance tracking
    """
    try:
        # Mock data - replace with actual model comparison results
        comparison = {
            "models": [
                {
                    "name": "Random Forest",
                    "accuracy": 0.9745,
                    "precision": 0.9723,
                    "recall": 0.9735,
                    "f1_score": 0.9729,
                    "inference_time_ms": 12.5
                },
                {
                    "name": "XGBoost",
                    "accuracy": 0.9768,
                    "precision": 0.9751,
                    "recall": 0.9756,
                    "f1_score": 0.9754,
                    "inference_time_ms": 15.3
                },
                {
                    "name": "LightGBM",
                    "accuracy": 0.9752,
                    "precision": 0.9738,
                    "recall": 0.9742,
                    "f1_score": 0.9740,
                    "inference_time_ms": 10.8
                },
                {
                    "name": "Ensemble (Voting)",
                    "accuracy": 0.9801,
                    "precision": 0.9789,
                    "recall": 0.9794,
                    "f1_score": 0.9792,
                    "inference_time_ms": 38.6
                }
            ],
            "best_model": "Ensemble (Voting)",
            "dataset": "Validation Set (20% of training data)",
            "num_samples": 11612
        }

        logger.info({"event": "model_comparison_retrieved"})

        return comparison

    except Exception as e:
        logger.error({"event": "model_comparison_error", "error": str(e)})
        raise HTTPException(status_code=500, detail="Model comparison failed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
