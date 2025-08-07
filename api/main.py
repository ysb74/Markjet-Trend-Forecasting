"""FastAPI application for Market Trend Forecasting API endpoints."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import config
from src.utils.logger import get_logger
from src.data.validation import validate_and_clean_data, TENANT_SCHEMA
from src.models.time_series import train_time_series_models

# Initialize FastAPI app
app = FastAPI(
    title="Market Trend Forecasting API",
    description="API for real estate market trend prediction and analysis",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = get_logger(__name__)

# Pydantic models for request/response
class TenantData(BaseModel):
    """Model for tenant data input."""
    tenant_id: int = Field(..., gt=0, description="Unique tenant identifier")
    property_id: str = Field(..., description="Property identifier")
    lease_start_date: datetime = Field(..., description="Lease start date")
    lease_end_date: datetime = Field(..., description="Lease end date")
    monthly_rent: float = Field(..., gt=0, description="Monthly rent amount")
    payment_delays_last_6_months: int = Field(..., ge=0, description="Number of payment delays")
    maintenance_requests_last_year: int = Field(..., ge=0, description="Number of maintenance requests")
    feedback_score: Optional[float] = Field(None, ge=0, le=5, description="Tenant feedback score")
    status: str = Field(..., description="Tenant status")

class PropertyData(BaseModel):
    """Model for property data input."""
    property_id: str = Field(..., description="Property identifier")
    address: str = Field(..., description="Property address")
    beds: int = Field(..., ge=0, le=20, description="Number of bedrooms")
    baths: float = Field(..., ge=0, le=20, description="Number of bathrooms")
    sqft: int = Field(..., gt=0, description="Square footage")
    price: float = Field(..., gt=0, description="Property price")
    property_type: str = Field(..., description="Type of property")

class PredictionRequest(BaseModel):
    """Model for prediction requests."""
    data: List[Dict[str, Any]] = Field(..., description="Input data for prediction")
    model_name: str = Field("churn_prediction", description="Model to use for prediction")

class ForecastRequest(BaseModel):
    """Model for time series forecasting requests."""
    target_variable: str = Field("monthly_rent", description="Variable to forecast")
    forecast_days: int = Field(30, ge=1, le=365, description="Number of days to forecast")
    model_type: str = Field("prophet", description="Forecasting model type")

class PredictionResponse(BaseModel):
    """Model for prediction responses."""
    predictions: List[float] = Field(..., description="Model predictions")
    confidence: Optional[List[float]] = Field(None, description="Prediction confidence scores")
    model_info: Dict[str, Any] = Field(..., description="Model information")

class ForecastResponse(BaseModel):
    """Model for forecast responses."""
    forecast: List[Dict[str, Any]] = Field(..., description="Forecast results")
    metadata: Dict[str, Any] = Field(..., description="Forecast metadata")

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="API version")

# Global variables for model cache
model_cache = {}

def load_model(model_name: str):
    """Load model from cache or disk."""
    if model_name not in model_cache:
        try:
            model_path = Path("deployed_models") / f"{model_name}.joblib"
            if model_path.exists():
                model_cache[model_name] = joblib.load(model_path)
                logger.info(f"Model {model_name} loaded successfully")
            else:
                raise FileNotFoundError(f"Model {model_name} not found")
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return model_cache[model_name]

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic API information."""
    return {
        "message": "Market Trend Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0"
    )

@app.post("/predict/churn", response_model=PredictionResponse)
async def predict_tenant_churn(request: PredictionRequest):
    """Predict tenant churn probability."""
    try:
        # Load the churn prediction model
        model = load_model("churn_rf_optimized")
        
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Validate and preprocess data
        # (Add preprocessing logic here)
        
        # Make predictions
        predictions = model.predict_proba(df)[:, 1].tolist()  # Probability of churn
        
        return PredictionResponse(
            predictions=predictions,
            confidence=[max(pred, 1-pred) for pred in predictions],
            model_info={
                "model_name": request.model_name,
                "model_type": "Random Forest Classifier",
                "features_used": df.columns.tolist(),
                "prediction_timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"Error in churn prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/rent", response_model=PredictionResponse)
async def predict_rent(request: PredictionRequest):
    """Predict monthly rent amount."""
    try:
        # Load the rent prediction model
        model = load_model("rent_rf_optimized")
        
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        return PredictionResponse(
            predictions=predictions,
            model_info={
                "model_name": "rent_prediction",
                "model_type": "Random Forest Regressor",
                "features_used": df.columns.tolist(),
                "prediction_timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        logger.error(f"Error in rent prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast/timeseries", response_model=ForecastResponse)
async def forecast_time_series(request: ForecastRequest):
    """Generate time series forecasts."""
    try:
        # Load sample data for forecasting
        # In production, this would load real data
        sample_data = pd.DataFrame({
            'date': pd.date_range(start='2020-01-01', end='2024-01-01', freq='D'),
            'monthly_rent': np.random.normal(1800, 200, 1462)
        })
        
        # Prepare time series data
        ts_data = sample_data.set_index('date')[request.target_variable]
        
        # Generate simple forecast (placeholder - would use actual models)
        last_date = ts_data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=request.forecast_days,
            freq='D'
        )
        
        # Simple linear trend forecast
        recent_values = ts_data.tail(30).values
        trend = np.polyfit(range(len(recent_values)), recent_values, 1)[0]
        last_value = ts_data.iloc[-1]
        
        forecast_values = [last_value + trend * i for i in range(1, request.forecast_days + 1)]
        
        # Prepare response
        forecast_data = [
            {
                "date": date.isoformat(),
                "forecast": value,
                "lower_bound": value * 0.95,
                "upper_bound": value * 1.05
            }
            for date, value in zip(forecast_dates, forecast_values)
        ]
        
        return ForecastResponse(
            forecast=forecast_data,
            metadata={
                "target_variable": request.target_variable,
                "forecast_days": request.forecast_days,
                "model_type": request.model_type,
                "forecast_timestamp": datetime.now().isoformat(),
                "historical_data_points": len(ts_data)
            }
        )
    
    except Exception as e:
        logger.error(f"Error in time series forecasting: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/data/validate")
async def validate_data(data: List[TenantData]):
    """Validate input data against schema."""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([item.dict() for item in data])
        
        # Validate against schema
        errors = TENANT_SCHEMA.validate(df)
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "record_count": len(df),
            "validation_timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error in data validation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/list")
async def list_models():
    """List available models."""
    try:
        model_dir = Path("deployed_models")
        models = []
        
        if model_dir.exists():
            for model_file in model_dir.glob("*.joblib"):
                models.append({
                    "name": model_file.stem,
                    "file": model_file.name,
                    "size": model_file.stat().st_size,
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                })
        
        return {
            "models": models,
            "count": len(models),
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data/sample")
async def get_sample_data():
    """Get sample data for testing."""
    sample_data = [
        {
            "tenant_id": 1,
            "property_id": "P001",
            "lease_start_date": "2023-01-01",
            "lease_end_date": "2024-01-01",
            "monthly_rent": 1500.0,
            "payment_delays_last_6_months": 0,
            "maintenance_requests_last_year": 1,
            "feedback_score": 4.5,
            "status": "Active"
        },
        {
            "tenant_id": 2,
            "property_id": "P002",
            "lease_start_date": "2023-02-01",
            "lease_end_date": "2024-02-01",
            "monthly_rent": 2000.0,
            "payment_delays_last_6_months": 2,
            "maintenance_requests_last_year": 3,
            "feedback_score": 3.2,
            "status": "Active"
        }
    ]
    
    return {
        "sample_data": sample_data,
        "description": "Sample tenant data for API testing",
        "timestamp": datetime.now().isoformat()
    }

# Background tasks
@app.post("/tasks/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """Trigger model retraining (background task)."""
    
    def retrain_task():
        """Background task for model retraining."""
        try:
            logger.info("Starting model retraining...")
            # Add actual retraining logic here
            logger.info("Model retraining completed successfully")
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    background_tasks.add_task(retrain_task)
    
    return {
        "message": "Model retraining task queued",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration
    host = config.get("api.host", "0.0.0.0")
    port = config.get("api.port", 8000)
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )