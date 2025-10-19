from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Autism Prediction API",
    description="AI-powered autism screening tool for toddlers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models for request/response
class PredictionInput(BaseModel):
    A10: int = Field(..., ge=0, le=1, description="Gets upset by minor changes in routine (0=No, 1=Yes)")
    Sex: str = Field(..., description="Gender (m/f)")
    Ethnicity: str = Field(..., description="Ethnic background")
    Jaundice: int = Field(..., ge=0, le=1, description="Born with jaundice (0=No, 1=Yes)")
    Family_mem_with_ASD: int = Field(..., ge=0, le=1, description="Family member with ASD (0=No, 1=Yes)")
    Who_completed_the_test: str = Field(..., alias="Who completed the test", description="Who completed the test")
    Age: int = Field(..., ge=1, le=3, description="Age of the child")
    
    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "A10": 1,
                "Sex": "m",
                "Ethnicity": "White-European",
                "Jaundice": 0,
                "Family_mem_with_ASD": 0,
                "Who completed the test": "family member",
                "Age": 3,
            }
        }

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Prediction result (0=No ASD traits, 1=ASD traits detected)")
    confidence: float = Field(..., description="Prediction confidence (0.0-1.0)")
    risk_level: str = Field(..., description="Risk assessment (Low/Moderate/High)")
    model_used: str = Field(..., description="ML model used for prediction")
    timestamp: str = Field(..., description="Prediction timestamp")
    recommendations: list = Field(..., description="Recommended next steps")

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    models_loaded: int
    uptime: str

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str

# Global variables for models
models = {}
feature_info = None
start_time = datetime.now()

# Load models on startup
@app.on_event("startup")
async def load_models():
    """Load all ML models on application startup"""
    global models, feature_info
    
    try:
        model_files = {
            'random_forest': '../saved_models/random_forest_pipeline.joblib',
            'logistic_regression': '../saved_models/logistic_regression_pipeline.joblib',
            'xgboost': '../saved_models/xgboost_pipeline.joblib',
            'naive_bayes': '../saved_models/naive_bayes_pipeline.joblib'
        }
        
        for name, path in model_files.items():
            if os.path.exists(path):
                models[name] = joblib.load(path)
                logger.info(f"Loaded model: {name}")
            else:
                logger.warning(f"Model file not found: {path}")
        
        # Load feature info
        if os.path.exists('saved_models/feature_info.joblib'):
            feature_info = joblib.load('saved_models/feature_info.joblib')
            logger.info("Feature information loaded")
        
        logger.info(f"Successfully loaded {len(models)} models")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise ex

def get_risk_level(confidence: float, prediction: int) -> str:
    """Determine risk level based on prediction and confidence"""
    if prediction == 0:
        return "Low"
    else:
        if confidence >= 0.8:
            return "High"
        elif confidence >= 0.6:
            return "Moderate"
        else:
            return "Low"

def get_recommendations(prediction: int, risk_level: str) -> list:
    """Generate recommendations based on prediction results"""
    if prediction == 1:
        if risk_level == "High":
            return [
                "Consult with a pediatrician immediately",
                "Schedule comprehensive autism evaluation",
                "Consider early intervention programs",
                "Document behavioral observations",
                "Seek support from autism specialists"
            ]
        elif risk_level == "Moderate":
            return [
                "Schedule follow-up with pediatrician",
                "Monitor child's development closely",
                "Consider developmental screening",
                "Maintain regular check-ups"
            ]
        else:  # Low risk but prediction == 1
            return [
                "Monitor child's development closely",
                "Schedule follow-up screening",
                "Maintain regular pediatric check-ups"
            ]
    else:
        return [
            "Continue regular developmental monitoring",
            "Maintain routine pediatric check-ups", 
            "Support healthy development activities",
            "Stay aware of developmental milestones"
        ]

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key (optional authentication)"""
    # For production, implement proper API key validation
    # For now, we'll make it optional
    if credentials:
        # Add your API key validation logic here
        return credentials.credentials
    return "anonymous"

# API Endpoints
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Autism Prediction API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "models": "/models",
            "docs": "/docs"
        },
        "disclaimer": "This tool is for screening purposes only and should not replace professional medical diagnosis."
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    uptime = str(datetime.now() - start_time)
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        uptime=uptime
    )

@app.get("/models")
async def get_available_models():
    """Get list of available models with their information"""
    model_info = {
        "random_forest": {
            "name": "Random Forest",
            "description": "Ensemble method using multiple decision trees",
            "loaded": "random_forest" in models
        },
        "xgboost": {
            "name": "XGBoost",
            "description": "Gradient boosting framework optimized for performance",
            "loaded": "xgboost" in models
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "description": "Statistical model for binary classification",
            "loaded": "logistic_regression" in models
        },
        "naive_bayes": {
            "name": "Naive Bayes",
            "description": "Probabilistic classifier based on Bayes theorem",
            "loaded": "naive_bayes" in models
        }
    }
    
    return {
        "available_models": model_info,
        "total_loaded": len(models),
        "default_model": "xgboost"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_autism(
    input_data: PredictionInput,
    model_name: str = "xgboost",
    api_key: str = Depends(verify_api_key)
):
    """
    Make autism prediction using the specified model
    
    Args:
        input_data: Patient information for prediction
        model_name: Model to use (random_forest, xgboost, logistic_regression, naive_bayes)
        api_key: API key for authentication (optional)
    
    Returns:
        Prediction results with confidence and recommendations
    """
    try:
        # Validate model availability
        if model_name not in models:
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_name}' not available. Available models: {list(models.keys())}"
            )
        
        # Convert input to DataFrame
        input_dict = input_data.dict(by_alias=True)
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        selected_model = models[model_name]
        prediction = int(selected_model.predict(input_df)[0])
        prediction_proba = selected_model.predict_proba(input_df)[0]
        
        # Get confidence for the predicted class
        confidence = float(prediction_proba[prediction])
        
        # Determine risk level
        risk_level = get_risk_level(confidence, prediction)
        
        # Get recommendations
        recommendations = get_recommendations(prediction, risk_level)
        
        # Log prediction (without personal data)
        logger.info(f"Prediction made: model={model_name}, result={prediction}, confidence={confidence:.3f}")
        
        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            risk_level=risk_level,
            model_used=model_name,
            timestamp=datetime.now().isoformat(),
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict")
async def batch_predict(
    input_data: list[PredictionInput],
    model_name: str = "xgboost",
    api_key: str = Depends(verify_api_key)
):
    """Batch prediction endpoint for multiple cases"""
    if len(input_data) > 100:
        raise HTTPException(status_code=400, detail="Batch size limited to 100 predictions")
    
    results = []
    for data in input_data:
        try:
            result = await predict_autism(data, model_name, api_key)
            results.append(result)
        except Exception as e:
            results.append({"error": str(e)})
    
    return {"predictions": results, "total": len(results)}

@app.get("/analytics")
async def get_analytics(api_key: str = Depends(verify_api_key)):
    """Get usage analytics (implement based on your needs)"""
    # This would typically connect to a database for real analytics
    return {
        "total_predictions": "Not implemented - would require database",
        "model_usage": "Not implemented - would require database",
        "note": "Add database integration for real analytics"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Exception",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )