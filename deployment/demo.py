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


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(
    title="Autism Prediction API",
    description = "AI-powered autism screening tool for toddlers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

security = HTTPBearer(auto_error=False)

    
"""Load models and feature information at startup"""
models = {}
feature_info = None
start_time = datetime.now()

@app.on_event("startup")
async def load_models():
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
                logger.info(f"{name} model loaded successfully")
            else:
                logger.warning(f"Model file not found: {path}")
        logger.info(f"{len(models)} Models loaded successfully")

    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise e


"""Create prediction input model"""

class PredictionInput(BaseModel):
    A10: int = Field(..., ge=0, le=1, description="Gets upset by minor changes in routine (0=No, 1=Yes)")
    Sex: str = Field(..., description="Gender (M/F)")
    Ethnicity: str = Field(..., description="Ethnic background")
    Jaundice: int = Field(..., ge=0, le=1, description="Was your child/you born with jaundice? (0=No, 1=Yes)")
    Family_mem_with_ASD: int = Field(..., ge=0, le=1, description="Does your family have a member with autism? (0=No, 1=Yes)")
    Who_completed_the_test: str = Field(...,alias="Who completed the test", description="Who completed the test (Parent/Child)")
    Age: int = Field(..., ge=0, le=2, description="Age in Years")
    Case_No: Optional[int] = Field(default=1, description="Case number for reference")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "A10": 1,
                "Sex": "m",
                "Ethnicity": "White-European",
                "Jaundice": 0,
                "family_member_with_ASD": 0,
                "Who_completed_the_test": "family member",
                "Age": 2,
                "Case_No": 1
            }   
        }

class HealthResponse(BaseModel):
    status:str
    timestamp:str
    models_loaded:int
    uptime:str

class PredictionResponse(BaseModel):
    prediction:int
    confidence:float
    risk_level:str
    recommendations:list
    model_used:str
    timestamp:str

class ErrorResponse(BaseModel):
    error: str
    message: str
    timestamp: str


def get_risk_level(confidence: float, prediction:int) -> str:
    if prediction == 0:
        return "Low"
    else:
        if confidence>= 0.8:
            return "High"
        elif confidence>= 0.5:
            return "Moderate"
        else:
            return "Low"

def get_recommendations(prediction: int, risk_level: str) -> list:
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

"""_______________SECURITY_______________"""
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    # in production, you would want to verify the API key against a database
    if credentials:
        return credentials.credentials
    return "anonymous"    

@app.get("/", response_model=Dict[str, Any], tags=["Root"])

async def root():
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

@app.get("/health", response_model = HealthResponse)
async def health_check():
    uptime = str(datetime.now() - start_time)
    return HealthResponse(
        status="Healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(models),
        uptime=uptime
    )

@app.get("/models")
async def get_available_models():

    model_info = {
        "random_forest": {
            "name":"Random Forest",
            "description":"Ensemble method using multiple decision trees",
            "loaded": "random_forest" in models
        },
        "logistic_regression": {
            "name":"Logistic Regression",
            "description":"Statistical model for binary classification",
            "loaded": "logistic_regression" in models
        },
        "xgboost": {
            "name":"XGBoost",
            "description":"Gradient boosting framework optimized for performance",
            "loaded": "xgboost" in models
        },
        "naive_bayes": {
            "name":"Naive Bayes",
            "description":"Probabilistic classifier based on Bayes theorem",
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
    """Make autism prediction using the specified model"""
    try:
        if model_name not in models:
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} not found in available models: {list(models.keys())}"
            )
            #convert input data into dataframe
        input_dict = input_data.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])

        model_columns = ['A10', 'Sex', 'Ethnicity', 'Jaundice', 
                 'Family_mem_with_ASD', 'Who completed the test', 'Age']
                 # Select and reorder columns to match training data
        input_df = input_df[model_columns]

        #make prediction
        selected_model = models[model_name]
        prediction = int(selected_model.predict(input_df)[0])
        prediction_proba = selected_model.predict_proba(input_df)[0]

        confidence = float(prediction_proba[prediction])
        risk_level = get_risk_level(confidence, prediction)
        recommendations = get_recommendations(prediction, risk_level)

        return PredictionResponse(
            prediction=prediction,
            confidence=round(confidence, 2),
            risk_level=risk_level,
            recommendations=recommendations,
            model_used = model_name,
            timestamp = datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

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