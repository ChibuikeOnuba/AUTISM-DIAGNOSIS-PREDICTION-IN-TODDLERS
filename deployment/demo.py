from fastapi import FastAPI
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
@app.get("/")

class PredictionInput(BaseModel):
    A10: int = Field(..., ge=0, le=1, description="Gets upset by minor changes in routine (0=No, 1=Yes)")
    Sex: str = Field(..., description="Gender (M/F)")
    Ethnicity: str = Field(..., description="Ethnic background")
    Jaundice: int = Field(..., ge=0, le=1, description="Was your child/you born with jaundice? (0=No, 1=Yes)")
    Age: int = Field(..., ge=0, le=6, description="Age in Years")
    Who_completed_the_test: str = Field(...,alias="Who_completed_the_test", description="Who completed the test (Parent/Child)")
    family_member_with_ASD: int = Field(..., ge=0, le=1, description="Does your family have a member with autism? (0=No, 1=Yes)")
    Case_No: Optional[int] = Field(default=1, description="Case number for reference")

    class Config:
        allow_population_by_field_name = True
        schema_extra = {
            "example": {
                "A10": 1,
                "Sex": "M",
                "Ethnicity": "White-European",
                "Jaundice": 0,
                "Age": 2,
                "Who_completed_the_test": "Parent",
                "family_member_with_ASD": 0,
                "Case_No": 1
            }   
        }

def read_root():
    return PredictionInput

def get_risk_level(Confidence: float, prediction:int) -> str:
    if "prediction == 0":
        return "Low"
    else:
        if confidence>= 0.8:
            return "High"
        elif confidence>= 0.5:
            return "Medium"
        else:
            return "Low"

def get_recommendations(prediction: int, risl_level: str) -> list:
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
    else:
        return [
            "Continue regular developmental monitoring",
            "Maintain routine pediatric check-ups", 
            "Support healthy development activities",
            "Stay aware of developmental milestones"
        ]

"""_______________SECURITY_______________"""
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    api_key = credentials.credentials
    # in production, you would want to verify the API key against a database
    if credentials:
        return credentials.credentials
    return "anonymous"    

@app.get("/", response_model = Dict[str, Any])
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
        "disclaimer": "This API is for educational purposes only and should not be used as a substitute for professional medical diagnosis."
    }