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