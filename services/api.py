"""
===============================================================================
Project       : Resilience Early Warning
Module        : api.py
Description   : FastAPI app for real-time scoring of household risk probabilities.
===============================================================================
Author        : Apoorv Pal
Created On    : 2025-10-06
Last Modified : 2025-10-06
Modified By   : Apoorv Pal
Version       : 1.0.0
Dependencies  : os, pydantic, fastapi, joblib, numpy
Status        : Production-Ready (Sample)
===============================================================================
Change Log:
-------------------------------------------------------------------------------
Date        Author        Description
----------  ------------- -----------------------------------------------------
2025-10-06  Apoorv Pal    Initial FastAPI endpoint for online inference.
===============================================================================
"""
import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

BUNDLE_PATH = os.getenv("MODEL_BUNDLE", "models/model.joblib")
bundle = joblib.load(BUNDLE_PATH)
model = bundle["model"]
feature_cols = bundle["features"]

app = FastAPI(title="Resilience Early Warning API")

class HouseholdFeatures(BaseModel):
    land_area_hectares: float
    livestock_units: float
    floodplain_exposure: int
    secondary_house: int
    head_age: int
    head_gender_female: int
    head_education_years: int
    head_disability: int
    drought_roll3: float = 0
    flood_roll3: float = 0
    illness_roll3: float = 0
    crop_disease_roll3: float = 0
    shock_count: float = 0
    shock_count_roll3: float = 0

@app.get("/health")
def health():
    return {"status": "ok", "features": feature_cols}

@app.post("/score")
def score(payload: HouseholdFeatures):
    x = np.array([[getattr(payload, c, 0) for c in feature_cols]]).astype(float)
    prob = float(model.predict_proba(x)[:,1][0])
    return {"prob_food_insecure_next_month": prob}
