"""
===============================================================================
Project       : Resilience Early Warning
Module        : generate_synthetic_data.py
Description   : Generates synthetic baseline and monthly shock survey data to
                mimic a resilience measurement context (baseline + monthly
                follow-ups), without referencing any specific real project.
===============================================================================
Author        : Apoorv Pal
Created On    : 2025-10-06
Last Modified : 2025-10-06
Modified By   : Apoorv Pal
Version       : 1.0.0
Dependencies  : os, pandas, numpy, datetime
Status        : Demo
===============================================================================
Change Log:
-------------------------------------------------------------------------------
Date        Author        Description
----------  ------------- -----------------------------------------------------
2025-10-06  Apoorv Pal    Initial implementation for synthetic data generation.
===============================================================================
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime
RAW_DIR = os.getenv("RAW_DIR", "data/raw")
np.random.seed(42)
os.makedirs(RAW_DIR, exist_ok=True)

N_HH = 1500
START = datetime(2016, 1, 1)
MONTHS = 18

baseline = pd.DataFrame({
    "household_id": np.arange(1, N_HH+1),
    "land_area_hectares": np.round(np.random.gamma(2.0, 1.2, N_HH), 2),
    "livestock_units": np.random.poisson(2, N_HH),
    "floodplain_exposure": np.random.binomial(1, 0.35, N_HH),
    "secondary_house": np.random.binomial(1, 0.15, N_HH),
    "head_age": np.random.randint(18, 80, N_HH),
    "head_gender_female": np.random.binomial(1, 0.42, N_HH),
    "head_education_years": np.random.randint(0, 16, N_HH),
    "head_disability": np.random.binomial(1, 0.06, N_HH),
})
baseline.to_csv(os.path.join(RAW_DIR, "baseline.csv"), index=False)

records = []
for m in range(MONTHS):
    date = (START + pd.DateOffset(months=m)).to_pydatetime().date()
    drought_p = 0.25 + 0.15*np.sin(2*np.pi*(m/12.0))
    flood_p   = 0.10 + 0.10*np.cos(2*np.pi*(m/12.0))
    illness_p = 0.20
    crop_p    = 0.12
    for hh in range(1, N_HH+1):
        records.append({
            "household_id": hh,
            "report_date": date.isoformat(),
            "drought": np.random.binomial(1, max(0,min(1,drought_p))),
            "flood": np.random.binomial(1, max(0,min(1,flood_p))),
            "illness": np.random.binomial(1, illness_p),
            "crop_disease": np.random.binomial(1, crop_p),
        })
monthly = pd.DataFrame(records)
monthly.to_csv(os.path.join(RAW_DIR, "monthly_shocks.csv"), index=False)
print("Synthetic data written to:", RAW_DIR)
