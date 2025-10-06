"""
===============================================================================
Project       : Resilience Early Warning
Module        : preprocess.py
Description   : Cleans, merges baseline and monthly shocks, engineers features,
                and produces labeled samples for next-month food insecurity risk.
===============================================================================
Author        : Apoorv Pal
Created On    : 2025-10-06
Last Modified : 2025-10-06
Modified By   : Apoorv Pal
Version       : 1.1.0
Dependencies  : os, yaml, pandas, numpy
Status        : Production-Ready (Sample)
===============================================================================
Change Log:
-------------------------------------------------------------------------------
Date        Author        Description
----------  ------------- -----------------------------------------------------
2025-10-06  Apoorv Pal    Initial version of preprocessing and feature creation.
===============================================================================
"""
import os
import yaml
import pandas as pd
import numpy as np

CFG_PATH = "config/config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
cols = cfg["columns"]
features = cfg["features"]
os.makedirs(paths["processed_dir"], exist_ok=True)

baseline = pd.read_csv(cfg["data"]["baseline_file"])
monthly = pd.read_csv(cfg["data"]["monthly_file"], parse_dates=[cols["date_col"]])

monthly = monthly.sort_values([cols["id_col"], cols["date_col"]]).copy()
shock_cols = cfg["columns"]["monthly_shock_events"]

def with_rolling(df):
    df = df.copy()
    for c in shock_cols:
        df[f"{c}_roll3"] = (
            df.groupby(cols["id_col"])[c]
              .rolling(features["shock_rolling_months"], min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
    if features.get("include_shock_counts", True):
        df["shock_count"] = df[shock_cols].sum(axis=1)
        df["shock_count_roll3"] = (
            df.groupby(cols["id_col"])["shock_count"]
              .rolling(features["shock_rolling_months"], min_periods=1).sum()
              .reset_index(level=0, drop=True)
        )
    return df

monthly_feat = with_rolling(monthly)
df = monthly_feat.merge(baseline, on=cols["id_col"], how="left")

# Simulated label for demonstration only (replace with real next-month target)
def simulate_label(row):
    base = 0.1
    shock_load = 0.15*row.get("shock_count_roll3", 0) + 0.20*row.get("drought_roll3", 0) + 0.10*row.get("flood_roll3", 0)
    capacity = ((row.get("land_area_hectares", 0) < 1.0)*0.10
                + (row.get("livestock_units", 0) < 1)*0.10
                + (row.get("head_gender_female", 0) == 1)*0.08
                + (row.get("head_disability", 0) == 1)*0.12
                + (row.get("floodplain_exposure", 0) == 1)*0.05)
    p = base + shock_load + capacity
    p = max(0.01, min(0.95, p))
    return np.random.binomial(1, p)

df = df.sort_values([cols["id_col"], cols["date_col"]])
df[cols["label_col"]] = df.apply(simulate_label, axis=1)

proc_path = os.path.join(paths["processed_dir"], "dataset.parquet")
df.to_parquet(proc_path, index=False)
print("Processed dataset:", proc_path)
