"""
===============================================================================
Project       : Resilience Early Warning
Module        : predict_batch.py
Description   : Generates batch predictions for the most recent panel month and
                writes probabilities to CSV for outreach planning.
===============================================================================
Author        : Apoorv Pal
Created On    : 2025-10-06
Last Modified : 2025-10-06
Modified By   : Apoorv Pal
Version       : 1.0.0
Dependencies  : os, yaml, joblib, pandas, numpy
Status        : Production-Ready (Sample)
===============================================================================
Change Log:
-------------------------------------------------------------------------------
Date        Author        Description
----------  ------------- -----------------------------------------------------
2025-10-06  Apoorv Pal    Initial batch prediction job.
===============================================================================
"""
import os
import yaml
import joblib
import pandas as pd

CFG_PATH = "config/config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
cols = cfg["columns"]

os.makedirs(paths["predictions_dir"], exist_ok=True)

bundle = joblib.load(os.path.join(paths["model_dir"], "model.joblib"))
model = bundle["model"]
feature_cols = bundle["features"]

df = pd.read_parquet(os.path.join(paths["processed_dir"], "dataset.parquet"))
df_latest = df.sort_values([cols["id_col"], cols["date_col"]]).groupby(cols["id_col"]).tail(1)

X = df_latest[feature_cols].fillna(0.0).astype(float)
df_latest["prob_food_insecure_next_month"] = model.predict_proba(X)[:,1]

out_path = os.path.join(paths["predictions_dir"], "latest_predictions.csv")
df_latest[[cols["id_col"], "prob_food_insecure_next_month"]].to_csv(out_path, index=False)
print("Wrote predictions:", out_path)
