"""
===============================================================================
Project       : Resilience Early Warning
Module        : train_model.py
Description   : Trains a classifier to predict next-month food insecurity risk
                from engineered features (baseline + rolling shocks).
===============================================================================
Author        : Apoorv Pal
Created On    : 2025-10-06
Last Modified : 2025-10-06
Modified By   : Apoorv Pal
Version       : 1.0.0
Dependencies  : os, yaml, pandas, numpy, scikit-learn, joblib
Status        : Production-Ready (Sample)
===============================================================================
Change Log:
-------------------------------------------------------------------------------
Date        Author        Description
----------  ------------- -----------------------------------------------------
2025-10-06  Apoorv Pal    Initial training pipeline (logistic regression).
===============================================================================
"""
import os
import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

CFG_PATH = "config/config.yaml"
with open(CFG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

paths = cfg["paths"]
cols = cfg["columns"]

os.makedirs(paths["model_dir"], exist_ok=True)

df = pd.read_parquet(os.path.join(paths["processed_dir"], "dataset.parquet"))

feature_cols = (
    cfg["columns"]["baseline_features"]
    + [c for c in df.columns if c.endswith("_roll3") or c in ("shock_count","shock_count_roll3")]
)

X = df[feature_cols].fillna(0.0).astype(float)
y = df[cols["label_col"]].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=cfg["training"]["test_size"],
    stratify=y if cfg["training"]["stratify"] else None,
    random_state=cfg["random_seed"]
)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(C=cfg["model"]["params"]["logistic_regression"]["C"],
                              max_iter=cfg["model"]["params"]["logistic_regression"]["max_iter"]))
])

pipe.fit(X_train, y_train)
probs = pipe.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, probs)

print("Held-out AUC:", round(auc, 4))
print(classification_report(y_test, (probs>=0.5).astype(int)))

joblib.dump({"model": pipe, "features": feature_cols},
            os.path.join(paths["model_dir"], "model.joblib"))
print("Saved model to:", os.path.join(paths["model_dir"], "model.joblib"))
