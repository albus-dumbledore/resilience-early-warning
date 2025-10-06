"""
===============================================================================
Project       : Resilience Early Warning
Module        : dashboard.py
Description   : Streamlit dashboard to visualize monthly shocks and risk scores.
===============================================================================
Author        : Apoorv Pal
Created On    : 2025-10-06
Last Modified : 2025-10-06
Modified By   : Apoorv Pal
Version       : 1.0.0
Dependencies  : streamlit, pandas, matplotlib
Status        : Demo
===============================================================================
Change Log:
-------------------------------------------------------------------------------
Date        Author        Description
----------  ------------- -----------------------------------------------------
2025-10-06  Apoorv Pal    Initial dashboard with shocks & predictions views.
===============================================================================
"""
import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Resilience Early Warning", layout="wide")

DATA_DIR = "data"
RAW = os.path.join(DATA_DIR, "raw")
PROC = os.path.join(DATA_DIR, "processed")
PRED = "predictions"

st.title("Resilience Early Warning")
st.caption("Baseline + monthly shocks → features → ML risk predictions")

col1, col2 = st.columns(2)

with col1:
    st.header("Monthly Shocks")
    monthly_path = os.path.join(RAW, "monthly_shocks.csv")
    if os.path.exists(monthly_path):
        monthly = pd.read_csv(monthly_path, parse_dates=["report_date"])
        opt = st.selectbox("Choose shock", ["drought","flood","illness","crop_disease"], index=0)
        grp = monthly.groupby("report_date")[opt].sum().reset_index()
        fig, ax = plt.subplots()
        ax.plot(grp["report_date"], grp[opt])
        ax.set_title(f"Monthly {opt} counts")
        ax.set_xlabel("Month")
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("Load synthetic or real data to view trends.")

with col2:
    st.header("Latest Predictions")
    pred_path = os.path.join(PRED, "latest_predictions.csv")
    if os.path.exists(pred_path):
        preds = pd.read_csv(pred_path)
        st.dataframe(preds.head(50))
        st.write(f"Households scored: **{len(preds)}**")
        st.write(f"High-risk (p≥0.5): **{(preds['prob_food_insecure_next_month']>=0.5).mean():.0%}**")
    else:
        st.info("Run training and batch prediction to see scores.")

st.divider()
st.markdown("**Tip:** Replace CSV ingestion with your real survey system connector (e.g., CommCare, Kobo, DHIS2).")
