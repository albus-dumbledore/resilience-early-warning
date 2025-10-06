# Resilience Early Warning (REW)

A lightweight, end-to-end system for **predicting household food insecurity risk** using **machine learning** on baseline and monthly survey data.  
Designed for easy setup, reproducibility, and extension across countries or survey platforms.

---

## Overview

**REW** combines:
- **Baseline characteristics** — demographics, assets, and capacities  
- **Monthly shock data** — drought, flood, illness, crop disease  
- **Machine learning** — predicts next-month food insecurity risk  
- **Dashboard & API** — visualize trends and get real-time predictions  

Built for organizations tracking community resilience and early-warning signals.

---

## System Architecture

```
Data (CSV / API)
     │
     ▼
Feature Engineering ──▶ Model Training ──▶ Risk Predictions
     │                                   │
     ▼                                   ▼
Dashboard (Streamlit)              API (FastAPI)
```

---

## Quick Start

### 1. Setup

```bash
git clone https://github.com/yourusername/resilience-early-warning.git
cd resilience-early-warning

python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Example Data

```bash
python scripts/generate_synthetic_data.py
```

### 3. Run the Pipeline

```bash
python pipelines/preprocess.py
python pipelines/train_model.py
python pipelines/predict_batch.py
```

### 4. Launch Dashboard

```bash
streamlit run services/dashboard.py
```

### 5. Use the API

```bash
uvicorn services.api:app --reload
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive API.

---

## Folder Structure

```
resilience-early-warning/
│
├── scripts/          # Data generation
├── pipelines/        # Preprocessing, training, prediction
├── services/         # API and dashboard
├── config/           # Configuration files
├── data/             # Data folders (raw, processed, predictions)
└── models/           # Saved ML models
```

---

## Key Features

- **Synthetic demo data** to test end-to-end flow  
- **Rolling shock features** and household-level risk modeling  
- **Simple, reproducible pipeline** using pandas + scikit-learn  
- **Dashboard and REST API** for quick visualization and deployment  

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ to help identify risk before crisis hits.**
