# BIA678-Team1
BIA 678 Project Contents

## Setup

1. Install PostgreSQL
2. Create database: df_db
3. Set environment variable:

DATABASE_URL=postgresql+psycopg2://postgres:password@localhost:5432/df_db

4. Install dependencies:
pip install pandas requests sqlalchemy
        optional: run installation tests
                  python 1-setup.py

6. Run ingestion:
python 2-data_ingestion.py
----

## A Repository Structure and Script Organization

```bash
BIA678-TEAM1/
├── main/
│   └── README.md
│
├── setup/
│   └── setup.py
│
├── data-ingestion/
│   ├── data-ingestion.py
│   └── data_2026-04-36.csv
│
├── feature-engineering/
│   ├── data-ingestion.py
│   ├── feature-engineering.py
│   ├── cleaned_violations.csv
│   ├── feature_engineered_violations.csv
│   ├── model_ready_features.csv
│   ├── clustering_features.csv
│   └── pipeline_output_new.txt
│
├── model-training/
│   ├── model-training-kmeans.py
│   ├── model-training-random-forest-classifier.py
│   ├── prophet-model.py
│   ├── xgboost-time-series.py
│   ├── clustered_data_KMeans_clustering.csv
│   ├── elbow-method-for-kmeans-clustering.png
│   ├── feature_engineered_violations.csv
│   ├── feature_importance_random_forest.csv
│   ├── feature_importance_random_forest.png
│   ├── xgboost_monthly_actual_vs_predicted.png
│   ├── xgboost_monthly_metrics.txt
│   ├── xgboost_monthly_predictions.csv
│   └── README.md
│
└── dashboard-reporting/
    ├── app.py
    ├── prophet1.png
    ├── prophet2.png
    ├── clustered_data_KMeans_clustering.csv
    ├── elbow-method-for-kmeans-clustering.png
    ├── feature_engineered_violations.csv
    ├── feature_importance_random_forest.csv
    ├── feature_importance_random_forest.png
    ├── xgboost_monthly_actual_vs_predicted.png
    ├── xgboost_monthly_metrics.txt
    └── xgboost_monthly_predictions.csv
```

--- 
# 🚗 NYC Violations Dashboard — Live Demo Guide

## Purpose

This dashboard is the **live reporting component** of our project.
It integrates outputs from all modeling steps (KMeans, Random Forest, XGBoost) into one interactive interface.

The goal is to allow users (e.g., Department of Finance, policymakers) to:

* Explore violation data
* View model results
* Understand trends and insights in real time

---

## How to Run the Dashboard

### 1. app.py is deploy so you can use direct link to access it
https://nyc-violations-dashboard.streamlit.app 
---

## How to Use the Dashboard (Demo Flow)

### 1. Overview Tab

* Shows total records, revenue, and averages
* Explain project goal and stakeholders

---

### 2. Data Explorer Tab

* Filter violations by county
* Show top violation types
* Highlight how users can interact with the data

---

### 3. KMeans Tab

* Displays clustered data
* Show Elbow Method plot
* Explain grouping of violation patterns

---

### 4. Random Forest Tab

* Show feature importance plot
* Explain key predictors of violations

---

### 5. XGBoost Tab

* Show actual vs predicted time series
* Discuss RMSE, MAE, R²
* Explain limitations due to small dataset

---

## Required Files

```
app.py
feature_engineered_violations.csv
clustering_features.csv
model_ready_features.csv
clustered_data_KMeans_clustering.csv
Elbow Method for KMeans Clustering.png
feature_importance_Random_Forest_Clustering.png
xgboost_monthly_predictions.csv
xgboost_monthly_metrics.txt
xgboost_monthly_actual_vs_predicted.png
```

---

## Notes

* Do NOT move or rename files — dashboard depends on exact names
* All models use the same dataset (1,000 rows) for consistency
* This is a **proof-of-concept dashboard**, not a production system

---

## Demo Talking Point

> “This dashboard integrates our full pipeline, allowing stakeholders to explore violation patterns, understand model outputs, and gain insights into revenue and enforcement trends.”
---

## Organization

The project repository is organized using separate GitHub branches to modularize each stage of the pipeline and improve reproducibility.

### 🔹 Main Branch
- Contains the main `README.md` with project overview.

### 🔹 Setup Branch
- `setup.py` — defines required packages and environment setup.

### 🔹 Data Ingestion Branch
- `data-ingestion.py` — extracts NYC parking violation data from the Socrata API.
- `data_2026-04-36.csv` — raw output dataset.

### 🔹 Feature Engineering Branch
- `data-ingestion.py` — included for reproducibility.
- `feature-engineering.py` — cleans data and creates features.
- `cleaned_violations.csv` — cleaned dataset.
- `feature_engineered_violations.csv` — full engineered dataset.
- `model_ready_features.csv` — dataset for supervised models.
- `clustering_features.csv` — dataset for clustering.
- `pipeline_output_new.txt` — pipeline logs.

### 🔹 Model Training Branch
- `model-training-kmeans.py` — K-Means clustering.
- `model-training-random-forest-classifier.py` — Random Forest classification.
- `prophet-model.py` — Prophet forecasting.
- `xgboost-time-series.py` — XGBoost time series model.
- Outputs include:
  - Cluster assignments
  - Feature importance (CSV + PNG)
  - XGBoost predictions and metrics
  - Elbow method visualization

### 🔹 Dashboard Reporting Branch
- `app.py` — Streamlit dashboard.
- Uses model outputs for visualization.
- Includes Prophet plots (`prophet1.png`, `prophet2.png`).

---
