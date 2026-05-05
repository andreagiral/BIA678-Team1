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
BIA678-TEAM1/
│
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
    └── (model outputs reused from model-training) 
	│   
├── clustered_data_KMeans_clustering.csv
├── elbow-method-for-kmeans-clustering.png
├── feature_engineered_violations.csv
├── feature_importance_random_forest.csv
├── feature_importance_random_forest.png
├── xgboost_monthly_actual_vs_predicted.png
├── xgboost_monthly_metrics.txt
├── xgboost_monthly_predictions.csv
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

## 📁 Required Files

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
The project repository was organized using separate GitHub branches to keep each stage of the pipeline modular and easier to track.
Main Branch
The main branch contains the project README.md file, which provides a general overview of the repository.
Setup Branch
The setup branch contains the required installation/setup file:
•	setup.py — defines the required packages and setup configuration for running the project.
Data Ingestion Branch
The data-ingestion branch contains the script used to extract the NYC parking violation data:
•	data-ingestion.py — retrieves violation-level records from the NYC OpenData Socrata API.
•	data_2026-04-36.csv — output file generated from the data ingestion process.
Feature Engineering Branch
The feature-engineering branch contains the data ingestion script, feature engineering script, and generated feature outputs:
•	data-ingestion.py — copy of the ingestion script used to maintain reproducibility.
•	feature-engineering.py — cleans the raw violation data and creates engineered features.
•	cleaned_violations.csv — cleaned dataset after preprocessing.
•	feature_engineered_violations.csv — full engineered dataset.
•	model_ready_features.csv — final feature set prepared for supervised modeling.
•	clustering_features.csv — feature subset prepared for clustering analysis.
•	pipeline_output_new.txt — saved terminal/log output from the feature engineering pipeline.
Model Training Branch
The model-training branch contains the scripts and outputs for the machine learning models:
•	model-training-kmeans.py — K-Means clustering model script.
•	model-training-random-forest-classifier.py — Random Forest classification model script.
•	prophet-model.py — Prophet time series forecasting model script.
•	xgboost-time-series.py — XGBoost monthly time series model script.
•	clustered_data_KMeans_clustering.csv — output dataset with K-Means cluster assignments.
•	elbow-method-for-kmean-clustering.png — elbow plot used to evaluate K-Means cluster selection.
•	feature_engineered_violations.csv — input dataset used for modeling.
•	feature_importance_random_forest.csv — Random Forest feature importance output.
•	feature_importance_random_forest.png — Random Forest feature importance visualization.
•	xgboost_monthly_actual_vs_predicted.png — XGBoost actual vs. predicted monthly violation count plot.
•	xgboost_monthly_metrics.txt — XGBoost evaluation metrics and feature importance.
•	xgboost_monthly_predictions.csv — XGBoost test-period prediction outputs.
•	README.md — model training branch documentation.
Dashboard Reporting Branch
The dashboard-reporting branch contains the Streamlit dashboard and visual/model outputs used for reporting:
•	app.py — Streamlit dashboard application.
•	Modeling outputs from the model-training branch — used to populate dashboard visuals and summaries.
•	prophet1.png — Prophet visualization output.
•	prophet2.png — Prophet visualization output.
This branch structure allowed the project to separate setup, ingestion, feature engineering, modeling, and dashboard reporting into clear stages while preserving outputs for reproducibility.
