# 🚗 NYC Violations Dashboard — Live Demo Guide

## 🎯 Purpose

This dashboard is the **live reporting component** of our project.
It integrates outputs from all modeling steps (KMeans, Random Forest, XGBoost) into one interactive interface.

The goal is to allow users (e.g., Department of Finance, policymakers) to:

* Explore violation data
* View model results
* Understand trends and insights in real time

---

## ▶️ How to Run the Dashboard

### 1. Open terminal in project folder

Make sure you are in:

```
BIA678-TEAM1/5-dashboard_repoorting
```

### 3. Run Streamlit app

```
streamlit run app.py
```

### 4. Open in browser

A browser tab will automatically open (usually http://localhost:8501)

---

## 🛑 How to Stop the App

In terminal:

```
Ctrl + C
```

If prompted:

```
Terminate batch job (Y/N)? → Y
```
For faster kill just click Trashcan Icon and open anew terminal;
sometimes Steamlit lags!!
 
---

## 🧭 How to Use the Dashboard (Demo Flow)

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

## ⚠️ Notes

* Do NOT move or rename files — dashboard depends on exact names
* All models use the same dataset (1,000 rows) for consistency
* This is a **proof-of-concept dashboard**, not a production system

---

## 💬 Demo Talking Point

> “This dashboard integrates our full pipeline, allowing stakeholders to explore violation patterns, understand model outputs, and gain insights into revenue and enforcement trends.”

---

## ✅ Status

* Pipeline complete
* Models in progress 
* Dashboard ready for demo (waiting on models + make it pretty)

