import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="NYC Violations Dashboard", layout="wide")

st.title("🚗 NYC Parking & Camera Violations Dashboard")

# Load data
df = pd.read_csv("feature_engineered_violations.csv")
kmeans_df = pd.read_csv("clustered_data_KMeans_clustering.csv")
xgb_preds = pd.read_csv("xgboost_monthly_predictions.csv")

# --------------------------
# TABS
# --------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Data Explorer",
    "KMeans Clustering",
    "Random Forest",
    "XGBoost Time Series"
])

# --------------------------
# OVERVIEW
# --------------------------
with tab1:
    st.header("Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Revenue", f"${df['total_charges'].sum():,.2f}")
    col3.metric("Avg Fine", f"${df['fine_amount'].mean():,.2f}")

# --------------------------
# DATA EXPLORER
# --------------------------
with tab2:
    st.header("Data Explorer")

    county = st.selectbox("Select County", df["county"].dropna().unique())

    filtered = df[df["county"] == county]

    st.write(filtered.head(50))
    st.bar_chart(filtered["violation"].value_counts().head(10))

# --------------------------
# KMEANS
# --------------------------
with tab3:
    st.header("KMeans Clustering")

    st.subheader("Clustered Data Preview")
    st.dataframe(kmeans_df.head(50))

    st.subheader("Elbow Method")
    st.image("Elbow_Method_for_KMeans_Clustering.png")

# --------------------------
# RANDOM FOREST
# --------------------------
with tab4:
    st.header("Random Forest")

    st.subheader("Feature Importance")
    st.image("feature_importance_Random_Forest_Clustering.png")

    st.write("Random Forest used for classification of violation patterns.")

# --------------------------
# XGBOOST
# --------------------------
with tab5:
    st.header("XGBoost Time Series")

    st.subheader("Metrics")
    st.text(open("xgboost_monthly_metrics.txt").read())

    st.subheader("Actual vs Predicted")
    st.image("xgboost_monthly_actual_vs_predicted.png")

    st.subheader("Predictions Table")
    st.dataframe(xgb_preds)