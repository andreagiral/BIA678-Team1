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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Data Explorer",
    "KMeans Clustering",
    "Random Forest",
    "XGBoost Time Series"
    "Prophet Forecasting"
])

# --------------------------
# OVERVIEW
# --------------------------
with tab1:
    st.header("Project Overview")
    
    st.write(
        """
        This dashboard summarizes NYC parking and camera violation patterns using
        exploratory reporting and machine learning model outputs. It is designed
        as a live report for stakeholders such as the NYC Department of Finance
        and lawmakers.
        """
    )
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Revenue", f"${df['total_charges'].sum():,.2f}")
    col3.metric("Avg Fine", f"${df['fine_amount'].mean():,.2f}")
    
    st.subheader("Dashboard Sections")
    st.markdown(
        """
        - **Data Explorer:** filter and inspect violation patterns by county.
        - **KMeans:** view clustered violation pattern outputs.
        - **Random Forest:** review classification feature importance.
        - **XGBoost:** evaluate monthly time-series model results.
        - **Prophet:** view additional forecasting visuals.
        """
    )
# --------------------------
# DATA EXPLORER
# --------------------------
with tab2:
    st.header("Data Explorer")
    
    st.write(
        """
        This section lets users interactively filter the feature-engineered dataset.
        It helps answer questions like: Which counties have the most violations?
        What are the most common violation types in each county?
        """
    )

    county = st.selectbox("Select County", df["county"].dropna().unique())

    filtered = df[df["county"] == county]

    st.write(filtered.head(50))
    st.bar_chart(filtered["violation"].value_counts().head(10))

# --------------------------
# KMEANS
# --------------------------
with tab3:
    st.header("KMeans Clustering")

    st.write(
        """
        KMeans was used to group similar violation records based on engineered
        numeric features. This helps identify common patterns in violation behavior.
        """
    )

    st.subheader("Clustered Data Preview")
    st.dataframe(kmeans_df.head(50))

    st.subheader("Elbow Method")
    st.image("Elbow_Method_for_KMeans_Clustering.png")

# --------------------------
# RANDOM FOREST
# --------------------------
with tab4:
    st.header("Random Forest")
    
    st.write(
        """
        Random Forest was used as a classification model to identify important
        predictors related to violation patterns.
        """
    )    

    st.subheader("Feature Importance")
    st.image("feature_importance_Random_Forest_Clustering.png")

    st.write("Random Forest used for classification of violation patterns.")

# --------------------------
# XGBOOST
# --------------------------
with tab5:
    st.header("XGBoost Time Series")
    
    st.write(
        """
        XGBoost was used to model monthly violation activity using lag features,
        rolling averages, and calendar-based features.
        """
    )

    st.subheader("Metrics")
    st.text(open("xgboost_monthly_metrics.txt").read())

    st.subheader("Actual vs Predicted")
    st.image("xgboost_monthly_actual_vs_predicted.png")

    st.subheader("Predictions Table")
    st.dataframe(xgb_preds)

# --------------------------
# PROPHET
# --------------------------
with tab6:
    st.header("Prophet Forecasting")

    st.write(
        """
        Prophet was used as an additional forecasting model to compare time-series
        behavior and visualize expected violation trends over time.
        """
    )

    st.subheader("Prophet Forecast Plot")
    if os.path.exists("prophet_1.png"):
        st.image("prophet_1.png", use_container_width=True)
    else:
        st.warning("prophet_1.png not found.")

    st.subheader("Prophet Components / Trend Plot")
    if os.path.exists("prophet_2.png"):
        st.image("prophet_2.png", use_container_width=True)
    else:
        st.warning("prophet_2.png not found.")