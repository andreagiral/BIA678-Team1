import streamlit as st
import pandas as pd
from PIL import Image

st.set_page_config(page_title="NYC Violations Dashboard", layout="wide")

st.markdown(
    "<h1 style='text-align: center;'>NYC Parking & Camera Violations Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<h5 style='text-align: center;'>BIA 678-WS | Team 1: Andrea Giraldo-Puerta, Olga Melissa Avram & Jeffrey Cheng</h5>",
    unsafe_allow_html=True
)

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #F7F3EC;
}

/* Titles */
h1, h2, h3 {
    color: #3A2A1A;
    font-family: "Segoe UI", sans-serif;
}

/* Body text */
p, div, span {
    font-family: "Segoe UI", sans-serif;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #FFFFFF;
    padding: 18px;
    border-radius: 18px;
    border: 1px solid #E5D8C8;
    box-shadow: 0px 3px 12px rgba(70, 50, 30, 0.08);
}

/* Tabs */
button[data-baseweb="tab"] {
    background-color: #EFE3D4;
    border-radius: 14px;
    padding: 10px 18px;
    margin-right: 6px;
    color: #3A2A1A;
    font-weight: 600;
}

button[data-baseweb="tab"][aria-selected="true"] {
    background-color: #8B5E3C;
    color: white;
}

/* Dataframes / containers */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    border: 1px solid #E5D8C8;
}

/* Buttons / select boxes */
.stSelectbox > div > div {
    background-color: #FFFFFF;
    border-radius: 12px;
}

/* Info/warning boxes soften */
[data-testid="stAlert"] {
    border-radius: 14px;
}

/* Horizontal divider style */
hr {
    border: none;
    height: 1px;
    background-color: #D8C3AA;
}
</style>
""", unsafe_allow_html=True)

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
    "XGBoost Time Series",
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
    st.divider()
    
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Records", len(df))
    col2.metric("Total Revenue", f"${df['total_charges'].sum():,.2f}")
    col3.metric("Avg Fine", f"${df['fine_amount'].mean():,.2f}")
    
    top_borough = df["county"].value_counts().idxmax()
    top_violation = df["violation"].value_counts().idxmax()

    col4, col5 = st.columns(2)
    col4.metric("Top Borough", top_borough)
    col5.metric("Top Violation", top_violation)
    
    st.divider()
    # --------------------------
    # KEY INSIGHTS
    # --------------------------    
    st.subheader("Key Insights")
    st.markdown("""
    - Violation activity shows strong short-term trends (rolling averages are key predictors)
    - Certain boroughs consistently generate higher violation volumes
    - Clustering reveals distinct violation behavior groups
    - Time-series models capture general trends but struggle with sudden spikes
    """)
    st.divider()
    # --------------------------
    # Visualization
    # --------------------------
    col1, col2 = st.columns(2)
    
    with col1: 
        st.subheader("Violations by Borough")
        st.bar_chart(df["county"].value_counts())
        
    with col2:
        st.subheader("Revenue by Borough")
        import plotly.express as px
        revenue_by_borough = df.groupby("county")["total_charges"].sum().reset_index()
        fig = px.bar(revenue_by_borough, x="county", 
                    y="total_charges", 
                    labels={"county": "Borough", "total_charges": "Revenue"}
        )
        
        fig.update_layout(
            yaxis_tickprefix="$",
            yaxis_tickformat=","
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    
    # --------------------------
    # DASHBOARD SECTIONS
    # --------------------------
    st.subheader("Dashboard Sections")
    st.markdown(
        """
        - **Data Explorer:** filter and inspect violation patterns by borough.
        - **KMeans:** view clustered violation pattern outputs.
        - **Random Forest:** review classification feature importance.
        - **XGBoost:** evaluate monthly time-series model results.
        - **Prophet:** view additional forecasting visuals.
        """
    )
    st.divider()
    
    # --------------------------
    # LIMITATIONS
    # --------------------------
    st.subheader(" Limitations")
    st.markdown("""
    - Dataset limited to ~1,000 records (API sample)
    - Sparse time series data leads to unstable predictions
    - Extreme spikes are difficult to model
    - Models are proof-of-concept and would improve with more data
    """)
# --------------------------
# DATA EXPLORER
# --------------------------
with tab2:
    st.header("Data Explorer")
    
    st.write(
        """
        This section lets users interactively filter the feature-engineered dataset.
        It helps answer questions like: Which boroughs have the most violations?
        What are the most common violation types in each borough?
        """
    )

    # Data column is called county, but dashboard label says Borough
    selected_borough = st.selectbox(
        "Select Borough",
        sorted(df["county"].dropna().unique())
    )
    
    filtered = df[df["county"] == selected_borough]

    st.write(filtered.head(50))
    st.bar_chart(filtered["violation"].value_counts().head(10))

    st.divider()

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

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.subheader("Elbow Method")
        st.image("Elbow_Method_for_KMeans_Clustering.png")
        
        st.subheader("Cluster Distribution")
        st.bar_chart(kmeans_df["cluster"].value_counts())
    st.divider()
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
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.subheader("Feature Importance")
        st.image("feature_importance_Random_Forest_Clustering.png")
    st.divider()
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

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("### Model Output: Actual vs Predicted")
        st.image("xgboost_monthly_actual_vs_predicted.png")

    st.subheader("Predictions Table")
    st.dataframe(xgb_preds)
    st.divider()
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

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prophet Forecast Plot")
        st.image("prophet_1.png")

    with col2:
        st.subheader("Prophet Components / Trend Plot")
        st.image("prophet_2.png")
    st.divider()