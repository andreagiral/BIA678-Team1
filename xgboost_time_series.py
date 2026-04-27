# =============================================================================
# PURPOSE:
#   Predict daily_total_charges (revenue) using XGBoost with time series features.
#   This is Step 4 in the pipeline:
#   ETL → Feature Engineering → [THIS SCRIPT] Time Series Modeling → Dashboard
#
# INPUT:
#   feature_engineered_violations.csv  (from 3-feature_engineering branch)
#
# OUTPUTS:
#   xgboost_time_series_predictions.csv
#   xgboost_time_series_metrics.txt
#   xgboost_actual_vs_predicted.png
# =============================================================================
 
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import os
 
warnings.filterwarnings("ignore")
 
# =============================================================================
# CONFIGURATION
# =============================================================================
 
INPUT_FILE          = "feature_engineered_violations.csv"
OUTPUT_PREDICTIONS  = "xgboost_time_series_predictions.csv"
OUTPUT_METRICS      = "xgboost_time_series_metrics.txt"
OUTPUT_PLOT         = "xgboost_actual_vs_predicted.png"
 
TRAIN_RATIO = 0.80   # 80% train / 20% test (time-based, no shuffle)
TARGET_COL  = "daily_total_charges"
 
# XGBoost hyperparameters
# These are sensible defaults for a small dataset — see BONUS section at bottom
XGBOOST_PARAMS = {
    "n_estimators":    200,
    "learning_rate":   0.05,
    "max_depth":       4,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    42,
    "verbosity":       0,
}
 
 
# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
 
def load_data(filepath):
    """Load the feature-engineered violations CSV."""
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
 
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] '{filepath}' not found.\n"
            "Make sure you ran:\n"
            "  git checkout 4-modeling\n"
            "  git checkout 3-feature_engineering -- feature_engineered_violations.csv"
        )
 
    df = pd.read_csv(filepath, low_memory=False)
    print(f"[OK] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df
 
 
# =============================================================================
# STEP 2: CONVERT issue_date TO DATETIME
# =============================================================================
 
def parse_dates(df):
    """Convert issue_date to proper datetime so we can aggregate by day."""
    print("\n[STEP 2] Parsing issue_date...")
 
    if "issue_date" not in df.columns:
        raise KeyError("[ERROR] 'issue_date' column not found in dataset.")
 
    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
 
    bad_dates = df["issue_date"].isnull().sum()
    if bad_dates > 0:
        print(f"Dropped {bad_dates} rows with unparseable dates.")
        df = df.dropna(subset=["issue_date"])
 
    print(f"Date range: {df['issue_date'].min().date()} -> {df['issue_date'].max().date()}")
    return df
 
 
# =============================================================================
# STEP 3: AGGREGATE BY DAY
# =============================================================================
 
def aggregate_by_day(df):
    """
    WHY AGGREGATE?
    XGBoost time series works on a single timeline — one row per time step.
    Our raw data has many rows per day (one per violation ticket).
    We collapse them into daily totals so the model can learn:
    "Given what revenue looked like the past N days, what will tomorrow look like?"
 
    We create three daily metrics:
      - daily_total_charges  : total revenue generated that day (our TARGET)
      - daily_violation_count: how many tickets were issued
      - daily_avg_fine       : average fine amount per ticket
    """
    print("\n[STEP 3] Aggregating data by day...")
 
    # Make sure total_charges exists (it should from feature engineering)
    if "total_charges" not in df.columns:
        print("  [WARN] 'total_charges' not found — computing from fine+penalty+interest")
        for col in ["fine_amount", "penalty_amount", "interest_amount"]:
            if col not in df.columns:
                df[col] = 0
        df["total_charges"] = df["fine_amount"] + df["penalty_amount"] + df["interest_amount"]
 
    daily = df.groupby("issue_date").agg(
        daily_total_charges  = ("total_charges", "sum"),
        daily_violation_count= ("total_charges", "count"),
        daily_avg_fine       = ("fine_amount",   "mean"),
    ).reset_index()
 
    # Sort chronologically — critical for time series
    daily = daily.sort_values("issue_date").reset_index(drop=True)
 
    print(f"Daily time series: {len(daily)} days")
    print(f"Date range:        {daily['issue_date'].min().date()} -> {daily['issue_date'].max().date()}")
    print(f"Avg daily revenue: ${daily['daily_total_charges'].mean():,.2f}")
    print(f"Max daily revenue: ${daily['daily_total_charges'].max():,.2f}")
 
    return daily
 
 
# =============================================================================
# STEP 4: CREATE TIME SERIES FEATURES
# =============================================================================
 
def create_time_series_features(daily):
    """
    WHY LAG FEATURES?
    A lag feature is simply "yesterday's value" (lag_1), "7 days ago" (lag_7), etc.
    They let the model answer: "revenue tends to be similar to what it was last week."
    Without lag features, the model has no memory of the past — it can't do time series.
 
    WHY ROLLING FEATURES?
    Rolling mean/std smooth out day-to-day noise and capture trends:
    - rolling_mean_7: "what was the average revenue over the past week?"
    - rolling_std_7:  "how volatile was revenue over the past week?"
 
    WHY CALENDAR FEATURES?
    Revenue follows predictable calendar patterns:
    - More tickets issued mid-week vs weekends
    - More violations in Q2 (spring/summer) than winter
    These are free information the model can use.
 
    IMPORTANT — NO DATA LEAKAGE:
    All lag and rolling features use .shift(1) or later, meaning they only
    look at PAST values. We never let the model see the future when training.
    """
    print("\n[STEP 4] Creating time series features...")
 
    df = daily.copy()
 
    # --- Lag features ---
    # shift(1) = value from 1 day ago, shift(7) = 7 days ago, etc.
    df["lag_1_revenue"]  = df[TARGET_COL].shift(1)
    df["lag_7_revenue"]  = df[TARGET_COL].shift(7)
    df["lag_30_revenue"] = df[TARGET_COL].shift(30)
 
    # --- Rolling features ---
    # min_periods=1 avoids NaN at the start of the series
    # closed="left" ensures we never include the current day in the window (no leakage)
    df["rolling_mean_7_revenue"]  = df[TARGET_COL].shift(1).rolling(window=7,  min_periods=1).mean()
    df["rolling_mean_30_revenue"] = df[TARGET_COL].shift(1).rolling(window=30, min_periods=1).mean()
    df["rolling_std_7_revenue"]   = df[TARGET_COL].shift(1).rolling(window=7,  min_periods=1).std().fillna(0)
 
    # --- Calendar features ---
    df["day_of_week"] = df["issue_date"].dt.dayofweek   # 0=Monday, 6=Sunday
    df["month"]       = df["issue_date"].dt.month
    df["quarter"]     = df["issue_date"].dt.quarter
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
 
    # Drop rows with NaN introduced by lag features (first 30 rows)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    print(f"  Dropped {dropped} rows due to lag/rolling NaN (expected).")
    print(f"  Usable rows for modeling: {len(df)}")
 
    features_created = [
        "lag_1_revenue", "lag_7_revenue", "lag_30_revenue",
        "rolling_mean_7_revenue", "rolling_mean_30_revenue", "rolling_std_7_revenue",
        "day_of_week", "month", "quarter", "is_weekend"
    ]
    print(f"Features created: {features_created}")
    return df, features_created
 
 
# =============================================================================
# STEP 5: TRAIN / TEST SPLIT (TIME-BASED)
# =============================================================================
 
def time_based_split(df, features):
    """
    WHY TIME-BASED SPLIT?
    In regular ML we shuffle data randomly. For time series, shuffling is WRONG —
    it creates data leakage: the model would train on "future" data and test on
    "past" data, making accuracy look great but the model useless in practice.
 
    Instead we cut the timeline at 80%:
      - Everything before the cutoff → training set
      - Everything after              → test set
    This simulates how the model will actually be used: trained on history,
    predicting the future.
    """
    print("\n[STEP 5] Time-based train/test split (80/20, no shuffle)...")
 
    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()
 
    print(f" Train: {len(train)} days  ({train['issue_date'].min().date()} -> {train['issue_date'].max().date()})")
    print(f" Test:  {len(test)} days  ({test['issue_date'].min().date()} -> {test['issue_date'].max().date()})")
 
    X_train = train[features]
    y_train = train[TARGET_COL]
    X_test  = test[features]
    y_test  = test[TARGET_COL]
 
    return train, test, X_train, y_train, X_test, y_test
 
 
# =============================================================================
# STEP 6: TRAIN XGBOOST MODEL
# =============================================================================
 
def train_model(X_train, y_train):
    """Train the XGBoost regressor on the training set."""
    print("\n[STEP 6] Training XGBoost model...")
    print(f" Parameters: {XGBOOST_PARAMS}")
 
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)
 
    print("[OK] Model trained.")
    return model
 
 
# =============================================================================
# STEP 7: EVALUATE MODEL
# =============================================================================
 
def evaluate_model(model, X_test, y_test, test, features):
    """
    Calculate RMSE, MAE, and R² on the hold-out test set.
 
    RMSE  — Root Mean Squared Error: average prediction error in dollars.
            Penalizes large errors more than small ones.
    MAE   — Mean Absolute Error: average absolute dollar difference.
            More intuitive / easier to explain.
    R²    — How much variance the model explains (1.0 = perfect, 0 = no better than mean).
    """
    print("\n[STEP 7] Evaluating model...")
 
    y_pred = model.predict(X_test)
    # Clamp predictions to >= 0 (revenue can't be negative)
    y_pred = np.maximum(y_pred, 0)
 
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
 
    print(f"\n  ── Evaluation Metrics ──────────────────")
    print(f"  RMSE : ${rmse:,.2f}")
    print(f"  MAE  : ${mae:,.2f}")
    print(f"  R²   : {r2:.4f}")
 
    # Feature importance
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"\n  ── Feature Importance ──────────────────")
    for feat, score in importance.items():
        bar = "█" * int(score * 40)
        print(f"  {feat:<30} {score:.4f}  {bar}")
 
    return y_pred, rmse, mae, r2, importance
 
 
# =============================================================================
# STEP 8: SAVE OUTPUTS
# =============================================================================
 
def save_outputs(test, y_pred, rmse, mae, r2, importance, train, features):
    """Save predictions CSV, metrics text file, and actual vs predicted plot."""
    print("\n[STEP 8] Saving output files...")
 
    # ---- 1. Predictions CSV ----
    predictions_df = test[["issue_date", TARGET_COL]].copy()
    predictions_df["predicted_total_charges"] = y_pred
    predictions_df["error"] = predictions_df[TARGET_COL] - predictions_df["predicted_total_charges"]
    predictions_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f" [1] Saved predictions -> '{OUTPUT_PREDICTIONS}'")
 
    # ---- 2. Metrics text file ----
    with open(OUTPUT_METRICS, "w") as f:
        f.write("XGBoost Time Series Model — Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target variable: {TARGET_COL}\n")
        f.write(f"Training days: {len(train)}\n")
        f.write(f"Test days: {len(test)}\n")
        f.write(f"Train date range: {train['issue_date'].min().date()} -> {train['issue_date'].max().date()}\n")
        f.write(f"Test date range: {test['issue_date'].min().date()} -> {test['issue_date'].max().date()}\n\n")
        f.write(f"RMSE: ${rmse:,.2f}\n")
        f.write(f"MAE: ${mae:,.2f}\n")
        f.write(f"R²: {r2:.4f}\n\n")
        f.write("Feature Importance:\n")
        for feat, score in importance.items():
            f.write(f"  {feat:<30} {score:.4f}\n")
        f.write("\nModel Parameters:\n")
        for k, v in XGBOOST_PARAMS.items():
            f.write(f"  {k}: {v}\n")
    print(f" [2] Saved metrics -> '{OUTPUT_METRICS}'")
 
    # ---- 3. Actual vs Predicted plot ----
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("XGBoost Time Series — NYC Parking Violation Revenue", fontsize=14, fontweight="bold")
 
    # Top chart: full timeline (train shaded, test highlighted)
    ax1 = axes[0]
    ax1.plot(train["issue_date"], train[TARGET_COL],
             color="#adb5bd", linewidth=1, label="Train (actual)", alpha=0.7)
    ax1.plot(test["issue_date"], test[TARGET_COL],
             color="#2196F3", linewidth=1.5, label="Test (actual)")
    ax1.plot(test["issue_date"], y_pred,
             color="#F44336", linewidth=1.5, linestyle="--", label="Test (predicted)")
    ax1.axvline(x=test["issue_date"].min(), color="black", linestyle=":", linewidth=1, alpha=0.5)
    ax1.set_title("Full Timeline: Train + Test Actual vs Predicted")
    ax1.set_ylabel("Daily Total Charges ($)")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(True, alpha=0.3)
 
    # Bottom chart: test period zoomed in
    ax2 = axes[1]
    ax2.plot(test["issue_date"], test[TARGET_COL],
             color="#2196F3", linewidth=2, label="Actual", marker="o", markersize=3)
    ax2.plot(test["issue_date"], y_pred,
             color="#F44336", linewidth=2, linestyle="--", label="Predicted", marker="x", markersize=3)
    ax2.fill_between(test["issue_date"], test[TARGET_COL], y_pred, alpha=0.15, color="#9C27B0")
    ax2.set_title(f"Test Period Zoom — RMSE: ${rmse:,.2f}  |  MAE: ${mae:,.2f}  |  R²: {r2:.4f}")
    ax2.set_ylabel("Daily Total Charges ($)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(True, alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f" [3] Saved plot ->'{OUTPUT_PLOT}'")
 
 
# =============================================================================
# STEP 9: PRINT FINAL SUMMARY
# =============================================================================
 
def print_summary(daily, train, test, rmse, mae, r2, importance, features):
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Dataset days after aggregation: {len(daily)}")
    print(f"Days used after lag drop: {len(train) + len(test)}")
    print(f"Training days: {len(train)}")
    print(f"Test days: {len(test)}")
    print(f"Train range: {train['issue_date'].min().date()} -> {train['issue_date'].max().date()}")
    print(f"Test range: {test['issue_date'].min().date()} -> {test['issue_date'].max().date()}")
    print(f"\n RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R²: {r2:.4f}")
    print(f"\n Top 3 most important features:")
    for feat, score in importance.head(3).items():
        print(f" -> {feat} ({score:.4f})")
    print(f"\n Outputs saved:")
    print(f" -> {OUTPUT_PREDICTIONS}")
    print(f" -> {OUTPUT_METRICS}")
    print(f" ->{OUTPUT_PLOT}")
    print("="*60 + "\n")
 
 
# =============================================================================
# MAIN
# =============================================================================
 
def main():
    # Step 1: Load
    df = load_data(INPUT_FILE)
 
    # Step 2: Parse dates
    df = parse_dates(df)
 
    # Step 3: Aggregate by day
    daily = aggregate_by_day(df)
 
    # Step 4: Create time series features
    daily_featured, features = create_time_series_features(daily)
 
    # Step 5: Split
    train, test, X_train, y_train, X_test, y_test = time_based_split(daily_featured, features)
 
    # Step 6: Train
    model = train_model(X_train, y_train)
 
    # Step 7: Evaluate
    y_pred, rmse, mae, r2, importance = evaluate_model(model, X_test, y_test, test, features)
 
    # Step 8: Save outputs
    save_outputs(test, y_pred, rmse, mae, r2, importance, train, features)
 
    # Step 9: Summary
    print_summary(daily, train, test, rmse, mae, r2, importance, features)
 
 
if __name__ == "__main__":
    main()
 