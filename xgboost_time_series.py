# PURPOSE:
#   Predict monthly_violation_count using XGBoost with time series features.
#
# WHY VIOLATION COUNT INSTEAD OF REVENUE?
#   Revenue (total_charges) varies wildly per ticket — a $50 speed camera
#   ticket vs a $200 street cleaning fine makes monthly revenue noisy.
#   Violation count is a direct, consistent signal: every row in our dataset
#   IS a violation, so counts are real and stable regardless of fine amount.
#
# WHY FILTER TO 2022 ONWARD?
#   The NYC API returns recent data much more densely than older data.
#   Our 1,000 rows span 2017-2025, but 2017-2021 contributes only ~10 rows
#   total — mostly 1-2 violations per month. Training on those near-zero
#   months and testing on the dense 2024-2025 period means training and test
#   look completely different, which guarantees bad predictions.
#   Filtering to 2022+ gives us a consistent data density across the full
#   train/test window (0-356 violations/month in both sets).
#
# PIPELINE:
#   ETL -> Feature Engineering -> [THIS SCRIPT] -> Dashboard
#
# INPUT  : feature_engineered_violations.csv
# OUTPUTS: xgboost_monthly_predictions.csv
#          xgboost_monthly_metrics.txt
#          xgboost_monthly_actual_vs_predicted.png
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
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

INPUT_FILE         = "feature_engineered_violations.csv"
OUTPUT_PREDICTIONS = "xgboost_monthly_predictions.csv"
OUTPUT_METRICS     = "xgboost_monthly_metrics.txt"
OUTPUT_PLOT        = "xgboost_monthly_actual_vs_predicted.png"

TRAIN_RATIO  = 0.80
TARGET_COL   = "monthly_violation_count"
LOG_TARGET   = "log_monthly_violation_count"

# Only use data from this date onward — avoids the sparse 2017-2021 period
# where the API returned almost no rows, which would make training/test
# distributions completely different.
START_DATE   = "2022-01-01"

XGBOOST_PARAMS = {
    "n_estimators":     300,
    "learning_rate":    0.05,
    "max_depth":        3,       # shallow = less overfitting on small data
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,       # requires at least 2 samples per leaf
    "random_state":     42,
    "verbosity":        0,
}


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data(filepath):
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] '{filepath}' not found.\n"
            "Run: git checkout 3-feature_engineering -- feature_engineered_violations.csv"
        )

    df = pd.read_csv(filepath, low_memory=False)
    print(f"[OK] Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# =============================================================================
# STEP 2: PARSE DATES + FILTER TO DENSE PERIOD
# =============================================================================

def parse_and_filter(df):
    print("\n" + "="*60)
    print("STEP 2: PARSE DATES + FILTER TO 2022+")
    print("="*60)

    if "issue_date" not in df.columns:
        raise KeyError("[ERROR] 'issue_date' column not found.")

    df["issue_date"] = pd.to_datetime(df["issue_date"], errors="coerce")
    df = df.dropna(subset=["issue_date"])

    before = len(df)
    df = df[df["issue_date"] >= START_DATE].copy()
    after  = len(df)

    print(f" Rows before filter: {before:,}")
    print(f" Rows after 2022+ : {after:,}  (dropped {before - after:,} pre-2022 rows)")
    print(f" Date range: {df['issue_date'].min().date()} to {df['issue_date'].max().date()}")
    return df


# =============================================================================
# STEP 3: AGGREGATE BY MONTH + CONTINUOUS CALENDAR
# =============================================================================

def aggregate_by_month(df):
    """
    Aggregate individual violation rows into monthly totals.
    Then reindex to a full monthly calendar so there are no gaps —
    a missing month means 0 violations, not a skipped time step.
    Without this, lag_1 could mean 'the previous month that had data'
    instead of 'exactly 1 calendar month ago.'
    """
    print("\n" + "="*60)
    print("STEP 3: MONTHLY AGGREGATION + CONTINUOUS CALENDAR")
    print("="*60)

    # Snap each row to the 1st of its month
    df["issue_month"] = df["issue_date"].dt.to_period("M").dt.to_timestamp()

    monthly = df.groupby("issue_month").agg(
        monthly_violation_count = ("issue_date", "count"),
        monthly_avg_fine        = ("fine_amount", "mean"),
    ).reset_index()

    monthly = monthly.rename(columns={"issue_month": "issue_date"})
    monthly = monthly.sort_values("issue_date").reset_index(drop=True)
    observed_months = len(monthly)

    # Reindex to full monthly calendar (no gaps)
    monthly = monthly.set_index("issue_date")
    full_range = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS"
    )
    monthly = monthly.reindex(full_range)
    monthly.index.name = "issue_date"

    monthly["monthly_violation_count"] = monthly["monthly_violation_count"].fillna(0)
    monthly["monthly_avg_fine"]        = monthly["monthly_avg_fine"].fillna(0)
    monthly = monthly.reset_index()

    monthly["monthly_violation_count"] = monthly["monthly_violation_count"].clip(upper=300)
    
    calendar_months = len(monthly)
    missing_added   = calendar_months - observed_months

    print(f" Observed months (with violations): {observed_months}")
    print(f" Full calendar months : {calendar_months}")
    print(f" Missing months filled with zeros: {missing_added}")
    print(f" Avg monthly violations: {monthly['monthly_violation_count'].mean():.1f}")
    print(f" Max monthly violations: {monthly['monthly_violation_count'].max():.0f}")
    print(f" Min monthly violations: {monthly['monthly_violation_count'].min():.0f}")

    return monthly


# =============================================================================
# STEP 4: FEATURE ENGINEERING
# =============================================================================

def create_features(monthly):
    """
    LAG FEATURES — model memory of past months:
      lag_1 : violations last month
      lag_3 : violations 3 months ago (one quarter)
      lag_6 : violations 6 months ago (half year)

    ROLLING FEATURES — recent trend and volatility:
      rolling_mean_3 : avg violations over past 3 months
      rolling_mean_6 : avg violations over past 6 months
      rolling_std_3  : volatility over past 3 months

    CALENDAR FEATURES:
      year, month, quarter

    NO DATA LEAKAGE: all lag/rolling use .shift(1) minimum —
    the model never sees the current month's count while training.
    """
    print("\n" + "="*60)
    print("STEP 4: FEATURE ENGINEERING")
    print("="*60)

    df = monthly.copy()

    # log1p transform: compresses the large spikes (e.g. 356 violations in one month)
    # so training doesn't fixate on outlier months. expm1() reverses it after prediction.
    df[LOG_TARGET] = np.log1p(df[TARGET_COL])
    print(f"  log1p transform: '{TARGET_COL}' -> '{LOG_TARGET}'")

    # Lag features (on log scale to match training target)
    df["lag_1_count"] = df[LOG_TARGET].shift(1)
    df["lag_3_count"] = df[LOG_TARGET].shift(3)
    df["lag_6_count"] = df[LOG_TARGET].shift(6)

    # Rolling features (.shift(1) before rolling = no leakage)
    df["rolling_mean_3_count"] = (
        df[LOG_TARGET].shift(1).rolling(window=3, min_periods=1).mean()
    )
    df["rolling_mean_6_count"] = (
        df[LOG_TARGET].shift(1).rolling(window=6, min_periods=1).mean()
    )
    df["rolling_std_3_count"] = (
        df[LOG_TARGET].shift(1).rolling(window=3, min_periods=1).std().fillna(0)
    )

    # Calendar features
    df["year"]    = df["issue_date"].dt.year
    df["month"]   = df["issue_date"].dt.month
    df["quarter"] = df["issue_date"].dt.quarter

    # Drop NaN rows from lag_6 (first 6 months)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped = before - len(df)
    print(f"  Dropped {dropped} rows for lag NaN (first {dropped} months).")
    print(f"  Usable months for modeling: {len(df)}")

    features = [
        "lag_1_count", "lag_3_count", "lag_6_count",
        "rolling_mean_3_count", "rolling_mean_6_count", "rolling_std_3_count",
        "year", "month", "quarter",
    ]
    print(f"Features ({len(features)}): {features}")
    return df, features


# =============================================================================
# STEP 5: TIME-BASED TRAIN/TEST SPLIT
# =============================================================================

def time_based_split(df, features):
    """
    First 80% of months -> training. Last 20% -> test. NO shuffle.
    Shuffling would let the model train on future months and test on past ones,
    making accuracy look good but the model useless in practice.
    """
    print("\n" + "="*60)
    print("STEP 5: TIME-BASED TRAIN/TEST SPLIT (80/20)")
    print("="*60)

    split_idx = int(len(df) * TRAIN_RATIO)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    print(f" Train: {len(train)} months  "
          f"({train['issue_date'].min().date()} to {train['issue_date'].max().date()})")
    print(f" Test:  {len(test)} months  "
          f"({test['issue_date'].min().date()} to {test['issue_date'].max().date()})")
    print(f" Train count range: {train[TARGET_COL].min():.0f} - {train[TARGET_COL].max():.0f}")
    print(f" Test  count range: {test[TARGET_COL].min():.0f} - {test[TARGET_COL].max():.0f}")

    X_train = train[features]
    y_train = train[LOG_TARGET]
    X_test  = test[features]
    y_test  = test[LOG_TARGET]

    return train, test, X_train, y_train, X_test, y_test
# =============================================================================
# STEP 6: TRAIN MODEL
# =============================================================================

def train_model(X_train, y_train):
    print("\n" + "="*60)
    print("STEP 6: TRAINING XGBOOST REGRESSOR")
    print("="*60)
    print(f" Parameters: {XGBOOST_PARAMS}")

    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train)

    print("[OK] Model trained.")
    return model


# =============================================================================
# STEP 7: EVALUATE
# =============================================================================

def evaluate_model(model, X_test, y_test, test, features):
    """
    Predict in log space, convert back with expm1(), report metrics in
    real violation counts. R² computed in log space (model's native scale).
    """
    print("\n" + "="*60)
    print("STEP 7: EVALUATION")
    print("="*60)

    y_pred_log = model.predict(X_test)
    r2 = r2_score(y_test.values, y_pred_log)

    # Convert back to violation counts
    y_pred = np.maximum(np.round(np.expm1(y_pred_log)), 0)
    y_true = np.expm1(y_test.values)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)

    print(f"\n RMSE: {rmse:.1f} violations (avg prediction error)")
    print(f"MAE: {mae:.1f} violations (avg absolute error)")
    print(f"R2: {r2:.4f} (fit in log space)")

    importance = pd.Series(
        model.feature_importances_, index=features
    ).sort_values(ascending=False)

    print(f"\n Feature Importance:")
    for feat, score in importance.items():
        bar = "+" * int(score * 40)
        print(f"  {feat:<28} {score:.4f}  {bar}")

    return y_pred, y_true, rmse, mae, r2, importance


# =============================================================================
# STEP 8: SAVE OUTPUTS
# =============================================================================

def save_outputs(test, y_pred, y_true, rmse, mae, r2, importance, train):
    print("\n" + "="*60)
    print("STEP 8: SAVING OUTPUTS")
    print("="*60)

    # 1. Predictions CSV
    pred_df = test[["issue_date"]].copy()
    pred_df["actual_violation_count"]    = y_true.astype(int)
    pred_df["predicted_violation_count"] = y_pred.astype(int)
    pred_df["error"] = (y_true - y_pred).astype(int)
    pred_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"  [1] Saved -> '{OUTPUT_PREDICTIONS}'")

    # 2. Metrics text file
    with open(OUTPUT_METRICS, "w") as f:
        f.write("XGBoost Monthly Time Series -- Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"Data window: {START_DATE} onward\n")
        f.write(f"Training months: {len(train)}\n")
        f.write(f"Test months: {len(test)}\n")
        f.write(f"Train range: {train['issue_date'].min().date()} to "
                f"{train['issue_date'].max().date()}\n")
        f.write(f"Test range: {test['issue_date'].min().date()} to "
                f"{test['issue_date'].max().date()}\n\n")
        f.write(f"RMSE: {rmse:.1f} violations\n")
        f.write(f"MAE: {mae:.1f} violations\n")
        f.write(f"R2: {r2:.4f}\n\n")
        f.write("Feature Importance:\n")
        for feat, score in importance.items():
            f.write(f"  {feat:<28} {score:.4f}\n")
        f.write("\nModel Parameters:\n")
        for k, v in XGBOOST_PARAMS.items():
            f.write(f"  {k}: {v}\n")
    print(f"  [2] Saved -> '{OUTPUT_METRICS}'")

    # 3. Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle(
        "XGBoost Monthly Time Series -- NYC Parking Violation Count",
        fontsize=14, fontweight="bold"
    )

    ax1 = axes[0]
    ax1.plot(train["issue_date"], train[TARGET_COL],
             color="#adb5bd", linewidth=1, label="Train (actual)", alpha=0.8)
    ax1.plot(test["issue_date"], y_true,
             color="#2196F3", linewidth=1.5, label="Test (actual)")
    ax1.plot(test["issue_date"], y_pred,
             color="#F44336", linewidth=1.5, linestyle="--", label="Test (predicted)")
    ax1.axvline(x=test["issue_date"].min(), color="black",
                linestyle=":", linewidth=1, alpha=0.5, label="Train/Test split")
    ax1.set_title("Full Timeline: Train + Test Actual vs Predicted")
    ax1.set_ylabel("Monthly Violation Count")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(test["issue_date"], y_true,
             color="#2196F3", linewidth=2, label="Actual",
             marker="o", markersize=6)
    ax2.plot(test["issue_date"], y_pred,
             color="#F44336", linewidth=2, linestyle="--", label="Predicted",
             marker="x", markersize=6)
    ax2.fill_between(test["issue_date"], y_true, y_pred,
                     alpha=0.15, color="#9C27B0")
    ax2.set_title(
        f"Test Period -- RMSE: {rmse:.1f}  |  MAE: {mae:.1f}  |  R2: {r2:.4f}"
    )
    ax2.set_ylabel("Monthly Violation Count")
    ax2.set_xlabel("Month")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [3] Saved -> '{OUTPUT_PLOT}'")


# =============================================================================
# STEP 9: SUMMARY
# =============================================================================

def print_summary(monthly, train, test, rmse, mae, r2, importance):
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    obs = (monthly[TARGET_COL] > 0).sum()
    print(f"  Data window: {START_DATE} onward")
    print(f"  Observed months with violations: {obs}")
    print(f"  Full calendar months (continuous) : {len(monthly)}")
    print(f"  Months after lag drop: {len(train) + len(test)}")
    print(f"  Training months: {len(train)}")
    print(f"  Test months: {len(test)}")
    print(f"  Train range: {train['issue_date'].min().date()} to {train['issue_date'].max().date()}")
    print(f"  Test range: {test['issue_date'].min().date()} to {test['issue_date'].max().date()}")
    print(f"\n  RMSE: {rmse:.1f} violations")
    print(f"  MAE: {mae:.1f} violations")
    print(f"  R2: {r2:.4f}")
    print(f"\n  Top 3 features:")
    for feat, score in importance.head(3).items():
        print(f" -> {feat}  ({score:.4f})")
    print(f"\n  Output files:")
    print(f" -> {OUTPUT_PREDICTIONS}")
    print(f" -> {OUTPUT_METRICS}")
    print(f" -> {OUTPUT_PLOT}")
    print("="*60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    df = load_data(INPUT_FILE)
    df = parse_and_filter(df)
    monthly = aggregate_by_month(df)
    monthly_feat, features = create_features(monthly)
    train, test, X_tr, y_tr, X_te, y_te = time_based_split(monthly_feat, features)
    model = train_model(X_tr, y_tr)
    y_pred, y_true, rmse, mae, r2, imp = evaluate_model(model, X_te, y_te, test, features)
    save_outputs(test, y_pred, y_true, rmse, mae, r2, imp, train)
    print_summary(monthly, train, test, rmse, mae, r2, imp)
if __name__ == "__main__":
    main()