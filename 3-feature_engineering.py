# =============================================================================
# 3-feature_engineering.py
# NYC Open Parking and Camera Violations — Feature Engineering Script
# Big Data Class Project
# =============================================================================
# PURPOSE:
#   This script is Step 3 in the data pipeline:
#   Extract -> Transform/Load (ETL) -> [THIS SCRIPT] Feature Engineering -> Modeling
#
#   It takes the already-loaded violations data (from CSV / PostgreSQL export),
#   cleans it for modeling purposes, engineers new features, and saves output files
#   ready for K-Means clustering, Random Forest classification, and XGBoost/time-series.
# =============================================================================

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
import os

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Input file — the CSV exported from PostgreSQL (violations_transf table)
INPUT_FILE = "DATA_2026-04-36.csv"

# Output files
OUTPUT_CLEANED     = "cleaned_violations.csv"              # after cleaning, before feature engineering
OUTPUT_ENGINEERED  = "feature_engineered_violations.csv"   # full engineered dataset
OUTPUT_MODEL_READY = "model_ready_features.csv"            # for Random Forest / XGBoost
OUTPUT_CLUSTERING  = "clustering_features.csv"             # for K-Means

# Z-score threshold for outlier detection (standard choice: values beyond ±3 std)
Z_SCORE_THRESHOLD = 3

# Numeric columns that contain dollar amounts
NUMERIC_COLS = [
    "fine_amount", "penalty_amount", "interest_amount",
    "reduction_amount", "payment_amount", "amount_due"
]

# Date columns
DATE_COLS = ["issue_date", "judgment_entry_date"]

# Categorical columns we will encode for modeling
CATEGORICAL_COLS = ["county", "state", "license_type", "issuing_agency", "violation_status"]

# Key financial + time fields — rows missing ALL of these will be dropped
KEY_FIELDS = ["fine_amount", "penalty_amount", "payment_amount", "amount_due", "issue_date"]

# =============================================================================
# TRACKING VARIABLES (for summary report at the end)
# =============================================================================
original_shape    = None
cleaned_shape     = None
final_shape       = None
missing_handled   = 0
outliers_capped   = 0
new_features      = []


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

def load_data(filepath):
    """
    Load the violations CSV into a Pandas DataFrame.
    Handles common file errors gracefully.
    """
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] File not found: '{filepath}'\n"
            "Make sure DATA_2026-04-36.csv is in the same folder as this script."
        )

    try:
        df = pd.read_csv(filepath, dtype=str, low_memory=False)
        print(f"[OK] Data loaded successfully from: {filepath}")
        print(f"     Rows: {df.shape[0]:,}  |  Columns: {df.shape[1]}")
        return df
    except Exception as e:
        raise RuntimeError(f"[ERROR] Could not read file: {e}")


# =============================================================================
# STEP 2: INITIAL DATA INSPECTION
# =============================================================================

def inspect_data(df):
    print("\n" + "="*60)
    print("STEP 2: INITIAL DATA INSPECTION")
    print("="*60)

    print(f"\n[Shape]  {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n[Columns]")
    for col in df.columns:
        print(f"  - {col}")

    print("\n[Missing Values per Column]")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
    missing_df = missing_df[missing_df["Missing Count"] > 0]
    if missing_df.empty:
        print("No missing values found.")
    else:
        print(missing_df.to_string())

    print("\n[Data Types]")
    print(df.dtypes.to_string())

    # Also show a sample of county values because they are inconsistent
    if "county" in df.columns:
        print("\n[Unique County Values (raw)]")
        print(" ", df["county"].dropna().unique())


# =============================================================================
# STEP 3: DATA CLEANING
# =============================================================================

def clean_data(df):
    """
    Perform cleaning to prepare the data for feature engineering.
    This is NOT a repeat of ingestion cleaning — it focuses on:
      - Removing completely unusable rows
      - Fixing data types (numeric, datetime)
      - Imputing missing values
      - Standardizing inconsistent categorical values
    """
    global missing_handled

    print("\n" + "="*60)
    print("STEP 3: DATA CLEANING")
    print("="*60)

    # ---- 3a. Strip whitespace from all string columns ----
    print("\n[3a] Stripping whitespace from string columns...")
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # ---- 3b. Replace common "empty-looking" strings with NaN ----
    # Some fields come through as empty strings or dashes instead of true NaN
    print("[3b] Replacing empty strings / placeholders with NaN...")
    df.replace(["", " ", "N/A", "n/a", "NA", "null", "NULL", "none", "None", "-"], np.nan, inplace=True)

    # ---- 3c. Drop rows where ALL key financial + time fields are missing ----
    # These rows contain no usable information for modeling
    print("[3c] Dropping rows where ALL key fields are missing...")
    key_cols_present = [c for c in KEY_FIELDS if c in df.columns]
    before = len(df)
    df.dropna(subset=key_cols_present, how="all", inplace=True)
    dropped = before - len(df)
    print(f"     Dropped {dropped:,} completely empty rows. Remaining: {len(df):,}")

    # ---- 3d. Convert numeric columns safely ----
    print("[3d] Converting numeric columns to float...")
    for col in NUMERIC_COLS:
        if col in df.columns:
            before_nulls = df[col].isnull().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # errors="coerce" turns anything that can't be a number into NaN
            after_nulls = df[col].isnull().sum()
            new_nulls = after_nulls - before_nulls
            if new_nulls > 0:
                print(f"     '{col}': {new_nulls} non-numeric values coerced to NaN")

    # ---- 3e. Convert date columns safely ----
    print("[3e] Converting date columns to datetime...")
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            nulls = df[col].isnull().sum()
            print(f"     '{col}': {nulls:,} unparseable dates set to NaT")

    # ---- 3f. Standardize county values ----
    # Raw data mixes abbreviations and full names for the same NYC borough.
    # We map everything to a clean, consistent borough name.
    print("[3f] Standardizing inconsistent county values...")
    county_map = {
        # Manhattan
        "NY": "Manhattan",   "MN": "Manhattan",   "NEW YORK": "Manhattan",
        "MANHATTAN": "Manhattan",

        # Brooklyn
        "K":  "Brooklyn",    "BK": "Brooklyn",    "KINGS": "Brooklyn",
        "Kings": "Brooklyn", "BROOKLYN": "Brooklyn",

        # Queens
        "QN": "Queens",      "QU": "Queens",      "QUEENS": "Queens",
        "QNS": "Queens",     "Q": "Queens",

        # Bronx
        "BX": "Bronx",       "BRONX": "Bronx",

        # Staten Island
        "ST": "Staten Island", "R": "Staten Island", "RICHMOND": "Staten Island",
        "STATEN ISLAND": "Staten Island",
    }
    if "county" in df.columns:
        # Upper-case before mapping so partial matches (e.g. "kings") are caught
        df["county"] = df["county"].str.upper().map(
            {k.upper(): v for k, v in county_map.items()}
        )
        # Values not found in the map become NaN -> will be filled as "Unknown" below
        still_unknown = df["county"].isnull().sum()
        print(f"     {still_unknown:,} county values could not be mapped -> will be 'Unknown'")

    # ---- 3g. Standardize other categorical columns (upper-case trim) ----
    print("[3g] Standardizing other categorical columns...")
    for col in ["state", "license_type", "issuing_agency", "violation_status", "violation"]:
        if col in df.columns:
            df[col] = df[col].str.upper().str.strip()

    # ---- 3h. Handle missing values ----
    print("[3h] Imputing missing values...")

    # Numeric -> fill with column mean
    # Mean imputation is simple, preserves column statistics, and keeps row count intact.
    for col in NUMERIC_COLS:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                col_mean = df[col].mean()
                df[col].fillna(col_mean, inplace=True)
                missing_handled += null_count
                print(f"     '{col}': filled {null_count:,} NaNs with mean ({col_mean:.2f})")

    # Categorical -> fill with "Unknown"
    for col in CATEGORICAL_COLS + ["violation", "violation_time"]:
        if col in df.columns:
            null_count = df[col].isnull().sum()
            if null_count > 0:
                df[col].fillna("UNKNOWN", inplace=True)
                missing_handled += null_count
                print(f"     '{col}': filled {null_count:,} NaNs with 'UNKNOWN'")

    # Date columns — we can't impute dates meaningfully, leave as NaT
    # (time features derived from these will simply be NaN for those rows)

    # ---- 3i. Remove rows with negative fine_amount (data error) ----
    if "fine_amount" in df.columns:
        neg_count = (df["fine_amount"] < 0).sum()
        if neg_count > 0:
            df = df[df["fine_amount"] >= 0]
            print(f"[3i] Removed {neg_count:,} rows with negative fine_amount (data errors)")

    print(f"\n[Cleaning done]  Rows remaining: {len(df):,}")
    return df


# =============================================================================
# STEP 4: OUTLIER HANDLING (Z-SCORE CAPPING / WINSORIZATION)
# =============================================================================

def handle_outliers(df):
    """
    Detect and CAP (winsorize) outliers in numeric columns using Z-scores.

    WHY Z-SCORE?
        The Z-score measures how many standard deviations a value is from the mean.
        Values beyond ±3 standard deviations are considered extreme outliers.
        This is a standard, interpretable method and works well for financial data.

    WHY CAP (WINSORIZE) INSTEAD OF DROP?
        Dropping outlier rows loses real data — a $1,000 fine IS a real fine.
        Capping replaces extreme values with the boundary value (the 1st or 99th
        percentile), so the row stays in the dataset but the extreme value no
        longer distorts model training. This is especially important for
        K-Means (sensitive to outliers) and linear models.
    """
    global outliers_capped

    print("\n" + "="*60)
    print("STEP 4: OUTLIER HANDLING (Z-SCORE CAPPING)")
    print("="*60)

    numeric_present = [c for c in NUMERIC_COLS if c in df.columns]

    for col in numeric_present:
        col_data = df[col].dropna()

        if len(col_data) == 0:
            continue

        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[col].fillna(df[col].mean())))

        # Count how many values exceed the threshold
        outlier_mask = z_scores > Z_SCORE_THRESHOLD
        count_outliers = outlier_mask.sum()

        if count_outliers > 0:
            # Cap at 1st and 99th percentiles (winsorization)
            lower_cap = df[col].quantile(0.01)
            upper_cap = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_cap, upper=upper_cap)
            outliers_capped += count_outliers
            print(f"  '{col}': {count_outliers:,} outliers capped "
                  f"[{lower_cap:.2f}, {upper_cap:.2f}]")
        else:
            print(f"  '{col}': no outliers detected")

    return df


# =============================================================================
# STEP 5: FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Create all new features for modeling.
    Organized into sub-sections: Time, Financial, Target, Categorical Encoding,
    and Clustering feature selection.
    """
    global new_features

    print("\n" + "="*60)
    print("STEP 5: FEATURE ENGINEERING")
    print("="*60)

    # -------------------------------------------------------
    # 5A. TIME FEATURES  (from issue_date)
    # -------------------------------------------------------
    # These help XGBoost/time-series models learn seasonal patterns
    # (e.g., more tickets on Mondays after street cleaning)
    # and help K-Means find time-based violation clusters.
    print("\n[5A] Extracting time features from issue_date...")

    if "issue_date" in df.columns:
        df["issue_year"]      = df["issue_date"].dt.year
        df["issue_month"]     = df["issue_date"].dt.month
        df["issue_day"]       = df["issue_date"].dt.day
        df["issue_dayofweek"] = df["issue_date"].dt.dayofweek   # 0=Monday, 6=Sunday
        df["issue_quarter"]   = df["issue_date"].dt.quarter
        df["is_weekend"]      = df["issue_dayofweek"].apply(
            lambda x: 1 if pd.notna(x) and x >= 5 else 0
        )
        time_feats = ["issue_year","issue_month","issue_day",
                      "issue_dayofweek","issue_quarter","is_weekend"]
        new_features.extend(time_feats)
        print(f"  Created: {time_feats}")
    else:
        print("[SKIP] 'issue_date' column not found.")

    # -------------------------------------------------------
    # 5B. FINANCIAL FEATURES
    # -------------------------------------------------------
    # These directly support NYC DOF revenue analysis.
    print("\n[5B] Creating financial features...")

    # total_charges: the full amount the violator was originally charged
    if all(c in df.columns for c in ["fine_amount","penalty_amount","interest_amount"]):
        df["total_charges"] = (
            df["fine_amount"] + df["penalty_amount"] + df["interest_amount"]
        )
        new_features.append("total_charges")
        print("Created: total_charges = fine + penalty + interest")

    # net_paid: how much was actually paid after any reduction
    if all(c in df.columns for c in ["payment_amount","reduction_amount"]):
        df["net_paid"] = df["payment_amount"] - df["reduction_amount"]
        new_features.append("net_paid")
        print("Created: net_paid = payment - reduction")

    # unpaid_balance_ratio: proportion of charges still owed
    # (0 = fully paid, 1 = nothing paid, >1 shouldn't happen but we guard for it)
    if all(c in df.columns for c in ["amount_due","total_charges"]):
        df["unpaid_balance_ratio"] = df.apply(
            lambda row: row["amount_due"] / row["total_charges"]
            if pd.notna(row["total_charges"]) and row["total_charges"] > 0
            else 0.0,
            axis=1
        )
        new_features.append("unpaid_balance_ratio")
        print("Created: unpaid_balance_ratio = amount_due / total_charges")

    # -------------------------------------------------------
    # 5C. CLASSIFICATION TARGET VARIABLE
    # -------------------------------------------------------
    # high_fine_flag: binary label for Random Forest classification.
    # "Is this violation above the median fine?" (1 = yes, 0 = no)
    print("\n[5C] Creating classification target: high_fine_flag...")

    if "fine_amount" in df.columns:
        median_fine = df["fine_amount"].median()
        df["high_fine_flag"] = (df["fine_amount"] > median_fine).astype(int)
        new_features.append("high_fine_flag")
        print(f"  Created: high_fine_flag  (median fine = ${median_fine:.2f})")
        print(f"  Distribution: {df['high_fine_flag'].value_counts().to_dict()}")
    else:
        print("[SKIP] 'fine_amount' column not found.")

    # -------------------------------------------------------
    # 5D. CATEGORICAL ENCODING
    # -------------------------------------------------------
    # Machine learning models cannot use raw text strings —
    # we must convert them to numbers.
    # We use Label Encoding here (assigns each unique value an integer).
    # One-hot encoding would work too but creates many extra columns;
    # label encoding is simpler and fine for tree-based models.
    print("\n[5D] Encoding categorical columns (Label Encoding)...")

    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            encoded_col = col + "_encoded"
            # Fill any remaining NaN just in case
            df[col] = df[col].fillna("UNKNOWN")
            df[encoded_col] = le.fit_transform(df[col].astype(str))
            new_features.append(encoded_col)
            print(f"  '{col}' -> '{encoded_col}'  ({df[col].nunique()} unique values)")
        else:
            print(f"  [SKIP] '{col}' not found.")

    print(f"\n[Feature Engineering complete]  {len(new_features)} new features created.")
    return df


# =============================================================================
# STEP 6: BUILD OUTPUT DATAFRAMES AND SAVE FILES
# =============================================================================

def build_and_save_outputs(df):
    """
    Create three output files:
      1. cleaned_violations.csv     — clean data before feature engineering
      2. feature_engineered_violations.csv  — full dataset with all new features
      3. model_ready_features.csv   — cleaned up for RF / XGBoost (no IDs, no raw dates)
      4. clustering_features.csv    — numeric-only subset for K-Means
    """
    print("\n" + "="*60)
    print("STEP 6: SAVING OUTPUT FILES")
    print("="*60)

    # ---- Output 1: Cleaned data (after cleaning + outlier handling, BEFORE new features) ----
    # Saves a snapshot of the data in its clean state so teammates or auditors
    # can inspect what the data looked like before feature columns were added.
    cleaned_cols = [c for c in df.columns if c not in new_features]
    df[cleaned_cols].to_csv(OUTPUT_CLEANED, index=False)
    print(f"[1] Saved cleaned dataset -> '{OUTPUT_CLEANED}'  ({len(df):,} rows, {len(cleaned_cols)} columns)")
    
    # ---- Output 2: Full engineered dataset ----
    df.to_csv(OUTPUT_ENGINEERED, index=False)
    print(f"[2] Saved full engineered dataset -> '{OUTPUT_ENGINEERED}'  ({len(df):,} rows)")

    # ---- Output 3: Model-ready features (for Random Forest / XGBoost) ----
    # Drop high-cardinality ID columns (plate, summons_number) — these are
    # identifiers, not predictors, and would cause the model to memorize, not learn.
    # Also drop raw date objects (we already extracted year/month/day etc.)
    # and raw categorical columns (we have the encoded versions).
    drop_for_model = (
        ["plate", "summons_number", "violation_time"]      # IDs / raw time string
        + DATE_COLS                                         # raw datetime objects
        + CATEGORICAL_COLS                                  # raw strings (encoded versions kept)
    )
    drop_for_model = [c for c in drop_for_model if c in df.columns]
    model_df = df.drop(columns=drop_for_model, errors="ignore")

    # Keep only numeric columns for clean model input
    model_df = model_df.select_dtypes(include=[np.number])
    model_df.to_csv(OUTPUT_MODEL_READY, index=False)
    print(f"[3] Saved model-ready features -> '{OUTPUT_MODEL_READY}'  "
          f"({model_df.shape[1]} features, {len(model_df):,} rows)")
    print(f"    Columns: {list(model_df.columns)}")

    # ---- Output 4: Clustering features (for K-Means) ----
    # K-Means needs purely numeric, non-NaN data.
    # We select the most meaningful numeric features for finding violation clusters.
    clustering_candidates = [
        "fine_amount", "penalty_amount", "interest_amount",
        "payment_amount", "amount_due",
        "total_charges", "net_paid", "unpaid_balance_ratio",
        "issue_year", "issue_month", "issue_day",
        "issue_dayofweek", "issue_quarter", "is_weekend",
        "county_encoded", "issuing_agency_encoded",
        "violation_status_encoded", "license_type_encoded",
    ]
    clustering_cols = [c for c in clustering_candidates if c in df.columns]
    cluster_df = df[clustering_cols].dropna()
    cluster_df.to_csv(OUTPUT_CLUSTERING, index=False)
    print(f"[4] Saved clustering features -> '{OUTPUT_CLUSTERING}'  "
          f"({cluster_df.shape[1]} features, {len(cluster_df):,} rows)")
    print(f"    Columns: {list(cluster_df.columns)}")

    return model_df, cluster_df


# =============================================================================
# STEP 7: SUMMARY REPORT
# =============================================================================

def print_summary(model_df, cluster_df):
    """Print a clean summary of everything that was done."""

    print("\n" + "="*60)
    print("STEP 7: SUMMARY REPORT")
    print("="*60)

    print(f"\n  Original shape: {original_shape[0]:,} rows x {original_shape[1]} columns")
    print(f"  After cleaning: {cleaned_shape[0]:,} rows x {cleaned_shape[1]} columns")
    print(f"  Final engineered shape: {final_shape[0]:,} rows x {final_shape[1]} columns")
    print(f"\n  Missing values handled: {missing_handled:,}")
    print(f"  Outliers capped: {outliers_capped:,}")
    print(f"\n  New features created ({len(new_features)}):")
    for feat in new_features:
        print(f"    + {feat}")

    print(f"\n  Model-ready features: {model_df.shape[1]} columns")
    print(f"  Clustering features: {cluster_df.shape[1]} columns")
    print("\n  Output files saved:")
    print(f" -> {OUTPUT_ENGINEERED}")
    print(f" -> {OUTPUT_MODEL_READY}")
    print(f" -> {OUTPUT_CLUSTERING}")
    print("\n" + "="*60)
    print("Feature engineering complete!")
    print("="*60 + "\n")


# =============================================================================
# MAIN — RUN ALL STEPS IN ORDER
# =============================================================================

def main():
    global original_shape, cleaned_shape, final_shape

    # STEP 1: Load
    df = load_data(INPUT_FILE)
    original_shape = df.shape

    # STEP 2: Inspect (prints info, doesn't modify df)
    inspect_data(df)

    # STEP 3: Clean
    df = clean_data(df)
    cleaned_shape = df.shape

    # STEP 4: Outlier handling
    df = handle_outliers(df)

    # STEP 5: Feature engineering
    df = engineer_features(df)
    final_shape = df.shape

    # STEP 6: Save outputs
    model_df, cluster_df = build_and_save_outputs(df)

    # STEP 7: Summary
    print_summary(model_df, cluster_df)


if __name__ == "__main__":
    main()