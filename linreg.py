
# =============================================================
# linreg.py — LASSO Linear Regression
#
# WHY LASSO before deep learning:
#   LASSO is fast to train and easy to interpret.
#   If LASSO cannot beat S-Median, your features are broken
#   and there is no point training an MLP or LSTM.
#   LASSO acts as a feature validation step.
#
# WHY LASSO (L1) not Ridge (L2):
#   With 42 features, many may be irrelevant or redundant.
#   L1 regularization forces irrelevant weights exactly to
#   zero — automatic feature selection.
#   L2 shrinks all weights toward zero but never exactly
#   to zero, keeping all features active.
#
# WHY per-store models (not global):
#   The paper explicitly states: "linear regression models
#   are fitted per time series as pooling did not improve
#   the results." A global linear model cannot capture
#   store-specific intercepts without dummy variables for
#   every store — which would be 1115 extra columns.
#
# WHAT it produces:
#   linreg_results.csv — MAE and MASE per SD type
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_absolute_error

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    N_CV_FOLDS, RANDOM_SEED
)
from evaluation import (
    evaluate_by_sd_type, save_results,
    build_results_table, print_results_table
)
from preprocessing import inverse_transform_target


# =============================================================
# SECTION 1: Feature columns for linear regression
# =============================================================

# WHY a separate feature list for LASSO:
#   LASSO works best with standardised continuous features.
#   We exclude Store_enc and sd_type as raw integers
#   because their ordinal encoding is arbitrary — Store 1
#   is not "less than" Store 2 in any meaningful way.
#   We include the binary SD indicators instead.

LINREG_FEATURE_COLS = [
    # Lag features (standardised in preprocessing)
    "Sales_lag_2", "Sales_lag_3", "Sales_lag_4",
    "Sales_lag_5", "Sales_lag_6", "Sales_lag_7",
    "Sales_lag_14",
    # Rolling features
    "Sales_rolling_median", "Sales_rolling_std",
    # Calendar — cyclical encodings
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "WeekOfYear_sin", "WeekOfYear_cos",
    # Calendar — binary/ordinal
    "IsWeekend", "IsStateHoliday",
    "SchoolHoliday", "Promo",
    # Store features
    "StoreType_enc", "Assortment_enc",
    "Promo2", "CompetitionDistance_log",
    # SD type indicators
    "IsSD1", "IsSD2", "IsSD3", "IsSD4",
    # SD-specific features (paper's core contribution)
    "sd_level", "sd_abs_change", "sd_rel_change",
    "sd_rel_change_storetype",
    "sd_level_other", "sd_abs_change_other",
    "sd_rel_change_other", "sd_rel_change_storetype_other",
]


# =============================================================
# SECTION 2: Single store LASSO training
# =============================================================

def train_lasso_for_store(store_train, store_test,
                           feature_cols, target_scaler):
    """
    Trains a LASSO model for a single store and returns
    predictions in real sales units.

    WHY LassoCV:
      Automatically selects the best regularization
      strength (alpha/lambda) using cross-validation.
      No manual tuning required.

    WHY we predict on scaled target then inverse transform:
      The target scaler was fitted on all training data.
      We must use the same scaler for consistency across
      stores. Fitting a separate scaler per store would
      give different scales and make MASE incomparable.

    Returns: (y_pred_real, y_true_real, sd_types)
    """
    # Get available feature columns
    available = [c for c in feature_cols
                 if c in store_train.columns]

    X_train = store_train[available].values
    X_test  = store_test[available].values
    y_train = store_train["Sales_scaled"].values
    y_test  = store_test["Sales_scaled"].values

    # Skip if not enough training data
    if len(X_train) < 30:
        return None, None, None

    # Fit LassoCV — automatically finds best alpha
    # cv=5 for speed (paper uses 10 but 5 is sufficient here)
    model = LassoCV(
        cv=5,
        max_iter=5000,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    try:
        model.fit(X_train, y_train)
    except Exception:
        return None, None, None

    # Predict on test set (scaled)
    y_pred_scaled = model.predict(X_test)

    # Clip to valid scale range before inverse transform
    y_pred_scaled = np.clip(y_pred_scaled, -0.5, 0.5)

    # Inverse transform to real sales units
    y_pred_real = inverse_transform_target(
        y_pred_scaled, target_scaler
    )
    y_true_real = inverse_transform_target(
        y_test, target_scaler
    )

    sd_types = store_test["sd_type"].values

    return y_pred_real, y_true_real, sd_types


# =============================================================
# SECTION 3: Run LASSO on all stores
# =============================================================

def run_linreg(sample_stores=None):
    """
    Trains LASSO per store and evaluates results.

    Args:
        sample_stores: list of Store IDs (None = all)

    Returns: results dict from evaluate_by_sd_type
    """
    print("=" * 50)
    print("Running LASSO Linear Regression")
    print("=" * 50)

    # Load data
    train_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet")
    )
    test_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

    # Load target scaler
    with open(os.path.join(
        MODELS_DIR, "target_scaler.pkl"), "rb"
    ) as f:
        target_scaler = pickle.load(f)

    # Subset stores if requested
    if sample_stores is not None:
        train_df = train_df[
            train_df["Store"].isin(sample_stores)
        ].copy()
        test_df = test_df[
            test_df["Store"].isin(sample_stores)
        ].copy()
        print(f"Running on {len(sample_stores)} stores")
    else:
        print(f"Running on all "
              f"{train_df['Store'].nunique()} stores")

    print(f"Train: {len(train_df):,} rows")
    print(f"Test : {len(test_df):,} rows")

    # Get feature columns available in data
    feature_cols = [
        c for c in LINREG_FEATURE_COLS
        if c in train_df.columns
    ]
    print(f"Features: {len(feature_cols)}")

    # Train per store and collect predictions
    all_y_true  = []
    all_y_pred  = []
    all_sd_types = []
    stores_done  = 0
    stores_skip  = 0

    stores = train_df["Store"].unique()

    for store in stores:
        store_train = train_df[
            train_df["Store"] == store
        ].copy()
        store_test = test_df[
            test_df["Store"] == store
        ].copy()

        if len(store_test) == 0:
            stores_skip += 1
            continue

        y_pred, y_true, sd_types = train_lasso_for_store(
            store_train, store_test,
            feature_cols, target_scaler
        )

        if y_pred is None:
            stores_skip += 1
            continue

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_sd_types.extend(sd_types)
        stores_done += 1

        if stores_done % 10 == 0:
            print(f"  Processed {stores_done} stores...")

    print(f"\nCompleted: {stores_done} stores "
          f"({stores_skip} skipped)")

    # Evaluate
    all_y_true   = np.array(all_y_true)
    all_y_pred   = np.array(all_y_pred)
    all_sd_types = np.array(all_sd_types)

    results = evaluate_by_sd_type(
        y_true   = all_y_true,
        y_pred   = all_y_pred,
        sd_types = all_sd_types,
        train_df = train_df
    )

    save_results(results, "linreg")
    print_results_table(
        build_results_table({"LIN-REG": results}),
        "LASSO Results"
    )

    return results, all_y_true, all_y_pred, all_sd_types


# =============================================================
# SECTION 4: Verification
# =============================================================

def verify_linreg(results, baseline_results):
    """
    Verifies LASSO results against expected patterns.

    Key check: LASSO must beat S-Median on overall MAE.
    If it does not, features are not carrying enough signal.
    """
    print("=" * 50)
    print("Verification — LASSO")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # LASSO must beat S-Median overall
    lasso_mae   = results["Overall"]["MAE"]
    smedian_mae = baseline_results["S-Median"]["Overall"]["MAE"]
    checks["LASSO MAE < S-Median MAE overall"] = (
        lasso_mae < smedian_mae
    )

    # LASSO must beat S-Naive on SD2 and SD3
    # (paper shows linear models improve on neighboring days)
    for sd in ["SD2", "SD3"]:
        lasso_sd   = results[sd]["MAE"]
        snaive_sd  = baseline_results["S-Naive"][sd]["MAE"]
        if not np.isnan(lasso_sd):
            checks[f"LASSO {sd} MAE < S-Naive {sd} MAE"] = (
                lasso_sd < snaive_sd
            )

    # All MAE values positive
    for sd_label, metrics in results.items():
        mae = metrics.get("MAE", np.nan)
        if not np.isnan(mae):
            checks[f"LASSO {sd_label} MAE > 0"] = mae > 0

    # LASSO MASE should be < 0.9 overall
    # (should meaningfully beat seasonal naive)
    checks["LASSO MASE < 0.9"] = (
        results["Overall"]["MASE"] < 0.9
    )

    # Print results
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    # Print relative performance vs S-Median
    rel = lasso_mae / smedian_mae
    print(f"\n  LASSO MAE relative to S-Median: {rel:.4f}")
    print(f"  (< 1.0 means LASSO beats S-Median)")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: lgbm_model.py")
    else:
        print("Some checks FAILED.")

    return all_pass
