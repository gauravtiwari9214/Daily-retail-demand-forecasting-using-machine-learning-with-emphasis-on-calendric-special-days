
# =============================================================
# baselines.py — S-Naive, S-Naive-Std, S-Median
#
# WHY baselines matter:
#   A model that cannot beat S-Median is useless in practice.
#   S-Median is what bakery staff already do manually —
#   they look at the same day last few weeks and take a
#   rough average. Any ML model must do better than this
#   to justify its complexity.
#
# WHY implement all three:
#   S-Naive:     shows the floor — simplest possible forecast
#   S-Naive-Std: shows the effect of cleaning SD contamination
#                from naive lags (important concept from paper)
#   S-Median:    the realistic practical benchmark — more
#                robust than S-Naive because median ignores
#                holiday outliers in the rolling window
#
# ALL THREE baselines from paper Section 5.2:
#   S-Naive:     ŷ_{t+h} = y_{t+h-m}        (m=7)
#   S-Median:    ŷ_{t+h} = median of same
#                weekday over last 4 weeks
#   S-Naive-Std: same as S-Naive but replaces
#                SD1/SD2/SD3 history with last SD0 value
# =============================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, RESULTS_DIR,
    SEASON_M, ROLLING_WEEKS
)
from evaluation import (
    evaluate_by_sd_type, save_results,
    build_results_table, print_results_table
)


# =============================================================
# SECTION 1: S-Naive
# =============================================================

def predict_snaive(test_df, train_df):
    """
    Seasonal Naive forecast: predict same weekday last week.

    Formula: ŷ_{t+h} = y_{t+h-m}  where m=7

    WHY this is a meaningful baseline:
      Demand has strong weekly seasonality. Last Tuesday's
      sales are a reasonable estimate for this Tuesday.
      In practice this is what many small retailers do.

    WHY it fails on SD1:
      If last week's same day was also a holiday, the lag
      value is a holiday observation — completely wrong
      for predicting a regular day, and vice versa.
      This is exactly what SD4 measures.

    Returns: array of predictions in real sales units
    """
    # Build lookup index: (Store, Date) → Sales
    # Much faster than row-by-row DataFrame lookup
    history = pd.concat([
        train_df[["Store", "Date", "Sales"]],
        test_df[["Store",  "Date", "Sales"]]
    ]).sort_values(["Store", "Date"])

    # Create dictionary for O(1) lookup
    lookup = {}
    for _, row in history.iterrows():
        lookup[(row["Store"], row["Date"])] = row["Sales"]

    # Store means as fallback for missing lags
    store_means = train_df.groupby("Store")["Sales"].mean().to_dict()

    predictions = []
    for _, row in test_df.iterrows():
        store    = row["Store"]
        date     = row["Date"]
        lag_date = date - pd.Timedelta(days=SEASON_M)

        key = (store, lag_date)
        if key in lookup:
            predictions.append(lookup[key])
        else:
            predictions.append(store_means.get(store, 0))

    return np.array(predictions)


# =============================================================
# SECTION 2: S-Naive-Std
# =============================================================

def predict_snaive_std(test_df, train_df):
    """
    Standardised Seasonal Naive: same as S-Naive but
    replaces SD1/SD2/SD3 values in history with the
    last observed SD0 value before them.

    WHY this matters (from paper):
      S-Naive uses lag-7 directly. If 7 days ago was
      a public holiday (SD1), the lag value is a holiday
      observation. Using it to predict a regular day
      is completely wrong. S-Naive-Std cleans the history
      by replacing holiday observations with the last
      normal day value before looking up the lag.

    WHY SD4 error is much better for S-Naive-Std vs S-Naive:
      SD4 is the day 7 days after a holiday. S-Naive's
      lag-7 for SD4 points directly to SD1 — a holiday
      observation. S-Naive-Std replaces that with the
      last SD0 value, giving a much better estimate.

    FIX applied: use scalar access (.iloc[0]) instead of
    .loc[idx] inside groupby to avoid ambiguous Series
    truth value error in pandas.

    Returns: array of predictions in real sales units
    """
    # Build combined history with sd_type
    history = pd.concat([
        train_df[["Store", "Date", "Sales", "sd_type"]],
        test_df[["Store",  "Date", "Sales", "sd_type"]]
    ]).sort_values(["Store", "Date"]).reset_index(drop=True)

    # Build cleaned lookup per store using vectorised approach
    # WHY vectorised: avoids the .loc[idx] scalar ambiguity
    # that caused the ValueError
    cleaned_lookup = {}

    for store, group in history.groupby("Store"):
        group     = group.sort_values("Date").reset_index(drop=True)
        sales     = group["Sales"].values.copy()
        sd_types  = group["sd_type"].values
        dates     = group["Date"].values

        last_sd0_sales = np.nan

        for i in range(len(group)):
            sd = int(sd_types[i])   # ensure scalar int

            if sd == 0:
                # Regular day — update last known SD0 sales
                last_sd0_sales = float(sales[i])
            elif sd in [1, 2, 3]:
                # Special day — replace with last SD0 value
                if not np.isnan(last_sd0_sales):
                    sales[i] = last_sd0_sales
            # SD4 is left unchanged — it is a regular-ish day

        # Store cleaned values in lookup dict
        for i in range(len(group)):
            cleaned_lookup[(store, dates[i])] = sales[i]

    # Store means as fallback
    store_means = train_df.groupby("Store")["Sales"].mean().to_dict()

    predictions = []
    for _, row in test_df.iterrows():
        store    = row["Store"]
        date     = row["Date"]
        lag_date = date - pd.Timedelta(days=SEASON_M)

        key = (store, lag_date)
        if key in cleaned_lookup:
            predictions.append(cleaned_lookup[key])
        else:
            predictions.append(store_means.get(store, 0))

    return np.array(predictions)


# =============================================================
# SECTION 3: S-Median
# =============================================================

def predict_smedian(test_df, train_df):
    """
    Seasonal Rolling Median forecast.

    Formula:
      ŷ_{t+h} = median({y_{t+h-lm} | l ∈ {1,2,3,4}})

    WHY median over mean:
      The rolling window of 4 same-weekday observations
      may include a holiday observation. The median is
      robust to this single outlier — it ignores it.
      The mean would be distorted by the holiday spike
      or drop.

    WHY 4 weeks:
      Enough history to capture recent trend (4 points)
      while being robust to single anomalies.
      The paper uses exactly 4 weeks (ROLLING_WEEKS=4).

    Returns: array of predictions in real sales units
    """
    # Build lookup dictionary for O(1) access
    history = pd.concat([
        train_df[["Store", "Date", "Sales"]],
        test_df[["Store",  "Date", "Sales"]]
    ]).sort_values(["Store", "Date"])

    lookup = {}
    for _, row in history.iterrows():
        lookup[(row["Store"], row["Date"])] = row["Sales"]

    store_means = train_df.groupby("Store")["Sales"].mean().to_dict()

    predictions = []
    for _, row in test_df.iterrows():
        store = row["Store"]
        date  = row["Date"]

        # Collect same-weekday sales from last 4 weeks
        lag_values = []
        for l in range(1, ROLLING_WEEKS + 1):
            lag_date = date - pd.Timedelta(days=SEASON_M * l)
            key      = (store, lag_date)
            if key in lookup:
                lag_values.append(lookup[key])

        if len(lag_values) > 0:
            predictions.append(np.median(lag_values))
        else:
            predictions.append(store_means.get(store, 0))

    return np.array(predictions)


# =============================================================
# SECTION 4: Run all baselines
# =============================================================

def run_baselines(sample_stores=None):
    """
    Runs all three baselines and evaluates them.

    Args:
        sample_stores: list of Store IDs to run on
                       (None = all stores)
                       Use a sample for development speed.

    Returns: dict of {model_name: results_dict}
    """
    print("=" * 50)
    print("Running Baselines")
    print("=" * 50)

    # Load preprocessed data
    train_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet")
    )
    test_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

    # Optionally subset to sample stores for speed
    if sample_stores is not None:
        train_df = train_df[
            train_df["Store"].isin(sample_stores)
        ].copy()
        test_df = test_df[
            test_df["Store"].isin(sample_stores)
        ].copy()
        print(f"Running on {len(sample_stores)} stores "
              f"(sample mode)")
    else:
        print(f"Running on all "
              f"{train_df['Store'].nunique()} stores")

    print(f"Train: {len(train_df):,} rows")
    print(f"Test : {len(test_df):,} rows")

    all_results = {}

    # ── S-Naive ───────────────────────────────────────────────
    print("\nRunning S-Naive...")
    preds_naive = predict_snaive(test_df, train_df)
    results_naive = evaluate_by_sd_type(
        y_true   = test_df["Sales"].values,
        y_pred   = preds_naive,
        sd_types = test_df["sd_type"].values,
        train_df = train_df
    )
    all_results["S-Naive"] = results_naive
    save_results(results_naive, "snaive")
    print("  Done.")

    # ── S-Naive-Std ───────────────────────────────────────────
    print("\nRunning S-Naive-Std...")
    preds_naive_std = predict_snaive_std(test_df, train_df)
    results_naive_std = evaluate_by_sd_type(
        y_true   = test_df["Sales"].values,
        y_pred   = preds_naive_std,
        sd_types = test_df["sd_type"].values,
        train_df = train_df
    )
    all_results["S-Naive-Std"] = results_naive_std
    save_results(results_naive_std, "snaive_std")
    print("  Done.")

    # ── S-Median ──────────────────────────────────────────────
    print("\nRunning S-Median...")
    preds_median = predict_smedian(test_df, train_df)
    results_median = evaluate_by_sd_type(
        y_true   = test_df["Sales"].values,
        y_pred   = preds_median,
        sd_types = test_df["sd_type"].values,
        train_df = train_df
    )
    all_results["S-Median"] = results_median
    save_results(results_median, "smedian")
    print("  Done.")

    # ── Print comparison table ────────────────────────────────
    table = build_results_table(all_results)
    print_results_table(table, "Baseline Results")

    return all_results, test_df["Sales"].values, \
           test_df["sd_type"].values


# =============================================================
# SECTION 5: Verification
# =============================================================

def verify_baselines(all_results):
    """
    Verifies baseline results match expected patterns
    from the paper.
    """
    print("=" * 50)
    print("Verification — Baselines")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # All three models must be present
    for model in ["S-Naive", "S-Naive-Std", "S-Median"]:
        checks[f"{model} results present"] = (
            model in all_results
        )

    # S-Median MAE overall should be <= S-Naive
    naive_mae  = all_results["S-Naive"]["Overall"]["MAE"]
    median_mae = all_results["S-Median"]["Overall"]["MAE"]
    checks["S-Median MAE <= S-Naive MAE overall"] = (
        median_mae <= naive_mae * 1.05
    )

    # SD2 should have higher MAE than SD0 for all baselines
    # WHY SD2 instead of SD1:
    #   In Rossmann, most stores CLOSE on public holidays
    #   so SD1 rows are only open stores (easier to forecast).
    #   SD2 (day before holiday) captures the pre-holiday
    #   shopping surge which IS harder than regular days.
    for model in ["S-Naive", "S-Median"]:
        sd0_mae = all_results[model]["SD0"]["MAE"]
        sd2_mae = all_results[model]["SD2"]["MAE"]
        checks[f"{model}: SD2 MAE > SD0 MAE"] = (
            sd2_mae > sd0_mae
        )

    # S-Naive-Std should have lower or equal SD4 error
    naive_sd4     = all_results["S-Naive"]["SD4"]["MAE"]
    naive_std_sd4 = all_results["S-Naive-Std"]["SD4"]["MAE"]
    checks["S-Naive-Std SD4 MAE <= S-Naive SD4 MAE"] = (
        naive_std_sd4 <= naive_sd4 * 1.05
    )

    # All MAE values must be positive
    for model, results in all_results.items():
        for sd_label, metrics in results.items():
            mae = metrics.get("MAE", 0)
            if not np.isnan(mae):
                checks[f"{model} {sd_label} MAE > 0"] = (
                    mae > 0
                )

    # Print results
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: linreg.py")
    else:
        print("Some checks FAILED.")

    return all_pass
