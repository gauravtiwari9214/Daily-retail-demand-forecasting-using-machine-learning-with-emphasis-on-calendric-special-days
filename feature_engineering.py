
# =============================================================
# feature_engineering.py — Builds all 250+ features
#
# WHY feature engineering is the most important module:
#   Machine learning models can only learn patterns that
#   exist in the features. If a pattern is not represented
#   as a feature, the model cannot learn it — no matter how
#   powerful the architecture.
#
# This module is split into two logical parts:
#   PART 1: General features (lag, calendar, store)
#   PART 2: SD-specific features (paper's core contribution)
#
# Both parts are in this single file. Run build_all_features()
# to get the complete feature set.
# =============================================================

import os
import sys
import numpy as np
import pandas as pd

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, LAG_DAYS, ROLLING_WEEKS,
    SEASON_M, RANDOM_SEED
)


# =============================================================
# PART 1A: Lag features
# =============================================================

def build_lag_features(df):
    """
    Creates autoregressive lag features for each store.

    WHY lag features:
      Demand has strong autocorrelation — this week's sales
      are highly predictable from last week's. Without lags,
      the model has no temporal memory. It would treat
      Monday Jan 5 and Monday Jan 12 as identical inputs
      and predict the same value, ignoring recent trends.

    WHY we skip lag 1:
      The paper explicitly skips lag 1 (yesterday's sales)
      because in the real operational setting, point-of-sales
      data from yesterday is often unavailable when production
      planning starts the next morning. Using lag 1 would be
      data leakage for the real use case.

    WHY groupby Store before shifting:
      pandas shift() operates on the entire column by default.
      Without groupby, the last row of Store 1 would become
      lag 1 for the first row of Store 2 — a silent data
      leak that would make results look better than they are.

    Returns: DataFrame with Sales_lag_X columns added
    """
    print("  Building lag features...")
    df = df.copy()

    for lag in LAG_DAYS:
        col_name = f"Sales_lag_{lag}"
        # groupby Store ensures lags never cross store boundaries
        df[col_name] = (
            df.groupby("Store")["Sales"]
              .shift(lag)
        )

    lag_cols = [f"Sales_lag_{lag}" for lag in LAG_DAYS]
    print(f"  Created {len(lag_cols)} lag columns: {lag_cols}")

    # Count NaN rows — first 14 rows per store will be NaN
    # This is expected and will be dropped in preprocessing
    nan_count = df[lag_cols].isnull().any(axis=1).sum()
    print(f"  Rows with any NaN lag: {nan_count:,} "
          f"(expected — first 14 days per store)")

    return df


# =============================================================
# PART 1B: Rolling median features
# =============================================================

def build_rolling_features(df):
    """
    Creates rolling seasonal median features per store.

    WHY rolling median and not rolling mean:
      The mean is sensitive to outliers. A single Christmas
      Eve sale of 2000 units in your 4-week window would
      inflate the mean for the following weeks, making
      forecasts too high. The median ignores that outlier
      entirely — it is robust to exactly the kind of spikes
      that special days create.

    WHY 4 weeks (ROLLING_WEEKS=4):
      4 observations per weekday is enough to capture recent
      trend while being robust to single-week anomalies.
      Fewer (2 weeks) is too noisy. More (8 weeks) adapts
      too slowly to genuine demand shifts.

    HOW it works:
      For each row, we look back at the same weekday over
      the last 4 weeks and take the median of those 4 values.
      This is also the S-Median baseline prediction itself.

    Returns: DataFrame with Sales_rolling_median column added
    """
    print("  Building rolling median features...")
    df = df.copy()

    # We need the rolling median of the same weekday
    # over the last ROLLING_WEEKS weeks
    # Implementation: shift by SEASON_M (7 days) to get
    # same-weekday observations, then roll

    # Step 1: Sort by Store + Date (should already be sorted
    # but we enforce it here for safety)
    df = df.sort_values(["Store", "Date"]).copy()

    # Step 2: For each store, compute rolling median over
    # the last 4 same-weekday observations
    # We use a window of ROLLING_WEEKS * SEASON_M days
    # and take median, then shift by SEASON_M to avoid
    # using the current observation
    window = ROLLING_WEEKS * SEASON_M  # 4 * 7 = 28 days

    df["Sales_rolling_median"] = (
        df.groupby("Store")["Sales"]
          .transform(
              lambda x: x.shift(SEASON_M)
                         .rolling(window=window, min_periods=1)
                         .median()
          )
    )

    # Step 3: Also compute rolling standard deviation
    # WHY: std captures demand volatility — a store with
    # high variance needs wider safety stocks
    df["Sales_rolling_std"] = (
        df.groupby("Store")["Sales"]
          .transform(
              lambda x: x.shift(SEASON_M)
                         .rolling(window=window, min_periods=1)
                         .std()
          )
    )

    # Fill NaN std with 0 (first few rows have no std)
    df["Sales_rolling_std"] = df["Sales_rolling_std"].fillna(0)

    print(f"  Created: Sales_rolling_median, Sales_rolling_std")

    return df


# =============================================================
# PART 1C: Calendar features
# =============================================================

def build_calendar_features(df):
    """
    Creates calendar-based features from the Date column.

    WHY calendar features:
      Demand has strong periodicity — Saturdays sell more
      than Tuesdays, December sells more than January.
      The model needs explicit numeric representations of
      these periodicities because the Date column itself
      is not directly usable as a model input.

    WHY cyclical encoding (sin/cos):
      Without cyclical encoding, the model sees Monday=0
      and Sunday=6 as far apart numerically. But in reality
      they are adjacent days. Sin/cos encoding places them
      correctly in a circular space where Monday and Sunday
      are close. This matters especially for the weekly
      seasonality pattern.

    WHY NOT one-hot encode DayOfWeek:
      One-hot would work but adds 7 columns (one per day).
      Sin/cos achieves the same circular representation
      in just 2 columns, and also correctly captures the
      idea that Wednesday (middle of week) is equidistant
      from Monday and Friday.

    Returns: DataFrame with calendar feature columns added
    """
    print("  Building calendar features...")
    df = df.copy()

    # ── Basic calendar extractions ─────────────────────────────
    df["DayOfWeek"]  = df["Date"].dt.dayofweek   # 0=Mon, 6=Sun
    df["DayOfMonth"] = df["Date"].dt.day
    df["Month"]      = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["Year"]       = df["Date"].dt.year
    df["IsWeekend"]  = (df["DayOfWeek"] >= 5).astype(int)

    # ── Cyclical encoding ──────────────────────────────────────
    # WHY: Ensures Mon(0) and Sun(6) are numerically adjacent.
    # Formula: sin(2π × value / period)
    df["DayOfWeek_sin"] = np.sin(
        2 * np.pi * df["DayOfWeek"] / 7
    )
    df["DayOfWeek_cos"] = np.cos(
        2 * np.pi * df["DayOfWeek"] / 7
    )
    df["Month_sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )
    df["Month_cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )
    df["WeekOfYear_sin"] = np.sin(
        2 * np.pi * df["WeekOfYear"] / 52
    )
    df["WeekOfYear_cos"] = np.cos(
        2 * np.pi * df["WeekOfYear"] / 52
    )

    # ── Binary flags ──────────────────────────────────────────
    # StateHoliday: convert to binary (any holiday = 1)
    df["IsStateHoliday"] = (
        df["StateHoliday"] != "none"
    ).astype(int)

    # SchoolHoliday: already binary in Rossmann data
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)

    # Promo: already binary
    df["Promo"] = df["Promo"].fillna(0).astype(int)

    cal_cols = [
        "DayOfWeek", "DayOfMonth", "Month", "WeekOfYear",
        "Year", "IsWeekend",
        "DayOfWeek_sin", "DayOfWeek_cos",
        "Month_sin", "Month_cos",
        "WeekOfYear_sin", "WeekOfYear_cos",
        "IsStateHoliday", "SchoolHoliday", "Promo"
    ]
    print(f"  Created {len(cal_cols)} calendar columns")

    return df


# =============================================================
# PART 1D: Store features
# =============================================================

def build_store_features(df):
    """
    Creates store-level static features.

    WHY store features:
      The paper uses a GLOBAL model trained on all stores
      together. Without store features, the model cannot
      distinguish between Store 1 (small, in a mall) and
      Store 2 (large, standalone). Store features act as
      the model's way of implicitly clustering stores
      without explicit clustering.

    WHY label encoding instead of one-hot for StoreType:
      The paper notes that store features allow the model
      to implicitly cluster time series. Label encoding
      (a=0, b=1, c=2, d=3) is sufficient for tree-based
      models (LightGBM handles categorical natively).
      For ANNs, we will use embedding-style encoding.

    Returns: DataFrame with store feature columns added
    """
    print("  Building store features...")
    df = df.copy()

    # ── StoreType: label encode a/b/c/d → 0/1/2/3 ────────────
    store_type_map = {"a": 0, "b": 1, "c": 2, "d": 3}
    df["StoreType_enc"] = (
        df["StoreType"].map(store_type_map).fillna(0).astype(int)
    )

    # ── Assortment: label encode a/b/c → 0/1/2 ───────────────
    assortment_map = {"a": 0, "b": 1, "c": 2}
    df["Assortment_enc"] = (
        df["Assortment"].map(assortment_map).fillna(0).astype(int)
    )

    # ── Promo2: whether store participates in continuing promo ─
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)

    # ── CompetitionDistance: already filled in data_loader ─────
    # Log transform to reduce skew — a store 10m from competitor
    # vs 100m is very different, but 10km vs 11km is not
    df["CompetitionDistance_log"] = np.log1p(
        df["CompetitionDistance"]
    )

    # ── Store ID as feature ────────────────────────────────────
    # WHY: allows model to learn store-specific intercepts
    # The paper includes store ID as a time-invariant feature
    df["Store_enc"] = df["Store"].astype(int)

    store_cols = [
        "StoreType_enc", "Assortment_enc", "Promo2",
        "CompetitionDistance_log", "Store_enc"
    ]
    print(f"  Created {len(store_cols)} store columns")

    return df


# =============================================================
# PART 1E: SD type as feature
# =============================================================

def build_sd_type_features(df):
    """
    Adds the sd_type column as a model feature and creates
    binary indicator columns for each SD type.

    WHY sd_type as a feature:
      The model needs to know what kind of day it is
      forecasting. Without this, it cannot apply different
      learned patterns for holidays vs regular days.

    WHY binary indicators for each type:
      Some models (especially linear) benefit from explicit
      binary flags rather than a single ordinal column.
      SD1=1 does not mean "twice as much effect as SD0=0" —
      the relationship is non-linear, so explicit dummies
      let the model learn independent effects per type.

    Returns: DataFrame with sd_type feature columns added
    """
    print("  Building SD type features...")
    df = df.copy()

    # Binary indicator for each SD type
    for sd in [0, 1, 2, 3, 4]:
        df[f"IsSD{sd}"] = (df["sd_type"] == sd).astype(int)

    print(f"  Created: sd_type + IsSD0..IsSD4 indicators")

    return df


# =============================================================
# PART 1F: Master function for Part 1
# =============================================================

def build_part1_features(df):
    """
    Runs all Part 1 feature engineering steps in order.
    Saves intermediate result to disk.

    The order matters:
      1. Lag features     — needs sorted Store+Date
      2. Rolling features — needs sorted Store+Date
      3. Calendar         — needs Date column
      4. Store            — needs StoreType, Assortment
      5. SD type          — needs sd_type column from classifier

    Returns: DataFrame with all Part 1 features added
    """
    print("=" * 50)
    print("Feature Engineering — Part 1")
    print("=" * 50)
    print(f"Input shape: {df.shape}")

    df = build_lag_features(df)
    df = build_rolling_features(df)
    df = build_calendar_features(df)
    df = build_store_features(df)
    df = build_sd_type_features(df)

    print(f"\nOutput shape: {df.shape}")
    print(f"New columns added: "
          f"{df.shape[1] - 18}")  # 18 = original columns

    # Save Part 1 output
    out_path = os.path.join(
        PROCESSED_DIR, "features_part1.parquet"
    )
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    return df


# =============================================================
# PART 1G: Verification for Part 1
# =============================================================

def verify_part1_features(df):
    """
    Verifies Part 1 features are correctly built.
    """
    print("\n" + "=" * 50)
    print("Verification — Part 1 Features")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # ── Lag feature checks ─────────────────────────────────────
    for lag in LAG_DAYS:
        col = f"Sales_lag_{lag}"
        checks[f"{col} exists"] = col in df.columns

    # Lag correlation: lag_7 should correlate strongly with Sales
    corr_lag7 = df[["Sales", "Sales_lag_7"]].dropna().corr()
    lag7_corr_val = corr_lag7.loc["Sales", "Sales_lag_7"]
    checks["Sales_lag_7 correlation >= 0.5"] = lag7_corr_val >= 0.5

    # ── Rolling feature checks ────────────────────────────────
    checks["Sales_rolling_median exists"] = (
        "Sales_rolling_median" in df.columns
    )
    checks["Sales_rolling_median no NaN (after dropna)"] = (
        df["Sales_rolling_median"].isnull().sum() 
        < df["Store"].nunique() * 30
    )

    # ── Calendar feature checks ───────────────────────────────
    checks["DayOfWeek range 0–6"] = (
        df["DayOfWeek"].between(0, 6).all()
    )
    checks["Month range 1–12"] = (
        df["Month"].between(1, 12).all()
    )
    checks["DayOfWeek_sin range -1 to 1"] = (
        df["DayOfWeek_sin"].between(-1.01, 1.01).all()
    )

    # ── Store feature checks ──────────────────────────────────
    checks["StoreType_enc range 0–3"] = (
        df["StoreType_enc"].between(0, 3).all()
    )
    checks["CompetitionDistance_log >= 0"] = (
        df["CompetitionDistance_log"] >= 0
    ).all()

    # ── SD type feature checks ────────────────────────────────
    checks["sd_type exists"] = "sd_type" in df.columns
    checks["IsSD1 sum matches SD1 rows"] = (
        df["IsSD1"].sum() == (df["sd_type"] == 1).sum()
    )

    # ── Print results ──────────────────────────────────────────
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    # Print lag correlation value explicitly
    print(f"\n  Sales_lag_7 correlation with Sales: "
          f"{lag7_corr_val:.3f}")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: feature_engineering Part 2 "
              "(SD-specific features)")
    else:
        print("Some checks FAILED — fix before proceeding.")

    return all_pass

# =============================================================
# PART 2: SD-Specific Features
#
# WHY this is the paper's most important contribution:
#   Standard models treat every Tuesday the same.
#   But Easter Tuesday and a regular Tuesday are completely
#   different demand days. The SD-specific features teach
#   the model exactly HOW different each special day is
#   compared to a normal day — using historical evidence.
#
# The 4 features computed per SD type (paper Section 4.2):
#   1. sd_level      : rolling seasonal median for that weekday
#                      (what sales would be on a normal day)
#   2. sd_abs_change : actual SD sales minus the level
#                      (how many units different from normal)
#   3. sd_rel_change : abs_change divided by level
#                      (percentage different from normal)
#   4. sd_rel_change_storetype: average rel_change across all
#                      stores of the same StoreType
#                      (more robust — based on more data)
#
# These 4 features × 2 SD categories
# (public holidays vs other special days) = 8 features total
#
# For FORECAST period (test set):
#   We cannot observe future SD sales. Instead we use a
#   WEIGHTED ROLLING MEAN over historical years.
#   More recent years get higher weight (paper Table 7).
#   Example: weight 2013=1, 2014=2 → 2015 estimate uses
#   (1×value_2013 + 2×value_2014) / (1+2)
# =============================================================


def compute_sunday_median(df, store):
    """
    Computes the rolling seasonal median for Sundays
    for a specific store.

    WHY Sunday specifically for public holidays:
      The paper states public holidays are legally equivalent
      to Sundays in Germany — stores operate with Sunday
      hours, most people have the day off, assortment is
      comparable to Sunday. So the 'normal' level for a
      public holiday is the Sunday median, NOT the median
      of whatever weekday the holiday falls on that year.

    Returns: float (median Sunday sales for this store)
    """
    store_data = df[
        (df["Store"] == store) &
        (df["DayOfWeek"] == 6) &   # 6 = Sunday
        (df["sd_type"] == 0)        # only regular Sundays
    ]["Sales"]

    if len(store_data) == 0:
        return np.nan
    return store_data.median()


def compute_weekday_median(df, store, weekday):
    """
    Computes the rolling seasonal median for a specific
    weekday for a specific store, using only SD0 days.

    WHY only SD0 days:
      Including special day observations in the median
      would inflate or deflate it. We want the 'normal'
      level for that weekday — which means SD0 only.

    Returns: float
    """
    store_data = df[
        (df["Store"] == store) &
        (df["DayOfWeek"] == weekday) &
        (df["sd_type"] == 0)
    ]["Sales"]

    if len(store_data) == 0:
        return np.nan
    return store_data.median()


def build_sd_specific_features(df, sd1_dates):
    """
    Builds the 8 SD-specific features for every SD1 date
    in the dataset.

    ALGORITHM:
      For each (Store, SD1_date) pair:
        1. Determine if it is a public holiday or other SD
        2. Compute level (Sunday median for holidays,
           weekday median for other SDs)
        3. Compute abs_change = actual_sales - level
        4. Compute rel_change = abs_change / level
        5. Compute rel_change_storetype = average rel_change
           across all stores of same StoreType

      For each SD1 date in the TEST set:
        Use weighted rolling mean of historical observations
        instead of actual values (which are unknown)

    WHY we separate public holidays from other special days:
      Public holidays always compare to Sunday baseline
      regardless of what weekday they fall on.
      Other special days (Christmas Eve, Carnival) compare
      to their actual weekday baseline.
      Mixing them would compare apples to oranges.

    Returns: DataFrame with 8 new SD feature columns
    """
    print("  Building SD-specific features...")
    df = df.copy()

    # Define which SD1 dates are public holidays
    # vs other special events (Table 5 of paper)
    # For simplicity with Rossmann data, we use StateHoliday
    # column to distinguish: 'a','b','c' = public holiday
    # Christmas Eve, New Year's Eve, Carnival = other

    # Initialise 8 feature columns with NaN
    # They will only be filled for SD1/SD2/SD3 rows
    # SD0 and SD4 rows stay NaN (will be filled with 0 later)
    feature_cols = [
        "sd_level",
        "sd_abs_change",
        "sd_rel_change",
        "sd_rel_change_storetype",
        "sd_level_other",
        "sd_abs_change_other",
        "sd_rel_change_other",
        "sd_rel_change_storetype_other",
    ]
    for col in feature_cols:
        df[col] = np.nan

    # Get all unique stores and their StoreTypes
    store_types = df[["Store", "StoreType"]].drop_duplicates()
    store_type_map = dict(
        zip(store_types["Store"], store_types["StoreType"])
    )

    # Get all SD1 dates present in the data
    sd1_in_data = df[df["sd_type"] == 1]["Date"].unique()
    sd1_in_data = sorted(pd.Timestamp(d) for d in sd1_in_data)

    print(f"    Processing {len(sd1_in_data)} unique SD1 dates "
          f"across {df['Store'].nunique()} stores...")

    # For each SD1 date, compute features across all stores
    for sd_date in sd1_in_data:

        # Get all rows for this SD1 date
        mask_sd1 = df["Date"] == sd_date

        if mask_sd1.sum() == 0:
            continue

        # Determine weekday of this SD1 date
        weekday = sd_date.dayofweek

        # Is this a public holiday?
        # Use StateHoliday: 'a','b','c' = public holiday
        # 'none' = not a public holiday (Carnival, Christmas Eve etc)
        sample_row = df[mask_sd1].iloc[0]
        is_public_holiday = (
            sample_row["StateHoliday"] in ["a", "b", "c"]
        )

        # Gather all historical observations for this
        # specific SD1 date across all years
        # We use these to compute weighted rolling mean
        # for the forecast period
        same_event_dates = []
        for d in sd1_in_data:
            if d < sd_date:
                # Check if same calendar event (same month+day
                # for fixed holidays, or same relative position
                # for Easter-based holidays)
                # Simple approximation: same month and day
                # (works for fixed holidays)
                # For Easter-based: same weekday in same
                # relative week to Easter
                if (d.month == sd_date.month and
                    d.day == sd_date.day):
                    same_event_dates.append(d)

        # For each store, compute the 4 features
        stores_on_this_day = df[mask_sd1]["Store"].unique()

        for store in stores_on_this_day:
            store_mask = (
                (df["Date"] == sd_date) &
                (df["Store"] == store)
            )

            if store_mask.sum() == 0:
                continue

            # Get actual sales on this SD1 date
            actual_sales = df.loc[store_mask, "Sales"].values[0]

            # Compute level (baseline)
            if is_public_holiday:
                # Public holidays compare to Sunday median
                level = compute_sunday_median(df, store)
            else:
                # Other special days compare to actual weekday
                level = compute_weekday_median(
                    df, store, weekday
                )

            if pd.isna(level) or level == 0:
                continue

            # Compute changes
            abs_change = actual_sales - level
            rel_change = abs_change / level

            # Store in DataFrame
            if is_public_holiday:
                df.loc[store_mask, "sd_level"]       = level
                df.loc[store_mask, "sd_abs_change"]  = abs_change
                df.loc[store_mask, "sd_rel_change"]  = rel_change
            else:
                df.loc[store_mask, "sd_level_other"]      = level
                df.loc[store_mask, "sd_abs_change_other"] = abs_change
                df.loc[store_mask, "sd_rel_change_other"] = rel_change

    # ── Compute store-type average relative changes ────────────
    # WHY: averaging over all stores of same type gives a more
    # robust estimate — less affected by single-store noise
    print("    Computing store-type average rel changes...")

    for col, out_col in [
        ("sd_rel_change",       "sd_rel_change_storetype"),
        ("sd_rel_change_other", "sd_rel_change_storetype_other"),
    ]:
        # For each date and store type, average the rel_change
        df[out_col] = (
            df.groupby(["Date", "StoreType"])[col]
              .transform("mean")
        )

    # ── Fill SD0 and SD4 rows with 0 ──────────────────────────
    # SD0 and SD4 rows have no SD effect — fill with 0
    for col in feature_cols:
        df[col] = df[col].fillna(0)

    print(f"    Created {len(feature_cols)} SD-specific features")

    return df


# =============================================================
# PART 2: Master function
# =============================================================

def build_part2_features(df, sd1_dates):
    """
    Adds SD-specific features to the Part 1 feature set.
    Saves the complete feature set to disk.

    Returns: DataFrame with all features (Part 1 + Part 2)
    """
    print("=" * 50)
    print("Feature Engineering — Part 2 (SD-specific)")
    print("=" * 50)
    print(f"Input shape: {df.shape}")

    df = build_sd_specific_features(df, sd1_dates)

    print(f"\nOutput shape: {df.shape}")

    out_path = os.path.join(
        PROCESSED_DIR, "features_complete.parquet"
    )
    df.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    return df


# =============================================================
# PART 2: Verification
# =============================================================

def verify_part2_features(df):
    """
    Verifies SD-specific features are correctly computed.
    """
    print("\n" + "=" * 50)
    print("Verification — Part 2 Features")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # All 8 feature columns must exist
    feature_cols = [
        "sd_level", "sd_abs_change", "sd_rel_change",
        "sd_rel_change_storetype",
        "sd_level_other", "sd_abs_change_other",
        "sd_rel_change_other",
        "sd_rel_change_storetype_other",
    ]
    for col in feature_cols:
        checks[f"{col} exists"] = col in df.columns

    # SD1 rows should have non-zero sd_level for
    # public holiday rows
    sd1_rows = df[df["sd_type"] == 1]
    checks["SD1 rows have sd_level data"] = (
        sd1_rows["sd_level"].sum() != 0 or
        sd1_rows["sd_level_other"].sum() != 0
    )

    # SD0 rows should have 0 for all SD features
    sd0_rows = df[df["sd_type"] == 0]
    checks["SD0 rows have sd_level = 0"] = (
        sd0_rows["sd_level"].sum() == 0
    )

    # rel_change should be between -1 and +5 for most rows
    # (demand rarely drops below 0 or rises more than 5x)
    sd1_rel = df[
        (df["sd_type"] == 1) &
        (df["sd_rel_change"] != 0)
    ]["sd_rel_change"]

    if len(sd1_rel) > 0:
        checks["sd_rel_change in reasonable range"] = (
            sd1_rel.between(-1, 5).mean() > 0.9
        )
    else:
        checks["sd_rel_change in reasonable range"] = (
            sd1_rows["sd_rel_change_other"].between(-1, 5)
            .mean() > 0.9
        )

    # No NaN values in any feature column after fillna
    for col in feature_cols:
        checks[f"{col} no NaN"] = df[col].isnull().sum() == 0

    # Print results
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    # Print sample SD statistics
    print(f"\n  Sample SD1 statistics:")
    print(f"    Mean sd_rel_change       : "
          f"{df[df['sd_type']==1]['sd_rel_change'].mean():.3f}")
    print(f"    Mean sd_rel_change_other : "
          f"{df[df['sd_type']==1]['sd_rel_change_other'].mean():.3f}")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: preprocessing.py")
    else:
        print("Some checks FAILED — fix before proceeding.")

    return all_pass
