
# =============================================================
# preprocessing.py — Prepares data for all models
#
# Produces:
#   train.parquet          — training feature matrix
#   test.parquet           — test feature matrix
#   target_scaler.pkl      — fitted MinMaxScaler for Sales
#   feature_scaler.pkl     — fitted StandardScaler for features
#   bin_edges.npy          — 124 bin boundaries for classification
#
# WHY this module exists as a separate step:
#   Every model (LASSO, LightGBM, MLP, LSTM) needs the
#   same preprocessed data. Doing this once and saving
#   to disk means:
#   1. All models are trained on identical inputs
#   2. No risk of accidentally refitting scalers on test data
#   3. Preprocessing bugs are fixed in one place
#
# THE GOLDEN RULE:
#   Everything fitted here uses ONLY training data.
#   The test set is transformed using training statistics.
#   Violating this = data leakage = falsely optimistic results.
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, MODELS_DIR,
    TRAIN_END, TEST_START,
    SCALE_MIN, SCALE_MAX,
    N_BINS, N_CV_FOLDS,
    RANDOM_SEED, LAG_DAYS
)


# =============================================================
# SECTION 1: Feature column definitions
# =============================================================

# These are the exact columns fed to every model.
# Defined here once so all models use identical feature sets.

# Columns that are NOT features (metadata + raw target)
NON_FEATURE_COLS = [
    "Store", "Date", "Sales", "Customers",
    "Open", "StateHoliday", "StoreType", "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
    "Promo2SinceWeek", "Promo2SinceYear", "PromoInterval",
    "DayOfWeek",   # raw — we use encoded version
]

# Columns to scale with StandardScaler (continuous numeric)
COLS_TO_STANDARDIZE = (
    [f"Sales_lag_{lag}" for lag in LAG_DAYS] +
    ["Sales_rolling_median", "Sales_rolling_std",
     "CompetitionDistance_log",
     "sd_level", "sd_abs_change",
     "sd_level_other", "sd_abs_change_other"]
)

# Columns kept as-is (binary, categorical integers, cyclical)
COLS_AS_IS = [
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "WeekOfYear_sin", "WeekOfYear_cos",
    "DayOfMonth", "Month", "WeekOfYear", "Year",
    "IsWeekend", "IsStateHoliday", "SchoolHoliday",
    "Promo", "Promo2",
    "StoreType_enc", "Assortment_enc",
    "Store_enc",
    "sd_type",
    "IsSD0", "IsSD1", "IsSD2", "IsSD3", "IsSD4",
    "sd_rel_change", "sd_rel_change_storetype",
    "sd_rel_change_other", "sd_rel_change_storetype_other",
]


def get_feature_cols(df):
    """
    Returns the list of feature columns available in df.
    Filters out non-feature columns dynamically so the
    function works even if some optional columns are missing.
    """
    all_feature_cols = COLS_TO_STANDARDIZE + COLS_AS_IS
    # Only include columns that actually exist in df
    available = [c for c in all_feature_cols if c in df.columns]
    return available


# =============================================================
# SECTION 2: Train / test split
# =============================================================

def temporal_split(df):
    """
    Splits data into train and test sets by date.

    WHY strict temporal split (not random):
      This is time series data. Shuffling would let the model
      train on future observations and predict past ones —
      that is information leakage. In production, you always
      train on the past and predict the future.

    WHY these specific dates:
      Rossmann public data ends July 2015.
      We use the last month (July) as test — it contains
      meaningful sales variation without being too small.

    Returns: (train_df, test_df)
    """
    train = df[df["Date"] <= TRAIN_END].copy()
    test  = df[df["Date"] >= TEST_START].copy()

    print(f"  Train: {len(train):,} rows "
          f"({train['Date'].min().date()} to "
          f"{train['Date'].max().date()})")
    print(f"  Test : {len(test):,} rows "
          f"({test['Date'].min().date()} to "
          f"{test['Date'].max().date()})")

    # Verify no overlap
    assert train["Date"].max() < test["Date"].min(), (
        "Train and test dates overlap — check TRAIN_END/TEST_START"
    )

    return train, test


# =============================================================
# SECTION 3: Drop NaN rows from lag features
# =============================================================

def drop_lag_nans(df, split_name=""):
    """
    Drops rows where any lag feature is NaN.

    WHY we drop instead of fill:
      The first 14 rows per store have NaN lags because
      there is no history to look back on. Filling with 0
      would be wrong — it would make the model think sales
      were 0 in the pre-history period, corrupting the
      autoregressive signal.

    Dropping is safe because these rows only occur at the
    very start of each store's history — not randomly.
    """
    lag_cols    = [f"Sales_lag_{lag}" for lag in LAG_DAYS]
    before      = len(df)
    df          = df.dropna(subset=lag_cols).copy()
    after       = len(df)
    dropped     = before - after
    print(f"  {split_name}: Dropped {dropped:,} NaN lag rows "
          f"({before:,} → {after:,})")
    return df


# =============================================================
# SECTION 4: Target transformation
# =============================================================

def fit_target_scaler(train_sales):
    """
    Fits a log + MinMax scaler on training Sales values.

    WHY log transform first:
      Raw sales are heavily right-skewed (most days 50–300
      units, rare days 2000+). The neural network loss is
      dominated by large values. Log transform compresses
      the range so all values contribute equally to the
      gradient.

    WHY then scale to [-0.5, 0.5]:
      LeCun et al. (2012) show this range is optimal for
      backpropagation. It keeps inputs in the most linear
      region of sigmoid/tanh activations, preventing
      gradient saturation.

    WHY MinMaxScaler fitted on TRAIN only:
      If we fit on test data, the model would "know" the
      test set's value range — that is data leakage.

    Returns: fitted MinMaxScaler (operates on log-sales)
    """
    log_sales = np.log1p(train_sales.values.reshape(-1, 1))
    scaler    = MinMaxScaler(
        feature_range=(SCALE_MIN, SCALE_MAX)
    )
    scaler.fit(log_sales)
    return scaler


def transform_target(sales, scaler):
    """Applies log then MinMax scaling to sales values."""
    log_sales = np.log1p(sales.values.reshape(-1, 1))
    return scaler.transform(log_sales).flatten()


def inverse_transform_target(scaled_sales, scaler):
    """
    Reverses scaling: unscale → exp → real sales units.

    WHY this must be called before computing MAE/MASE:
      If you compute error on scaled values, your numbers
      are meaninglessly small and appear perfect.
      Always inverse transform before evaluation.
    """
    unscaled  = scaler.inverse_transform(
        scaled_sales.reshape(-1, 1)
    )
    real_sales = np.expm1(unscaled).flatten()
    # Clip to 0 — predictions can sometimes go slightly
    # negative after inverse transform due to floating point
    return np.clip(real_sales, 0, None)


# =============================================================
# SECTION 5: Feature scaling
# =============================================================

def fit_feature_scaler(X_train, cols_to_scale):
    """
    Fits StandardScaler on continuous numeric features
    of the training set.

    WHY StandardScaler (zero mean, unit variance):
      Neural networks are sensitive to feature scale.
      If one feature has range [0, 2000] and another [0, 1],
      the large-valued feature dominates gradient updates.
      StandardScaler puts all continuous features on the
      same scale so each contributes equally.

    WHY only on COLS_TO_STANDARDIZE:
      Binary features (0/1), cyclical features (sin/cos
      already in [-1,1]), and ordinal encodings do not
      need scaling — they are already in a reasonable range.

    Returns: fitted StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train[cols_to_scale])
    return scaler


def apply_feature_scaler(X, scaler, cols_to_scale):
    """
    Applies fitted StandardScaler to a feature matrix.
    Only scales COLS_TO_STANDARDIZE — leaves others unchanged.
    """
    X = X.copy()
    available = [c for c in cols_to_scale if c in X.columns]
    X[available] = scaler.transform(X[available])
    return X


# =============================================================
# SECTION 6: Classification bins
# =============================================================

def build_classification_bins(train_sales_scaled, n_bins=N_BINS):
    """
    Creates bin edges for the classification target.

    WHY classification instead of regression (paper insight):
      Regression predicts one point estimate with no
      uncertainty information. Classification outputs a
      full probability distribution over 124 bins — the
      model can be uncertain (spread distribution) or
      confident (peaked distribution). This uncertainty
      information improves accuracy when decoded with
      CL(median).

    HOW bins are created (from paper Section 4):
      1. Start with one bin per percentile (100 bins)
      2. For any two adjacent bins where the relative
         increase exceeds 10%, add an extra split point
      3. This gives ~124 bins total — denser at low sales
         values (common) and sparser at high values (rare)

    WHY fitted on training data only:
      The bin boundaries encode the training sales
      distribution. Using test data would let the model
      "know" future sales ranges.

    Returns: (bin_edges array, bin_midpoints array)
    """
    # Unscale to get real sales values for percentile calc
    # We build bins on the scaled values directly
    values = np.sort(train_sales_scaled)

    # Step 1: Create percentile-based bin edges
    percentiles   = np.linspace(0, 100, 101)
    initial_edges = np.percentile(values, percentiles)
    initial_edges = np.unique(initial_edges)  # remove duplicates

    # Step 2: Add extra splits where relative increase > 10%
    final_edges = [initial_edges[0]]
    for i in range(1, len(initial_edges)):
        prev = initial_edges[i-1]
        curr = initial_edges[i]
        # Relative increase: (curr - prev) / abs(prev)
        # Avoid division by zero
        if abs(prev) > 1e-10:
            rel_increase = (curr - prev) / abs(prev)
            if rel_increase > 0.10:
                # Add midpoint as extra split
                final_edges.append((prev + curr) / 2)
        final_edges.append(curr)

    bin_edges = np.array(final_edges)

    # Step 3: Compute bin midpoints (class values)
    # WHY midpoints: the predicted value for a class is
    # the mean of its interval — minimises quantisation error
    bin_midpoints = np.array([
        (bin_edges[i] + bin_edges[i+1]) / 2
        for i in range(len(bin_edges) - 1)
    ])

    actual_n_bins = len(bin_midpoints)
    print(f"  Classification bins: {actual_n_bins} "
          f"(target: {n_bins})")

    return bin_edges, bin_midpoints


def sales_to_bin_index(sales_scaled, bin_edges):
    """
    Converts a scaled sales value to its bin index.
    Used to create the classification target vector.
    """
    # np.digitize returns index of bin each value falls in
    indices = np.digitize(sales_scaled, bin_edges) - 1
    # Clip to valid range [0, n_bins-1]
    indices = np.clip(indices, 0, len(bin_edges) - 2)
    return indices


def bin_index_to_sales(bin_index, bin_midpoints):
    """
    Converts bin index back to scaled sales value.
    Used for CL(max): pick class with highest probability.
    """
    return bin_midpoints[bin_index]


def proba_to_sales_median(proba, bin_midpoints):
    """
    Converts probability distribution to sales value
    using the MEDIAN of the distribution.

    WHY CL(median) beats CL(max) — the key insight:
      CL(max) picks the mode — the single most likely class.
      But MAE is minimised by the MEDIAN, not the mode.
      For asymmetric distributions (which demand data has),
      the median and mode can be very different.
      The paper shows CL(median) reduces error by 6-9%.

    HOW it works:
      1. Compute cumulative sum of probabilities
      2. Find first bin where cumsum >= 0.5
      3. That bin's midpoint is the median prediction

    Returns: scalar sales value
    """
    cumsum = np.cumsum(proba)
    median_idx = np.searchsorted(cumsum, 0.5)
    median_idx = min(median_idx, len(bin_midpoints) - 1)
    return bin_midpoints[median_idx]


# =============================================================
# SECTION 7: Cross-validation fold creation
# =============================================================

def create_cv_folds(train_df, n_folds=N_CV_FOLDS):
    """
    Creates stratified temporal CV folds.

    WHY stratified by sd_type:
      Special days are rare (16% of data). Random splitting
      might put all SD1 rows in one fold, making some folds
      impossible to evaluate on special days.
      Stratification ensures each fold has a representative
      proportion of each SD type.

    WHY temporal within each fold:
      The validation set must always come after the training
      set in time — even within CV. Otherwise we are testing
      the model on data it has "seen the future of".

    HOW it works:
      We split the training dates into n_folds windows.
      For each fold, 80% of dates = train, last 20% = val.
      We stratify by ensuring each fold has similar SD
      type proportions.

    Returns: list of (train_idx, val_idx) tuples
    """
    folds       = []
    dates       = sorted(train_df["Date"].unique())
    n_dates     = len(dates)
    fold_size   = n_dates // n_folds

    for i in range(n_folds):
        # Each fold uses a different temporal window
        start_idx   = i * fold_size
        end_idx     = start_idx + fold_size
        fold_dates  = dates[start_idx:end_idx]

        # 80% train, 20% validation — temporal split
        split_point = int(len(fold_dates) * 0.8)
        train_dates = fold_dates[:split_point]
        val_dates   = fold_dates[split_point:]

        train_idx = train_df[
            train_df["Date"].isin(train_dates)
        ].index
        val_idx = train_df[
            train_df["Date"].isin(val_dates)
        ].index

        folds.append((train_idx, val_idx))

    print(f"  Created {len(folds)} CV folds")
    for i, (tr, vl) in enumerate(folds):
        print(f"    Fold {i+1}: "
              f"train={len(tr):,} val={len(vl):,}")

    return folds


# =============================================================
# SECTION 8: Master preprocessing function
# =============================================================

def run_preprocessing(save=True):
    """
    Runs the complete preprocessing pipeline.
    Loads features_complete.parquet, produces all outputs.

    Returns: (train_df, test_df, feature_cols,
               target_scaler, feature_scaler,
               bin_edges, bin_midpoints, cv_folds)
    """
    print("=" * 50)
    print("Preprocessing")
    print("=" * 50)

    # ── Load complete feature set ─────────────────────────────
    in_path = os.path.join(
        PROCESSED_DIR, "features_complete.parquet"
    )
    df = pd.read_parquet(in_path)
    print(f"Loaded: {df.shape}")

    # ── Temporal split ────────────────────────────────────────
    print("\nSplitting train/test...")
    train, test = temporal_split(df)

    # ── Drop NaN lag rows ─────────────────────────────────────
    print("\nDropping NaN lag rows...")
    train = drop_lag_nans(train, "Train")
    test  = drop_lag_nans(test,  "Test")

    # ── Get feature columns ───────────────────────────────────
    feature_cols = get_feature_cols(train)
    print(f"\nFeature columns: {len(feature_cols)}")

    # ── Fit and apply target scaler ───────────────────────────
    print("\nFitting target scaler (train only)...")
    target_scaler = fit_target_scaler(train["Sales"])
    train["Sales_scaled"] = transform_target(
        train["Sales"], target_scaler
    )
    test["Sales_scaled"] = transform_target(
        test["Sales"], target_scaler
    )
    print(f"  Train Sales_scaled range: "
          f"[{train['Sales_scaled'].min():.3f}, "
          f"{train['Sales_scaled'].max():.3f}]")

    # ── Fit and apply feature scaler ──────────────────────────
    print("\nFitting feature scaler (train only)...")
    cols_to_scale = [
        c for c in COLS_TO_STANDARDIZE
        if c in feature_cols
    ]
    feature_scaler = fit_feature_scaler(train, cols_to_scale)
    train_X = apply_feature_scaler(
        train[feature_cols], feature_scaler, cols_to_scale
    )
    test_X  = apply_feature_scaler(
        test[feature_cols],  feature_scaler, cols_to_scale
    )

    # Add metadata back for evaluation
    for col in ["Store", "Date", "Sales",
                "Sales_scaled", "sd_type"]:
        train_X[col] = train[col].values
        test_X[col]  = test[col].values

    # ── Build classification bins ─────────────────────────────
    print("\nBuilding classification bins...")
    bin_edges, bin_midpoints = build_classification_bins(
        train["Sales_scaled"].values
    )

    # Add bin index as classification target
    train_X["bin_index"] = sales_to_bin_index(
        train["Sales_scaled"].values, bin_edges
    )
    test_X["bin_index"] = sales_to_bin_index(
        test["Sales_scaled"].values, bin_edges
    )

    # ── Create CV folds ───────────────────────────────────────
    print("\nCreating CV folds...")
    cv_folds = create_cv_folds(train_X)

    # ── Save everything ───────────────────────────────────────
    if save:
        os.makedirs(MODELS_DIR, exist_ok=True)

        train_path = os.path.join(PROCESSED_DIR, "train.parquet")
        test_path  = os.path.join(PROCESSED_DIR, "test.parquet")
        train_X.to_parquet(train_path, index=False)
        test_X.to_parquet(test_path,   index=False)
        print(f"\nSaved train : {train_path}")
        print(f"Saved test  : {test_path}")

        with open(os.path.join(
            MODELS_DIR, "target_scaler.pkl"), "wb"
        ) as f:
            pickle.dump(target_scaler, f)

        with open(os.path.join(
            MODELS_DIR, "feature_scaler.pkl"), "wb"
        ) as f:
            pickle.dump(feature_scaler, f)

        np.save(
            os.path.join(MODELS_DIR, "bin_edges.npy"),
            bin_edges
        )
        np.save(
            os.path.join(MODELS_DIR, "bin_midpoints.npy"),
            bin_midpoints
        )

        print(f"Saved scalers and bins to: {MODELS_DIR}")

    print(f"\nTrain shape : {train_X.shape}")
    print(f"Test shape  : {test_X.shape}")
    print(f"Feature cols: {len(feature_cols)}")
    print(f"Bins        : {len(bin_midpoints)}")

    return (train_X, test_X, feature_cols,
            target_scaler, feature_scaler,
            bin_edges, bin_midpoints, cv_folds)


# =============================================================
# SECTION 9: Verification
# =============================================================

def verify_preprocessing(train_X, test_X, feature_cols,
                          target_scaler, bin_edges,
                          bin_midpoints):
    """
    Verifies the preprocessing pipeline is correct.
    """
    print("\n" + "=" * 50)
    print("Verification — Preprocessing")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # No NaN in feature columns
    train_nans = train_X[feature_cols].isnull().sum().sum()
    test_nans  = test_X[feature_cols].isnull().sum().sum()
    checks["Train features: no NaN"] = train_nans == 0
    checks["Test features: no NaN"]  = test_nans  == 0

    # Target scaled to [-0.5, 0.5]
    checks["Train Sales_scaled in [-0.5, 0.5]"] = (
        train_X["Sales_scaled"].between(
            SCALE_MIN - 0.01, SCALE_MAX + 0.01
        ).all()
    )

    # Temporal split: no overlap
    checks["Train ends before test starts"] = (
        train_X["Date"].max() < test_X["Date"].min()
    )

    # Inverse transform recovers original sales
    sample_scaled = train_X["Sales_scaled"].values[:100]
    sample_real   = inverse_transform_target(
        sample_scaled, target_scaler
    )
    sample_orig   = train_X["Sales"].values[:100]
    max_error     = np.abs(sample_real - sample_orig).max()
    checks["Inverse transform recovers sales (error < 1)"] = (
        max_error < 1.0
    )

    # Bin edges are monotonically increasing
    checks["Bin edges monotonically increasing"] = (
        np.all(np.diff(bin_edges) > 0)
    )

    # Bin count close to N_BINS
    checks[f"Bin count between 80 and 200"] = (
        80 <= len(bin_midpoints) <= 200
    )

    # bin_index is in valid range
    max_bin = len(bin_midpoints) - 1
    checks["Train bin_index in valid range"] = (
        train_X["bin_index"].between(0, max_bin).all()
    )

    # Print results
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n  Train shape     : {train_X.shape}")
    print(f"  Test shape      : {test_X.shape}")
    print(f"  Feature columns : {len(feature_cols)}")
    print(f"  Actual bin count: {len(bin_midpoints)}")
    print(f"  Max inverse error: {max_error:.4f}")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: evaluation.py")
    else:
        print("Some checks FAILED — fix before proceeding.")

    return all_pass
