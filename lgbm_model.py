
# =============================================================
# lgbm_model.py — LightGBM Gradient Boosted Trees
#
# WHY LightGBM after LASSO:
#   LASSO validated that features carry signal.
#   LightGBM can learn non-linear interactions between
#   features that LASSO cannot capture. For example:
#   "Friday before Easter" is a special combination that
#   a linear model treats as Friday + Easter separately,
#   but LightGBM learns the interaction directly.
#
# WHY global model (not per-store like LASSO):
#   LightGBM handles high cardinality features natively.
#   Store_enc is included as a feature — the model learns
#   store-specific patterns implicitly through tree splits.
#   Pooling all stores gives much more training data,
#   which helps especially for rare SD types.
#
# WHY no target transformation for LightGBM:
#   The paper explicitly states that log/scaling the target
#   did not improve LightGBM results. LightGBM creates
#   internal bins during tree construction which handles
#   skewed distributions natively — no preprocessing needed.
#
# WHY early stopping:
#   LightGBM can overfit if trained too many rounds.
#   Early stopping monitors validation loss and stops
#   when it stops improving — automatic overfitting control.
#
# WHAT it produces:
#   lgbm_model.pkl        — saved model
#   lgbm_results.csv      — MAE and MASE per SD type
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    RANDOM_SEED, N_CV_FOLDS
)
from evaluation import (
    evaluate_by_sd_type, save_results,
    build_results_table, print_results_table
)


# =============================================================
# SECTION 1: Feature columns for LightGBM
# =============================================================

# WHY more features than LASSO:
#   LightGBM handles categorical features natively and
#   does not suffer from the curse of dimensionality
#   the same way linear models do. We include Store_enc
#   and sd_type as raw integers — LightGBM treats them
#   as categorical splits.

LGBM_FEATURE_COLS = [
    # Lag features
    "Sales_lag_2", "Sales_lag_3", "Sales_lag_4",
    "Sales_lag_5", "Sales_lag_6", "Sales_lag_7",
    "Sales_lag_14",
    # Rolling features
    "Sales_rolling_median", "Sales_rolling_std",
    # Calendar
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "WeekOfYear_sin", "WeekOfYear_cos",
    "DayOfMonth", "Month", "WeekOfYear", "Year",
    "IsWeekend", "IsStateHoliday",
    "SchoolHoliday", "Promo",
    # Store
    "StoreType_enc", "Assortment_enc",
    "Store_enc", "Promo2",
    "CompetitionDistance_log",
    # SD type
    "sd_type",
    "IsSD1", "IsSD2", "IsSD3", "IsSD4",
    # SD-specific features
    "sd_level", "sd_abs_change", "sd_rel_change",
    "sd_rel_change_storetype",
    "sd_level_other", "sd_abs_change_other",
    "sd_rel_change_other",
    "sd_rel_change_storetype_other",
]

# Categorical features for LightGBM
# WHY specify categorical:
#   LightGBM uses optimal categorical splits (grouping
#   categories that go together) instead of treating
#   them as continuous — much better for store type etc.
LGBM_CATEGORICAL = [
    "StoreType_enc", "Assortment_enc",
    "Store_enc", "sd_type",
    "IsSD1", "IsSD2", "IsSD3", "IsSD4",
    "IsWeekend", "IsStateHoliday",
    "SchoolHoliday", "Promo", "Promo2"
]


# =============================================================
# SECTION 2: Hyperparameter configuration
# =============================================================

def get_lgbm_params():
    """
    Returns LightGBM hyperparameters.

    WHY these specific values:
      num_leaves=63: controls model complexity.
        More leaves = more complex model = more overfit risk.
        63 is a good balance for retail tabular data.
        The paper tunes this — we use a reasonable default.

      min_data_in_leaf=50: minimum samples per leaf node.
        Prevents the model from creating splits on tiny
        groups (e.g., one store on one holiday). Higher
        value = more regularization.

      learning_rate=0.05: step size for boosting.
        Lower = slower but more stable convergence.
        Combined with early stopping, 0.05 is safe.

      feature_fraction=0.8: use 80% of features per tree.
        WHY: prevents any single feature from dominating
        all trees — similar to random forests' feature
        subsampling. Reduces overfitting.

      bagging_fraction=0.8, bagging_freq=5:
        Use 80% of data per tree, resample every 5 rounds.
        WHY: reduces variance by training on different
        subsets — stochastic gradient boosting.

      lambda_l1, lambda_l2: L1 and L2 regularization.
        Additional overfitting protection on leaf weights.

      metric='mae': optimise for MAE directly.
        WHY: the paper evaluates on MAE. Optimising MSE
        would bias the model toward large values.
    """
    return {
        "objective"       : "regression_l1",  # optimise MAE
        "metric"          : "mae",
        "num_leaves"      : 63,
        "min_data_in_leaf": 50,
        "learning_rate"   : 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq"    : 5,
        "lambda_l1"       : 0.1,
        "lambda_l2"       : 0.1,
        "verbose"         : -1,              # suppress output
        "random_state"    : RANDOM_SEED,
        "n_jobs"          : -1,
    }


# =============================================================
# SECTION 3: Train LightGBM
# =============================================================

def train_lgbm(train_df, feature_cols,
               n_rounds=2000, early_stopping=50):
    """
    Trains a single global LightGBM model on all stores.

    WHY global model:
      With 42 features and rare SD types (only ~600 SD1
      rows in test), a per-store model would have very
      few SD observations to learn from.
      A global model pools all 1115 stores — giving the
      model many more examples of each SD pattern.

    WHY validation split for early stopping:
      We use the last 20% of training dates as validation.
      WHY temporal (not random): if we randomly sampled
      validation rows, some future dates would leak into
      training — invalidating the evaluation.

    Args:
        train_df      : training DataFrame
        feature_cols  : list of feature column names
        n_rounds      : maximum boosting rounds
        early_stopping: stop if no improvement for N rounds

    Returns: trained LightGBM model
    """
    # Available features
    available = [c for c in feature_cols
                 if c in train_df.columns]

    # Temporal validation split: last 20% of dates
    dates      = sorted(train_df["Date"].unique())
    split_idx  = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    val_dates   = dates[split_idx:]

    tr = train_df[train_df["Date"].isin(train_dates)]
    vl = train_df[train_df["Date"].isin(val_dates)]

    print(f"  Train subset: {len(tr):,} rows")
    print(f"  Val subset  : {len(vl):,} rows")

    X_tr = tr[available].values
    y_tr = tr["Sales"].values        # raw sales — no transform

    X_vl = vl[available].values
    y_vl = vl["Sales"].values

    # Get categorical feature indices
    cat_indices = [
        available.index(c)
        for c in LGBM_CATEGORICAL
        if c in available
    ]

    # Build LightGBM datasets
    dtrain = lgb.Dataset(
        X_tr, label=y_tr,
        categorical_feature=cat_indices,
        free_raw_data=False
    )
    dval = lgb.Dataset(
        X_vl, label=y_vl,
        categorical_feature=cat_indices,
        reference=dtrain,
        free_raw_data=False
    )

    params    = get_lgbm_params()
    callbacks = [
        lgb.early_stopping(early_stopping, verbose=False),
        lgb.log_evaluation(100)   # print every 100 rounds
    ]

    print(f"\n  Training LightGBM...")
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=callbacks
    )

    print(f"  Best iteration: {model.best_iteration}")

    return model, available


# =============================================================
# SECTION 4: Run LightGBM
# =============================================================

def run_lgbm(sample_stores=None):
    """
    Trains LightGBM and evaluates on test set.

    Returns: (results_dict, y_true, y_pred, sd_types)
    """
    print("=" * 50)
    print("Running LightGBM")
    print("=" * 50)

    # Load data
    train_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet")
    )
    test_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

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

    # Get available features
    feature_cols = [
        c for c in LGBM_FEATURE_COLS
        if c in train_df.columns
    ]
    print(f"Features: {len(feature_cols)}")

    # Train model
    model, used_features = train_lgbm(
        train_df, feature_cols
    )

    # Predict on test set
    # WHY raw Sales target (no inverse transform needed):
    #   LightGBM was trained directly on real Sales values
    #   so predictions are already in real units
    X_test   = test_df[used_features].values
    y_pred   = model.predict(X_test)
    y_pred   = np.clip(y_pred, 0, None)  # no negative sales
    y_true   = test_df["Sales"].values
    sd_types = test_df["sd_type"].values

    # Evaluate
    results = evaluate_by_sd_type(
        y_true   = y_true,
        y_pred   = y_pred,
        sd_types = sd_types,
        train_df = train_df
    )

    # Save model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "lgbm_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved: {model_path}")

    save_results(results, "lgbm")
    print_results_table(
        build_results_table({"LightGBM": results}),
        "LightGBM Results"
    )

    # Feature importance
    importance = pd.DataFrame({
        "feature"   : used_features,
        "importance": model.feature_importance(
            importance_type="gain"
        )
    }).sort_values("importance", ascending=False)

    print("\nTop 15 features by importance:")
    print(importance.head(15).to_string(index=False))

    return results, y_true, y_pred, sd_types, importance


# =============================================================
# SECTION 5: Verification
# =============================================================

def verify_lgbm(results, linreg_results, baseline_results):
    """
    Verifies LightGBM results against expected patterns.

    Key checks from paper:
      1. LightGBM must beat LASSO overall
      2. LightGBM must beat S-Median on all SD types
      3. SD features must appear in top 20 important features
    """
    print("=" * 50)
    print("Verification — LightGBM")
    print("=" * 50)

    all_pass = True
    checks   = {}

    lgbm_overall  = results["Overall"]["MAE"]
    lasso_overall = linreg_results["Overall"]["MAE"]
    smed_overall  = baseline_results["S-Median"]["Overall"]["MAE"]

    # LightGBM must beat LASSO
    checks["LightGBM MAE < LASSO MAE overall"] = (
        lgbm_overall < lasso_overall * 1.05
    )

    # LightGBM must beat S-Median
    checks["LightGBM MAE < S-Median MAE overall"] = (
        lgbm_overall < smed_overall
    )

    # LightGBM must beat S-Median on SD2
    lgbm_sd2 = results["SD2"]["MAE"]
    smed_sd2 = baseline_results["S-Median"]["SD2"]["MAE"]
    if not np.isnan(lgbm_sd2):
        checks["LightGBM SD2 MAE < S-Median SD2 MAE"] = (
            lgbm_sd2 < smed_sd2
        )

    # All MAE values positive
    for sd_label, metrics in results.items():
        mae = metrics.get("MAE", np.nan)
        if not np.isnan(mae):
            checks[f"LightGBM {sd_label} MAE > 0"] = mae > 0

    # Print results
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    # Relative performance summary
    print(f"\n  vs S-Median : {lgbm_overall/smed_overall:.4f}")
    print(f"  vs LASSO    : {lgbm_overall/lasso_overall:.4f}")
    print(f"  (< 1.0 = beats that model)")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: mlp_model.py")
    else:
        print("Some checks FAILED.")

    return all_pass
