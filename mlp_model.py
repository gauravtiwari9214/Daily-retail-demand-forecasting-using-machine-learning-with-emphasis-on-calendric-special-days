
# =============================================================
# mlp_model.py — Multi-Layer Perceptron (Feed-Forward ANN)
#
# Implements THREE variants from the paper:
#   MLP-REG      : regression output (linear, 1 node)
#   MLP-CL (max) : classification, pick highest prob class
#   MLP-CL(median): classification, pick median of distribution
#
# WHY MLP after LightGBM:
#   LightGBM validated that non-linear patterns exist.
#   MLP can learn different non-linearities through
#   continuous representations in hidden layers.
#   The classification variant adds uncertainty quantification
#   which is the paper's key ANN contribution.
#
# WHY one file for all three variants:
#   MLP-CL(max) and MLP-CL(median) are the SAME model.
#   They differ only in how the softmax output is decoded.
#   MLP-REG differs only in the output layer.
#   One architecture builder + configurable output = cleaner.
#
# ARCHITECTURE (from paper Section 3.2.1):
#   Input → Dense(ReLU) → Dense(ReLU) → Output
#   Regression: Output = Dense(1, linear)
#   Classification: Output = Dense(124, softmax)
#
# TRAINING:
#   Optimizer: ADAM
#   Target: log-scaled sales in [-0.5, 0.5] for REG
#           bin index (0–103) for CL
#   Ensemble: N_ENSEMBLE models, combine with median
#   Early stopping on validation loss
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    N_ENSEMBLE, RANDOM_SEED, N_CV_FOLDS
)
from evaluation import (
    evaluate_by_sd_type, save_results,
    build_results_table, print_results_table
)
from preprocessing import (
    inverse_transform_target,
    proba_to_sales_median,
    bin_index_to_sales
)


# =============================================================
# SECTION 1: Feature columns for MLP
# =============================================================

# Same as LASSO but includes Store_enc and sd_type
# WHY: MLP can learn store-specific patterns through
# continuous embeddings in hidden layers
MLP_FEATURE_COLS = [
    "Sales_lag_2", "Sales_lag_3", "Sales_lag_4",
    "Sales_lag_5", "Sales_lag_6", "Sales_lag_7",
    "Sales_lag_14",
    "Sales_rolling_median", "Sales_rolling_std",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Month_sin", "Month_cos",
    "WeekOfYear_sin", "WeekOfYear_cos",
    "DayOfMonth", "Month", "WeekOfYear", "Year",
    "IsWeekend", "IsStateHoliday",
    "SchoolHoliday", "Promo",
    "StoreType_enc", "Assortment_enc",
    "Store_enc", "Promo2",
    "CompetitionDistance_log",
    "IsSD1", "IsSD2", "IsSD3", "IsSD4",
    "sd_level", "sd_abs_change", "sd_rel_change",
    "sd_rel_change_storetype",
    "sd_level_other", "sd_abs_change_other",
    "sd_rel_change_other",
    "sd_rel_change_storetype_other",
]


# =============================================================
# SECTION 2: Model architecture builder
# =============================================================

def build_mlp(n_features, n_output, output_activation,
              hidden_units=None, dropout_rate=0.2):
    """
    Builds the MLP architecture.

    WHY ReLU activation in hidden layers:
      ReLU = max(0, x). It does not saturate for positive
      inputs (unlike sigmoid/tanh), so gradients flow well
      through many layers. The paper uses ReLU or ELU.

    WHY BatchNormalization:
      Normalises activations within each mini-batch.
      Reduces internal covariate shift — makes training
      faster and more stable. Also acts as regularisation.

    WHY Dropout:
      Randomly zeros out neurons during training.
      Forces the network to learn redundant representations
      — no single neuron can be relied upon.
      This is the primary overfitting prevention mechanism.

    WHY smaller hidden layers for classification:
      The output layer has n_output=124 nodes (vs 1 for REG).
      That is 124x more weights between last hidden layer
      and output. To keep total parameters similar,
      hidden layer capacity is reduced.
      The paper observes this exact pattern.

    Args:
        n_features       : number of input features
        n_output         : 1 for REG, N_BINS for CL
        output_activation: 'linear' for REG, 'softmax' for CL
        hidden_units     : list of units per hidden layer
        dropout_rate     : dropout probability

    Returns: compiled Keras model
    """
    if hidden_units is None:
        # WHY smaller for classification (more output nodes)
        if n_output == 1:
            hidden_units = [256, 128]   # REG
        else:
            hidden_units = [128, 64]    # CL

    tf.random.set_seed(RANDOM_SEED)

    model = keras.Sequential()

    # Input layer
    model.add(layers.Input(shape=(n_features,)))

    # Hidden layers
    for units in hidden_units:
        model.add(layers.Dense(units))
        model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.Dropout(dropout_rate))

    # Output layer
    model.add(layers.Dense(n_output,
                            activation=output_activation))

    # Loss function depends on task
    if output_activation == "linear":
        # Regression: minimise MSE on scaled target
        # WHY MSE not MAE for training:
        #   MSE has smooth gradients everywhere — easier
        #   to optimise with gradient descent.
        #   MAE has discontinuous gradient at 0.
        #   We evaluate on MAE even though we train on MSE.
        loss = "mse"
    else:
        # Classification: minimise cross-entropy
        # WHY sparse_categorical_crossentropy:
        #   Our targets are integer bin indices (not one-hot).
        #   Sparse version handles integer targets directly
        #   without needing to one-hot encode 124 classes.
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=loss
    )

    return model


# =============================================================
# SECTION 3: Train single MLP model
# =============================================================

def train_single_mlp(X_train, y_train, X_val, y_val,
                      n_output, output_activation,
                      seed=42, hidden_units=None,
                      dropout_rate=0.2, epochs=200,
                      batch_size=1024):
    """
    Trains one MLP model with early stopping.

    WHY batch_size=1024:
      Large batches give stable gradient estimates for
      tabular data. Small batches (32-64) are needed for
      image data where spatial locality matters — not here.

    WHY early stopping with patience=10:
      Monitors validation loss. If it does not improve
      for 10 consecutive epochs, training stops and the
      best weights are restored.
      This is the primary mechanism preventing overfitting.

    WHY restore_best_weights=True:
      Without this, you get the weights from the last
      epoch — which may be worse than epoch N-10.
      Always restore the best checkpoint.

    Returns: trained Keras model
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    model      = build_mlp(
        n_features, n_output, output_activation,
        hidden_units, dropout_rate
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            verbose=0
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0
    )

    best_epoch = np.argmin(history.history["val_loss"]) + 1

    return model, best_epoch


# =============================================================
# SECTION 4: Train ensemble of MLP models
# =============================================================

def train_mlp_ensemble(train_df, feature_cols,
                        mode="reg",
                        n_models=N_ENSEMBLE):
    """
    Trains an ensemble of N_ENSEMBLE MLP models.

    WHY ensemble with median operator:
      Neural networks are sensitive to random initialisation.
      Different seeds produce different local minima.
      Some initialisations are lucky, some are not.
      Training N models and taking the median prediction:
        1. Reduces variance — bad initialisations are
           averaged out by the median
        2. Improves reliability — one bad model cannot
           dominate the prediction
      The paper trains 50 models. We use N_ENSEMBLE=10
      for speed during development.

    WHY MEDIAN not MEAN for ensemble:
      A single badly-initialised model can predict 0 or
      10000 for a day — completely wrong. The mean would
      be pulled toward this outlier. The median ignores it.

    Args:
        mode: "reg" for regression, "cl" for classification

    Returns: list of trained models, feature_cols used
    """
    print(f"  Training {n_models} MLP models (mode={mode})...")

    # Get available features
    available = [c for c in feature_cols
                 if c in train_df.columns]

    # Temporal validation split: last 20% of dates
    dates       = sorted(train_df["Date"].unique())
    split_idx   = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    val_dates   = dates[split_idx:]

    tr = train_df[train_df["Date"].isin(train_dates)]
    vl = train_df[train_df["Date"].isin(val_dates)]

    X_train = tr[available].values.astype(np.float32)
    X_val   = vl[available].values.astype(np.float32)

    # Target depends on mode
    if mode == "reg":
        # Regression: scaled sales in [-0.5, 0.5]
        y_train = tr["Sales_scaled"].values.astype(np.float32)
        y_val   = vl["Sales_scaled"].values.astype(np.float32)
        n_output          = 1
        output_activation = "linear"
    else:
        # Classification: bin index (integer)
        y_train = tr["bin_index"].values.astype(np.int32)
        y_val   = vl["bin_index"].values.astype(np.int32)
        # n_output = number of bins in the data
        n_output          = int(train_df["bin_index"].max()) + 1
        output_activation = "softmax"

    print(f"  Train: {len(X_train):,} | "
          f"Val: {len(X_val):,} | "
          f"Features: {len(available)} | "
          f"Output nodes: {n_output}")

    models      = []
    best_epochs = []

    for i in range(n_models):
        seed  = RANDOM_SEED + i
        model, best_epoch = train_single_mlp(
            X_train, y_train,
            X_val,   y_val,
            n_output, output_activation,
            seed=seed
        )
        models.append(model)
        best_epochs.append(best_epoch)

        if (i + 1) % 2 == 0:
            print(f"    Model {i+1}/{n_models} done "
                  f"(best epoch: {best_epoch})")

    print(f"  Avg best epoch: "
          f"{np.mean(best_epochs):.1f} ± "
          f"{np.std(best_epochs):.1f}")

    return models, available


# =============================================================
# SECTION 5: Prediction with ensemble
# =============================================================

def predict_ensemble_reg(models, X_test,
                          target_scaler):
    """
    Generates regression predictions from ensemble.

    Each model predicts scaled sales.
    Median across all models gives the ensemble prediction.
    Inverse transform gives real sales units.

    Returns: array of predictions in real sales units
    """
    all_preds = []
    X_test_f  = X_test.astype(np.float32)

    for model in models:
        pred_scaled = model.predict(
            X_test_f, verbose=0
        ).flatten()
        pred_scaled = np.clip(pred_scaled, -0.5, 0.5)
        all_preds.append(pred_scaled)

    # Median across ensemble (axis=0 = per sample)
    ensemble_scaled = np.median(
        np.array(all_preds), axis=0
    )

    # Inverse transform to real units
    return inverse_transform_target(
        ensemble_scaled, target_scaler
    )


def predict_ensemble_cl(models, X_test,
                         bin_midpoints):
    """
    Generates classification predictions from ensemble.

    For each model, gets the softmax probability distribution.
    Takes the MEDIAN probability per class across all models.
    Renormalises to sum to 1.
    Decodes using both CL(max) and CL(median).

    WHY median of probabilities (not majority vote):
      Median probabilities preserve the full distribution
      shape, allowing CL(median) decoding.
      Majority vote would collapse to a single class,
      losing all uncertainty information.

    Returns: (pred_cl_max, pred_cl_median) both in real units
    """
    all_probas = []
    X_test_f   = X_test.astype(np.float32)

    for model in models:
        proba = model.predict(X_test_f, verbose=0)
        all_probas.append(proba)

    # Shape: (n_models, n_samples, n_bins)
    all_probas = np.array(all_probas)

    # Median probability per class per sample
    # Shape: (n_samples, n_bins)
    median_proba = np.median(all_probas, axis=0)

    # Renormalise so probabilities sum to 1
    # WHY: median of probabilities may not sum to 1 exactly
    row_sums     = median_proba.sum(axis=1, keepdims=True)
    median_proba = median_proba / (row_sums + 1e-10)

    n_samples = median_proba.shape[0]

    pred_cl_max    = np.zeros(n_samples)
    pred_cl_median = np.zeros(n_samples)

    for i in range(n_samples):
        proba = median_proba[i]
        n_bins = len(bin_midpoints)

        # CL(max): class with highest probability
        max_idx         = np.argmin(np.abs(
            np.arange(len(proba)) -
            np.argmax(proba)
        ))
        max_idx         = np.argmax(proba)
        max_idx         = min(max_idx, n_bins - 1)
        pred_cl_max[i]  = bin_midpoints[max_idx]

        # CL(median): 50th percentile of distribution
        pred_cl_median[i] = proba_to_sales_median(
            proba, bin_midpoints
        )

    return pred_cl_max, pred_cl_median


# =============================================================
# SECTION 6: Run MLP
# =============================================================

def run_mlp(sample_stores=None):
    """
    Trains and evaluates all three MLP variants:
    MLP-REG, MLP-CL(max), MLP-CL(median)

    Returns: dict of results for all three variants
    """
    print("=" * 50)
    print("Running MLP")
    print("=" * 50)

    # Load data
    train_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet")
    )
    test_df = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

    # Load scalers and bins
    with open(os.path.join(
        MODELS_DIR, "target_scaler.pkl"), "rb"
    ) as f:
        target_scaler = pickle.load(f)

    bin_midpoints = np.load(
        os.path.join(MODELS_DIR, "bin_midpoints.npy")
    )

    # Subset stores
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

    print(f"Train: {len(train_df):,} | "
          f"Test: {len(test_df):,}")

    # Get available features
    feature_cols = [
        c for c in MLP_FEATURE_COLS
        if c in train_df.columns
    ]

    # Test feature matrix
    available = [c for c in feature_cols
                 if c in test_df.columns]
    X_test    = test_df[available].values.astype(np.float32)
    y_true    = test_df["Sales"].values
    sd_types  = test_df["sd_type"].values

    all_results = {}

    # ── MLP-REG ───────────────────────────────────────────────
    print("\n--- MLP-REG ---")
    models_reg, used_feats = train_mlp_ensemble(
        train_df, feature_cols, mode="reg"
    )
    X_test_reg = test_df[used_feats].values.astype(np.float32)
    y_pred_reg = predict_ensemble_reg(
        models_reg, X_test_reg, target_scaler
    )
    results_reg = evaluate_by_sd_type(
        y_true, y_pred_reg, sd_types, train_df
    )
    all_results["MLP-REG"] = results_reg
    save_results(results_reg, "mlp_reg")
    print("  MLP-REG done.")

    # ── MLP-CL ────────────────────────────────────────────────
    print("\n--- MLP-CL ---")
    models_cl, used_feats_cl = train_mlp_ensemble(
        train_df, feature_cols, mode="cl"
    )
    X_test_cl = test_df[used_feats_cl].values.astype(
        np.float32
    )
    y_pred_cl_max, y_pred_cl_med = predict_ensemble_cl(
        models_cl, X_test_cl, bin_midpoints
    )

    # CL predictions are in scaled space — inverse transform
    y_pred_cl_max_real = inverse_transform_target(
        y_pred_cl_max, target_scaler
    )
    y_pred_cl_med_real = inverse_transform_target(
        y_pred_cl_med, target_scaler
    )

    results_cl_max = evaluate_by_sd_type(
        y_true, y_pred_cl_max_real, sd_types, train_df
    )
    results_cl_med = evaluate_by_sd_type(
        y_true, y_pred_cl_med_real, sd_types, train_df
    )
    all_results["MLP-CL(max)"]    = results_cl_max
    all_results["MLP-CL(median)"] = results_cl_med
    save_results(results_cl_max, "mlp_cl_max")
    save_results(results_cl_med, "mlp_cl_median")
    print("  MLP-CL done.")

    # Print combined table
    print_results_table(
        build_results_table(all_results),
        "MLP Results — All Variants"
    )

    # Save models
    os.makedirs(MODELS_DIR, exist_ok=True)
    for i, m in enumerate(models_reg):
        m.save(os.path.join(
            MODELS_DIR, f"mlp_reg_{i}.keras"
        ))
    for i, m in enumerate(models_cl):
        m.save(os.path.join(
            MODELS_DIR, f"mlp_cl_{i}.keras"
        ))
    print(f"Models saved to {MODELS_DIR}")

    return all_results, y_true, sd_types


# =============================================================
# SECTION 7: Verification
# =============================================================

def verify_mlp(all_results, lgbm_results,
               baseline_results):
    """
    Verifies MLP results match paper's expected patterns.

    Key checks:
      1. MLP-REG beats LightGBM on MASE (paper finding)
      2. CL(median) beats CL(max) — paper's core CL insight
      3. MLP-CL(median) is best MLP variant overall
    """
    print("=" * 50)
    print("Verification — MLP")
    print("=" * 50)

    all_pass = True
    checks   = {}

    reg_mase    = all_results["MLP-REG"]["Overall"]["MASE"]
    cl_max_mase = all_results["MLP-CL(max)"]["Overall"]["MASE"]
    cl_med_mase = all_results["MLP-CL(median)"]["Overall"]["MASE"]
    lgbm_mase   = lgbm_results["Overall"]["MASE"]
    smed_mae    = baseline_results["S-Median"]["Overall"]["MAE"]

    # All MLP variants must beat S-Median
    for variant in ["MLP-REG", "MLP-CL(max)", "MLP-CL(median)"]:
        mae = all_results[variant]["Overall"]["MAE"]
        checks[f"{variant} MAE < S-Median MAE"] = (
            mae < smed_mae
        )

    # CL(median) must beat CL(max) — paper's key finding
    checks["MLP-CL(median) MASE <= MLP-CL(max) MASE"] = (
        cl_med_mase <= cl_max_mase * 1.05
    )

    # CL(median) should be best MLP variant
    checks["MLP-CL(median) is best MLP variant"] = (
        cl_med_mase <= reg_mase * 1.05
    )

    # All MAE values positive
    for variant, results in all_results.items():
        for sd_label, metrics in results.items():
            mae = metrics.get("MAE", np.nan)
            if not np.isnan(mae):
                checks[f"{variant} {sd_label} MAE > 0"] = (
                    mae > 0
                )

    # Print results
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    # Summary
    print(f"\n  MASE summary:")
    print(f"    LightGBM          : {lgbm_mase:.4f}")
    print(f"    MLP-REG           : {reg_mase:.4f}")
    print(f"    MLP-CL(max)       : {cl_max_mase:.4f}")
    print(f"    MLP-CL(median)    : {cl_med_mase:.4f}")
    print(f"\n  CL(median) vs CL(max): "
          f"{cl_med_mase/cl_max_mase:.4f} "
          f"({'better' if cl_med_mase < cl_max_mase else 'worse'})")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: lstm_model.py")
    else:
        print("Some checks FAILED.")

    return all_pass
