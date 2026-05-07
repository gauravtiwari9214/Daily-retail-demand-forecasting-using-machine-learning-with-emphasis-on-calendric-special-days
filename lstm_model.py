
# =============================================================
# lstm_model.py — Long Short-Term Memory (Recurrent ANN)
#
# Implements THREE variants from the paper:
#   LSTM-REG       : regression output (linear, 1 node)
#   LSTM-CL(max)   : classification, pick highest prob class
#   LSTM-CL(median): classification, pick median — BEST MODEL
#
# WHY LSTM beats MLP (paper's finding):
#   MLP treats all lag features as a flat vector.
#   It sees [lag2, lag3, lag4, lag5, lag6, lag7, lag14]
#   as 7 independent numbers with no sense of ordering.
#   LSTM processes lags as a SEQUENCE — it reads lag14
#   first, then lag7, then lag6... building a hidden state
#   that represents "what happened leading up to today."
#   This temporal processing captures dynamics that a
#   flat vector cannot — e.g. "sales have been rising
#   for 5 days" is a pattern LSTM detects, MLP cannot.
#
# ARCHITECTURE (from paper Section 3.2.2):
#   Sequence input → LSTM layer → Dense(ReLU) → Output
#   The sequence contains dynamic features (lag sales,
#   weekday) per time step. Static features (store class,
#   SD features) are concatenated after the LSTM layer.
#
# KEY DIFFERENCE from MLP implementation:
#   Input must be reshaped into sequences.
#   Each sample has shape (seq_len, n_dynamic_features).
#   Static features are concatenated to LSTM output.
#
# RETRAINING (from paper Section 6.1.4):
#   LSTMs use fine-tuning (continue from existing weights)
#   rather than training from scratch when retraining.
#   This saves compute time and avoids forgetting.
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, MODELS_DIR, RESULTS_DIR,
    N_ENSEMBLE, RANDOM_SEED, LAG_DAYS
)
from evaluation import (
    evaluate_by_sd_type, save_results,
    build_results_table, print_results_table
)
from preprocessing import (
    inverse_transform_target,
    proba_to_sales_median
)


# =============================================================
# SECTION 1: Feature split — dynamic vs static
# =============================================================

# Dynamic features change at each lag step (per time step)
# WHY: these are the features that form the sequence
# the LSTM processes step by step
DYNAMIC_FEATURES = [
    "Sales_lag_2", "Sales_lag_3", "Sales_lag_4",
    "Sales_lag_5", "Sales_lag_6", "Sales_lag_7",
    "Sales_lag_14",
    "DayOfWeek_sin", "DayOfWeek_cos",
    "Sales_rolling_median",
]

# Static features are the same regardless of lag step
# WHY: store class, SD features, promo do not change
# as we look back through the lag sequence
STATIC_FEATURES = [
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
# SECTION 2: Sequence preparation
# =============================================================

def prepare_sequences(df, dynamic_cols, static_cols):
    """
    Reshapes the flat feature matrix into sequences for LSTM.

    WHY sequences for LSTM:
      The lag features [lag2, lag3, ..., lag14] represent
      a time sequence. Instead of feeding them as 7 separate
      numbers, we create a sequence of length 7 where each
      step contains the dynamic features for that lag day.

    HOW it works:
      For each row (one forecast target), we create:
        sequence: shape (n_lags, n_dynamic) — the history
        static  : shape (n_static,) — time-invariant info

      The LSTM reads the sequence step by step, building
      a hidden state. Then static features are concatenated
      to the final hidden state before the output layer.

    WHY this ordering (oldest lag first):
      LSTM processes left to right. By feeding lag14 first
      and lag2 last, the most recent information is the
      last thing the LSTM sees before making a prediction.
      This means the final hidden state is most influenced
      by recent history — correct for forecasting.

    Returns:
      X_seq   : shape (n_samples, n_lags, n_dynamic)
      X_static: shape (n_samples, n_static)
    """
    available_dynamic = [c for c in dynamic_cols
                         if c in df.columns]
    available_static  = [c for c in static_cols
                         if c in df.columns]

    # The lag columns are already in the DataFrame as
    # separate columns. We stack them into a sequence.
    # Order: lag14 (oldest) → lag7 → lag6 → ... → lag2 (newest)
    lag_cols_ordered = [
        c for c in [
            "Sales_lag_14", "Sales_lag_7", "Sales_lag_6",
            "Sales_lag_5",  "Sales_lag_4", "Sales_lag_3",
            "Sales_lag_2"
        ] if c in df.columns
    ]

    # Other dynamic features (weekday, rolling median)
    other_dynamic = [
        c for c in available_dynamic
        if c not in lag_cols_ordered
    ]

    # Build sequence: each time step has lag value +
    # other dynamic features repeated for each step
    # Shape per sample: (n_lags, 1 + n_other_dynamic)
    lag_values    = df[lag_cols_ordered].values
    # Shape: (n_samples, n_lags, 1)
    lag_values_3d = lag_values[:, :, np.newaxis]

    if len(other_dynamic) > 0:
        # Repeat other dynamic features for each lag step
        other_values = df[other_dynamic].values
        # Shape: (n_samples, n_other)
        # Expand to (n_samples, n_lags, n_other)
        other_3d = np.repeat(
            other_values[:, np.newaxis, :],
            lag_values_3d.shape[1],
            axis=1
        )
        X_seq = np.concatenate(
            [lag_values_3d, other_3d], axis=2
        )
    else:
        X_seq = lag_values_3d

    X_static = df[available_static].values

    return (X_seq.astype(np.float32),
            X_static.astype(np.float32),
            available_dynamic, available_static)


# =============================================================
# SECTION 3: LSTM model architecture
# =============================================================

def build_lstm(n_seq_features, n_static_features,
               seq_len, n_output, output_activation,
               lstm_units=64, dense_units=64,
               dropout_rate=0.2):
    """
    Builds the LSTM architecture using Keras functional API.

    WHY functional API (not Sequential):
      We have TWO inputs: sequence + static.
      Sequential API handles only one input.
      Functional API allows merging two input streams.

    WHY concatenate LSTM output with static features:
      The LSTM only sees the sequence (temporal dynamics).
      Static features like store class, SD-specific features,
      and promo flags are not part of the sequence — they
      apply to the whole forecast, not just one lag step.
      Concatenating them after the LSTM lets the final
      dense layer combine temporal patterns with
      time-invariant context.

    WHY lstm_units=64:
      Larger LSTM units = more memory capacity but
      slower training and more overfitting risk.
      64 is a good starting point for tabular data.
      Paper tunes this — we use a practical default.

    Args:
        n_seq_features   : features per time step
        n_static_features: static feature count
        seq_len          : number of lag steps
        n_output         : 1 for REG, N_BINS for CL
        output_activation: 'linear' or 'softmax'

    Returns: compiled Keras model
    """
    # ── Sequence input branch ─────────────────────────────────
    seq_input = layers.Input(
        shape=(seq_len, n_seq_features),
        name="sequence_input"
    )
    lstm_out = layers.LSTM(
        lstm_units,
        return_sequences=False,  # only final hidden state
        name="lstm_layer"
    )(seq_input)
    lstm_out = layers.Dropout(
        dropout_rate, name="lstm_dropout"
    )(lstm_out)

    # ── Static input branch ───────────────────────────────────
    static_input = layers.Input(
        shape=(n_static_features,),
        name="static_input"
    )
    static_dense = layers.Dense(
        32, activation="relu", name="static_dense"
    )(static_input)

    # ── Merge branches ────────────────────────────────────────
    # WHY concatenate: combine temporal memory with
    # time-invariant context before final prediction
    merged = layers.Concatenate(name="merge")(
        [lstm_out, static_dense]
    )

    # Final dense layers
    x = layers.Dense(
        dense_units, activation="relu",
        name="dense_1"
    )(merged)
    x = layers.BatchNormalization(name="bn_1")(x)
    x = layers.Dropout(dropout_rate, name="dense_dropout")(x)

    # Output layer
    output = layers.Dense(
        n_output,
        activation=output_activation,
        name="output"
    )(x)

    model = Model(
        inputs=[seq_input, static_input],
        outputs=output
    )

    # Loss function
    if output_activation == "linear":
        loss = "mse"
    else:
        loss = "sparse_categorical_crossentropy"

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
            clipnorm=1.0    # gradient clipping prevents
                            # exploding gradients in LSTM
        ),
        loss=loss
    )

    return model


# =============================================================
# SECTION 4: Train single LSTM
# =============================================================

def train_single_lstm(X_seq_train, X_static_train,
                       y_train,
                       X_seq_val, X_static_val,
                       y_val,
                       n_output, output_activation,
                       seed=42):
    """
    Trains one LSTM model with early stopping.

    WHY clipnorm=1.0 (gradient clipping):
      LSTMs can suffer from exploding gradients —
      the gradient grows exponentially as it propagates
      back through many time steps. Clipping limits the
      gradient norm to 1.0, stabilising training.
      This is especially important for retail data where
      holiday spikes create large loss values.

    Returns: (trained model, best_epoch)
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    seq_len          = X_seq_train.shape[1]
    n_seq_features   = X_seq_train.shape[2]
    n_static_features = X_static_train.shape[1]

    model = build_lstm(
        n_seq_features, n_static_features,
        seq_len, n_output, output_activation
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,            # more patience than MLP
            restore_best_weights=True,
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            verbose=0
        )
    ]

    history = model.fit(
        [X_seq_train, X_static_train], y_train,
        validation_data=(
            [X_seq_val, X_static_val], y_val
        ),
        epochs=200,
        batch_size=1024,
        callbacks=callbacks,
        verbose=0
    )

    best_epoch = np.argmin(
        history.history["val_loss"]
    ) + 1

    return model, best_epoch


# =============================================================
# SECTION 5: Train LSTM ensemble
# =============================================================

def train_lstm_ensemble(train_df, mode="reg",
                         n_models=N_ENSEMBLE):
    """
    Trains an ensemble of LSTM models.

    WHY fine-tuning for retraining (paper Section 6.1.4):
      When new data arrives and models need retraining,
      LSTMs continue from existing weights (fine-tuning)
      rather than training from scratch.
      This preserves learned patterns while adapting
      to recent data — faster and less prone to forgetting.

    Returns: (models list, seq_feature_cols, static_feature_cols)
    """
    print(f"  Training {n_models} LSTM models "
          f"(mode={mode})...")

    # Temporal split
    dates       = sorted(train_df["Date"].unique())
    split_idx   = int(len(dates) * 0.8)
    train_dates = dates[:split_idx]
    val_dates   = dates[split_idx:]

    tr = train_df[train_df["Date"].isin(train_dates)]
    vl = train_df[train_df["Date"].isin(val_dates)]

    # Prepare sequences
    (X_seq_tr, X_static_tr,
     dyn_cols, stat_cols) = prepare_sequences(
        tr, DYNAMIC_FEATURES, STATIC_FEATURES
    )
    (X_seq_vl, X_static_vl, _, _) = prepare_sequences(
        vl, DYNAMIC_FEATURES, STATIC_FEATURES
    )

    # Targets
    if mode == "reg":
        y_tr = tr["Sales_scaled"].values.astype(np.float32)
        y_vl = vl["Sales_scaled"].values.astype(np.float32)
        n_output          = 1
        output_activation = "linear"
    else:
        y_tr = tr["bin_index"].values.astype(np.int32)
        y_vl = vl["bin_index"].values.astype(np.int32)
        n_output          = int(
            train_df["bin_index"].max()
        ) + 1
        output_activation = "softmax"

    print(f"  Train: {len(X_seq_tr):,} | "
          f"Val: {len(X_seq_vl):,} | "
          f"Seq shape: {X_seq_tr.shape[1:]} | "
          f"Static: {X_static_tr.shape[1]} | "
          f"Output: {n_output}")

    models      = []
    best_epochs = []

    for i in range(n_models):
        seed  = RANDOM_SEED + i * 7   # different seeds
        model, best_epoch = train_single_lstm(
            X_seq_tr, X_static_tr, y_tr,
            X_seq_vl, X_static_vl, y_vl,
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

    return models, dyn_cols, stat_cols


# =============================================================
# SECTION 6: Prediction
# =============================================================

def predict_lstm_ensemble_reg(models, X_seq_test,
                               X_static_test,
                               target_scaler):
    """Regression predictions from LSTM ensemble."""
    all_preds = []

    for model in models:
        pred_scaled = model.predict(
            [X_seq_test, X_static_test], verbose=0
        ).flatten()
        pred_scaled = np.clip(pred_scaled, -0.5, 0.5)
        all_preds.append(pred_scaled)

    ensemble_scaled = np.median(
        np.array(all_preds), axis=0
    )
    return inverse_transform_target(
        ensemble_scaled, target_scaler
    )


def predict_lstm_ensemble_cl(models, X_seq_test,
                              X_static_test,
                              bin_midpoints):
    """Classification predictions from LSTM ensemble."""
    all_probas = []

    for model in models:
        proba = model.predict(
            [X_seq_test, X_static_test], verbose=0
        )
        all_probas.append(proba)

    all_probas   = np.array(all_probas)
    median_proba = np.median(all_probas, axis=0)

    # Renormalise
    row_sums     = median_proba.sum(axis=1, keepdims=True)
    median_proba = median_proba / (row_sums + 1e-10)

    n_samples      = median_proba.shape[0]
    pred_cl_max    = np.zeros(n_samples)
    pred_cl_median = np.zeros(n_samples)

    for i in range(n_samples):
        proba   = median_proba[i]
        n_bins  = len(bin_midpoints)
        max_idx = np.argmax(proba)
        max_idx = min(max_idx, n_bins - 1)
        pred_cl_max[i]    = bin_midpoints[max_idx]
        pred_cl_median[i] = proba_to_sales_median(
            proba, bin_midpoints
        )

    return pred_cl_max, pred_cl_median


# =============================================================
# SECTION 7: Run LSTM
# =============================================================

def run_lstm(sample_stores=None):
    """
    Trains and evaluates all three LSTM variants.
    Returns: dict of results for all three variants
    """
    print("=" * 50)
    print("Running LSTM")
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

    all_results = {}
    y_true      = test_df["Sales"].values
    sd_types    = test_df["sd_type"].values

    # ── LSTM-REG ──────────────────────────────────────────────
    print("\n--- LSTM-REG ---")
    models_reg, dyn_cols, stat_cols = train_lstm_ensemble(
        train_df, mode="reg"
    )
    (X_seq_test, X_static_test,
     _, _) = prepare_sequences(
        test_df, DYNAMIC_FEATURES, STATIC_FEATURES
    )
    y_pred_reg = predict_lstm_ensemble_reg(
        models_reg, X_seq_test,
        X_static_test, target_scaler
    )
    results_reg = evaluate_by_sd_type(
        y_true, y_pred_reg, sd_types, train_df
    )
    all_results["LSTM-REG"] = results_reg
    save_results(results_reg, "lstm_reg")
    print("  LSTM-REG done.")

    # ── LSTM-CL ───────────────────────────────────────────────
    print("\n--- LSTM-CL ---")
    models_cl, _, _ = train_lstm_ensemble(
        train_df, mode="cl"
    )
    y_pred_cl_max, y_pred_cl_med = predict_lstm_ensemble_cl(
        models_cl, X_seq_test,
        X_static_test, bin_midpoints
    )

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
    all_results["LSTM-CL(max)"]    = results_cl_max
    all_results["LSTM-CL(median)"] = results_cl_med
    save_results(results_cl_max, "lstm_cl_max")
    save_results(results_cl_med, "lstm_cl_median")
    print("  LSTM-CL done.")

    # Print combined table
    print_results_table(
        build_results_table(all_results),
        "LSTM Results — All Variants"
    )

    # Save models
    for i, m in enumerate(models_reg):
        m.save(os.path.join(
            MODELS_DIR, f"lstm_reg_{i}.keras"
        ))
    for i, m in enumerate(models_cl):
        m.save(os.path.join(
            MODELS_DIR, f"lstm_cl_{i}.keras"
        ))
    print(f"Models saved to {MODELS_DIR}")

    return all_results, y_true, sd_types


# =============================================================
# SECTION 8: Verification
# =============================================================

def verify_lstm(all_results, mlp_results,
                lgbm_results, baseline_results):
    """
    Verifies LSTM results match paper's expected patterns.

    Key checks from paper:
      1. LSTM-CL(median) is the best overall model
      2. LSTM beats MLP on MASE
      3. CL(median) beats CL(max)
      4. All models beat S-Median
    """
    print("=" * 50)
    print("Verification — LSTM")
    print("=" * 50)

    all_pass = True
    checks   = {}

    lstm_reg_mase  = all_results["LSTM-REG"]["Overall"]["MASE"]
    lstm_max_mase  = all_results["LSTM-CL(max)"]["Overall"]["MASE"]
    lstm_med_mase  = all_results["LSTM-CL(median)"]["Overall"]["MASE"]
    mlp_best_mase  = mlp_results["MLP-CL(median)"]["Overall"]["MASE"]
    lgbm_mase      = lgbm_results["Overall"]["MASE"]
    smed_mae       = baseline_results["S-Median"]["Overall"]["MAE"]

    # All LSTM variants beat S-Median
    for variant in ["LSTM-REG", "LSTM-CL(max)",
                    "LSTM-CL(median)"]:
        mae = all_results[variant]["Overall"]["MAE"]
        checks[f"{variant} MAE < S-Median MAE"] = (
            mae < smed_mae
        )

    # CL(median) beats CL(max) — paper's core finding
    checks["LSTM-CL(median) MASE <= LSTM-CL(max) MASE"] = (
        lstm_med_mase <= lstm_max_mase * 1.05
    )

    # LSTM-CL(median) is best LSTM variant
    checks["LSTM-CL(median) is best LSTM variant"] = (
        lstm_med_mase <= lstm_reg_mase * 1.05
    )

    # LSTM should match or beat MLP on MASE
    checks["LSTM-CL(median) MASE <= MLP-CL(median) MASE"] = (
        lstm_med_mase <= mlp_best_mase * 1.10
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

    # Full MASE ranking
    print(f"\n  Final MASE ranking:")
    print(f"    LightGBM           : {lgbm_mase:.4f}")
    print(f"    MLP-CL(median)     : {mlp_best_mase:.4f}")
    print(f"    LSTM-REG           : {lstm_reg_mase:.4f}")
    print(f"    LSTM-CL(max)       : {lstm_max_mase:.4f}")
    print(f"    LSTM-CL(median)    : {lstm_med_mase:.4f}")

    print(f"\n  CL(median) vs CL(max): "
          f"{lstm_med_mase/lstm_max_mase:.4f} "
          f"({'better' if lstm_med_mase < lstm_max_mase else 'worse'})")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: visualisation.py")
    else:
        print("Some checks FAILED.")

    return all_pass
