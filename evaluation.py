
# =============================================================
# evaluation.py — Computes MASE and MAE per SD type
#
# WHY this module exists as a standalone file:
#   All 9 models must be evaluated with identical logic.
#   If evaluation code is duplicated inside each model file,
#   a bug fix requires changing 9 files. Here it is fixed
#   in one place.
#
# PRIMARY METRICS (from paper Section 5.1):
#   MASE — Seasonal Mean Absolute Scaled Error
#   MAE  — Mean Absolute Error
#
# WHY report per SD type (not just overall):
#   The paper's key finding is that ML methods improve
#   accuracy SPECIFICALLY on special days. If you only
#   report overall accuracy, special day improvements
#   are hidden by the large volume of regular days.
#   A model could be terrible on Christmas and still
#   look "good" overall.
#
# WHY MASE specifically:
#   MAE is scale-dependent — a store selling 1000 units/day
#   has naturally higher MAE than one selling 100 units/day.
#   MASE scales each error by that store's own baseline
#   difficulty, making results comparable across stores.
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import RESULTS_DIR, SEASON_M, SD_TYPES


# =============================================================
# SECTION 1: Core metric functions
# =============================================================

def compute_mae(y_true, y_pred):
    """
    Mean Absolute Error.

    MAE = (1/N) * Σ |y_true - y_pred|

    WHY MAE over MSE:
      MSE penalises large errors quadratically — a single
      very wrong prediction on Christmas day would dominate
      the entire score. MAE treats all errors linearly,
      giving a more honest picture of typical accuracy.

    Args:
        y_true: array of actual sales values (real units)
        y_pred: array of predicted sales values (real units)

    Returns: float
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def compute_mase(y_true, y_pred, y_train_sd0, m=SEASON_M):
    """
    Seasonal Mean Absolute Scaled Error (Hyndman & Koehler 2006)
    — the paper's primary metric.

    Formula:
      MASE = MAE(model) / MAE(seasonal_naive_on_SD0_train)

    Denominator =
      (1/|T_SD0|) * Σ_{t in SD0} |y_t - y_{t-7}|

    WHY the denominator uses ONLY SD0 rows:
      We want a scaling factor that represents "how hard is
      a typical REGULAR day to forecast in this series?"
      Including special days would inflate the denominator
      (they are harder), making MASE look artificially small.
      SD0-only denominator gives a fair, stable baseline.

    WHY m=7 (weekly):
      Retail demand has strong weekly seasonality.
      The seasonal naive forecast (same weekday last week)
      is the natural benchmark for this data.

    MASE < 1: model beats seasonal naive on regular days
    MASE = 1: model equals seasonal naive
    MASE > 1: model is worse than seasonal naive

    Args:
        y_true      : actual sales (real units, any SD type)
        y_pred      : predicted sales (real units, any SD type)
        y_train_sd0 : SD0 training sales for this time series
                      (used to compute denominator)
        m           : seasonality period (7 for weekly)

    Returns: float
    """
    y_true      = np.array(y_true)
    y_pred      = np.array(y_pred)
    y_train_sd0 = np.array(y_train_sd0)

    # Numerator: MAE of this model
    numerator = np.mean(np.abs(y_true - y_pred))

    # Denominator: MAE of seasonal naive on SD0 training data
    if len(y_train_sd0) <= m:
        # Not enough history for seasonal naive
        # Fall back to MAE of naive (mean prediction)
        denominator = np.mean(
            np.abs(y_train_sd0 - np.mean(y_train_sd0))
        )
    else:
        naive_errors = np.abs(
            y_train_sd0[m:] - y_train_sd0[:-m]
        )
        denominator = np.mean(naive_errors)

    if denominator == 0:
        return np.nan

    return numerator / denominator


# =============================================================
# SECTION 2: Per-SD-type evaluation
# =============================================================

def evaluate_by_sd_type(y_true, y_pred, sd_types,
                         train_df, feature_cols=None):
    """
    Computes MAE and MASE for each SD type separately.

    WHY separate evaluation per SD type:
      The paper's Figures 2-13 all show results broken down
      by SD type. This is how you verify your model matches
      the paper's findings:
        - ML methods should show biggest improvement on SD2+SD3
        - SD1 improvement should be smaller but present
        - SD0 and SD4 should be close to each other

    Args:
        y_true     : array of actual sales (real units)
        y_pred     : array of predicted sales (real units)
        sd_types   : array of SD type labels (0-4) per row
        train_df   : training DataFrame (for MASE denominator)

    Returns: dict with structure:
        {
          'SD0': {'MAE': float, 'MASE': float, 'N': int},
          'SD1': {'MAE': float, 'MASE': float, 'N': int},
          ...
          'Overall': {'MAE': float, 'MASE': float, 'N': int}
        }
    """
    y_true   = np.array(y_true)
    y_pred   = np.array(y_pred)
    sd_types = np.array(sd_types)

    # Get SD0 training sales for MASE denominator
    y_train_sd0 = train_df[
        train_df["sd_type"] == 0
    ]["Sales"].values

    results = {}

    for sd in SD_TYPES:
        mask = sd_types == sd
        if mask.sum() == 0:
            results[f"SD{sd}"] = {
                "MAE": np.nan, "MASE": np.nan, "N": 0
            }
            continue

        mae  = compute_mae(y_true[mask], y_pred[mask])
        mase = compute_mase(
            y_true[mask], y_pred[mask], y_train_sd0
        )

        results[f"SD{sd}"] = {
            "MAE" : round(mae,  2),
            "MASE": round(mase, 4),
            "N"   : int(mask.sum())
        }

    # Overall (all SD types combined)
    results["Overall"] = {
        "MAE" : round(compute_mae(y_true, y_pred), 2),
        "MASE": round(
            compute_mase(y_true, y_pred, y_train_sd0), 4
        ),
        "N"   : len(y_true)
    }

    return results


# =============================================================
# SECTION 3: Results table builder
# =============================================================

def build_results_table(all_model_results):
    """
    Builds a summary DataFrame comparing all models.

    Args:
        all_model_results: dict of {model_name: results_dict}
            where results_dict comes from evaluate_by_sd_type()

    Returns: pandas DataFrame with models as columns,
             (SD_type, metric) as rows

    Example:
        results = {
            'S-Naive':  evaluate_by_sd_type(...),
            'LightGBM': evaluate_by_sd_type(...),
        }
        table = build_results_table(results)
    """
    rows = []
    sd_labels = [f"SD{i}" for i in SD_TYPES] + ["Overall"]

    for sd_label in sd_labels:
        for metric in ["MAE", "MASE"]:
            row = {"SD_Type": sd_label, "Metric": metric}
            for model_name, model_results in (
                all_model_results.items()
            ):
                if sd_label in model_results:
                    row[model_name] = (
                        model_results[sd_label].get(metric, np.nan)
                    )
                else:
                    row[model_name] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.set_index(["SD_Type", "Metric"])
    return df


def print_results_table(table, title="Results"):
    """
    Prints the results table in a readable format.
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(table.to_string())
    print(f"{'='*60}\n")


def save_results(results_dict, model_name):
    """
    Saves a single model's results to CSV.

    Args:
        results_dict: output of evaluate_by_sd_type()
        model_name  : string name for the file
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows = []
    for sd_label, metrics in results_dict.items():
        row = {"SD_Type": sd_label}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(
        RESULTS_DIR, f"{model_name}_results.csv"
    )
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# =============================================================
# SECTION 4: Relative performance
# =============================================================

def compute_relative_performance(results_dict,
                                   baseline_results,
                                   metric="MAE"):
    """
    Computes model performance relative to a baseline.

    WHY relative performance:
      The paper reports all ML results as relative to the
      best baseline (ETS pct-cl). This is also done for
      confidentiality — actual sales numbers are not published.
      relative_error < 1.0 means model beats baseline.

    Formula: relative_error = model_metric / baseline_metric

    Returns: dict of {SD_type: relative_error}
    """
    relative = {}
    for sd_label in results_dict:
        model_val    = results_dict[sd_label].get(metric, np.nan)
        baseline_val = baseline_results.get(
            sd_label, {}
        ).get(metric, np.nan)

        if baseline_val and baseline_val != 0:
            relative[sd_label] = round(
                model_val / baseline_val, 4
            )
        else:
            relative[sd_label] = np.nan

    return relative


# =============================================================
# SECTION 5: Statistical significance test
# =============================================================

def wilcoxon_test(errors_model, errors_baseline,
                   alpha=0.05):
    """
    Wilcoxon signed-rank test for statistical significance.

    WHY Wilcoxon (not t-test):
      The paper uses Wilcoxon signed-rank test (Section 5.1).
      It is non-parametric — does not assume normal
      distribution of errors. Forecast errors are often
      non-normal (heavy tails from special day spikes).

    Returns: (statistic, p_value, is_significant bool)
    """
    errors_model    = np.array(errors_model)
    errors_baseline = np.array(errors_baseline)

    # Absolute errors
    abs_model    = np.abs(errors_model)
    abs_baseline = np.abs(errors_baseline)

    try:
        stat, p_value = stats.wilcoxon(
            abs_model, abs_baseline
        )
        is_sig = p_value < alpha
        return stat, p_value, is_sig
    except Exception:
        return np.nan, np.nan, False


# =============================================================
# SECTION 6: Verification
# =============================================================

def verify_evaluation():
    """
    Tests evaluation functions with known synthetic data.
    """
    print("=" * 50)
    print("Verification — evaluation.py")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # ── Test MAE ──────────────────────────────────────────────
    y_true = np.array([100, 200, 150, 300])
    y_pred = np.array([110, 190, 160, 280])
    mae    = compute_mae(y_true, y_pred)
    # Expected: (10+10+10+20)/4 = 12.5
    checks["MAE computes correctly (12.5)"] = (
        abs(mae - 12.5) < 0.01
    )

    # ── Test MASE ─────────────────────────────────────────────
    # Simple case: if model equals seasonal naive, MASE = 1
    y_train_sd0 = np.array(
        [100, 120, 110, 90, 100, 120, 110,  # week 1
          95, 115, 105, 85,  95, 115, 105]   # week 2
    )
    # Seasonal naive predictions for week 2 = week 1 values
    y_true_test = y_train_sd0[7:]
    y_pred_naive = y_train_sd0[:7]
    mase = compute_mase(y_true_test, y_pred_naive, y_train_sd0)
    checks["MASE ≈ 1 for seasonal naive"] = (
        abs(mase - 1.0) < 0.1
    )

    # ── Test evaluate_by_sd_type ──────────────────────────────
    import pandas as pd
    n = 100
    np.random.seed(42)
    mock_true    = np.random.randint(50, 500, n).astype(float)
    mock_pred    = mock_true + np.random.randn(n) * 20
    mock_sd      = np.random.choice([0,1,2,3,4], n)
    mock_train   = pd.DataFrame({
        "sd_type": [0]*200,
        "Sales"  : np.random.randint(50, 500, 200).astype(float)
    })

    results = evaluate_by_sd_type(
        mock_true, mock_pred, mock_sd, mock_train
    )
    checks["evaluate_by_sd_type returns all SD types"] = (
        all(f"SD{i}" in results for i in range(5))
    )
    checks["Overall key present"] = "Overall" in results
    checks["MAE > 0"] = results["Overall"]["MAE"] > 0

    # ── Test relative performance ─────────────────────────────
    baseline = {"Overall": {"MAE": 100.0}}
    model    = {"Overall": {"MAE":  80.0}}
    rel      = compute_relative_performance(model, baseline)
    checks["Relative performance = 0.8"] = (
        abs(rel["Overall"] - 0.8) < 0.01
    )

    # ── Test results table builder ────────────────────────────
    all_results = {
        "Model_A": results,
        "Model_B": results,
    }
    table = build_results_table(all_results)
    checks["Results table has correct shape"] = (
        table.shape[0] == 12  # 6 SD labels × 2 metrics
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
        print("Next step: baselines.py")
    else:
        print("Some checks FAILED — fix before proceeding.")

    return all_pass
