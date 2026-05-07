
# =============================================================
# visualisation.py — Replicates paper's key figures
#
# Produces 4 plots:
#   1. Full results table (all models × all SD types)
#   2. MAE per SD type bar chart (Figures 2-7 equivalent)
#   3. MASE per SD type bar chart
#   4. Feature importance from LightGBM
#
# WHY visualise per SD type separately:
#   The paper's main argument is that ML improves accuracy
#   SPECIFICALLY on special days. Aggregated numbers hide
#   this. The per-SD plots make the story visible.
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import RESULTS_DIR, PLOTS_DIR, MODELS_DIR
from evaluation import build_results_table


# =============================================================
# SECTION 1: Load all results
# =============================================================

def load_all_results():
    """
    Loads all model results from CSV files.
    Returns: dict of {model_name: results_dict}
    """
    def load_csv(name):
        path = os.path.join(
            RESULTS_DIR, f"{name}_results.csv"
        )
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found — skipping")
            return None
        df = pd.read_csv(path)
        results = {}
        for _, row in df.iterrows():
            results[row["SD_Type"]] = {
                "MAE" : row["MAE"],
                "MASE": row["MASE"],
                "N"   : row["N"]
            }
        return results

    models = {
        "S-Naive"        : load_csv("snaive"),
        "S-Naive-Std"    : load_csv("snaive_std"),
        "S-Median"       : load_csv("smedian"),
        "LIN-REG"        : load_csv("linreg"),
        "LightGBM"       : load_csv("lgbm"),
        "MLP-REG"        : load_csv("mlp_reg"),
        "MLP-CL(max)"    : load_csv("mlp_cl_max"),
        "MLP-CL(median)" : load_csv("mlp_cl_median"),
        "LSTM-REG"       : load_csv("lstm_reg"),
        "LSTM-CL(max)"   : load_csv("lstm_cl_max"),
        "LSTM-CL(median)": load_csv("lstm_cl_median"),
    }

    # Remove None entries
    return {k: v for k, v in models.items() if v is not None}


# =============================================================
# SECTION 2: Full results table
# =============================================================

def plot_results_table(all_results):
    """
    Saves the complete results table as a formatted CSV
    and prints it to console.

    This is your main presentation table — equivalent to
    Figures 8-13 in the paper combined into one view.
    """
    print("=" * 70)
    print("COMPLETE RESULTS TABLE")
    print("=" * 70)

    # Build table
    sd_labels  = ["SD0", "SD1", "SD2", "SD3", "SD4", "Overall"]
    model_names = list(all_results.keys())

    # MAE table
    mae_rows = []
    for sd in sd_labels:
        row = {"SD_Type": sd}
        for model in model_names:
            val = all_results[model].get(sd, {}).get("MAE", np.nan)
            row[model] = round(val, 1) if not np.isnan(val) else "-"
        mae_rows.append(row)

    mae_df = pd.DataFrame(mae_rows).set_index("SD_Type")

    # MASE table
    mase_rows = []
    for sd in sd_labels:
        row = {"SD_Type": sd}
        for model in model_names:
            val = all_results[model].get(sd, {}).get("MASE", np.nan)
            row[model] = round(val, 4) if not np.isnan(val) else "-"
        mase_rows.append(row)

    mase_df = pd.DataFrame(mase_rows).set_index("SD_Type")

    print("\nMAE Table:")
    print(mae_df.to_string())
    print("\nMASE Table:")
    print(mase_df.to_string())

    # Save to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    mae_df.to_csv(os.path.join(RESULTS_DIR, "full_mae_table.csv"))
    mase_df.to_csv(os.path.join(RESULTS_DIR, "full_mase_table.csv"))
    print(f"\nSaved tables to {RESULTS_DIR}")

    return mae_df, mase_df


# =============================================================
# SECTION 3: Bar chart per SD type
# =============================================================

def plot_sd_comparison(all_results, metric="MAE"):
    """
    Creates a grouped bar chart showing model performance
    per SD type — equivalent to Figures 2-7 in the paper.

    WHY this plot matters for your presentation:
      It visually shows that ML methods provide the largest
      improvements specifically on SD2 and SD3 (neighboring
      days). This is the paper's main empirical finding.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sd_labels   = ["SD0", "SD1", "SD2", "SD3", "SD4", "Overall"]
    model_names = list(all_results.keys())
    n_models    = len(model_names)
    n_sd        = len(sd_labels)

    # Color scheme: baselines gray, ML models colored
    colors = {
        "S-Naive"        : "#B4B2A9",
        "S-Naive-Std"    : "#888780",
        "S-Median"       : "#5F5E5A",
        "LIN-REG"        : "#85B7EB",
        "LightGBM"       : "#1D9E75",
        "MLP-REG"        : "#AFA9EC",
        "MLP-CL(max)"    : "#7F77DD",
        "MLP-CL(median)" : "#534AB7",
        "LSTM-REG"       : "#F0997B",
        "LSTM-CL(max)"   : "#D85A30",
        "LSTM-CL(median)": "#993C1D",
    }

    fig, axes = plt.subplots(
        2, 3, figsize=(18, 10),
        sharey=False
    )
    axes = axes.flatten()

    for idx, sd in enumerate(sd_labels):
        ax     = axes[idx]
        values = []
        cols   = []
        names  = []

        for model in model_names:
            val = all_results[model].get(sd, {}).get(
                metric, np.nan
            )
            if not np.isnan(val):
                values.append(val)
                cols.append(
                    colors.get(model, "#888780")
                )
                names.append(model)

        x = np.arange(len(names))
        bars = ax.bar(x, values, color=cols,
                      width=0.7, edgecolor="white",
                      linewidth=0.5)

        ax.set_title(
            f"{sd} — {metric}",
            fontsize=11, fontweight="bold", pad=8
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            names, rotation=45, ha="right", fontsize=7
        )
        ax.set_ylabel(metric, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.01,
                f"{val:.0f}",
                ha="center", va="bottom",
                fontsize=6, color="#444441"
            )

    plt.suptitle(
        f"Model Comparison by SD Type — {metric}\n"
        f"(Lower is better)",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()

    path = os.path.join(PLOTS_DIR, f"sd_comparison_{metric.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# =============================================================
# SECTION 4: Relative improvement over S-Median
# =============================================================

def plot_relative_improvement(all_results, metric="MAE"):
    """
    Shows ML model improvement RELATIVE to S-Median.
    Values < 1.0 mean the model beats S-Median.

    WHY relative improvement:
      The paper reports relative errors for confidentiality.
      More importantly, relative plots immediately show
      WHICH SD types benefit most from ML.
      This is your strongest presentation slide.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sd_labels  = ["SD0", "SD1", "SD2", "SD3", "SD4", "Overall"]
    ml_models  = [
        "LIN-REG", "LightGBM",
        "MLP-REG", "MLP-CL(median)",
        "LSTM-REG", "LSTM-CL(median)"
    ]
    ml_models  = [m for m in ml_models
                  if m in all_results]

    colors = {
        "LIN-REG"        : "#85B7EB",
        "LightGBM"       : "#1D9E75",
        "MLP-REG"        : "#AFA9EC",
        "MLP-CL(median)" : "#534AB7",
        "LSTM-REG"       : "#F0997B",
        "LSTM-CL(median)": "#993C1D",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    x       = np.arange(len(sd_labels))
    width   = 0.13
    offsets = np.linspace(
        -(len(ml_models)-1)*width/2,
        (len(ml_models)-1)*width/2,
        len(ml_models)
    )

    for i, model in enumerate(ml_models):
        rel_vals = []
        for sd in sd_labels:
            model_val  = all_results[model].get(
                sd, {}
            ).get(metric, np.nan)
            smedian_val = all_results.get(
                "S-Median", {}
            ).get(sd, {}).get(metric, np.nan)

            if (not np.isnan(model_val) and
                not np.isnan(smedian_val) and
                smedian_val != 0):
                rel_vals.append(model_val / smedian_val)
            else:
                rel_vals.append(np.nan)

        ax.bar(
            x + offsets[i], rel_vals,
            width=width,
            label=model,
            color=colors.get(model, "#888780"),
            edgecolor="white", linewidth=0.5
        )

    # Reference line at 1.0 (= S-Median level)
    ax.axhline(
        y=1.0, color="#E24B4A",
        linewidth=1.5, linestyle="--",
        label="S-Median (baseline = 1.0)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(sd_labels, fontsize=11)
    ax.set_ylabel(f"Relative {metric} vs S-Median",
                  fontsize=11)
    ax.set_title(
        f"ML Model Performance Relative to S-Median\n"
        f"(< 1.0 = beats S-Median, lower is better)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, 1.5)

    plt.tight_layout()
    path = os.path.join(
        PLOTS_DIR,
        f"relative_improvement_{metric.lower()}.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# =============================================================
# SECTION 5: Feature importance
# =============================================================

def plot_feature_importance():
    """
    Plots LightGBM feature importance.
    Highlights SD-specific features in a different colour
    to show their contribution visually.
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model_path = os.path.join(MODELS_DIR, "lgbm_model.pkl")
    if not os.path.exists(model_path):
        print("LightGBM model not found — skipping")
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    importance = pd.DataFrame({
        "feature"   : model.feature_name(),
        "importance": model.feature_importance(
            importance_type="gain"
        )
    }).sort_values("importance", ascending=True).tail(25)

    # Colour SD-specific features differently
    sd_feature_keywords = [
        "sd_level", "sd_abs", "sd_rel",
        "IsSD", "sd_type"
    ]
    colors = [
        "#1D9E75" if any(
            kw in feat for kw in sd_feature_keywords
        ) else "#85B7EB"
        for feat in importance["feature"]
    ]

    fig, ax = plt.subplots(figsize=(10, 9))
    bars = ax.barh(
        importance["feature"],
        importance["importance"],
        color=colors, edgecolor="white", linewidth=0.5
    )

    ax.set_xlabel("Feature Importance (Gain)", fontsize=11)
    ax.set_title(
        "LightGBM — Top 25 Features\n"
        "(green = SD-specific features)",
        fontsize=12, fontweight="bold"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.3)

    # Legend
    sd_patch  = mpatches.Patch(
        color="#1D9E75", label="SD-specific features"
    )
    gen_patch = mpatches.Patch(
        color="#85B7EB", label="General features"
    )
    ax.legend(handles=[sd_patch, gen_patch],
              fontsize=10, loc="lower right")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# =============================================================
# SECTION 6: Summary heatmap
# =============================================================

def plot_mase_heatmap(all_results):
    """
    Creates a heatmap of MASE values — all models × SD types.
    The single most useful slide for your presentation.
    Green = good (low MASE), Red = bad (high MASE).
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    sd_labels   = ["SD0", "SD1", "SD2", "SD3", "SD4", "Overall"]
    model_order = [
        "S-Naive", "S-Naive-Std", "S-Median",
        "LIN-REG", "LightGBM",
        "MLP-REG", "MLP-CL(max)", "MLP-CL(median)",
        "LSTM-REG", "LSTM-CL(max)", "LSTM-CL(median)"
    ]
    model_order = [m for m in model_order
                   if m in all_results]

    # Build matrix
    matrix = []
    for model in model_order:
        row = []
        for sd in sd_labels:
            val = all_results[model].get(sd, {}).get(
                "MASE", np.nan
            )
            row.append(val)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(
        figsize=(10, len(model_order) * 0.7 + 2)
    )

    # Green-White-Red colormap (good=green, bad=red)
    im = ax.imshow(
        matrix, cmap="RdYlGn_r",
        aspect="auto", vmin=0.2, vmax=1.2
    )

    ax.set_xticks(np.arange(len(sd_labels)))
    ax.set_yticks(np.arange(len(model_order)))
    ax.set_xticklabels(sd_labels, fontsize=11)
    ax.set_yticklabels(model_order, fontsize=10)

    # Add value annotations
    for i in range(len(model_order)):
        for j in range(len(sd_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = (
                    "white" if val > 0.9 or val < 0.3
                    else "black"
                )
                ax.text(
                    j, i, f"{val:.3f}",
                    ha="center", va="center",
                    fontsize=9, color=text_color,
                    fontweight="bold"
                )

    plt.colorbar(im, ax=ax, shrink=0.8,
                 label="MASE (lower = better)")
    ax.set_title(
        "MASE Heatmap — All Models × SD Types\n"
        "(Green = better, Red = worse)",
        fontsize=12, fontweight="bold", pad=15
    )

    # Draw line separating baselines from ML models
    ax.axhline(y=2.5, color="white",
               linewidth=2, linestyle="-")
    ax.text(
        -0.8, 1.0, "Baselines",
        fontsize=8, color="#5F5E5A",
        rotation=90, va="center"
    )
    ax.text(
        -0.8, 6.0, "ML Models",
        fontsize=8, color="#534AB7",
        rotation=90, va="center"
    )

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "mase_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# =============================================================
# SECTION 7: Master function
# =============================================================

def run_visualisation():
    """
    Runs all visualisations and prints the full results table.
    """
    print("=" * 50)
    print("Visualisation")
    print("=" * 50)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Load all results
    all_results = load_all_results()
    print(f"\nLoaded results for: "
          f"{list(all_results.keys())}")

    # 1. Full results table
    print("\n--- Results Table ---")
    mae_df, mase_df = plot_results_table(all_results)

    # 2. MAE bar chart per SD type
    print("\n--- MAE Bar Chart ---")
    plot_sd_comparison(all_results, metric="MAE")

    # 3. MASE bar chart per SD type
    print("\n--- MASE Bar Chart ---")
    plot_sd_comparison(all_results, metric="MASE")

    # 4. Relative improvement over S-Median
    print("\n--- Relative Improvement Plot ---")
    plot_relative_improvement(all_results, metric="MAE")

    # 5. Feature importance
    print("\n--- Feature Importance ---")
    plot_feature_importance()

    # 6. MASE heatmap
    print("\n--- MASE Heatmap ---")
    plot_mase_heatmap(all_results)

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("\nProject complete.")

    return all_results
