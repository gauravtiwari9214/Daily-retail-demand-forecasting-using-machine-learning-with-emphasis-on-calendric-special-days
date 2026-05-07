
# =============================================================
# advanced_visualisation.py — Extended visualisations
#
# COMPLETELY SEPARATE from visualisation.py
# Does not import or modify any existing module.
# Safe to run at any point after the pipeline is complete.
#
# Contents:
#   1.  Dataset snapshot     — what the data looks like
#   2.  Sales distribution   — before/after log transform
#   3.  Weekly seasonality   — the pattern we are modelling
#   4.  Special day calendar — which days are SD1/2/3/4
#   5.  Feature correlation  — heatmap of feature relationships
#   6.  Lag autocorrelation  — why lag features work
#   7.  Model error timeline — MAE per day (Figure 1 replica)
#   8.  SD type deep dive    — error distribution per SD type
#   9.  CL median vs max     — probability distribution visual
#   10. Feature importance   — extended with SD highlighting
#   11. Learning curves      — training vs validation loss
#   12. Prediction vs actual — scatter plots per model
#   13. Animated sales chart — stock-market style time series
#   14. Model ranking radar  — spider chart across SD types
# =============================================================

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import (
    PROCESSED_DIR, RESULTS_DIR, PLOTS_DIR,
    MODELS_DIR, LAG_DAYS
)

ADV_PLOTS_DIR = "/kaggle/working/retail_forecasting/outputs/plots/advanced"
os.makedirs(ADV_PLOTS_DIR, exist_ok=True)


# =============================================================
# HELPER: load results from CSV
# =============================================================

def load_result(name):
    path = os.path.join(RESULTS_DIR, f"{name}_results.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        out[row["SD_Type"]] = {
            "MAE" : row["MAE"],
            "MASE": row["MASE"],
            "N"   : row["N"]
        }
    return out


def load_all_model_results():
    names = {
        "S-Naive"        : "snaive",
        "S-Naive-Std"    : "snaive_std",
        "S-Median"       : "smedian",
        "LIN-REG"        : "linreg",
        "LightGBM"       : "lgbm",
        "MLP-REG"        : "mlp_reg",
        "MLP-CL(max)"    : "mlp_cl_max",
        "MLP-CL(median)" : "mlp_cl_median",
        "LSTM-REG"       : "lstm_reg",
        "LSTM-CL(max)"   : "lstm_cl_max",
        "LSTM-CL(median)": "lstm_cl_median",
    }
    return {k: load_result(v)
            for k, v in names.items()
            if load_result(v) is not None}


# =============================================================
# 1. DATASET SNAPSHOT
# =============================================================

def plot_dataset_snapshot():
    """
    Shows the first 10 rows of each processed file so you
    can see exactly what the data looks like at each stage.
    Saves a formatted table as an image.
    """
    print("1. Dataset Snapshot...")

    files = {
        "Raw data"          : "rossmann_raw.parquet",
        "After SD labels"   : "rossmann_with_sd.parquet",
        "After features Pt1": "features_part1.parquet",
        "Complete features" : "features_complete.parquet",
        "Train (model input)": "train.parquet",
    }

    fig, axes = plt.subplots(
        len(files), 1,
        figsize=(20, len(files) * 3.5)
    )

    for ax, (title, fname) in zip(axes, files.items()):
        path = os.path.join(PROCESSED_DIR, fname)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"{fname} not found",
                    ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(title)
            continue

        df   = pd.read_parquet(path)
        head = df.head(6)

        # Select key columns to display (avoid overflow)
        key_cols = [c for c in [
            "Store", "Date", "Sales", "sd_type",
            "DayOfWeek", "Promo", "StoreType",
            "Sales_lag_7", "Sales_rolling_median",
            "sd_rel_change", "Sales_scaled", "bin_index"
        ] if c in head.columns][:10]

        head_display = head[key_cols].copy()

        # Round floats for readability
        for col in head_display.select_dtypes(
            include=[float]
        ).columns:
            head_display[col] = head_display[col].round(3)

        ax.axis("off")
        table = ax.table(
            cellText=head_display.values,
            colLabels=head_display.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)

        # Style header
        for j in range(len(head_display.columns)):
            table[0, j].set_facecolor("#534AB7")
            table[0, j].set_text_props(
                color="white", fontweight="bold"
            )

        # Alternate row colours
        for i in range(1, len(head_display) + 1):
            for j in range(len(head_display.columns)):
                if i % 2 == 0:
                    table[i, j].set_facecolor("#F1EFE8")

        ax.set_title(
            f"{title}  |  shape: {df.shape}  |  "
            f"columns: {df.shape[1]}",
            fontsize=10, fontweight="bold",
            pad=8, loc="left"
        )

    plt.suptitle(
        "Dataset at each pipeline stage",
        fontsize=14, fontweight="bold", y=1.002
    )
    plt.tight_layout()
    path = os.path.join(ADV_PLOTS_DIR,
                        "01_dataset_snapshot.png")
    plt.savefig(path, dpi=120,
                bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 2. SALES DISTRIBUTION — before and after transforms
# =============================================================

def plot_sales_distribution():
    """
    Shows why log transform is needed.
    Side by side: raw sales histogram vs log-transformed.
    """
    print("2. Sales distribution...")

    train = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet")
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Raw sales
    ax = axes[0]
    ax.hist(train["Sales"].clip(0, 10000),
            bins=80, color="#85B7EB",
            edgecolor="white", linewidth=0.3)
    ax.set_title("Raw Sales", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("Sales (units)")
    ax.set_ylabel("Frequency")
    ax.axvline(train["Sales"].mean(), color="#E24B4A",
               linestyle="--", linewidth=1.5,
               label=f"Mean: {train['Sales'].mean():.0f}")
    ax.axvline(train["Sales"].median(), color="#1D9E75",
               linestyle="--", linewidth=1.5,
               label=f"Median: {train['Sales'].median():.0f}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # Annotation
    ax.text(0.6, 0.8,
            "Right-skewed\nNeural nets struggle\nwith this shape",
            transform=ax.transAxes, fontsize=9,
            color="#993C1D",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FAECE7",
                      edgecolor="#993C1D", alpha=0.8))

    # Log-transformed
    log_sales = np.log1p(train["Sales"])
    ax = axes[1]
    ax.hist(log_sales, bins=80, color="#1D9E75",
            edgecolor="white", linewidth=0.3)
    ax.set_title("Log(Sales + 1)", fontsize=12,
                 fontweight="bold")
    ax.set_xlabel("log(Sales + 1)")
    ax.set_ylabel("Frequency")
    ax.axvline(log_sales.mean(), color="#E24B4A",
               linestyle="--", linewidth=1.5,
               label=f"Mean: {log_sales.mean():.2f}")
    ax.axvline(log_sales.median(), color="#534AB7",
               linestyle="--", linewidth=1.5,
               label=f"Median: {log_sales.median():.2f}")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.05, 0.8,
            "More symmetric\nNeural nets train\nmuch better now",
            transform=ax.transAxes, fontsize=9,
            color="#0F6E56",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#E1F5EE",
                      edgecolor="#0F6E56", alpha=0.8))

    # Scaled
    ax = axes[2]
    ax.hist(train["Sales_scaled"], bins=80,
            color="#534AB7",
            edgecolor="white", linewidth=0.3)
    ax.set_title("Scaled to [-0.5, 0.5]",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Scaled value")
    ax.set_ylabel("Frequency")
    ax.axvline(0, color="#E24B4A", linestyle="--",
               linewidth=1.5, label="Zero")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.05, 0.8,
            "Ready for neural\nnetwork training\n[-0.5, 0.5] range",
            transform=ax.transAxes, fontsize=9,
            color="#26215C",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#EEEDFE",
                      edgecolor="#26215C", alpha=0.8))

    plt.suptitle(
        "Why we transform the target variable",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(ADV_PLOTS_DIR,
                        "02_sales_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 3. WEEKLY SEASONALITY — the pattern we model
# =============================================================

def plot_weekly_seasonality():
    """
    Shows average sales by day of week for each store type.
    Demonstrates why lag-7 is such a strong feature.
    """
    print("3. Weekly seasonality...")

    df = pd.read_parquet(
        os.path.join(PROCESSED_DIR,
                     "features_complete.parquet")
    )
    sd0 = df[df["sd_type"] == 0]

    day_names = ["Mon", "Tue", "Wed",
                 "Thu", "Fri", "Sat", "Sun"]
    store_types = sorted(sd0["StoreType"].unique())
    colors = ["#534AB7", "#1D9E75", "#D85A30", "#BA7517"]

    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5)
    )

    # Left: by store type
    ax = axes[0]
    for i, stype in enumerate(store_types):
        subset = sd0[sd0["StoreType"] == stype]
        daily  = subset.groupby("DayOfWeek")["Sales"].mean()
        ax.plot(daily.index, daily.values,
                marker="o", linewidth=2,
                markersize=5, color=colors[i],
                label=f"Type {stype}")

    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_title("Average Sales by Day of Week\n"
                 "(SD0 regular days only)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Sales")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.95,
            "Strong weekly pattern =\nlag-7 is highly predictive",
            transform=ax.transAxes, fontsize=9,
            color="#534AB7", va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#EEEDFE",
                      edgecolor="#534AB7", alpha=0.8))

    # Right: SD0 vs SD1 vs SD2 comparison
    ax = axes[1]
    sd_colors = {
        0: "#85B7EB", 1: "#E24B4A",
        2: "#EF9F27", 3: "#1D9E75", 4: "#888780"
    }
    sd_names = {
        0: "SD0 Regular", 1: "SD1 Holiday",
        2: "SD2 Day Before", 3: "SD3 Day After",
        4: "SD4 Week After"
    }
    for sd in [0, 1, 2, 3, 4]:
        subset = df[df["sd_type"] == sd]
        if len(subset) < 100:
            continue
        daily = subset.groupby(
            "DayOfWeek"
        )["Sales"].mean()
        ax.plot(daily.index, daily.values,
                marker="o", linewidth=2,
                markersize=5,
                color=sd_colors[sd],
                label=sd_names[sd],
                alpha=0.85)

    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names)
    ax.set_title("Average Sales by Day of Week\n"
                 "Comparing all SD types",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Sales")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Weekly Seasonality Pattern",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(ADV_PLOTS_DIR,
                        "03_weekly_seasonality.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 4. LAG AUTOCORRELATION — why lag features work
# =============================================================

def plot_lag_autocorrelation():
    """
    Shows correlation between Sales and each lag feature.
    Visually proves why lag-7 is the strongest predictor.
    """
    print("4. Lag autocorrelation...")

    train = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "train.parquet")
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of correlations
    ax = axes[0]
    lag_cols = [f"Sales_lag_{l}" for l in LAG_DAYS
                if f"Sales_lag_{l}" in train.columns]
    corrs = [train["Sales"].corr(train[c])
             for c in lag_cols]
    lag_labels = [f"Lag {l}" for l in LAG_DAYS
                  if f"Sales_lag_{l}" in train.columns]

    bar_colors = ["#E24B4A" if c == max(corrs)
                  else "#85B7EB" for c in corrs]
    bars = ax.bar(lag_labels, corrs,
                  color=bar_colors,
                  edgecolor="white", linewidth=0.5)

    ax.set_title("Correlation: Sales vs Lag Features",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("Pearson Correlation")
    ax.set_xticklabels(lag_labels, rotation=30,
                       ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, corr in zip(bars, corrs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{corr:.3f}",
                ha="center", va="bottom",
                fontsize=8)

    ax.text(0.02, 0.05,
            "Lag-7 is strongest\n(same weekday = same pattern)",
            transform=ax.transAxes, fontsize=9,
            color="#A32D2D",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="#FCEBEB",
                      edgecolor="#A32D2D", alpha=0.8))

    # Right: scatter plot Sales vs lag-7
    ax = axes[1]
    sample = train.dropna(
        subset=["Sales_lag_7"]
    ).sample(3000, random_state=42)

    ax.scatter(sample["Sales_lag_7"],
               sample["Sales"],
               alpha=0.15, s=8,
               color="#534AB7")

    # Regression line
    x = sample["Sales_lag_7"].values
    y = sample["Sales"].values
    m, b = np.polyfit(x, y, 1)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, m * x_line + b,
            color="#E24B4A", linewidth=2,
            label=f"y = {m:.2f}x + {b:.0f}")

    ax.set_xlabel("Sales (7 days ago)")
    ax.set_ylabel("Sales (today)")
    ax.set_title("Sales Today vs Sales Last Week\n"
                 "(sample of 3,000 observations)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    corr_val = train["Sales"].corr(
        train["Sales_lag_7"]
    )
    ax.text(0.05, 0.92,
            f"r = {corr_val:.3f}",
            transform=ax.transAxes, fontsize=11,
            fontweight="bold", color="#534AB7")

    plt.suptitle(
        "Autocorrelation — why lag features are powerful",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(ADV_PLOTS_DIR,
                        "04_lag_autocorrelation.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 5. CYCLICAL ENCODING — visualising sin/cos
# =============================================================

def plot_cyclical_encoding():
    """
    Visualises why sin/cos encoding is better than
    raw integer encoding for day of week and month.
    """
    print("5. Cyclical encoding...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    days = np.arange(7)
    day_names = ["Mon","Tue","Wed",
                 "Thu","Fri","Sat","Sun"]
    sin_vals = np.sin(2 * np.pi * days / 7)
    cos_vals = np.cos(2 * np.pi * days / 7)

    # Left: raw integer encoding problem
    ax = axes[0]
    ax.plot(days, days, "o-",
            color="#85B7EB", linewidth=2, markersize=8)
    ax.set_xticks(days)
    ax.set_xticklabels(day_names)
    ax.set_title("Raw integer encoding\n"
                 "(Mon=0, Sun=6 look far apart)",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Encoded value")

    # Draw arrow showing Mon-Sun distance
    ax.annotate("",
                xy=(6, 6), xytext=(0, 0),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="#E24B4A",
                    lw=2
                ))
    ax.text(3, 3.2, "Distance = 6\n(but they are adjacent!)",
            ha="center", color="#E24B4A", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Middle: sin/cos on unit circle
    ax = axes[1]
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta),
            "k-", linewidth=0.8, alpha=0.3)

    colors_days = plt.cm.rainbow(np.linspace(0, 1, 7))
    for i, (s, c, name) in enumerate(
        zip(sin_vals, cos_vals, day_names)
    ):
        ax.plot(c, s, "o",
                color=colors_days[i],
                markersize=12, zorder=5)
        ax.annotate(name, (c, s),
                    textcoords="offset points",
                    xytext=(8, 0),
                    fontsize=8,
                    color=colors_days[i],
                    fontweight="bold")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_title("Sin/Cos encoding\n"
                 "(days on a circle — Mon and Sun adjacent)",
                 fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("cos(2π × day/7)")
    ax.set_ylabel("sin(2π × day/7)")

    # Arrow showing Mon-Sun are now close
    ax.annotate("",
                xy=(cos_vals[6], sin_vals[6]),
                xytext=(cos_vals[0], sin_vals[0]),
                arrowprops=dict(
                    arrowstyle="<->",
                    color="#1D9E75", lw=2
                ))
    dist = np.sqrt(
        (cos_vals[0]-cos_vals[6])**2 +
        (sin_vals[0]-sin_vals[6])**2
    )
    ax.text(-0.5, 0.3,
            f"Distance = {dist:.2f}\n(correctly small!)",
            ha="center", color="#1D9E75", fontsize=8)

    # Right: sin and cos values plotted
    ax = axes[2]
    ax.plot(day_names, sin_vals, "o-",
            color="#534AB7", linewidth=2,
            markersize=8, label="sin(2π×day/7)")
    ax.plot(day_names, cos_vals, "s-",
            color="#D85A30", linewidth=2,
            markersize=8, label="cos(2π×day/7)")
    ax.axhline(0, color="gray",
               linewidth=0.5, linestyle="--")
    ax.set_title("Sin and Cos values per day",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Value")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Cyclical Encoding — why sin/cos beats raw integers",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(ADV_PLOTS_DIR,
                        "05_cyclical_encoding.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 6. SD TYPE CALENDAR — visual calendar heatmap
# =============================================================

def plot_sd_calendar():
    """
    Shows a calendar heatmap of SD types for the test period.
    Makes it visually obvious which days are special.
    """
    print("6. SD calendar heatmap...")

    test = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

    # Get unique date-SD pairs
    date_sd = test.groupby("Date")["sd_type"].first().reset_index()
    date_sd["Date"] = pd.to_datetime(date_sd["Date"])
    date_sd = date_sd.sort_values("Date")

    # Build calendar grid
    dates     = date_sd["Date"].values
    sd_values = date_sd["sd_type"].values

    # Get unique months in test period
    months = sorted(set(
        pd.Timestamp(d).to_period("M")
        for d in dates
    ))

    sd_colors_map = {
        0: "#E6F1FB",   # light blue — regular
        1: "#E24B4A",   # red — holiday
        2: "#EF9F27",   # amber — day before
        3: "#1D9E75",   # green — day after
        4: "#888780",   # gray — week after
    }

    n_months = len(months)
    fig, axes = plt.subplots(
        1, n_months,
        figsize=(n_months * 4, 6)
    )
    if n_months == 1:
        axes = [axes]

    for ax, month in zip(axes, months):
        month_dates = [
            d for d in dates
            if pd.Timestamp(d).to_period("M") == month
        ]
        month_sds = [
            sd_values[list(dates).index(d)]
            for d in month_dates
        ]

        # Build 6×7 grid (6 weeks, 7 days)
        grid      = np.full((6, 7), -1)
        grid_sds  = np.full((6, 7), -1)

        first_date = pd.Timestamp(month_dates[0])
        # Find what weekday the 1st of month is
        first_of_month = pd.Timestamp(
            first_date.year, first_date.month, 1
        )
        start_col = first_of_month.dayofweek

        for ts, sd in zip(month_dates, month_sds):
            t   = pd.Timestamp(ts)
            day = t.day
            col = (start_col + day - 1) % 7
            row = (start_col + day - 1) // 7
            if row < 6:
                grid[row, col]     = day
                grid_sds[row, col] = sd

        # Plot calendar
        ax.set_xlim(-0.5, 6.5)
        ax.set_ylim(-0.5, 5.5)
        ax.invert_yaxis()

        for r in range(6):
            for c in range(7):
                if grid[r, c] == -1:
                    continue
                sd    = int(grid_sds[r, c])
                color = sd_colors_map.get(sd, "white")
                rect  = plt.Rectangle(
                    (c - 0.45, r - 0.45),
                    0.9, 0.9,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.5,
                    zorder=2
                )
                ax.add_patch(rect)
                ax.text(c, r,
                        str(int(grid[r, c])),
                        ha="center", va="center",
                        fontsize=8, zorder=3,
                        fontweight=(
                            "bold" if sd > 0 else "normal"
                        ),
                        color=(
                            "white" if sd == 1 else "#2C2C2A"
                        ))

        day_abbr = ["M","T","W","T","F","S","S"]
        for c, d in enumerate(day_abbr):
            ax.text(c, -0.5, d,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="#5F5E5A")

        ax.set_title(
            str(month), fontsize=11,
            fontweight="bold", pad=15
        )
        ax.axis("off")

    # Legend
    legend_elements = [
        mpatches.Patch(
            facecolor=sd_colors_map[i],
            edgecolor="gray",
            label=lbl
        )
        for i, lbl in [
            (0, "SD0 Regular"),
            (1, "SD1 Holiday"),
            (2, "SD2 Day Before"),
            (3, "SD3 Day After"),
            (4, "SD4 Week After"),
        ]
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5, fontsize=9,
        bbox_to_anchor=(0.5, -0.02)
    )

    plt.suptitle(
        "Special Day Calendar — Test Period",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR, "06_sd_calendar.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 7. MODEL ERROR TIMELINE — stock market style
# =============================================================

def plot_error_timeline():
    """
    Shows MAE over time for each model — like a stock chart.
    Vertical bands mark special day periods.
    This is the most visually impressive plot.
    """
    print("7. Error timeline (stock-market style)...")

    test = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

    # Load predictions from saved results
    # We reconstruct daily MAE from the test set
    # using the saved per-SD results as approximation

    all_results = load_all_model_results()

    # Build a daily error proxy: use Overall MAE
    # weighted by SD type distribution per date
    date_sd = test.groupby("Date")["sd_type"].agg(
        lambda x: x.mode()[0]
    ).reset_index()
    date_sd["Date"] = pd.to_datetime(date_sd["Date"])
    date_sd = date_sd.sort_values("Date")

    fig, ax = plt.subplots(figsize=(16, 7))

    model_colors = {
        "S-Naive"        : "#B4B2A9",
        "S-Median"       : "#5F5E5A",
        "LightGBM"       : "#1D9E75",
        "MLP-CL(median)" : "#534AB7",
        "LSTM-CL(median)": "#993C1D",
    }

    sd_mae_map = {}
    for model, results in all_results.items():
        if model not in model_colors:
            continue
        sd_mae_map[model] = {
            int(k[2]): v["MAE"]
            for k, v in results.items()
            if k.startswith("SD") and k != "Overall"
            and not np.isnan(v["MAE"])
        }

    # Plot daily MAE proxy per model
    for model, sd_map in sd_mae_map.items():
        daily_mae = []
        for _, row in date_sd.iterrows():
            sd  = int(row["sd_type"])
            mae = sd_map.get(sd, sd_map.get(0, 0))
            daily_mae.append(mae)

        # Apply rolling smoothing for visual clarity
        smoothed = pd.Series(daily_mae).rolling(
            3, center=True, min_periods=1
        ).mean().values

        ax.plot(
            date_sd["Date"], smoothed,
            color=model_colors[model],
            linewidth=1.8 if model != "S-Naive" else 1,
            alpha=0.9,
            label=model,
            zorder=3 if "LSTM" in model else 2
        )

    # Shade SD1 periods
    sd1_dates = date_sd[date_sd["sd_type"] == 1]["Date"]
    for d in sd1_dates:
        ax.axvspan(
            d - pd.Timedelta(days=0.5),
            d + pd.Timedelta(days=0.5),
            alpha=0.25, color="#E24B4A",
            zorder=1
        )

    # Shade SD2 periods
    sd2_dates = date_sd[date_sd["sd_type"] == 2]["Date"]
    for d in sd2_dates:
        ax.axvspan(
            d - pd.Timedelta(days=0.5),
            d + pd.Timedelta(days=0.5),
            alpha=0.15, color="#EF9F27",
            zorder=1
        )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("MAE (lower = better)", fontsize=11)
    ax.set_title(
        "Daily Forecast Error Over Time\n"
        "Red bands = SD1 holidays  |  "
        "Orange bands = SD2 day before",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR, "07_error_timeline.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 8. CL(MEDIAN) vs CL(MAX) — probability distribution
# =============================================================

def plot_cl_median_vs_max():
    """
    Visually explains WHY CL(median) beats CL(max).
    Shows symmetric vs asymmetric distributions and
    where the mode and median diverge.
    """
    print("8. CL(median) vs CL(max) explanation...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bin_mids = np.linspace(0, 1000, 104)

    # ── Case 1: Symmetric distribution ─────────────────────
    ax   = axes[0]
    mu   = 500
    proba = np.exp(-0.5 * ((bin_mids - mu)/80)**2)
    proba = proba / proba.sum()

    cumsum = np.cumsum(proba)
    median_idx = np.searchsorted(cumsum, 0.5)
    mode_idx   = np.argmax(proba)

    ax.bar(bin_mids, proba, width=10,
           color="#85B7EB", alpha=0.7,
           label="Probability distribution")
    ax.axvline(bin_mids[mode_idx], color="#E24B4A",
               linewidth=2.5, linestyle="--",
               label=f"Mode (CL max) = {bin_mids[mode_idx]:.0f}")
    ax.axvline(bin_mids[median_idx], color="#1D9E75",
               linewidth=2.5, linestyle="-",
               label=f"Median (CL med) = {bin_mids[median_idx]:.0f}")
    ax.set_title("Case 1: Symmetric\n"
                 "Mode = Median → same result",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Sales bins")
    ax.set_ylabel("Probability")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Case 2: Right-skewed (typical holiday day) ──────────
    ax   = axes[1]
    # Bimodal / skewed
    proba2 = (
        0.3 * np.exp(-0.5 * ((bin_mids - 300)/60)**2) +
        0.7 * np.exp(-0.5 * ((bin_mids - 700)/120)**2)
    )
    proba2 = proba2 / proba2.sum()

    cumsum2    = np.cumsum(proba2)
    median_idx2 = np.searchsorted(cumsum2, 0.5)
    mode_idx2   = np.argmax(proba2)

    ax.bar(bin_mids, proba2, width=10,
           color="#AFA9EC", alpha=0.7)
    ax.axvline(bin_mids[mode_idx2],
               color="#E24B4A", linewidth=2.5,
               linestyle="--",
               label=f"Mode (CL max) = {bin_mids[mode_idx2]:.0f}")
    ax.axvline(bin_mids[median_idx2],
               color="#1D9E75", linewidth=2.5,
               linestyle="-",
               label=f"Median (CL med) = {bin_mids[median_idx2]:.0f}")

    true_val = 650
    ax.axvline(true_val, color="#888780",
               linewidth=2, linestyle=":",
               label=f"True value = {true_val}")

    err_mode   = abs(true_val - bin_mids[mode_idx2])
    err_median = abs(true_val - bin_mids[median_idx2])
    ax.set_title(
        f"Case 2: Bimodal (holiday day)\n"
        f"Mode error={err_mode:.0f}  "
        f"Median error={err_median:.0f}  "
        f"→ Median wins",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Sales bins")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Case 3: Why median minimises MAE ───────────────────
    ax = axes[2]
    x  = np.linspace(-3, 3, 200)
    # Show MAE loss (|x-c|) vs MSE loss ((x-c)²)
    # for different values of c
    c_values = np.linspace(-2, 2, 100)
    # Sample from asymmetric distribution
    np.random.seed(42)
    samples  = np.concatenate([
        np.random.normal(-0.5, 0.5, 30),
        np.random.normal(1.5, 0.8, 70)
    ])

    mae_vals = [np.mean(np.abs(samples - c))
                for c in c_values]
    mse_vals = [np.mean((samples - c)**2)
                for c in c_values]

    # Normalise for comparison
    mae_n = np.array(mae_vals) / max(mae_vals)
    mse_n = np.array(mse_vals) / max(mse_vals)

    ax.plot(c_values, mae_n, color="#E24B4A",
            linewidth=2.5, label="MAE loss (minimised by median)")
    ax.plot(c_values, mse_n, color="#85B7EB",
            linewidth=2.5, label="MSE loss (minimised by mean)")

    median_c = np.median(samples)
    mean_c   = np.mean(samples)
    ax.axvline(median_c, color="#E24B4A",
               linewidth=2, linestyle="--",
               label=f"Median = {median_c:.2f}")
    ax.axvline(mean_c, color="#85B7EB",
               linewidth=2, linestyle="--",
               label=f"Mean = {mean_c:.2f}")

    ax.set_title(
        "Why median minimises MAE\n"
        "(and mean minimises MSE)",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Prediction value c")
    ax.set_ylabel("Normalised loss")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle(
        "Why CL(median) beats CL(max) — "
        "the mathematical intuition",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR, "08_cl_median_vs_max.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 9. MODEL RANKING RADAR CHART
# =============================================================

def plot_radar_chart():
    """
    Spider/radar chart comparing all models across
    all SD types simultaneously.
    One chart that tells the whole story.
    """
    print("9. Model radar chart...")

    all_results = load_all_model_results()

    # Use MASE for each SD type as axes
    categories  = ["SD0", "SD1", "SD2",
                   "SD3", "SD4", "Overall"]
    N           = len(categories)
    angles      = [n / float(N) * 2 * np.pi
                   for n in range(N)]
    angles     += angles[:1]  # close the polygon

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw=dict(polar=True)
    )

    # Models to show (not all — too crowded)
    models_to_show = {
        "S-Naive"        : ("#B4B2A9", "--", 1.5),
        "S-Median"       : ("#5F5E5A", "--", 1.5),
        "LightGBM"       : ("#1D9E75", "-",  2.5),
        "MLP-CL(median)" : ("#534AB7", "-",  2.5),
        "LSTM-CL(median)": ("#993C1D", "-",  2.5),
    }

    for model, (color, ls, lw) in models_to_show.items():
        if model not in all_results:
            continue
        values = [
            all_results[model].get(
                cat, {}
            ).get("MASE", np.nan)
            for cat in categories
        ]

        if any(np.isnan(v) for v in values):
            continue

        values += values[:1]  # close polygon

        ax.plot(angles, values,
                color=color, linewidth=lw,
                linestyle=ls, label=model)
        ax.fill(angles, values,
                color=color, alpha=0.08)

    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12,
                       fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ["0.2", "0.4", "0.6", "0.8", "1.0"],
        fontsize=8, color="gray"
    )
    ax.grid(color="gray", linewidth=0.5, alpha=0.5)

    ax.set_title(
        "Model Performance Radar\n"
        "MASE per SD Type\n(smaller = better)",
        fontsize=13, fontweight="bold",
        pad=30
    )
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.35, 1.15),
        fontsize=10
    )

    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR, "09_radar_chart.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 10. PREDICTION VS ACTUAL SCATTER
# =============================================================

def plot_prediction_scatter():
    """
    Scatter plot of predicted vs actual sales.
    Perfect model = all points on the diagonal.
    Shows systematic errors on special days.
    """
    print("10. Prediction vs actual scatter...")

    test = pd.read_parquet(
        os.path.join(PROCESSED_DIR, "test.parquet")
    )

    # Load LightGBM model for predictions
    lgbm_path = os.path.join(
        MODELS_DIR, "lgbm_model.pkl"
    )
    if not os.path.exists(lgbm_path):
        print("    LightGBM model not found — skipping")
        return None

    with open(lgbm_path, "rb") as f:
        lgbm_model = pickle.load(f)

    feature_names = lgbm_model.feature_name()
    avail = [f for f in feature_names
             if f in test.columns]
    X_test    = test[avail].values
    y_pred    = np.clip(
        lgbm_model.predict(X_test), 0, None
    )
    y_true    = test["Sales"].values
    sd_types  = test["sd_type"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sd_colors_map = {
        0: "#85B7EB", 1: "#E24B4A",
        2: "#EF9F27", 3: "#1D9E75", 4: "#888780"
    }
    sd_names = {
        0: "SD0 Regular", 1: "SD1 Holiday",
        2: "SD2 Day Before", 3: "SD3 Day After",
        4: "SD4 Week After"
    }

    # Left: all points coloured by SD type
    ax = axes[0]
    for sd in [4, 3, 2, 0, 1]:
        mask = sd_types == sd
        if mask.sum() == 0:
            continue
        sample_size = min(500, mask.sum())
        idx = np.where(mask)[0]
        np.random.seed(42)
        idx = np.random.choice(idx, sample_size,
                               replace=False)
        ax.scatter(y_true[idx], y_pred[idx],
                   color=sd_colors_map[sd],
                   alpha=0.4, s=8,
                   label=sd_names[sd],
                   zorder=sd+1)

    max_val = min(
        max(y_true.max(), y_pred.max()), 15000
    )
    ax.plot([0, max_val], [0, max_val],
            "k--", linewidth=1.5,
            label="Perfect forecast", alpha=0.5)
    ax.set_xlabel("Actual Sales")
    ax.set_ylabel("Predicted Sales")
    ax.set_title(
        "LightGBM: Predicted vs Actual\n"
        "(coloured by SD type)",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: residuals by SD type
    ax     = axes[1]
    errors = y_pred - y_true

    sd_labels_list = []
    error_lists    = []
    colors_list    = []

    for sd in [0, 1, 2, 3, 4]:
        mask = sd_types == sd
        if mask.sum() < 10:
            continue
        sd_labels_list.append(sd_names[sd])
        error_lists.append(errors[mask])
        colors_list.append(sd_colors_map[sd])

    bp = ax.boxplot(
        error_lists,
        labels=sd_labels_list,
        patch_artist=True,
        medianprops=dict(color="black",
                         linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(markersize=2,
                        alpha=0.3)
    )

    for patch, color in zip(
        bp["boxes"], colors_list
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(0, color="black",
               linewidth=1.5, linestyle="--",
               label="Zero error")
    ax.set_ylabel("Prediction Error (Predicted - Actual)")
    ax.set_title(
        "Error Distribution by SD Type\n"
        "(box = IQR, whiskers = 1.5×IQR)",
        fontsize=11, fontweight="bold"
    )
    ax.set_xticklabels(
        sd_labels_list, rotation=20, ha="right"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle(
        "LightGBM — Prediction Quality Analysis",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR, "10_prediction_scatter.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 11. ANIMATED SALES TIME SERIES (video)
# =============================================================

def create_animated_sales_chart(store_id=1,
                                  n_days=90):
    """
    Creates an animated GIF showing:
    - Sales rolling in like a stock chart
    - SD type markers appearing as vertical bands
    - Rolling median line updating in real time

    Saved as animated GIF — viewable in Kaggle notebooks.
    """
    print("11. Animated sales chart...")

    df = pd.read_parquet(
        os.path.join(PROCESSED_DIR,
                     "features_complete.parquet")
    )

    store_df = df[
        df["Store"] == store_id
    ].sort_values("Date").tail(n_days).reset_index(
        drop=True
    )

    if len(store_df) == 0:
        print(f"    Store {store_id} not found")
        return None

    dates     = store_df["Date"].values
    sales     = store_df["Sales"].values
    sd_types  = store_df["sd_type"].values
    rolling   = store_df["Sales_rolling_median"].values

    sd_colors_map = {
        0: None,
        1: "#FCEBEB",
        2: "#FAEEDA",
        3: "#EAF3DE",
        4: "#F1EFE8"
    }

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    line_sales,  = ax.plot([], [], color="#534AB7",
                            linewidth=2, label="Actual Sales",
                            zorder=4)
    line_rolling, = ax.plot([], [], color="#D85A30",
                             linewidth=1.5,
                             linestyle="--",
                             label="Rolling Median",
                             zorder=3, alpha=0.8)

    ax.set_xlim(0, n_days)
    ax.set_ylim(
        max(0, sales.min() * 0.8),
        sales.max() * 1.15
    )
    ax.set_title(
        f"Store {store_id} — Daily Sales"
        f" (animated)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Sales")
    ax.legend(fontsize=10, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.2)

    # Pre-draw SD bands
    sd_patches = []
    for i, (date, sd) in enumerate(
        zip(dates, sd_types)
    ):
        color = sd_colors_map.get(int(sd))
        if color:
            patch = ax.axvspan(
                i - 0.5, i + 0.5,
                alpha=0, color=color,
                zorder=1
            )
            sd_patches.append((patch, i, color))

    # Date labels on x-axis
    tick_indices = list(
        range(0, n_days, max(1, n_days // 10))
    )
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(
        [pd.Timestamp(dates[i]).strftime("%b %d")
         for i in tick_indices],
        rotation=30, ha="right", fontsize=8
    )

    # Annotation text
    ann_text = ax.text(
        0.02, 0.95, "",
        transform=ax.transAxes,
        fontsize=10, va="top",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white",
                  edgecolor="gray", alpha=0.8)
    )

    def init():
        line_sales.set_data([], [])
        line_rolling.set_data([], [])
        return [line_sales, line_rolling]

    def animate(frame):
        # Reveal data up to frame
        x_data = list(range(frame + 1))
        line_sales.set_data(x_data, sales[:frame+1])
        line_rolling.set_data(
            x_data, rolling[:frame+1]
        )

        # Show SD bands for revealed days
        for patch, i, color in sd_patches:
            if i <= frame:
                patch.set_alpha(0.35)

        # Update annotation
        sd = int(sd_types[frame])
        sd_label = {
            0: "Regular Day",
            1: "SD1 Holiday",
            2: "SD2 Day Before",
            3: "SD3 Day After",
            4: "SD4 Week After"
        }[sd]
        date_str = pd.Timestamp(
            dates[frame]
        ).strftime("%Y-%m-%d")
        ann_text.set_text(
            f"{date_str}\n"
            f"Sales: {sales[frame]:,.0f}\n"
            f"Type: {sd_label}"
        )
        ann_color = {
            0: "#E6F1FB", 1: "#FCEBEB",
            2: "#FAEEDA", 3: "#EAF3DE",
            4: "#F1EFE8"
        }[sd]
        ann_text.get_bbox_patch().set_facecolor(
            ann_color
        )

        return [line_sales, line_rolling, ann_text]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_days,
        interval=80,       # ms per frame
        blit=True
    )

    path = os.path.join(
        ADV_PLOTS_DIR,
        "11_animated_sales.gif"
    )
    writer = animation.PillowWriter(fps=12)
    anim.save(path, writer=writer, dpi=100)
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 12. GRADIENT BOOSTING EXPLAINED VISUALLY
# =============================================================

def plot_gradient_boosting_explanation():
    """
    Shows how gradient boosting works step by step
    using a simple example — makes the concept visual.
    """
    print("12. Gradient boosting explanation...")

    np.random.seed(42)
    x       = np.linspace(0, 10, 50)
    y_true  = 2 * np.sin(x) + 0.5 * x + \
              np.random.randn(50) * 0.3

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    # Simulate 5 rounds of boosting
    y_pred    = np.full_like(y_true, y_true.mean())
    residuals = y_true - y_pred
    lr        = 0.7

    titles = [
        "Step 0: Initial prediction\n(just the mean)",
        "Step 1: First tree fits residuals",
        "Step 2: Updated prediction",
        "Step 3: Second tree fits new residuals",
        "Step 4: Updated prediction",
        "Final: Ensemble prediction",
    ]

    for step, ax in enumerate(axes):
        if step == 0:
            ax.scatter(x, y_true, color="#534AB7",
                       s=20, alpha=0.7, label="True values", zorder=3)
            ax.axhline(y_pred[0], color="#D85A30",
                       linewidth=2, label="Prediction (mean)", zorder=2)
            residual_mae = np.mean(np.abs(residuals))
            ax.set_title(
                f"{titles[step]}\nMAE = {residual_mae:.2f}",
                fontsize=9, fontweight="bold"
            )

        elif step == 1:
            ax.scatter(x, residuals, color="#E24B4A",
                       s=20, alpha=0.7,
                       label="Residuals", zorder=3)
            ax.axhline(0, color="gray",
                       linewidth=1, linestyle="--")
            # Fit a simple curve to residuals
            coeffs    = np.polyfit(x, residuals, 3)
            tree1_pred = np.polyval(coeffs, x)
            ax.plot(x, tree1_pred, color="#1D9E75",
                    linewidth=2, label="Tree 1 prediction", zorder=2)
            ax.set_title(
                f"{titles[step]}",
                fontsize=9, fontweight="bold"
            )

        elif step == 2:
            coeffs    = np.polyfit(x, residuals, 3)
            tree1_pred = np.polyval(coeffs, x)
            y_pred    = y_pred + lr * tree1_pred
            residuals = y_true - y_pred

            ax.scatter(x, y_true, color="#534AB7",
                       s=20, alpha=0.7,
                       label="True values", zorder=3)
            ax.plot(x, y_pred, color="#D85A30",
                    linewidth=2, label="Updated prediction", zorder=2)
            mae = np.mean(np.abs(residuals))
            ax.set_title(
                f"{titles[step]}\nMAE = {mae:.2f} (improved!)",
                fontsize=9, fontweight="bold"
            )

        elif step == 3:
            ax.scatter(x, residuals, color="#E24B4A",
                       s=20, alpha=0.7,
                       label="New residuals", zorder=3)
            ax.axhline(0, color="gray",
                       linewidth=1, linestyle="--")
            coeffs2     = np.polyfit(x, residuals, 5)
            tree2_pred  = np.polyval(coeffs2, x)
            ax.plot(x, tree2_pred, color="#1D9E75",
                    linewidth=2, label="Tree 2 prediction", zorder=2)
            ax.set_title(
                f"{titles[step]}",
                fontsize=9, fontweight="bold"
            )

        elif step == 4:
            coeffs2    = np.polyfit(x, residuals, 5)
            tree2_pred = np.polyval(coeffs2, x)
            y_pred     = y_pred + lr * tree2_pred
            residuals  = y_true - y_pred

            ax.scatter(x, y_true, color="#534AB7",
                       s=20, alpha=0.7,
                       label="True values", zorder=3)
            ax.plot(x, y_pred, color="#D85A30",
                    linewidth=2,
                    label="Updated prediction", zorder=2)
            mae = np.mean(np.abs(residuals))
            ax.set_title(
                f"{titles[step]}\nMAE = {mae:.2f}",
                fontsize=9, fontweight="bold"
            )

        else:
            # Final
            ax.scatter(x, y_true, color="#534AB7",
                       s=20, alpha=0.7,
                       label="True values", zorder=3)
            ax.plot(x, y_pred, color="#1D9E75",
                    linewidth=2.5,
                    label="Final ensemble", zorder=4)
            mae = np.mean(np.abs(y_true - y_pred))
            ax.fill_between(
                x, y_pred - mae, y_pred + mae,
                alpha=0.15, color="#1D9E75",
                label="±MAE band"
            )
            ax.set_title(
                f"{titles[step]}\n"
                f"Final MAE = {mae:.2f}",
                fontsize=9, fontweight="bold"
            )

        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle(
        "How Gradient Boosting Works — Step by Step",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR,
        "12_gradient_boosting_explained.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# 13. LSTM MEMORY MECHANISM VISUAL
# =============================================================

def plot_lstm_memory():
    """
    Shows how the LSTM's cell state (memory) evolves
    over the lag sequence for a sample sequence.
    Makes the gate mechanism intuitive.
    """
    print("13. LSTM memory visualisation...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Gate diagram
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Draw the LSTM cell
    cell_rect = plt.Rectangle(
        (2, 1), 6, 6,
        facecolor="#EEEDFE",
        edgecolor="#534AB7",
        linewidth=2
    )
    ax.add_patch(cell_rect)
    ax.text(5, 7.3, "LSTM Cell",
            ha="center", fontsize=12,
            fontweight="bold", color="#534AB7")

    # Cell state (top horizontal line)
    ax.annotate("",
                xy=(8, 6.5), xytext=(2, 6.5),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#E24B4A", lw=3
                ))
    ax.text(5, 6.75, "Cell State c_t (long-term memory)",
            ha="center", fontsize=9,
            color="#E24B4A", fontweight="bold")

    # Gates
    gates = [
        (3.0, 3.5, "Forget\nGate f_t",
         "#F09595", "Erase old\nmemory"),
        (5.0, 3.5, "Input\nGate i_t",
         "#9FE1CB", "Write new\nmemory"),
        (7.0, 3.5, "Output\nGate o_t",
         "#FAC775", "Read from\nmemory"),
    ]
    for gx, gy, label, color, desc in gates:
        circle = plt.Circle(
            (gx, gy), 0.7,
            facecolor=color,
            edgecolor="gray",
            linewidth=1.5, zorder=3
        )
        ax.add_patch(circle)
        ax.text(gx, gy, label,
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                zorder=4)
        ax.text(gx, gy - 1.2, desc,
                ha="center", fontsize=7,
                color="#5F5E5A")

    # Input and hidden state
    ax.annotate("",
                xy=(2.5, 2), xytext=(0.5, 2),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#534AB7", lw=2
                ))
    ax.text(0.5, 1.7, "x_t\n(today's\nfeatures)",
            ha="center", fontsize=8,
            color="#534AB7")

    ax.annotate("",
                xy=(2.5, 4.5), xytext=(0.5, 4.5),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#1D9E75", lw=2
                ))
    ax.text(0.5, 4.8, "h_{t-1}\n(prev hidden\nstate)",
            ha="center", fontsize=8,
            color="#1D9E75")

    ax.annotate("",
                xy=(9.5, 2), xytext=(8, 2),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#534AB7", lw=2
                ))
    ax.text(9.5, 1.7, "h_t\n(output)",
            ha="center", fontsize=8,
            color="#534AB7")

    ax.set_title(
        "LSTM Cell — Gate Mechanism",
        fontsize=11, fontweight="bold"
    )

    # Right: Memory evolution over lag sequence
    ax = axes[1]
    lag_labels = ["Lag 14\n(2 wks ago)",
                  "Lag 7\n(1 wk ago)",
                  "Lag 6", "Lag 5", "Lag 4",
                  "Lag 3", "Lag 2\n(yesterday)"]

    # Simulate memory evolution (for illustration)
    np.random.seed(7)
    n      = len(lag_labels)
    memory = np.cumsum(np.random.randn(n) * 0.3) + 0.5
    memory = np.clip(memory, -1, 1)
    forget = np.random.uniform(0.3, 0.95, n)
    input_ = np.random.uniform(0.1, 0.8, n)
    output = np.random.uniform(0.4, 0.9, n)

    x = np.arange(n)
    ax.plot(x, memory, "o-",
            color="#E24B4A", linewidth=2.5,
            markersize=10, label="Cell state",
            zorder=4)
    ax.bar(x, forget, width=0.25,
           align="edge", color="#F09595",
           alpha=0.6, label="Forget gate")
    ax.bar(x + 0.25, input_, width=0.25,
           align="edge", color="#9FE1CB",
           alpha=0.6, label="Input gate")
    ax.bar(x + 0.5, output, width=0.25,
           align="edge", color="#FAC775",
           alpha=0.6, label="Output gate")

    ax.set_xticks(x)
    ax.set_xticklabels(lag_labels, fontsize=8)
    ax.set_ylabel("Gate activation / Cell state value")
    ax.set_title(
        "LSTM Memory Evolution Over Lag Sequence\n"
        "(illustrative — shows how gates control memory)",
        fontsize=10, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle(
        "LSTM — How Memory Works",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(
        ADV_PLOTS_DIR, "13_lstm_memory.png"
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   Saved: {path}")
    return path


# =============================================================
# MASTER RUNNER
# =============================================================

def run_all_advanced_visualisations():
    """
    Runs all advanced visualisations in sequence.
    Safe to run multiple times — all outputs go to
    /outputs/plots/advanced/ and never touch other files.
    """
    print("=" * 55)
    print("Advanced Visualisations")
    print("=" * 55)
    print(f"Output directory: {ADV_PLOTS_DIR}\n")

    saved_paths = []

    funcs = [
        plot_dataset_snapshot,
        plot_sales_distribution,
        plot_weekly_seasonality,
        plot_lag_autocorrelation,
        plot_cyclical_encoding,
        plot_sd_calendar,
        plot_error_timeline,
        plot_cl_median_vs_max,
        plot_radar_chart,
        plot_prediction_scatter,
        create_animated_sales_chart,
        plot_gradient_boosting_explanation,
        plot_lstm_memory,
    ]

    for func in funcs:
        try:
            path = func()
            if path:
                saved_paths.append(path)
        except Exception as e:
            print(f"   ERROR in {func.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*55}")
    print(f"Done. {len(saved_paths)} plots saved to:")
    print(f"  {ADV_PLOTS_DIR}")
    print(f"{'='*55}")

    return saved_paths
