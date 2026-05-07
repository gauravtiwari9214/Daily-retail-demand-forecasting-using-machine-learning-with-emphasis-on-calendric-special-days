
# =============================================================
# data_loader.py — Loads and merges Rossmann raw CSVs
# Produces: rossmann_raw.parquet
# =============================================================

import os
import sys
import pandas as pd

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import RAW_DIR, PROCESSED_DIR


def load_rossmann_data(save=True):
    """
    Loads train.csv + store.csv, merges them, filters to
    open stores with non-zero sales, sorts by Store+Date.

    Why we filter:
      - Closed days (Open=0) have Sales=0 by definition —
        including them would corrupt lag features and rolling
        medians because the zero would look like a real demand
        observation.
      - Zero-sales open days are stock-outs or data errors —
        not genuine demand signal.

    Returns: cleaned DataFrame
    """
    print("=" * 50)
    print("Loading raw Rossmann data...")
    print("=" * 50)

    # ── Load raw files ────────────────────────────────────────
    train = pd.read_csv(
        os.path.join(RAW_DIR, "train.csv"),
        parse_dates=["Date"],
        low_memory=False
    )
    store = pd.read_csv(
        os.path.join(RAW_DIR, "store.csv")
    )

    print(f"\nRaw shapes:")
    print(f"  train.csv : {train.shape}")
    print(f"  store.csv : {store.shape}")

    # ── Merge store metadata onto every sales row ─────────────
    # Why left merge: keep all train rows, attach store info
    df = train.merge(store, on="Store", how="left")
    print(f"\nAfter merge: {df.shape}")

    # ── Filter: open stores with real sales only ──────────────
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()
    print(f"After filter (Open=1, Sales>0): {df.shape}")

    # ── Sort: critical for lag features later ─────────────────
    # If not sorted by Store+Date, shift() will compute
    # lags across different stores — a silent data leak
    df.sort_values(["Store", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── Basic type fixes ──────────────────────────────────────
    # StateHoliday comes as mixed 0/"a"/"b"/"c" — normalise
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df["StateHoliday"] = df["StateHoliday"].replace("0", "none")

    # CompetitionDistance: fill NaN with median
    # (NaN means no competitor nearby — use large distance)
    median_dist = df["CompetitionDistance"].median()
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(median_dist)

    print(f"\nFinal dataset:")
    print(f"  Rows      : {len(df):,}")
    print(f"  Stores    : {df['Store'].nunique()}")
    print(f"  Date range: {df['Date'].min().date()} "
          f"to {df['Date'].max().date()}")
    print(f"  Columns   : {list(df.columns)}")

    # ── Save ──────────────────────────────────────────────────
    if save:
        out_path = os.path.join(PROCESSED_DIR, "rossmann_raw.parquet")
        df.to_parquet(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return df


def verify_raw_data(df):
    """
    Runs sanity checks on the loaded DataFrame.
    Prints PASS / FAIL for each check.
    All must pass before moving to day_classifier.py.
    """
    print("\n" + "=" * 50)
    print("Running verification checks...")
    print("=" * 50)

    checks = {
        "No negative sales"    : (df["Sales"] >= 0).all(),
        "No null dates"        : df["Date"].isnull().sum() == 0,
        "No null Store IDs"    : df["Store"].isnull().sum() == 0,
        "Sales > 0 (all rows)" : (df["Sales"] > 0).all(),
        "Open == 1 (all rows)" : (df["Open"] == 1).all(),
        "StoreType not null"   : df["StoreType"].isnull().sum() == 0,
        "Sorted by Store+Date" : df.groupby("Store")["Date"]
                                   .apply(lambda x:
                                       x.is_monotonic_increasing)
                                   .all(),
        "CompetitionDistance no NaN": 
                                 df["CompetitionDistance"]
                                   .isnull().sum() == 0,
    }

    all_pass = True
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: day_classifier.py")
    else:
        print("Some checks FAILED — fix before proceeding.")

    return all_pass
