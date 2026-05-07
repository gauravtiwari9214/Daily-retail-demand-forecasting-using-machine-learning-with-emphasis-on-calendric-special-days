
# =============================================================
# day_classifier.py — Labels every row with SD type (0–4)
#
# WHY this module exists:
#   The paper's entire contribution depends on correctly
#   identifying special days. A wrong SD label on even one
#   date means the model learns the wrong pattern for that
#   day — and since there are very few SD observations,
#   even one wrong label has outsized impact.
#
# WHAT it produces:
#   A new column 'sd_type' on every row:
#   SD0 = regular day
#   SD1 = public holiday / major event on day t
#   SD2 = day before public holiday (t-1)
#   SD3 = day after public holiday  (t+1)
#   SD4 = one week after holiday    (t+7) — sanity check
#
# PRECEDENCE RULE (from paper Section 4.1):
#   If a date qualifies for multiple types, assign only
#   the highest precedence:
#   SD1 > SD2 > SD3 > SD4 > SD0
#   Example: if Easter Monday (SD1) falls on a day that is
#   also t+1 of Good Friday (SD3), it gets SD1.
# =============================================================

import os
import sys
import pandas as pd
from datetime import date, timedelta

sys.path.append("/kaggle/working/retail_forecasting/src")
from config import PROCESSED_DIR


# =============================================================
# SECTION 1: Easter computation
# =============================================================

def compute_easter(year):
    """
    Computes Easter Sunday for a given year using the
    Anonymous Gregorian Algorithm.

    WHY we need this:
      Many German public holidays are defined relative to
      Easter (Good Friday = Easter-2, Easter Monday = Easter+1,
      Ascension = Easter+39, Whit Monday = Easter+50,
      Corpus Christi = Easter+60). Easter itself can shift
      by up to 35 days across years, so we cannot hardcode it.

    Returns: datetime.date object for Easter Sunday
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day   = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


# =============================================================
# SECTION 2: Build German holiday calendar
# =============================================================

def build_holiday_calendar(years):
    """
    Builds the complete set of SD1 dates for all given years.
    Based on Tables 4 and 5 of the paper.

    WHY we include both fixed and moveable holidays:
      Fixed holidays (Christmas, New Year) are on the same
      date every year but change weekday.
      Moveable holidays (Easter-based) change date entirely.
      Both require special treatment in the model.

    WHY we include non-public-holiday special days (Table 5):
      Christmas Eve and New Year's Eve are not public holidays
      but have vastly different demand patterns — stores close
      early, customers stockpile heavily.

    Returns: set of datetime.date objects that are SD1 days
    """
    sd1_dates = set()

    for year in years:
        easter = compute_easter(year)

        # ── Fixed public holidays (Table 4 of paper) ──────────
        fixed = [
            date(year, 1,  1),   # New Year's Day
            date(year, 5,  1),   # Labour Day
            date(year, 10, 3),   # Day of German Unity
            date(year, 12, 25),  # Christmas Day 1
            date(year, 12, 26),  # Christmas Day 2
        ]

        # ── Baden-Württemberg specific (Table 4) ──────────────
        # Rossmann stores are in Germany — we include BW holidays
        # as they affect a significant portion of stores
        bw_fixed = [
            date(year, 1,  6),   # Epiphany
            date(year, 11, 1),   # All Hallows
        ]

        # ── Easter-based moveable holidays (Table 4) ──────────
        moveable = [
            easter - timedelta(days=2),   # Good Friday
            easter,                        # Easter Sunday
            easter + timedelta(days=1),   # Easter Monday
            easter + timedelta(days=39),  # Ascension Day
            easter + timedelta(days=49),  # Whit Sunday
            easter + timedelta(days=50),  # Whit Monday
            easter + timedelta(days=60),  # Corpus Christi (BW)
        ]

        # ── Non-public-holiday special days (Table 5) ─────────
        # These are not legal holidays but have strongly
        # different demand patterns — paper explicitly includes them
        special_events = [
            date(year, 12, 24),  # Christmas Eve
            date(year, 12, 31),  # New Year's Eve
        ]

        # ── Carnival week (Table 5) ───────────────────────────
        # Carnival runs from Women's Carnival Day (Thu, 7 weeks
        # before Easter) through Ash Wednesday
        # WHY: Carnival is culturally major in German bakery
        # regions — demand patterns shift significantly
        carnival_thursday = easter - timedelta(weeks=7)
        # Carnival Thursday is always a Thursday
        # Ash Wednesday is 46 days before Easter
        ash_wednesday = easter - timedelta(days=46)
        carnival_day = carnival_thursday
        while carnival_day <= ash_wednesday:
            special_events.append(carnival_day)
            carnival_day += timedelta(days=1)

        # ── Combine all SD1 dates for this year ───────────────
        for d in fixed + bw_fixed + moveable + special_events:
            sd1_dates.add(d)

    return sd1_dates


# =============================================================
# SECTION 3: Core classification logic
# =============================================================

def classify_days(df, sd1_dates):
    """
    Assigns sd_type (0–4) to every row in the DataFrame.

    ALGORITHM (strictly follows paper Section 4.1):

    Step 1: Mark all SD1 dates
    Step 2: For each SD1, mark SD2 (t-1) and SD3 (t+1)
            Special case: if SD1 falls on Monday,
            Saturday (t-2) is ALSO SD2
    Step 3: Mark SD4 (t+7) for each SD1
    Step 4: Apply precedence: SD1 > SD2 > SD3 > SD4 > SD0

    WHY the Monday special case:
      If a public holiday falls on Monday, the previous
      Saturday is effectively the "day before" because Sunday
      is already a rest day. People do their pre-holiday
      shopping on Saturday, not Sunday.

    WHY SD4 exists:
      Autoregressive models use lag-7 (same weekday last week).
      If that lag observation was itself a special day, the
      model may underestimate or overestimate demand.
      SD4 lets us measure this contamination effect.

    Returns: DataFrame with new 'sd_type' column
    """
    df = df.copy()

    # Convert sd1_dates to pandas Timestamps for comparison
    sd1_ts  = set(pd.Timestamp(d) for d in sd1_dates)

    # Build SD2, SD3, SD4 sets from SD1
    sd2_ts  = set()
    sd3_ts  = set()
    sd4_ts  = set()

    for d in sd1_ts:
        # SD2: day before (t-1)
        sd2_ts.add(d - pd.Timedelta(days=1))

        # Special case: if SD1 is Monday, Saturday (t-2) is also SD2
        # WHY: Sunday is already a rest day, so the effective
        # "day before" for a Monday holiday is Saturday
        if d.dayofweek == 0:  # 0 = Monday
            sd2_ts.add(d - pd.Timedelta(days=2))

        # SD3: day after (t+1)
        sd3_ts.add(d + pd.Timedelta(days=1))

        # SD4: one week after (t+7) — sanity check day
        sd4_ts.add(d + pd.Timedelta(days=7))

    # ── Apply labels with precedence ──────────────────────────
    # Start everyone at SD0 (regular day)
    df["sd_type"] = 0

    # Apply in REVERSE precedence order so higher precedence
    # overwrites lower precedence
    # Order: SD4 first (lowest), then SD3, SD2, SD1 (highest)
    # This way SD1 always wins if there is a conflict

    df.loc[df["Date"].isin(sd4_ts), "sd_type"] = 4
    df.loc[df["Date"].isin(sd3_ts), "sd_type"] = 3
    df.loc[df["Date"].isin(sd2_ts), "sd_type"] = 2
    df.loc[df["Date"].isin(sd1_ts), "sd_type"] = 1

    return df


# =============================================================
# SECTION 4: Main entry point
# =============================================================

def run_day_classification(save=True):
    """
    Loads rossmann_raw.parquet, classifies every row,
    saves rossmann_with_sd.parquet.

    Returns: DataFrame with sd_type column added
    """
    print("=" * 50)
    print("Day Classification")
    print("=" * 50)

    # Load the output from data_loader.py
    in_path = os.path.join(PROCESSED_DIR, "rossmann_raw.parquet")
    assert os.path.exists(in_path), (
        f"File not found: {in_path}\n"
        f"Run data_loader.py first."
    )
    df = pd.read_parquet(in_path)
    print(f"Loaded: {in_path}")
    print(f"Shape : {df.shape}")

    # Get all unique years in the data
    years = sorted(df["Date"].dt.year.unique().tolist())
    print(f"Years in data: {years}")

    # Build holiday calendar for all years
    sd1_dates = build_holiday_calendar(years)
    print(f"SD1 dates defined: {len(sd1_dates)} total")

    # Classify every row
    df = classify_days(df, sd1_dates)

    # ── Print distribution ─────────────────────────────────────
    print("\nSD type distribution:")
    counts = df["sd_type"].value_counts().sort_index()
    total  = len(df)
    for sd_type, count in counts.items():
        pct = count / total * 100
        name = {0:"Regular",1:"Holiday",
                2:"Day Before",3:"Day After",
                4:"Week After"}[sd_type]
        print(f"  SD{sd_type} ({name:11s}): "
              f"{count:7,} rows  ({pct:.1f}%)")

    # Save
    if save:
        out_path = os.path.join(
            PROCESSED_DIR, "rossmann_with_sd.parquet"
        )
        df.to_parquet(out_path, index=False)
        print(f"\nSaved: {out_path}")

    return df, sd1_dates


# =============================================================
# SECTION 5: Verification
# =============================================================

def verify_day_classification(df, sd1_dates):
    """
    Verifies correctness of SD labels.
    Uses known dates to check the classification is right.

    WHY we verify specific known dates:
      Easter and Christmas are fixed reference points.
      If these are wrong, the entire SD feature engineering
      downstream will be wrong.
    """
    print("\n" + "=" * 50)
    print("Verification checks")
    print("=" * 50)

    all_pass = True
    checks   = {}

    # ── Check 1: SD0 should be ~80-90% of all rows ────────────
    sd0_pct = (df["sd_type"] == 0).mean() * 100
    checks["SD0 is 75–95% of rows"] = 75 <= sd0_pct <= 95

    # ── Check 2: SD1 should be 4–8% ───────────────────────────
    sd1_pct = (df["sd_type"] == 1).mean() * 100
    checks["SD1 is 2–10% of rows"] = 2 <= sd1_pct <= 10

    # ── Check 3: Easter 2014 — verify specific known dates ────
    # Easter Sunday 2014 = April 20
    easter_2014 = pd.Timestamp("2014-04-20")
    good_friday_2014  = pd.Timestamp("2014-04-18")  # SD1
    easter_mon_2014   = pd.Timestamp("2014-04-21")  # SD1
    day_before_gf     = pd.Timestamp("2014-04-17")  # SD2
    day_after_em      = pd.Timestamp("2014-04-22")  # SD3
    week_after_easter = pd.Timestamp("2014-04-27")  # SD4

    def get_sd(date_ts):
        rows = df[df["Date"] == date_ts]
        if len(rows) == 0:
            return None
        return rows["sd_type"].iloc[0]

    checks["Good Friday 2014 = SD1"] = (
        get_sd(good_friday_2014) == 1
    )
    checks["Easter Monday 2014 = SD1"] = (
        get_sd(easter_mon_2014) == 1
    )
    checks["Day before Good Friday = SD2"] = (
        get_sd(day_before_gf) == 2
    )
    checks["Day after Easter Monday = SD3"] = (
        get_sd(day_after_em) == 3
    )
    checks["Week after Easter Sunday = SD4"] = (
        get_sd(week_after_easter) == 4
    )

    # ── Check 4: Christmas 2014 ────────────────────────────────
    xmas1_2014   = pd.Timestamp("2014-12-25")  # SD1
    xmas_eve     = pd.Timestamp("2014-12-24")  # SD1 (Table 5)
    xmas2_2014   = pd.Timestamp("2014-12-26")  # SD1
    day_b4_xmas1 = pd.Timestamp("2014-12-24")  # SD1 (Christmas Eve)

    checks["Christmas Day 1 2014 = SD1"] = (
        get_sd(xmas1_2014) == 1
    )
    checks["Christmas Day 2 2014 = SD1"] = (
        get_sd(xmas2_2014) == 1
    )
    checks["Christmas Eve 2014 = SD1"] = (
        get_sd(xmas_eve) == 1
    )

    # ── Check 5: SD2/SD3 only appear next to SD1 ──────────────
    # For every SD2 row, the next day must be SD1
    # For every SD3 row, the previous day must be SD1
    dates_sd1 = set(df[df["sd_type"]==1]["Date"].unique())
    dates_sd2 = set(df[df["sd_type"]==2]["Date"].unique())
    dates_sd3 = set(df[df["sd_type"]==3]["Date"].unique())

    sd2_valid = all(
        (d + pd.Timedelta(days=1)) in dates_sd1
        or (d + pd.Timedelta(days=2)) in dates_sd1
        for d in dates_sd2
    )
    checks["All SD2 dates are adjacent to SD1"] = sd2_valid

    sd3_valid = all(
        (d - pd.Timedelta(days=1)) in dates_sd1
        for d in dates_sd3
    )
    checks["All SD3 dates are adjacent to SD1"] = sd3_valid

    # ── Print results ──────────────────────────────────────────
    for name, result in checks.items():
        status = "PASS" if result else "FAIL"
        if not result:
            all_pass = False
        print(f"  [{status}] {name}")

    print()
    if all_pass:
        print("All checks passed.")
        print("Next step: feature_engineering.py")
    else:
        print("Some checks FAILED — fix before proceeding.")

    return all_pass
