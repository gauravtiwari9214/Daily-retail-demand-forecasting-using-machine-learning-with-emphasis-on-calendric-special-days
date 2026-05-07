
# =============================================================
# config.py — Single source of truth for all constants
# Every other module imports from here. Never hardcode
# any value in any other file.
# =============================================================

# ── Paths ────────────────────────────────────────────────────
RAW_DIR       = "/kaggle/input/competitions/rossmann-store-sales"
PROCESSED_DIR = "/kaggle/working/retail_forecasting/data/processed"
MODELS_DIR    = "/kaggle/working/retail_forecasting/outputs/models"
RESULTS_DIR   = "/kaggle/working/retail_forecasting/outputs/results"
PLOTS_DIR     = "/kaggle/working/retail_forecasting/outputs/plots"
SRC_DIR       = "/kaggle/working/retail_forecasting/src"


# ── Date splits (exactly as paper: train Oct2014–Jan2017,
#    test Feb–Jun 2017) ────────────────────────────────────────
TRAIN_START = "2013-01-01"   # Rossmann data starts here
TRAIN_END   = "2015-03-31"   # Rossmann public train ends here
TEST_START  = "2015-04-01"   # we use last ~5 months as test
TEST_END    = "2015-06-30"   # Rossmann test period

# ── Seasonality ──────────────────────────────────────────────
SEASON_M      = 7            # weekly seasonality

# ── Lag feature days (paper uses 2–7 and 14, skips lag 1
#    because yesterday's data unavailable at planning time) ───
LAG_DAYS      = [2, 3, 4, 5, 6, 7, 14]

# ── Rolling median window (paper uses 4 weeks) ───────────────
ROLLING_WEEKS = 4

# ── Classification bins (paper uses 124) ─────────────────────
N_BINS        = 124

# ── Ensemble size (paper uses 50, we use 10 for dev speed) ───
N_ENSEMBLE    = 10

# ── Cross-validation folds (paper uses 10) ───────────────────
N_CV_FOLDS    = 10

# ── Target scaling range (paper: log then scale to
#    [−0.5, 0.5] for ANNs) ─────────────────────────────────────
SCALE_MIN     = -0.5
SCALE_MAX     =  0.5

# ── Special day type labels ───────────────────────────────────
SD_TYPES      = [0, 1, 2, 3, 4]
# 0 = regular day
# 1 = public holiday or major event (t)
# 2 = day before public holiday (t-1)
# 3 = day after public holiday  (t+1)
# 4 = one week after holiday    (t+7) — sanity check day

# ── Product categories (Rossmann equivalent) ─────────────────
STORE_TYPES   = ["a", "b", "c", "d"]

# ── Random seed for reproducibility ──────────────────────────
RANDOM_SEED   = 42
