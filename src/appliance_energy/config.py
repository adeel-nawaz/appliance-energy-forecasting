"""
Shared configuration for the appliance energy forecasting project.

Centralising paths and modelling constants here means every notebook,
script, and module refers to a single source of truth instead of
redefining them.
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

# src/appliance_energy/config.py -> parents[2] is the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
FORECAST_DIR = OUTPUT_DIR / "forecasts"
METRICS_DIR = OUTPUT_DIR / "metrics"
MODEL_DIR = OUTPUT_DIR / "model_objects"

REPORT_DIR = PROJECT_ROOT / "reports"

RAW_FILENAME = "energydata_complete.csv"
HOURLY_FILENAME = "appliance_hourly.csv"

RAW_DATA_PATH = RAW_DIR / RAW_FILENAME
HOURLY_DATA_PATH = PROCESSED_DIR / HOURLY_FILENAME

RAW_DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00374/energydata_complete.csv"
)

ALL_DIRS = [
    RAW_DIR, INTERIM_DIR, PROCESSED_DIR,
    FIGURE_DIR, FORECAST_DIR, METRICS_DIR, MODEL_DIR,
]


def ensure_dirs():
    """Create every project data/output directory if it doesn't exist."""
    for path in ALL_DIRS:
        path.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# Modelling constants
# ------------------------------------------------------------------

RANDOM_STATE = 0

TARGET = "Appliances"

# Hourly data: 24 observations = 1 day, 168 observations = 1 week.
DAILY_PERIOD = 24
WEEKLY_PERIOD = 168

# 24-hour-ahead forecasting task.
HORIZON = 24

# Recommended test period: final 14 days, on hourly data.
TEST_STEPS = 14 * 24

# Validation block sitting immediately before the test period, used to
# choose feature sets and hyper-parameters. Selecting on the test set is
# listed as data leakage in the brief, so model choices are made here.
VALID_STEPS = 14 * 24

# Candidate exogenous / covariate columns available in the raw dataset.
CANDIDATE_EXOG_COLS = [
    "T_out",
    "RH_out",
    "Windspeed",
    "Visibility",
    "Tdewpoint",
]

# Indoor sensor columns (temperature + humidity, rooms 1-9).
INDOOR_TEMP_COLS = [f"T{i}" for i in range(1, 10)]
INDOOR_HUMIDITY_COLS = [f"RH_{i}" for i in range(1, 10)]

SARIMAX_ORDER = (1, 0, 1)
SARIMAX_SEASONAL_ORDER = (1, 1, 1, DAILY_PERIOD)

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------
# Every lag is >= HORIZON. At a 24-hour horizon, a feature for the row
# at time t may only use target values from t - 24 or earlier: shorter
# lags refer to observations that have not been made yet at the forecast
# origin. See appliance_energy.features for the full explanation.
LAG_FEATURES = [24, 25, 26, 48, 72, 168]

# Rolling windows are applied after shifting the target by HORIZON, so
# these describe how far back the window extends beyond that shift.
ROLLING_WINDOWS = [24, 48, 168]

FEATURE_TABLE_PATH = PROCESSED_DIR / "feature_table.csv"

# ------------------------------------------------------------------
# Feature-based model
# ------------------------------------------------------------------
FEATURE_MODEL_CANDIDATES = ["xgboost", "histgb", "random_forest"]

FEATURE_MODEL_PATH = MODEL_DIR / "feature_model.pkl"
FOUNDATION_MODEL_NAME = "amazon/chronos-bolt-small"
FOUNDATION_FORECAST_PATH = FORECAST_DIR / "foundation_forecast.csv"
FEATURE_MODEL_SELECTION_PATH = METRICS_DIR / "feature_model_selection.json"
FEATURE_ABLATION_PATH = METRICS_DIR / "feature_group_ablation.csv"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "feature_importance.csv"

# ------------------------------------------------------------------
# SARIMAX order search
# ------------------------------------------------------------------
# The assignment requires looping over every combination of
# p in [0, 6], d in [0, 2], q in [0, 6]  ->  7 * 3 * 7 = 147 models.

P_RANGE = range(0, 7)
D_RANGE = range(0, 3)
Q_RANGE = range(0, 7)

# Stage 2: seasonal grid. D=1 is fixed because the EDA showed a strong, 
# repeating daily cycle, so one seasonal difference is well justified.
SEASONAL_P_RANGE = range(0, 2)
SEASONAL_D_RANGE = range(1, 2)
SEASONAL_Q_RANGE = range(0, 2)

# How many stage-1 orders to carry into the (expensive) stage-2 search.
SEASONAL_REFINE_TOP_K = 3

SEASONAL_EXTRA_ORDERS = [(1, 0, 1), (2, 0, 1), (2, 0, 2)]

# The screening grid runs with a low iteration cap so that 147 fits stay
# affordable, but that leaves complex models short of convergence and
# their AIC untrustworthy. 

VERIFY_TOP_N = 15

SEARCH_MAXITER = 50
FINAL_MAXITER = 200

# Cached search results.
SARIMAX_STAGE1_PATH = METRICS_DIR / "sarimax_grid_search_stage1.csv"
SARIMAX_STAGE1_VERIFIED_PATH = METRICS_DIR / "sarimax_grid_search_stage1_verified.csv"
SARIMAX_STAGE2_PATH = METRICS_DIR / "sarimax_grid_search_stage2.csv"
SARIMAX_BEST_PARAMS_PATH = METRICS_DIR / "sarimax_best_params.json"
SARIMAX_MODEL_PATH = MODEL_DIR / "sarimax_best.pkl"
