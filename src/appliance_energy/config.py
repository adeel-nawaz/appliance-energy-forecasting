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
