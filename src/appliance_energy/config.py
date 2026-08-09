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
