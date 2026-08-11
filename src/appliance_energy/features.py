"""
Feature engineering: calendar, lag, rolling, and sensor covariates.

Horizon safety (the key constraint here)
----------------------------------------
The task is a 24-hour-ahead forecast. At forecast origin `o` we predict
`o+1 ... o+24`. A feature attached to the row at time `t` may 
only use target values from `t - horizon` or earlier: when `t = o + 24`,
anything more recent than `t - 24 = o` has not happened yet.

This rules out the short lags (`lag_1`, `lag_2`, ...) that a naive
feature table would include. They are legitimate for one-step-ahead
forecasting, but for a 24-hour horizon they silently leak future
information and inflate test scores.

Every builder here takes a `horizon` and refuses to emit an unsafe
feature unless `allow_leaky=True` is passed explicitly, which exists
only so the effect can be demonstrated and quantified.

Covariate availability
----------------------
Features also differ in whether they would be *knowable* in deployment:

* `calendar`  - known indefinitely in advance (a true forecast input).
* `lag` / `rolling` - known, provided they respect the horizon rule.
* `indoor` / `outdoor` - measured sensor and weather readings. Their
  future values are NOT known at the forecast origin. Using realised
  test-set values makes the result a *conditional* forecast.

`FEATURE_GROUPS` records this so the distinction can be tested.
"""

import numpy as np
import pandas as pd

from appliance_energy import config

# Synthetic random columns shipped with the UCI dataset. They carry no
# information and are excluded by default, but are useful as a sanity
# check on feature importance (a good model should rank them near zero).
RANDOM_COLS = ["rv1", "rv2"]

CALENDAR_COLS = [
    "hour", "dayofweek", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

OUTDOOR_COLS = [
    "T_out", "RH_out", "Windspeed", "Visibility", "Tdewpoint", "Press_mm_hg",
]

# `lights` is a second energy meter (lighting circuit), not a sensor
# reading. It is highly informative about occupancy, but like the sensor
# columns its future values are unknown at the forecast origin.
HOUSE_COLS = ["lights"]

DERIVED_SENSOR_COLS = [
    "indoor_temp_mean", "indoor_temp_std", "indoor_temp_range",
    "indoor_rh_mean", "indoor_rh_std",
    "indoor_outdoor_temp_diff",
]


# ------------------------------------------------------------------
# Calendar features
# ------------------------------------------------------------------

def add_time_features(df):
    """
    Time-of-day and day-of-week features derived from the index.

    Hour and day-of-week are also encoded as sine/cosine pairs so that
    the model sees them as cyclical: hour 23 and hour 0 are adjacent,
    which a raw integer encoding cannot express.
    """

    out = df.copy()

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)

    return out


# ------------------------------------------------------------------
# Lag and rolling features
# ------------------------------------------------------------------

def horizon_safe_lags(lags, horizon=config.HORIZON):
    """Return only the lags that are computable `horizon` steps ahead."""
    return [lag for lag in lags if lag >= horizon]


def add_lag_features(df, target=config.TARGET, lags=None,
                     horizon=config.HORIZON, allow_leaky=False):
    """
    Add lagged target features.

    Lags shorter than `horizon` are dropped, because at a 24-hour
    horizon they refer to observations that have not been made yet.
    Pass `allow_leaky=True` to keep them anyway (for demonstrating the
    size of the leakage effect, not for reporting results).
    """

    if lags is None:
        lags = config.LAG_FEATURES

    if not allow_leaky:
        unsafe = [lag for lag in lags if lag < horizon]
        if unsafe:
            print(f"Dropping lags {unsafe}: shorter than the {horizon}-step "
                  f"horizon, so they would use unobserved values.")
        lags = horizon_safe_lags(lags, horizon)

    out = df.copy()

    for lag in lags:
        out[f"lag_{lag}"] = out[target].shift(lag)

    return out


def add_rolling_features(df, target=config.TARGET, windows=None,
                         horizon=config.HORIZON, allow_leaky=False):
    """
    Add rolling mean and standard deviation of the target.

    The series is shifted by `horizon` *before* the rolling window is
    applied, so a window ending at row `t` covers `t - horizon` and
    earlier only. Shifting by just one step -- as is common in
    one-step-ahead setups -- would leak at a 24-hour horizon.
    """

    if windows is None:
        windows = config.ROLLING_WINDOWS

    shift = 1 if allow_leaky else horizon

    out = df.copy()
    shifted = out[target].shift(shift)

    for window in windows:
        out[f"roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"roll_std_{window}"] = shifted.rolling(window).std()

    return out


# ------------------------------------------------------------------
# Sensor features
# ------------------------------------------------------------------

def add_sensor_aggregates(df):
    """
    Summarise the nine indoor sensors into house-level features.

    The individual room readings are strongly correlated with one
    another, so aggregates (mean, spread) capture most of the signal in
    far fewer columns, and the indoor/outdoor gradient is a more
    physically meaningful quantity than either reading alone.
    """

    out = df.copy()

    temp_cols = [c for c in config.INDOOR_TEMP_COLS if c in out.columns]
    rh_cols = [c for c in config.INDOOR_HUMIDITY_COLS if c in out.columns]

    # Spread statistics need at least two sensors: the standard deviation
    # of a single column is NaN for every row, which would silently empty
    # the table once missing rows are dropped.
    if temp_cols:
        out["indoor_temp_mean"] = out[temp_cols].mean(axis=1)

        if len(temp_cols) > 1:
            out["indoor_temp_std"] = out[temp_cols].std(axis=1)
            out["indoor_temp_range"] = (
                out[temp_cols].max(axis=1) - out[temp_cols].min(axis=1)
            )

    if rh_cols:
        out["indoor_rh_mean"] = out[rh_cols].mean(axis=1)

        if len(rh_cols) > 1:
            out["indoor_rh_std"] = out[rh_cols].std(axis=1)

    if temp_cols and "T_out" in out.columns:
        out["indoor_outdoor_temp_diff"] = out["indoor_temp_mean"] - out["T_out"]

    return out


# ------------------------------------------------------------------
# Assembly
# ------------------------------------------------------------------

def build_feature_table(df, target=config.TARGET, horizon=config.HORIZON,
                        lags=None, windows=None, allow_leaky=False,
                        include_random=False, dropna=True):
    """
    Build the full supervised-learning table.

    Combines calendar, lag, rolling, indoor sensor, and outdoor weather
    features. Rows containing NaNs introduced by the longest lag or
    rolling window are dropped by default, so the table is ready for a
    model that cannot accept missing values.
    """

    out = add_time_features(df)
    out = add_sensor_aggregates(out)
    out = add_lag_features(out, target=target, lags=lags, horizon=horizon,
                           allow_leaky=allow_leaky)
    out = add_rolling_features(out, target=target, windows=windows,
                               horizon=horizon, allow_leaky=allow_leaky)

    if not include_random:
        out = out.drop(columns=[c for c in RANDOM_COLS if c in out.columns])

    if dropna:
        out = out.dropna()

    return out


def feature_groups(columns, target=config.TARGET):
    """
    Map each feature column to its availability group.

    Returns a dict of {group_name: [columns]}, used both for the
    feature-group ablation and for reasoning about which covariates
    would genuinely be known at the forecast origin.
    """

    columns = [c for c in columns if c != target]

    groups = {
        "calendar": [c for c in columns if c in CALENDAR_COLS],
        "lag": [c for c in columns if c.startswith("lag_")],
        "rolling": [c for c in columns if c.startswith("roll_")],
        "indoor": [
            c for c in columns
            if c in config.INDOOR_TEMP_COLS
            or c in config.INDOOR_HUMIDITY_COLS
            or c in DERIVED_SENSOR_COLS
        ],
        "outdoor": [c for c in columns if c in OUTDOOR_COLS],
        "house": [c for c in columns if c in HOUSE_COLS],
    }

    groups = {name: cols for name, cols in groups.items() if cols}

    assigned = {c for cols in groups.values() for c in cols}
    other = [c for c in columns if c not in assigned]

    if other:
        groups["other"] = other

    return groups


# Groups whose future values are genuinely known at the forecast origin.
# Anything outside this set makes the result a conditional forecast.
KNOWN_AT_ORIGIN = ["calendar", "lag", "rolling"]


def known_at_origin_columns(columns, target=config.TARGET):
    """Columns that would genuinely be available in an operational forecast."""

    groups = feature_groups(columns, target=target)

    return [c for group in KNOWN_AT_ORIGIN for c in groups.get(group, [])]
