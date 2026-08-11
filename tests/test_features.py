import numpy as np
import pandas as pd
import pytest

from appliance_energy import config, features


@pytest.fixture
def hourly_frame():
    """A small hourly frame with a known target and a couple of sensors."""
    index = pd.date_range("2020-01-06", periods=24 * 21, freq="h")  # starts Monday
    rng = np.random.RandomState(0)

    return pd.DataFrame(
        {
            config.TARGET: np.arange(len(index), dtype=float),
            "T1": rng.normal(20, 1, len(index)),
            "T2": rng.normal(21, 1, len(index)),
            "RH_1": rng.normal(40, 2, len(index)),
            "RH_2": rng.normal(42, 2, len(index)),
            "T_out": rng.normal(5, 3, len(index)),
        },
        index=index,
    )


# ------------------------------------------------------------------
# Calendar features
# ------------------------------------------------------------------

def test_time_features_match_the_index(hourly_frame):
    out = features.add_time_features(hourly_frame)

    assert (out["hour"] == out.index.hour).all()
    assert (out["dayofweek"] == out.index.dayofweek).all()
    assert (out["is_weekend"] == (out.index.dayofweek >= 5).astype(int)).all()


def test_cyclical_encodings_wrap_around(hourly_frame):
    out = features.add_time_features(hourly_frame)

    # Hour 0 and hour 23 must be neighbours in sin/cos space, which is
    # the whole point of the cyclical encoding.
    h0 = out[out["hour"] == 0].iloc[0]
    h23 = out[out["hour"] == 23].iloc[0]
    h12 = out[out["hour"] == 12].iloc[0]

    dist_0_23 = np.hypot(h0["hour_sin"] - h23["hour_sin"], h0["hour_cos"] - h23["hour_cos"])
    dist_0_12 = np.hypot(h0["hour_sin"] - h12["hour_sin"], h0["hour_cos"] - h12["hour_cos"])

    assert dist_0_23 < dist_0_12


def test_cyclical_encodings_lie_on_unit_circle(hourly_frame):
    out = features.add_time_features(hourly_frame)

    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)
    assert np.allclose(out["dow_sin"] ** 2 + out["dow_cos"] ** 2, 1.0)


# ------------------------------------------------------------------
# Lag features: leakage
# ------------------------------------------------------------------

def test_lag_features_use_only_past_target_values(hourly_frame):
    out = features.add_lag_features(hourly_frame, lags=[24, 48], horizon=24)

    target = hourly_frame[config.TARGET]

    for lag in (24, 48):
        expected = target.shift(lag)
        assert out[f"lag_{lag}"].equals(expected)

    # And concretely: the lag_24 value at row t equals the target 24 rows back.
    t = out.index[100]
    assert out.loc[t, "lag_24"] == target.loc[out.index[100 - 24]]


def test_sub_horizon_lags_are_dropped(hourly_frame):
    out = features.add_lag_features(
        hourly_frame, lags=[1, 2, 12, 24, 168], horizon=24
    )

    for unsafe in ("lag_1", "lag_2", "lag_12"):
        assert unsafe not in out.columns

    for safe in ("lag_24", "lag_168"):
        assert safe in out.columns


def test_sub_horizon_lags_kept_only_when_explicitly_allowed(hourly_frame):
    out = features.add_lag_features(
        hourly_frame, lags=[1, 24], horizon=24, allow_leaky=True
    )

    assert "lag_1" in out.columns


def test_horizon_safe_lags_filters_correctly():
    assert features.horizon_safe_lags([1, 6, 23, 24, 25, 168], horizon=24) == [24, 25, 168]
    assert features.horizon_safe_lags([1, 2, 3], horizon=24) == []


# ------------------------------------------------------------------
# Rolling features: leakage
# ------------------------------------------------------------------

def test_rolling_features_exclude_the_horizon_window(hourly_frame):
    horizon = 24
    window = 24

    out = features.add_rolling_features(
        hourly_frame, windows=[window], horizon=horizon
    )

    target = hourly_frame[config.TARGET]
    expected = target.shift(horizon).rolling(window).mean()

    assert out[f"roll_mean_{window}"].equals(expected)


def test_rolling_mean_never_includes_current_or_recent_values(hourly_frame):
    # The target is 0, 1, 2, ... so a rolling mean is easy to verify by hand.
    horizon, window = 24, 24
    out = features.add_rolling_features(hourly_frame, windows=[window], horizon=horizon)

    position = 100
    t = out.index[position]

    # Window must cover positions [100-24-24+1, 100-24] = [53, 76].
    expected = np.mean(np.arange(position - horizon - window + 1, position - horizon + 1))

    assert out.loc[t, f"roll_mean_{window}"] == pytest.approx(expected)
    # Strictly less than the value 24 steps back, proving nothing recent leaked in.
    assert out.loc[t, f"roll_mean_{window}"] < position - horizon + 1


def test_no_feature_correlates_perfectly_with_the_target(hourly_frame):
    # A blunt leakage smoke test: on a monotonic target, any feature that
    # accidentally included the current value would show up here.
    table = features.build_feature_table(hourly_frame, horizon=24)

    target = table[config.TARGET]
    lag_and_roll = [c for c in table.columns if c.startswith(("lag_", "roll_"))]

    for col in lag_and_roll:
        assert not np.allclose(table[col].values, target.values)


# ------------------------------------------------------------------
# Sensor aggregates
# ------------------------------------------------------------------

def test_sensor_aggregates_are_computed_from_available_columns(hourly_frame):
    out = features.add_sensor_aggregates(hourly_frame)

    assert np.allclose(out["indoor_temp_mean"], hourly_frame[["T1", "T2"]].mean(axis=1))
    assert np.allclose(out["indoor_rh_mean"], hourly_frame[["RH_1", "RH_2"]].mean(axis=1))
    assert np.allclose(
        out["indoor_outdoor_temp_diff"], out["indoor_temp_mean"] - hourly_frame["T_out"]
    )


def test_single_sensor_does_not_produce_all_nan_spread_columns():
    # Regression test: std across one column is NaN for every row, which
    # would empty the whole table once NaN rows are dropped.
    index = pd.date_range("2020-01-01", periods=24 * 30, freq="h")
    frame = pd.DataFrame(
        {
            config.TARGET: np.arange(len(index), dtype=float),
            "T1": 20.0,
            "RH_1": 40.0,
        },
        index=index,
    )

    out = features.add_sensor_aggregates(frame)

    assert "indoor_temp_mean" in out.columns
    assert "indoor_temp_std" not in out.columns
    assert "indoor_rh_std" not in out.columns

    table = features.build_feature_table(frame, horizon=24)
    assert len(table) > 0


def test_sensor_aggregates_skip_missing_sensor_columns():
    index = pd.date_range("2020-01-01", periods=48, freq="h")
    frame = pd.DataFrame({config.TARGET: np.arange(48, dtype=float)}, index=index)

    out = features.add_sensor_aggregates(frame)

    assert "indoor_temp_mean" not in out.columns


# ------------------------------------------------------------------
# Table assembly and grouping
# ------------------------------------------------------------------

def test_feature_table_has_no_missing_values_and_keeps_target(hourly_frame):
    table = features.build_feature_table(hourly_frame, horizon=24)

    assert table.isna().sum().sum() == 0
    assert config.TARGET in table.columns
    assert len(table) < len(hourly_frame)  # warm-up rows dropped


def test_random_columns_excluded_by_default(hourly_frame):
    frame = hourly_frame.copy()
    frame["rv1"] = 1.0
    frame["rv2"] = 2.0

    assert "rv1" not in features.build_feature_table(frame, horizon=24).columns
    assert "rv1" in features.build_feature_table(frame, horizon=24, include_random=True).columns


def test_feature_groups_partition_all_columns(hourly_frame):
    table = features.build_feature_table(hourly_frame, horizon=24)
    groups = features.feature_groups(table.columns)

    grouped = [c for cols in groups.values() for c in cols]
    expected = [c for c in table.columns if c != config.TARGET]

    assert sorted(grouped) == sorted(expected)
    assert len(grouped) == len(set(grouped))  # no column in two groups


def test_known_at_origin_excludes_sensor_and_weather(hourly_frame):
    table = features.build_feature_table(hourly_frame, horizon=24)
    known = features.known_at_origin_columns(table.columns)

    # Calendar and lag features are genuinely known in advance.
    assert "hour_sin" in known
    assert "lag_24" in known

    # Realised sensor and weather readings are not.
    assert "T1" not in known
    assert "T_out" not in known
    assert "indoor_temp_mean" not in known
