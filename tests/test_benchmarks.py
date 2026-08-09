import numpy as np
import pandas as pd
import pytest

from appliance_energy.models import benchmarks


def _make_train_series(n=200):
    index = pd.date_range("2020-01-01", periods=n, freq="h")
    values = np.arange(n, dtype=float)
    return pd.Series(values, index=index)


def _test_index(train, horizon):
    return pd.date_range(
        train.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="h"
    )


@pytest.mark.parametrize("horizon", [1, 24, 48])
def test_forecast_length_matches_horizon(horizon):
    train = _make_train_series()
    index = _test_index(train, horizon)

    for forecast_fn in [
        benchmarks.mean_forecast,
        benchmarks.naive_forecast,
        benchmarks.drift_forecast,
    ]:
        pred = forecast_fn(train, horizon, index)
        assert len(pred) == horizon
        assert list(pred.index) == list(index)

    seasonal_pred = benchmarks.seasonal_naive_forecast(train, horizon, index, seasonality=24)
    assert len(seasonal_pred) == horizon
    assert list(seasonal_pred.index) == list(index)


def test_mean_forecast_is_constant_training_mean():
    train = _make_train_series()
    index = _test_index(train, 24)

    pred = benchmarks.mean_forecast(train, 24, index)

    assert (pred == train.mean()).all()


def test_naive_forecast_repeats_last_value():
    train = _make_train_series()
    index = _test_index(train, 24)

    pred = benchmarks.naive_forecast(train, 24, index)

    assert (pred == train.iloc[-1]).all()


def test_seasonal_naive_forecast_matches_lagged_value_within_seasonality():
    train = _make_train_series()
    horizon = 24
    seasonality = 24
    index = _test_index(train, horizon)

    pred = benchmarks.seasonal_naive_forecast(train, horizon, index, seasonality=seasonality)

    # For a horizon equal to the seasonality, every forecast step should
    # equal the training value exactly `seasonality` steps before it.
    expected = train.iloc[-seasonality:].values
    assert np.allclose(pred.values, expected)


def test_seasonal_naive_forecast_recurses_beyond_one_season():
    train = _make_train_series()
    seasonality = 24
    horizon = seasonality + 5
    index = _test_index(train, horizon)

    pred = benchmarks.seasonal_naive_forecast(train, horizon, index, seasonality=seasonality)

    # Steps beyond the first season should reuse forecasted values, not
    # raw training data (since training data alone can't reach that far).
    assert pred.iloc[seasonality] == pred.iloc[0]


def test_drift_forecast_extrapolates_constant_slope():
    train = _make_train_series()
    horizon = 10
    index = _test_index(train, horizon)

    pred = benchmarks.drift_forecast(train, horizon, index)

    slope = (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)
    expected = [train.iloc[-1] + slope * step for step in range(1, horizon + 1)]

    assert np.allclose(pred.values, expected)
