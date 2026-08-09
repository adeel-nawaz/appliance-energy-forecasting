"""Benchmark forecasting models: mean, naive, seasonal naive, drift."""

import pandas as pd


def mean_forecast(y_train, horizon, index):
    """Forecast every step as the training-period mean."""
    return pd.Series(y_train.mean(), index=index, name="mean")


def naive_forecast(y_train, horizon, index):
    """Forecast every step as the last observed training value."""
    return pd.Series(y_train.iloc[-1], index=index, name="naive")


def seasonal_naive_forecast(y_train, horizon, index, seasonality):
    """
    Recursive seasonal naive forecast.

    Each step repeats the value from `seasonality` steps earlier,
    appending its own forecasts to history so that later steps within
    a horizon longer than the seasonality can still look back.

    For hourly data: seasonality=24 gives same hour yesterday,
    seasonality=168 gives same hour last week.
    """

    values = []
    history = list(y_train.values)

    for _ in range(horizon):
        values.append(history[-seasonality])
        history.append(values[-1])

    return pd.Series(values, index=index)


def drift_forecast(y_train, horizon, index):
    """Linear extrapolation of the slope from the first to last training value."""

    slope = (y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)

    values = [y_train.iloc[-1] + slope * step for step in range(1, horizon + 1)]

    return pd.Series(values, index=index, name="drift")
