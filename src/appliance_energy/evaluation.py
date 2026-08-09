"""Forecast evaluation metrics: MAE, RMSE, MASE, Bias."""

import numpy as np
import pandas as pd

from appliance_energy import config


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias(y_true, y_pred):
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def mase(y_true, y_pred, y_train, seasonality=config.DAILY_PERIOD):
    """
    Mean absolute scaled error, scaled by the in-sample seasonal naive
    forecast error (lag = `seasonality`). MASE < 1 means the model
    beats seasonal naive on the training period's own errors.
    """

    y_train = pd.Series(y_train).astype(float)

    seasonal_errors = np.abs(
        y_train.iloc[seasonality:].values - y_train.iloc[:-seasonality].values
    )
    scale = seasonal_errors.mean()

    if scale == 0:
        return np.nan

    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / scale)


def evaluate_forecast(name, y_true, y_pred, y_train, seasonality=config.DAILY_PERIOD):
    """Compute MAE, RMSE, MASE, and Bias for a single forecast."""

    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred, index=y_true.index).astype(float)

    return {
        "model": name,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MASE": mase(y_true, y_pred, y_train, seasonality=seasonality),
        "Bias": bias(y_true, y_pred),
    }


def evaluate_all(forecasts, test, train, seasonality=config.DAILY_PERIOD):
    """
    Evaluate a dict of {model_name: forecast_series} against a shared
    test set, aligning indices and dropping any unmatched points (e.g.
    a feature-based model whose lag features need a warm-up period).

    Returns a results dataframe sorted by MASE (ascending = better).
    """

    results = []

    for name, pred in forecasts.items():
        pred = pred.reindex(test.index)
        valid = pred.notna() & test.notna()

        results.append(
            evaluate_forecast(
                name=name,
                y_true=test.loc[valid],
                y_pred=pred.loc[valid],
                y_train=train,
                seasonality=seasonality,
            )
        )

    return (
        pd.DataFrame(results)
        .sort_values("MASE")
        .reset_index(drop=True)
    )
