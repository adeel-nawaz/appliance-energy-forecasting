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


# ------------------------------------------------------------------
# Comparison against the strongest benchmark
# ------------------------------------------------------------------

BENCHMARK_MODELS = [
    "mean", "naive", "drift", "seasonal_naive_daily", "seasonal_naive_weekly",
]


def strongest_benchmark(results, benchmark_names=None, metric="MASE"):
    """
    Identify the best-performing benchmark model in a results frame.

    Every advanced model should be judged against this, not merely
    against the other advanced models, a model that beats SARIMAX but
    loses to seasonal naive has not earned its complexity.
    """

    if benchmark_names is None:
        benchmark_names = BENCHMARK_MODELS

    present = results[results["model"].isin(benchmark_names)]

    if present.empty:
        raise ValueError(
            f"No benchmark models found in results. Looked for: {benchmark_names}"
        )

    return present.sort_values(metric).iloc[0]


def skill_scores(results, baseline_model=None, metrics=("MAE", "RMSE", "MASE")):
    """
    Percentage improvement of each model over a baseline.

    A positive skill score means the model beats the baseline by that
    percentage on the metric, negative means it is worse.
    """

    if baseline_model is None:
        baseline_model = strongest_benchmark(results)["model"]

    baseline = results.loc[results["model"] == baseline_model].iloc[0]

    out = results[["model"]].copy()

    for metric in metrics:
        out[f"{metric}_improvement_%"] = (
            100.0 * (baseline[metric] - results[metric]) / baseline[metric]
        )

    out["beats_benchmark"] = out["MASE_improvement_%"] > 0
    out.attrs["baseline_model"] = baseline_model

    return out.reset_index(drop=True)


# ------------------------------------------------------------------
# Error diagnostics
# ------------------------------------------------------------------

def error_frame(forecasts, test):
    """Signed errors (forecast - actual) for every model, aligned to the test index."""

    return pd.DataFrame(
        {name: pred.reindex(test.index) - test for name, pred in forecasts.items()},
        index=test.index,
    )


def error_by_hour(forecasts, test, absolute=True):
    """
    Mean error by hour of day.

    Reveals 'when' each model fails, which aggregate metrics hide. most
    models here are accurate overnight and struggle during the evening
    demand peak.
    """

    errors = error_frame(forecasts, test)

    if absolute:
        errors = errors.abs()

    return errors.groupby(errors.index.hour).mean()


def error_by_step_ahead(forecasts, test, horizon=config.HORIZON, absolute=True):
    """
    Mean error by position within each rolling forecast block.

    IMPORTANT: with aligned 24-hour blocks, step position and hour of
    day are the same variable, so this returns `error_by_hour`'s numbers
    re-indexed. A rising profile does NOT show decay with forecast
    distance, later steps may just land on harder hours.

    Valid use is comparing models at a fixed step (all predict the same
    hour). Step `horizon` is best. every model is a full horizon ahead,
    so none has a shorter-horizon advantage. Separating horizon from
    time of day would need origins advancing by less than `horizon`.

    """

    errors = error_frame(forecasts, test)

    if absolute:
        errors = errors.abs()

    n_blocks = int(np.ceil(len(test) / horizon))
    step = np.tile(np.arange(1, horizon + 1), n_blocks)[: len(test)]

    return errors.groupby(step).mean().rename_axis("step_ahead")


def error_summary(forecasts, test):
    """
    Distribution of absolute errors per model.

    Median versus upper percentiles separates "usually accurate" from
    "never badly wrong", two very different properties that MAE and
    RMSE respectively reward.
    """

    errors = error_frame(forecasts, test).abs()

    return pd.DataFrame({
        "median": errors.median(),
        "p75": errors.quantile(0.75),
        "p90": errors.quantile(0.90),
        "p95": errors.quantile(0.95),
        "max": errors.max(),
    })


def residual_autocorrelation(forecasts, test, lags=(1, 24, 168)):
    """
    Autocorrelation of each model's forecast errors at selected lags.

    Errors from a well-specified forecast should be close to
    unpredictable. Strong autocorrelation at lag 24 means the model is
    still missing daily structure that could be misused.
    """

    errors = error_frame(forecasts, test)

    return pd.DataFrame(
        {f"lag_{lag}": errors.apply(lambda col: col.autocorr(lag)) for lag in lags}
    )
