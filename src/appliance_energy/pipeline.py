"""
Main orchestration pipeline.

Evaluation design
-----------------
The forecasting task is 24 hours ahead and the test period is the final
14 days. Every model is therefore evaluated with rolling-origin
(walk-forward) forecasting: forecast 24 hours, reveal the realised
values, advance the origin, repeat. This keeps all models on an
identical footing and measures the 24-hour accuracy the task actually
asks about, rather than 336-step-ahead accuracy.

This module is built up incrementally as each model class is added.
Currently implemented: data preparation, benchmarks, and SARIMAX. The
feature-based and foundation models are wired in at later stages.
"""

import pandas as pd

from appliance_energy import backtest, config, data, evaluation, plotting
from appliance_energy.models import benchmarks, sarimax


def build_benchmark_forecasts(train, test, horizon=config.HORIZON):
    """Rolling-origin forecasts for every benchmark model."""

    forecasts = {}

    forecasts["mean"] = backtest.rolling_origin_forecast(
        benchmarks.mean_forecast, train, test, horizon=horizon, name="mean"
    )
    forecasts["naive"] = backtest.rolling_origin_forecast(
        benchmarks.naive_forecast, train, test, horizon=horizon, name="naive"
    )
    forecasts["seasonal_naive_daily"] = backtest.rolling_origin_forecast(
        benchmarks.seasonal_naive_forecast, train, test, horizon=horizon,
        name="seasonal_naive_daily", seasonality=config.DAILY_PERIOD,
    )
    forecasts["seasonal_naive_weekly"] = backtest.rolling_origin_forecast(
        benchmarks.seasonal_naive_forecast, train, test, horizon=horizon,
        name="seasonal_naive_weekly", seasonality=config.WEEKLY_PERIOD,
    )
    forecasts["drift"] = backtest.rolling_origin_forecast(
        benchmarks.drift_forecast, train, test, horizon=horizon, name="drift"
    )

    return forecasts


def build_sarimax_forecast(train, test, horizon=config.HORIZON, use_cached=True):
    """
    Rolling-origin SARIMAX forecast.

    Reuses the model cached by `scripts/run_sarimax_search.py` when
    available, since a seasonal fit at period 24 takes several minutes.
    Returns the forecast frame (point forecast plus interval) and the
    fitted results object.
    """

    order, seasonal_order = sarimax.load_best_params()

    if use_cached and config.SARIMAX_MODEL_PATH.exists():
        print(f"Loading cached SARIMAX model from {config.SARIMAX_MODEL_PATH}")
        fit = sarimax.load_model()
    else:
        print(f"Fitting SARIMA{order}x{seasonal_order} (this can take several minutes) ...")
        fit = sarimax.fit_sarimax(train, order=order, seasonal_order=seasonal_order)
        sarimax.save_model(fit)

    forecast_df = sarimax.rolling_forecast_sarimax(
        fit, test, horizon=horizon, name="sarimax"
    )

    return forecast_df, fit


def run_pipeline(horizon=config.HORIZON, use_cached_sarimax=True):
    config.ensure_dirs()

    hourly = data.load_hourly_data()
    y = hourly[config.TARGET].asfreq("h")

    train, test = data.train_test_split_series(y, test_steps=config.TEST_STEPS)

    print("Train period:", train.index.min(), "to", train.index.max())
    print("Test period: ", test.index.min(), "to", test.index.max())
    print(f"Rolling-origin evaluation, horizon = {horizon} hours "
          f"({len(backtest.rolling_origin_indices(test, horizon))} blocks)\n")

    forecasts = build_benchmark_forecasts(train, test, horizon=horizon)

    sarimax_df, sarimax_fit = build_sarimax_forecast(
        train, test, horizon=horizon, use_cached=use_cached_sarimax
    )
    forecasts["sarimax"] = sarimax_df["sarimax"]

    # TODO (next stage): feature-based model, via models.feature_models
    # TODO (next stage): foundation model, via models.foundation

    results_df = evaluation.evaluate_all(
        forecasts, test, train, seasonality=config.DAILY_PERIOD
    )

    print("\nModel comparison (rolling-origin, 24-hour horizon):")
    print(results_df.round(3).to_string(index=False))

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(config.FORECAST_DIR / "all_forecasts.csv")
    results_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    # Keep the SARIMAX prediction intervals alongside the point forecasts.
    sarimax_df.to_csv(config.FORECAST_DIR / "sarimax_forecast_intervals.csv")

    fig = plotting.plot_forecasts(train, test, forecast_df)
    plotting.save_fig(fig, config.FIGURE_DIR / "forecast_comparison.png")

    print("\nSaved outputs:")
    print(" ", config.FORECAST_DIR / "all_forecasts.csv")
    print(" ", config.METRICS_DIR / "model_comparison.csv")
    print(" ", config.FIGURE_DIR / "forecast_comparison.png")

    return results_df, forecast_df
