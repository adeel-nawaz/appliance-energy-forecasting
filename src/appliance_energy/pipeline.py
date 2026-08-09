"""
Main orchestration pipeline.

This module is built up incrementally as each model class is added to
the project. Currently implemented: data preparation and benchmark
models. SARIMAX, the feature-based model, and the foundation model are
wired in at later stages (see README "Running the pipeline").
"""

import pandas as pd

from appliance_energy import config, data, evaluation, plotting
from appliance_energy.models import benchmarks


def run_pipeline():
    config.ensure_dirs()

    hourly = data.load_hourly_data()
    y = hourly[config.TARGET]

    train, test = data.train_test_split_series(y, test_steps=config.TEST_STEPS)
    horizon = len(test)

    print("Train period:", train.index.min(), "to", train.index.max())
    print("Test period:", test.index.min(), "to", test.index.max())

    forecasts = {}

    forecasts["mean"] = benchmarks.mean_forecast(train, horizon, test.index)
    forecasts["naive"] = benchmarks.naive_forecast(train, horizon, test.index)
    forecasts["seasonal_naive_daily"] = benchmarks.seasonal_naive_forecast(
        train, horizon, test.index, seasonality=config.DAILY_PERIOD
    )
    forecasts["seasonal_naive_weekly"] = benchmarks.seasonal_naive_forecast(
        train, horizon, test.index, seasonality=config.WEEKLY_PERIOD
    )
    forecasts["drift"] = benchmarks.drift_forecast(train, horizon, test.index)

    # TODO (next stage): SARIMAX forecasts, via appliance_energy.models.sarimax
    # TODO (next stage): feature-based model, via appliance_energy.models.feature_models
    # TODO (next stage): foundation model, via appliance_energy.models.foundation

    results_df = evaluation.evaluate_all(
        forecasts, test, train, seasonality=config.DAILY_PERIOD
    )

    print("\nModel comparison:")
    print(results_df.round(3))

    forecast_df = pd.DataFrame({"actual": test})
    for name, pred in forecasts.items():
        forecast_df[name] = pred.reindex(test.index)

    forecast_df.to_csv(config.FORECAST_DIR / "all_forecasts.csv")
    results_df.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)

    fig = plotting.plot_forecasts(train, test, forecast_df)
    plotting.save_fig(fig, config.FIGURE_DIR / "forecast_comparison.png")

    return results_df, forecast_df
