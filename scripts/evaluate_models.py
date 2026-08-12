# scripts/evaluate_models.py
#
# Evaluate every saved forecast: accuracy metrics, comparison against
# the strongest benchmark, error diagnostics, and figures.
#
# Reads outputs/forecasts/all_forecasts.csv (written by run_pipeline.py)
# so it can be re-run without refitting any model.
#
# Usage:
#   python scripts/evaluate_models.py

import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from appliance_energy import config, data, evaluation, plotting


def load_forecasts(path=None):
    """Load the saved forecast table and split it into actual + model columns."""

    path = path or (config.FORECAST_DIR / "all_forecasts.csv")

    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/run_pipeline.py` first."
        )

    frame = pd.read_csv(path, index_col=0, parse_dates=True)

    actual = frame["actual"]
    forecasts = {col: frame[col] for col in frame.columns if col != "actual"}

    return actual, forecasts


def main():
    config.ensure_dirs()

    hourly = data.load_hourly_data()
    y = hourly[config.TARGET].asfreq("h")
    train, _ = data.train_test_split_series(y, test_steps=config.TEST_STEPS)

    actual, forecasts = load_forecasts()

    print(f"Evaluating {len(forecasts)} models over {len(actual)} test hours")
    print(f"Test period: {actual.index.min()} to {actual.index.max()}\n")

    # ---- accuracy metrics -------------------------------------------
    results = evaluation.evaluate_all(
        forecasts, actual, train, seasonality=config.DAILY_PERIOD)

    print("Accuracy metrics (sorted by MASE):")
    print(results.round(3).to_string(index=False))

    # ---- comparison against the strongest benchmark -----------------
    baseline = evaluation.strongest_benchmark(results)
    skill = evaluation.skill_scores(results, baseline_model=baseline["model"])

    print(f"\nStrongest benchmark: {baseline['model']} (MASE {baseline['MASE']:.3f})")
    print("\nImprovement over that benchmark:")
    print(skill.round(2).to_string(index=False))

    # ---- error diagnostics ------------------------------------------
    errors = evaluation.error_frame(forecasts, actual)
    hourly_errors = evaluation.error_by_hour(forecasts, actual)
    step_errors = evaluation.error_by_step_ahead(
        forecasts, actual, horizon=config.HORIZON)
    summary = evaluation.error_summary(forecasts, actual)
    autocorr = evaluation.residual_autocorrelation(forecasts, actual)

    print("\nAbsolute error percentiles:")
    print(summary.round(2).to_string())

    print("\nForecast-error autocorrelation:")
    print(autocorr.round(3).to_string())

    # ---- figures -----------------------------------------------------
    fig = plotting.plot_error_diagnostics(errors, hourly_errors, step_errors, summary)
    plotting.save_fig(fig, config.FIGURE_DIR / "error_diagnostics.png")

    fig = plotting.plot_metric_comparison(results, baseline_mase=baseline["MASE"])
    plotting.save_fig(fig, config.FIGURE_DIR / "metric_comparison.png")

    fig = plotting.plot_skill_scores(skill)
    plotting.save_fig(fig, config.FIGURE_DIR / "skill_scores.png")

    # ---- persist -----------------------------------------------------
    results.to_csv(config.METRICS_DIR / "model_comparison.csv", index=False)
    skill.to_csv(config.METRICS_DIR / "skill_vs_benchmark.csv", index=False)
    summary.to_csv(config.METRICS_DIR / "error_percentiles.csv")
    hourly_errors.to_csv(config.METRICS_DIR / "error_by_hour.csv")
    step_errors.to_csv(config.METRICS_DIR / "error_by_step_ahead.csv")
    autocorr.to_csv(config.METRICS_DIR / "error_autocorrelation.csv")

    print("\nSaved metrics and diagnostic figures.")

    return results, skill


if __name__ == "__main__":
    main()
